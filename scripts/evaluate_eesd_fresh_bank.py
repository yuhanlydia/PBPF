#!/usr/bin/env python3
"""Evaluate a one-candidate fresh-policy bank on the full primary suite.

Generation must have completed before this evaluator sees the private task root.
No visible-result-based selection occurs: every source has exactly one candidate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from pbpf.apbpf.codearc_bank import file_sha, load_bank
from pbpf.apbpf.codearc_execution import execute_call
from pbpf.apbpf.rbr_execution import execute_stdin


def load_fresh_banks(paths, domain):
    runs = [load_bank(path) for path in paths]
    run, bank, first_sha = runs[0]
    expected_schema = f"apbpf-{domain}-generation-v1"
    if run.get("schema") != expected_schema or run.get("split") != "primary" or run.get("candidates") != 1:
        raise ValueError("fresh Pass@1 requires one-candidate primary generation banks")
    identity = ("schema", "split", "candidates", "model", "revision", "adapter_sha256",
                "public_tasks_sha256", "seed", "max_input_tokens", "max_new_tokens",
                "temperature", "top_p", "decode_policy")
    for shard, rows, _ in runs[1:]:
        if any(shard.get(key) != run.get(key) for key in identity):
            raise ValueError("fresh generation shards have different model or sampling identities")
        bank.extend(rows)
    if len({row["task_id"] for row in bank}) != len(bank):
        raise ValueError("duplicate task across fresh generation shards")
    return run, bank, first_sha, [digest for _, _, digest in runs]


def work(job):
    domain, task_id, source, candidate_id, code, tests, timeout, profile = job
    if profile == "direct-no-sandbox":
        from pbpf.apbpf.direct_execution import execute_stdin as direct_stdin, execute_call as direct_call
        execute = direct_stdin if domain == "rbr" else direct_call
    else:
        execute = execute_stdin if domain == "rbr" else execute_call
    results = [dict(test_id=str(test.get("id", i)), **execute(code, test, timeout=timeout))
               for i, test in enumerate(tests)]
    visible = [r for r in results if int(r["test_id"]) < 4]
    hidden = [r for r in results if int(r["test_id"]) >= 4]
    return {
        "task_id": task_id,
        "source_component_id": source,
        "candidate_id": candidate_id,
        "tests": results,
        "visible_all_pass": bool(visible) and all(r["outcome"] == "PASS" for r in visible),
        "hidden_all_pass": bool(hidden) and all(r["outcome"] == "PASS" for r in hidden),
        "all_tests_pass": all(r["outcome"] == "PASS" for r in results),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["rbr", "codearc"], required=True)
    p.add_argument("--evaluator-root", type=Path, required=True)
    p.add_argument("--bank", type=Path, action="append", required=True,
                   help="repeat for disjoint sealed primary generation shards")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--timeout", type=float, default=6.0)
    p.add_argument("--execution-profile", choices=["sandbox", "direct-no-sandbox"], default="sandbox")
    p.add_argument("--execution-lock", type=Path)
    p.add_argument("--execution-lock-sha256")
    args = p.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        raise ValueError("positive execution bounds required")
    if args.output.exists():
        raise FileExistsError("fresh evaluations are create-once")
    run, bank, bank_sha, bank_shas = load_fresh_banks(args.bank, args.domain)
    if args.execution_profile == "direct-no-sandbox":
        from pbpf.eesd.direct_runtime import probe_readiness
        if args.timeout != 6.0 or args.execution_lock is None or args.execution_lock_sha256 is None:
            raise ValueError("direct fresh evaluation requires the six-second execution lock")
        readiness = probe_readiness(args.execution_lock, args.execution_lock_sha256)
        if readiness.get("status") != "ready" or readiness.get("profile") != "direct-no-sandbox":
            raise ValueError("direct execution readiness failed")
    elif args.execution_lock is not None or args.execution_lock_sha256 is not None:
        raise ValueError("execution lock belongs only to the direct profile")

    root = args.evaluator_root
    manifest = json.loads((root / "manifest.json").read_text())
    expected_materialization = (
        "apbpf-rbr-generated-materialization-v1" if args.domain == "rbr"
        else "apbpf-codearc-replay-materialization-v1"
    )
    if manifest.get("schema") != expected_materialization:
        raise ValueError("wrong evaluator materialization")
    if file_sha(root / "tasks.jsonl") != manifest["evaluator_tasks_sha256"]:
        raise ValueError("evaluator task checksum mismatch")
    if run["public_tasks_sha256"] != manifest["public_tasks_sha256"]:
        raise ValueError("generation and evaluator public task identities differ")

    with (root / "tasks.jsonl").open() as stream:
        tasks = {row["task_id"]: row for row in map(json.loads, stream) if row["split"] == "primary"}
    expected_sources = {row["source_component_id"] for row in tasks.values()}
    bank_sources = [row["source_component_id"] for row in bank]
    if (
        len(bank_sources) != len(set(bank_sources))
        or set(bank_sources) != expected_sources
        or any(row["task_id"] not in tasks for row in bank)
    ):
        raise ValueError("fresh bank must contain exactly one representative for every primary source component")

    jobs = []
    for row in bank:
        task = tasks[row["task_id"]]
        if task["source_component_id"] != row["source_component_id"]:
            raise ValueError("source-component mismatch")
        candidate = row["candidates"][0]
        jobs.append((
            args.domain, row["task_id"], row["source_component_id"],
            candidate["candidate_id"], candidate["code"], task["tests"], args.timeout,
            args.execution_profile,
        ))

    args.output.mkdir(parents=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output / "records.jsonl").open("x") as stream:
        for index, row in enumerate(pool.map(work, jobs), 1):
            rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            if index % 25 == 0:
                print(json.dumps({"completed": index, "total": len(jobs)}), flush=True)

    n = len(rows)
    report = {
        "schema": "eesd-fresh-policy-eval-v1",
        "domain": args.domain,
        "model": run["model"],
        "revision": run["revision"],
        "adapter_sha256": run.get("adapter_sha256"),
        "bank_complete_sha256": bank_sha,
        "bank_complete_sha256s": bank_shas,
        "task_manifest_sha256": file_sha(root / "manifest.json"),
        "sources": n,
        "fresh_all_tests_pass_at_1": sum(r["all_tests_pass"] for r in rows) / n,
        "fresh_hidden_pass_at_1": sum(r["hidden_all_pass"] for r in rows) / n,
        "visible_all_pass_rate": sum(r["visible_all_pass"] for r in rows) / n,
        "all_tests_passes": sum(r["all_tests_pass"] for r in rows),
        "hidden_passes": sum(r["hidden_all_pass"] for r in rows),
        "records": rows,
        "selection": "none; exactly one fresh candidate per primary source",
        "execution_profile": args.execution_profile,
        "execution_lock_sha256": args.execution_lock_sha256,
    }
    (args.output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
