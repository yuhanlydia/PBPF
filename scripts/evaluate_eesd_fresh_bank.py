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


def work(job):
    domain, task_id, source, candidate_id, code, tests, timeout = job
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
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--timeout", type=float, default=6.0)
    args = p.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        raise ValueError("positive execution bounds required")
    if args.output.exists():
        raise FileExistsError("fresh evaluations are create-once")

    run, bank, bank_sha = load_bank(args.bank)
    expected_schema = f"apbpf-{args.domain}-generation-v1"
    if run.get("schema") != expected_schema or run.get("split") != "primary" or run.get("candidates") != 1:
        raise ValueError("fresh Pass@1 requires a one-candidate primary generation bank")

    root = args.evaluator_root
    manifest = json.loads((root / "manifest.json").read_text())
    expected_materialization = (
        "apbpf-rbr-generated-materialization-v1" if args.domain == "rbr"
        else "apbpf-codearc-materialization-v1"
    )
    if manifest.get("schema") != expected_materialization:
        raise ValueError("wrong evaluator materialization")
    if file_sha(root / "tasks.jsonl") != manifest["evaluator_tasks_sha256"]:
        raise ValueError("evaluator task checksum mismatch")
    if run["public_tasks_sha256"] != manifest["public_tasks_sha256"]:
        raise ValueError("generation and evaluator public task identities differ")

    with (root / "tasks.jsonl").open() as stream:
        tasks = {row["task_id"]: row for row in map(json.loads, stream) if row["split"] == "primary"}
    if {row["task_id"] for row in bank} != set(tasks):
        raise ValueError("fresh bank must cover the complete primary task population")

    jobs = []
    for row in bank:
        task = tasks[row["task_id"]]
        if task["source_component_id"] != row["source_component_id"]:
            raise ValueError("source-component mismatch")
        candidate = row["candidates"][0]
        jobs.append((
            args.domain, row["task_id"], row["source_component_id"],
            candidate["candidate_id"], candidate["code"], task["tests"], args.timeout,
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
        "task_manifest_sha256": file_sha(root / "manifest.json"),
        "sources": n,
        "fresh_all_tests_pass_at_1": sum(r["all_tests_pass"] for r in rows) / n,
        "fresh_hidden_pass_at_1": sum(r["hidden_all_pass"] for r in rows) / n,
        "visible_all_pass_rate": sum(r["visible_all_pass"] for r in rows) / n,
        "all_tests_passes": sum(r["all_tests_pass"] for r in rows),
        "hidden_passes": sum(r["hidden_all_pass"] for r in rows),
        "records": rows,
        "selection": "none; exactly one fresh candidate per primary source",
    }
    (args.output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
