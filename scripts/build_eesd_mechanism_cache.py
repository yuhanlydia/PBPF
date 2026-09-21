#!/usr/bin/env python3
"""Build a public-query/private-outcome EED mechanism cache from sealed candidates.

Candidate generation is already complete before this trusted evaluator opens the
full ten-test task materialization. The resulting cache exposes each test *input*
as a prediction query but redacts expected outputs, reference code, actual stdout,
and stderr. Outcomes are retained only as evaluation labels consumed by the
offline EED runner. This cache is a mechanism diagnostic, not a repair input.
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
    domain, row, task, timeout = job
    execute = execute_stdin if domain == "rbr" else execute_call
    candidate = row["candidates"][0]
    measured = [execute(candidate["code"], test, timeout=timeout) for test in task["tests"]]
    return {
        "task_id": candidate["candidate_id"],
        "problem_id": row["task_id"],
        "source_component_id": row["source_component_id"],
        "split": row["split"],
        "candidate_code_sha256": hashlib.sha256(candidate["code"].encode()).hexdigest(),
        "tests": [{"id": str(test["id"]), "input": test["input"]} for test in task["tests"]],
        "outcomes": [result["outcome"] for result in measured],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["rbr", "codearc"], required=True)
    p.add_argument("--evaluator-root", type=Path, required=True)
    p.add_argument("--bank", type=Path, action="append", required=True,
                   help="sealed one-candidate development/primary banks")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--timeout", type=float, default=6.0)
    args = p.parse_args()
    if args.workers < 1 or args.timeout <= 0:
        raise ValueError("positive execution bounds required")
    if args.output.exists():
        raise FileExistsError("mechanism caches are create-once")

    evaluator = args.evaluator_root
    manifest = json.loads((evaluator / "manifest.json").read_text())
    expected_schema = (
        "apbpf-rbr-generated-materialization-v1" if args.domain == "rbr"
        else "apbpf-codearc-replay-materialization-v1"
    )
    if manifest.get("schema") != expected_schema:
        raise ValueError("wrong evaluator materialization")
    if file_sha(evaluator / "tasks.jsonl") != manifest["evaluator_tasks_sha256"]:
        raise ValueError("evaluator task checksum mismatch")
    with (evaluator / "tasks.jsonl").open() as stream:
        tasks = {row["task_id"]: row for row in map(json.loads, stream)}

    jobs, bank_bindings, identities = [], [], set()
    source_seen = set()
    for bank_path in args.bank:
        run, bank, complete_sha = load_bank(bank_path)
        if (
            run.get("schema") != f"apbpf-{args.domain}-generation-v1"
            or run.get("split") not in {"development", "primary"}
            or run.get("candidates") != 1
        ):
            raise ValueError("mechanism builder requires sealed one-candidate development/primary banks")
        if run["public_tasks_sha256"] != manifest["public_tasks_sha256"]:
            raise ValueError("candidate generation and evaluator materialization differ")
        identities.add((run["model"], run["revision"], run.get("adapter_sha256"), run["seed"]))
        bank_bindings.append({
            "split": run["split"],
            "path": str(bank_path),
            "complete_sha256": complete_sha,
            "components": run["components"],
        })
        for row in bank:
            source = row["source_component_id"]
            if source in source_seen:
                raise ValueError("mechanism source appears in more than one split")
            source_seen.add(source)
            task = tasks.get(row["task_id"])
            if (
                task is None
                or task["source_component_id"] != source
                or task["split"] != row["split"]
                or len(task["tests"]) != 10
            ):
                raise ValueError("bank/evaluator source binding mismatch")
            jobs.append((args.domain, row, task, args.timeout))
    if len(identities) != 1 or {entry["split"] for entry in bank_bindings} != {"development", "primary"}:
        raise ValueError("one model identity and both mechanism splits are required")
    if not jobs:
        raise ValueError("empty mechanism population")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(work, jobs), 1):
            rows.append(row)
            if index % 100 == 0:
                print(json.dumps({"completed": index, "total": len(jobs)}), flush=True)

    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("development", "primary")
    }
    source_counts = {
        split: len({row["source_component_id"] for row in rows if row["split"] == split})
        for split in ("development", "primary")
    }
    payload = {
        "schema": "eesd-public-query-mechanism-cache-v1",
        "dataset": args.domain,
        "generator_identity": list(identities)[0],
        "tests_per_candidate": 10,
        "records": rows,
        "counts": counts,
        "source_counts": source_counts,
        "bank_bindings": bank_bindings,
        "evaluator_manifest_sha256": file_sha(evaluator / "manifest.json"),
        "visibility": (
            "candidate generator saw only four public tests; after candidate sealing, "
            "mechanism predictor receives all ten test inputs as queries; expected outputs, "
            "reference code, stdout/stderr, and future outcomes are not predictor features"
        ),
        "claim_scope": "public-query/private-outcome probability mechanism only; not hidden-test repair",
    }
    args.output.write_text(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
