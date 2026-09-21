#!/usr/bin/env python3
"""Build public correction opportunities from current-policy one-candidate banks.

This stage opens only the public task root. It executes the current policy's one
candidate on the four public tests and writes failed public histories for correction.
It never opens evaluator-only tests or reference implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pbpf.apbpf.codearc_bank import file_sha, load_bank
from pbpf.apbpf.codearc_execution import execute_call
from pbpf.apbpf.rbr_execution import execute_stdin


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["rbr", "codearc"], required=True)
    p.add_argument("--public-root", type=Path, required=True)
    p.add_argument("--bank", type=Path, action="append", required=True,
                   help="one-candidate train/development banks; pass once per split")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=6.0)
    args = p.parse_args()
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.output.exists():
        raise FileExistsError("public current-policy correction banks are create-once")

    root = args.public_root
    manifest = json.loads((root / "manifest.json").read_text())
    expected_schema = (
        "apbpf-rbr-generated-materialization-v1" if args.domain == "rbr"
        else "apbpf-codearc-replay-materialization-v1"
    )
    if manifest.get("schema") != expected_schema:
        raise ValueError("wrong public materialization")
    if file_sha(root / "tasks.jsonl") != manifest["public_tasks_sha256"]:
        raise ValueError("public task checksum mismatch")
    with (root / "tasks.jsonl").open() as stream:
        public_tasks = {row["task_id"]: row for row in map(json.loads, stream)}

    runs, rows, model_identity = [], [], None
    source_seen = set()
    for bank_path in args.bank:
        run, bank, complete_sha = load_bank(bank_path)
        if run.get("schema") != f"apbpf-{args.domain}-generation-v1":
            raise ValueError("bank domain differs")
        if run.get("split") not in {"train", "development"} or run.get("candidates") != 1:
            raise ValueError("recursive correction prep requires one-candidate train/development banks")
        identity = (run["model"], run["revision"], run.get("adapter_sha256"), run["seed"])
        if model_identity is None:
            model_identity = identity
        elif model_identity != identity:
            raise ValueError("cannot combine banks from different policy identities")
        if run["public_tasks_sha256"] != manifest["public_tasks_sha256"]:
            raise ValueError("bank and public task identities differ")
        runs.append({
            "path": str(bank_path),
            "complete_sha256": complete_sha,
            "split": run["split"],
        })
        for row in bank:
            source = row["source_component_id"]
            if source in source_seen:
                raise ValueError("source component duplicated across recursive splits")
            source_seen.add(source)
            if row["task_id"] not in public_tasks:
                raise ValueError("bank task is absent from public materialization")
            task = public_tasks[row["task_id"]]
            if task["source_component_id"] != source or task["split"] != row["split"]:
                raise ValueError("bank/public source binding mismatch")
            rows.append((run, row, task))

    if {run["split"] for run, _, _ in rows} != {"train", "development"}:
        raise ValueError("both train and development recursive banks are required")
    execute = execute_stdin if args.domain == "rbr" else execute_call
    prepared, excluded = [], []
    for run, row, task in rows:
        candidate = row["candidates"][0]
        public_tests = task["visible_tests"]
        if [str(t["id"]) for t in public_tests] != ["0", "1", "2", "3"]:
            raise ValueError("recursive EESD expects exactly four ordered public tests")
        measured = []
        for test in public_tests:
            result = execute(candidate["code"], test, timeout=args.timeout)
            measured.append({
                "id": str(test["id"]),
                "input": test["input"],
                "expected": test["expected"],
                **({"expected_error": bool(test["expected_error"])} if "expected_error" in test else {}),
                "actual": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "outcome": result["outcome"],
            })
        outcomes = [test["outcome"] for test in measured]
        if all(value == "PASS" for value in outcomes):
            excluded.append({
                "split": row["split"],
                "source_component_id": row["source_component_id"],
                "task_id": row["task_id"],
                "reason": "current_policy_passes_all_public_tests",
            })
            continue
        prepared.append({
            "schema": "eesd-public-correction-row-v1",
            "split": row["split"],
            "source_component_id": row["source_component_id"],
            "problem_id": row.get("problem_id", row["task_id"]),
            "task_id": row["task_id"],
            "task_text": task["task_text"],
            "candidate": candidate["code"],
            "candidate_code_sha256": hashlib.sha256(candidate["code"].encode()).hexdigest(),
            "tests": measured,
            "outcomes": outcomes,
            "selection": "single current-policy candidate; correction requested iff any public test fails",
            "generator": {
                "model": run["model"],
                "revision": run["revision"],
                "adapter_sha256": run.get("adapter_sha256"),
                "seed": run["seed"],
            },
        })

    if not prepared:
        raise ValueError("current policy produced no public correction opportunities")
    args.output.mkdir(parents=True)
    path = args.output / "public-corrections.jsonl"
    with path.open("x") as stream:
        for row in prepared:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    audit = {
        "schema": "eesd-recursive-public-bank-v1",
        "domain": args.domain,
        "model_identity": model_identity,
        "public_manifest_sha256": file_sha(root / "manifest.json"),
        "source_banks": runs,
        "rows": len(prepared),
        "split_counts": {
            split: sum(row["split"] == split for row in prepared)
            for split in ("train", "development")
        },
        "excluded": excluded,
        "public_corrections_sha256": sha(path),
        "evaluator_root_opened": False,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "excluded"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
