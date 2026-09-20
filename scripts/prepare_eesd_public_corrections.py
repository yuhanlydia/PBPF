#!/usr/bin/env python3
"""Prepare a public-only one-candidate-per-source correction bank.

This trusted preparation step may read a full execution cache, but it writes only
train/development task text, one selected candidate, and the first N public
executions. Future tests/outcomes and reference repairs are never written.
Candidate selection uses public outcomes only: among candidates that fail at least
one visible execution, choose the highest visible pass count, then task_id.
Sources for which every candidate passes all visible executions are recorded as
not requiring correction and excluded before any correction model is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--visible", type=int, default=4)
    args = p.parse_args()
    if args.visible < 1:
        raise ValueError("visible must be positive")

    payload = json.loads(args.cache.read_text())
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("execution cache requires nonempty records")

    groups = {}
    for row in records:
        split = row.get("original_split", row.get("split"))
        if split not in {"train", "development"}:
            continue
        tests, outcomes = row.get("tests"), row.get("outcomes")
        if (
            not isinstance(tests, list)
            or not isinstance(outcomes, list)
            or len(tests) != len(outcomes)
            or len(tests) < args.visible
            or "candidate" not in row
            or "task_text" not in row
        ):
            raise ValueError("cache row missing candidate/public execution fields")
        source = row.get("source_component_id", row.get("problem_id"))
        if not source:
            raise ValueError("source_component_id/problem_id required")
        groups.setdefault((split, source), []).append(row)

    args.output.mkdir(parents=True, exist_ok=False)
    rows, excluded = [], []
    for (split, source), candidates in sorted(groups.items()):
        repairable = []
        for row in candidates:
            visible_outcomes = row["outcomes"][: args.visible]
            passes = sum(value == "PASS" for value in visible_outcomes)
            if passes < args.visible:
                repairable.append((passes, str(row["task_id"]), row))
        if not repairable:
            excluded.append({"split": split, "source_component_id": source,
                             "reason": "all_candidates_pass_visible"})
            continue
        _, _, selected = sorted(repairable, key=lambda x: (-x[0], x[1]))[0]
        tests = []
        for i, test in enumerate(selected["tests"][: args.visible]):
            allowed = {
                "id": str(test.get("id", i)),
                "input": test.get("input", ""),
                "expected": test.get("expected", ""),
                "actual": test.get("actual", ""),
                "stderr": test.get("stderr", ""),
                "outcome": selected["outcomes"][i],
            }
            if "expected_error" in test:
                allowed["expected_error"] = bool(test["expected_error"])
            if not all(isinstance(allowed[k], str) for k in ("id","input","expected","actual","stderr","outcome")):
                raise ValueError("public execution text fields must be strings")
            tests.append(allowed)
        rows.append({
            "schema": "eesd-public-correction-row-v1",
            "split": split,
            "source_component_id": source,
            "problem_id": selected.get("problem_id"),
            "task_id": selected["task_id"],
            "task_text": selected["task_text"],
            "candidate": selected["candidate"],
            "candidate_code_sha256": hashlib.sha256(selected["candidate"].encode()).hexdigest(),
            "tests": tests,
            "outcomes": selected["outcomes"][: args.visible],
            "selection": "highest-public-pass-count-among-visible-failing-candidates; task_id tie-break",
        })

    if not rows:
        raise ValueError("no public correction opportunities")
    public_path = args.output / "public-corrections.jsonl"
    with public_path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    audit = {
        "schema": "eesd-public-correction-bank-v1",
        "source_cache_sha256": sha(args.cache),
        "visible": args.visible,
        "rows": len(rows),
        "sources": len({r["source_component_id"] for r in rows}),
        "split_counts": {s: sum(r["split"] == s for r in rows) for s in ("train","development")},
        "excluded": excluded,
        "public_bank_sha256": sha(public_path),
        "forbidden_fields_written": [
            "future tests", "future outcomes", "reference_code", "gold patch"
        ],
    }
    (args.output / "audit.json").write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n")
    print(json.dumps(audit, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
