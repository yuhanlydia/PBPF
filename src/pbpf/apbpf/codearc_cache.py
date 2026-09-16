"""Convert verified CodeARC train/development executions without primary data."""
from collections import Counter
import hashlib

from pbpf.real_gate import CODEARC_CACHE_SCHEMA, validate_rbr_cache
from .codearc_prompt import bounded_task_text


def build_cache(bank_rows, execution_rows, public_tasks, private_tasks):
    records = []
    evidence = {}
    for row in execution_rows:
        key = row["task_id"], row["candidate_id"]
        if key in evidence:
            raise ValueError("duplicate execution record")
        evidence[key] = row
    used = set()
    for group in bank_rows:
        if group["split"] not in {"train", "development"}:
            raise ValueError("development cache must never ingest primary candidates")
        public, private = public_tasks[group["task_id"]], private_tasks[group["task_id"]]
        if any(task["split"] != group["split"] or task["source_component_id"] != group["source_component_id"]
               for task in (public, private)):
            raise ValueError("source binding mismatch")
        tests = private["tests"]
        if [t["id"] for t in tests] != [str(i) for i in range(10)]:
            raise ValueError("all ten replay test IDs required")
        for candidate in group["candidates"]:
            key = group["task_id"], candidate["candidate_id"]
            if key in used or key not in evidence:
                raise ValueError("execution inventory differs from candidate bank")
            used.add(key)
            result = evidence[key]
            if [t["test_id"] for t in result["tests"]] != [t["id"] for t in tests]:
                raise ValueError("execution tests are missing or reordered")
            cases = []
            for test, actual in zip(tests, result["tests"], strict=True):
                cases.append({"id": test["id"], "input": test["input"],
                    "expected": test["expected"][:4096], "expected_error": test["expected_error"],
                    "expected_truncated": len(test["expected"]) > 4096,
                    "expected_sha256": hashlib.sha256(test["expected"].encode()).hexdigest(),
                    "actual": actual["stdout"][:4096], "stderr": actual["stderr"][:4096],
                    "returncode": actual["returncode"], "timed_out": actual["timed_out"],
                    "outcome": actual["outcome"]})
            records.append({"task_id": candidate["candidate_id"], "problem_id": group["task_id"],
                "source_component_id": group["source_component_id"], "split": group["split"],
                "task_text": bounded_task_text(public) if group.get("public_prompt_bounded") else public["task_text"],
                "candidate": candidate["code"] or "\n",
                "empty_candidate": candidate["code"] == "", "tests": cases,
                "candidate_code_sha256": hashlib.sha256(candidate["code"].encode()).hexdigest(),
                "outcomes": [t["outcome"] for t in cases]})
    if used != set(evidence):
        raise ValueError("unused execution records; population filtering forbidden")
    payload = {"schema": CODEARC_CACHE_SCHEMA, "dataset": "codearc_replay", "seed": 1701,
        "tests_per_candidate": 10, "records": records,
        "counts": dict(Counter(r["split"] for r in records)),
        "population_policy": "complete train/development generation inventories; no correctness filtering",
        "execution_fields_visibility": "evaluator-only; future expected outputs never predictor features"}
    return validate_rbr_cache(payload)
