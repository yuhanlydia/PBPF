from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .registry import DATASETS

EVALUATOR_ONLY_FIELDS = {
    "gold",
    "gold_solution",
    "gold_patch",
    "canonical_solution",
    "solution",
    "fixed_code",
    "patch",
    "output",
    "expected_output",
    "expected_outputs",
    "future_outcomes",
    "hidden_tests",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
}


@dataclass(frozen=True)
class BenchmarkRecord:
    task_id: str
    dataset: str
    prompt: str
    test_order: tuple[str, ...]
    metadata: dict[str, Any]


def _evaluator_paths(value: Any, prefix: str = "") -> list[str]:
    leaked: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in EVALUATOR_ONLY_FIELDS:
                leaked.append(path)
            leaked.extend(_evaluator_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaked.extend(_evaluator_paths(child, f"{prefix}[{index}]"))
    return leaked


def _reject_evaluator_fields(row: dict[str, Any]) -> None:
    leaked = sorted(_evaluator_paths(row))
    if leaked:
        raise ValueError(f"evaluator-only fields present in trainer input: {', '.join(leaked)}")


def load_benchmark_jsonl(path: str | Path, *, dataset: str) -> tuple[BenchmarkRecord, ...]:
    if dataset not in DATASETS:
        raise ValueError(f"unknown benchmark dataset: {dataset}")
    records: list[BenchmarkRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        _reject_evaluator_fields(row)
        task_id = row.get("task_id") or row.get("id") or row.get("instance_id")
        prompt = row.get("prompt") or row.get("problem") or row.get("buggy_code")
        tests = row.get("tests") or row.get("test_order")
        if not isinstance(task_id, str) or not isinstance(prompt, str):
            raise ValueError(f"line {line_number} lacks task_id/prompt")
        if not isinstance(tests, list) or not tests or not all(isinstance(item, str) for item in tests):
            raise ValueError(f"line {line_number} requires a non-empty ordered test identifier list")
        records.append(
            BenchmarkRecord(
                task_id,
                dataset,
                prompt,
                tuple(tests),
                {key: value for key, value in row.items() if key not in {"prompt", "problem", "buggy_code", "tests", "test_order"}},
            )
        )
    return tuple(records)


def load_runbugrun(path: str | Path) -> tuple[BenchmarkRecord, ...]:
    return load_benchmark_jsonl(path, dataset="runbugrun")


def load_codearc(path: str | Path) -> tuple[BenchmarkRecord, ...]:
    return load_benchmark_jsonl(path, dataset="codearc")


def load_evalplus(path: str | Path) -> tuple[BenchmarkRecord, ...]:
    return load_benchmark_jsonl(path, dataset="evalplus")


def load_livecodebench(path: str | Path) -> tuple[BenchmarkRecord, ...]:
    return load_benchmark_jsonl(path, dataset="livecodebench_v6")


def load_swebench(path: str | Path, *, verified: bool) -> tuple[BenchmarkRecord, ...]:
    dataset = "swebench_verified" if verified else "swebench_lite"
    return load_benchmark_jsonl(path, dataset=dataset)
