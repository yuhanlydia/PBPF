from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from .schema import (
    AdapterResult,
    EvaluatorTask,
    EvaluatorTest,
    PublicTask,
    PublicTest,
    resolve_task_rows,
)

PROTOCOL = "PBPF-EvalPlus"


_NON_EXECUTION_FIELDS = {
    "id",
    "partition",
    "provenance",
    "public_source",
    "source_dataset",
    "visibility",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _case_signature(case: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical(
            {key: value for key, value in case.items() if key not in _NON_EXECUTION_FIELDS}
        ).encode("utf-8")
    ).hexdigest()


def _deduplicate(cases: Any, seen: set[str]) -> list[Mapping[str, Any]]:
    if not isinstance(cases, list):
        raise ValueError("EvalPlus cases must be lists")
    if any(not isinstance(case, Mapping) for case in cases):
        raise ValueError("EvalPlus cases must be objects")
    result: list[Mapping[str, Any]] = []
    ordered = sorted(
        cases,
        key=lambda case: (
            hashlib.sha256(str(case.get("id", "")).encode("utf-8")).hexdigest(),
            str(case.get("id", "")),
        ),
    )
    for case in ordered:
        signature = _case_signature(case)
        if signature not in seen:
            seen.add(signature)
            result.append(case)
    return result


def adapt_evalplus(rows: Iterable[Mapping[str, Any]]) -> AdapterResult:
    raw_count, resolved_rows, exclusions = resolve_task_rows(rows)
    public: list[PublicTask] = []
    evaluator: list[EvaluatorTask] = []
    for row in resolved_rows:
        task_id = str(row["task_id"])
        try:
            signatures: set[str] = set()
            base = _deduplicate(row.get("base_tests"), signatures)
            plus = _deduplicate(row.get("plus_tests"), signatures)
        except ValueError:
            exclusions.append((task_id, "malformed_test_cases"))
            continue
        if not base or not plus:
            exclusions.append((task_id, "missing_base_or_plus_cases"))
            continue
        case_ids = [str(case.get("id") or f"base-{index}") for index, case in enumerate(base)]
        plus_ids = [str(case.get("id") or f"plus-{index}") for index, case in enumerate(plus)]
        if len(set(case_ids + plus_ids)) != len(case_ids) + len(plus_ids):
            exclusions.append((task_id, "duplicate_test_id"))
            continue
        if row.get("hidden_outcomes") is not None and not isinstance(
            row.get("hidden_outcomes"), Mapping
        ):
            exclusions.append((task_id, "malformed_record"))
            continue
        split = str(row.get("official_split") or "test")
        public.append(
            PublicTask(
                task_id,
                PROTOCOL,
                str(row.get("prompt") or row.get("task_text") or ""),
                str(row.get("candidate_code") or ""),
                tuple(
                    PublicTest(
                        case_ids[index],
                        case.get("source") if case.get("public_source") is True else None,
                    )
                    for index, case in enumerate(base)
                ),
                split,
                (f"evalplus:{split}:{task_id}",),
            )
        )
        hidden_outcomes = row.get("hidden_outcomes") or {}
        evaluator.append(
            EvaluatorTask(
                task_id,
                PROTOCOL,
                tuple(
                    [
                        EvaluatorTest(
                            case_ids[index],
                            case.get("source"),
                            case.get("expected_output"),
                            False,
                            payload=dict(case),
                        )
                        for index, case in enumerate(base)
                    ]
                    + [
                        EvaluatorTest(
                            plus_ids[index],
                            case.get("source"),
                            case.get("expected_output"),
                            True,
                            payload=dict(case),
                        )
                        for index, case in enumerate(plus)
                    ]
                ),
                None if row.get("gold_code") is None else str(row["gold_code"]),
                None if row.get("gold_patch") is None else str(row["gold_patch"]),
                tuple(sorted((str(key), str(value)) for key, value in hidden_outcomes.items())),
            )
        )
    return AdapterResult(
        PROTOCOL,
        tuple(public),
        tuple(evaluator),
        raw_count,
        tuple(sorted(exclusions)),
        (),
        {"exact_case_deduplication": True},
    )
