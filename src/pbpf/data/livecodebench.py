from __future__ import annotations

import hashlib
import json
from typing import AbstractSet, Any, Iterable, Mapping

from .schema import (
    AdapterResult,
    EvaluatorTask,
    EvaluatorTest,
    PublicTask,
    PublicTest,
    resolve_task_rows,
)

PROTOCOL = "LiveCodeBench"


def adapt_livecodebench(
    rows: Iterable[Mapping[str, Any]], *, release_v5_task_ids: AbstractSet[str]
) -> AdapterResult:
    if any(not isinstance(task_id, str) or not task_id for task_id in release_v5_task_ids):
        raise ValueError("release_v5_task_ids must contain non-empty strings")
    v5_ids = tuple(sorted(release_v5_task_ids))
    v5_hash = hashlib.sha256(
        json.dumps(v5_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    raw_count, resolved_rows, exclusions = resolve_task_rows(rows)
    public: list[PublicTask] = []
    evaluator: list[EvaluatorTask] = []
    for row in resolved_rows:
        task_id = str(row["task_id"])
        release = row.get("release")
        if (
            release is not None
            and str(release).lower() not in {"v6", "release_v6", "6"}
        ) or task_id in release_v5_task_ids:
            exclusions.append((task_id, "not_new_in_v6"))
            continue
        order_key = lambda case: (
            hashlib.sha256(str(case.get("id", "")).encode()).hexdigest(),
            str(case.get("id", "")),
        )
        if "public_tests" in row or "hidden_tests" in row:
            public_cases = row.get("public_tests")
            hidden_cases = row.get("hidden_tests")
            if not isinstance(public_cases, list) or not isinstance(hidden_cases, list):
                exclusions.append((task_id, "malformed_test_cases"))
                continue
            partitioned = True
        else:
            cases = row.get("tests")
            if not isinstance(cases, list):
                exclusions.append((task_id, "malformed_test_cases"))
                continue
            public_cases = cases
            hidden_cases = []
            partitioned = False
        if (
            not public_cases
            or any(not isinstance(case, Mapping) for case in public_cases)
            or any(not isinstance(case, Mapping) for case in hidden_cases)
        ):
            exclusions.append((task_id, "malformed_test_cases"))
            continue
        all_cases = [*public_cases, *hidden_cases]
        if any(
            not isinstance(case.get("id"), (str, int)) or not str(case["id"])
            for case in all_cases
        ):
            exclusions.append((task_id, "malformed_test_cases"))
            continue
        if partitioned:
            ordered_public = sorted(public_cases, key=order_key)
            ordered_hidden = sorted(hidden_cases, key=order_key)
        else:
            ordered = sorted(public_cases, key=order_key)
            ordered_public = ordered[:4]
            ordered_hidden = ordered[4:]
        public_cases = ordered_public
        hidden_cases = ordered_hidden
        if not hidden_cases:
            exclusions.append((task_id, "no_hidden_cases"))
            continue
        ordered = public_cases + hidden_cases
        identifiers = [str(case["id"]) for case in ordered]
        if len(identifiers) != len(set(identifiers)):
            exclusions.append((task_id, "duplicate_test_id"))
            continue
        split = "locked_test"
        visible_count = len(ordered_public)
        public.append(
            PublicTask(
                task_id,
                PROTOCOL,
                str(row.get("prompt") or row.get("task_text") or ""),
                str(row.get("candidate_code") or ""),
                tuple(
                    PublicTest(
                        identifiers[index],
                        ordered[index].get("source")
                        if ordered[index].get("public_source") is True
                        else None,
                    )
                    for index in range(visible_count)
                ),
                split,
                (f"livecodebench:{split}:{task_id}",),
            )
        )
        evaluator.append(
            EvaluatorTask(
                task_id,
                PROTOCOL,
                tuple(
                    EvaluatorTest(
                        identifiers[index],
                        case.get("source"),
                        case.get("expected_output"),
                        index >= visible_count,
                        payload=dict(case),
                    )
                    for index, case in enumerate(ordered)
                ),
                None if row.get("gold_code") is None else str(row["gold_code"]),
                None if row.get("gold_patch") is None else str(row["gold_patch"]),
            )
        )
    return AdapterResult(
        PROTOCOL,
        tuple(public),
        tuple(evaluator),
        raw_count,
        tuple(sorted(exclusions)),
        (),
        {
            "release_v5_task_ids_hash": v5_hash,
            "release_v5_task_count": len(v5_ids),
            "visible_partition": "upstream-or-first-four",
        },
    )
