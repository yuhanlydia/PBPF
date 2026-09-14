from __future__ import annotations

from typing import Any, Iterable, Mapping

from .schema import (
    AdapterResult,
    EvaluatorTask,
    EvaluatorTest,
    PublicTask,
    PublicTest,
    resolve_task_rows,
)

PROTOCOL = "CodeARC-Replay"


def adapt_codearc(rows: Iterable[Mapping[str, Any]]) -> AdapterResult:
    raw_count, resolved_rows, exclusions = resolve_task_rows(rows)
    public: list[PublicTask] = []
    evaluator: list[EvaluatorTask] = []
    for row in resolved_rows:
        task_id = str(row["task_id"])
        if row.get("anonymous") is not True:
            exclusions.append((task_id, "non_anonymous"))
            continue
        invocations = row.get("invocations")
        if not isinstance(invocations, list) or len(invocations) != 10 or any(
            not isinstance(case, Mapping) for case in invocations
        ):
            exclusions.append((task_id, "invalid_invocation_count"))
            continue
        invocation_ids = [str(case.get("id") or f"invocation-{index}") for index, case in enumerate(invocations)]
        if len(invocation_ids) != len(set(invocation_ids)):
            exclusions.append((task_id, "duplicate_invocation_id"))
            continue
        split = str(row.get("official_split") or "test")
        public.append(
            PublicTask(
                task_id=task_id,
                protocol=PROTOCOL,
                task_text=str(row.get("task_text") or row.get("prompt") or ""),
                candidate_code=str(row.get("candidate_code") or ""),
                visible_tests=tuple(
                    PublicTest(invocation_ids[index], invocations[index].get("input"), index)
                    for index in range(4)
                ),
                split=split,
                group_ids=(f"codearc:{split}:{task_id}",),
            )
        )
        evaluator.append(
            EvaluatorTask(
                task_id=task_id,
                protocol=PROTOCOL,
                tests=tuple(
                    EvaluatorTest(
                        invocation_ids[index],
                        invocations[index].get("input"),
                        invocations[index].get("expected_output"),
                        index >= 4,
                        index,
                    )
                    for index in range(10)
                ),
                gold_code=None if row.get("target_code") is None else str(row["target_code"]),
                gold_patch=None,
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
            "anonymous_only": True,
            "visible_invocation_indices": [0, 1, 2, 3],
            "evaluator_invocation_indices": [4, 5, 6, 7, 8, 9],
        },
    )
