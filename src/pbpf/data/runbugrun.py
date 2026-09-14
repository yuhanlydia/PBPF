from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from ..registry import OUTCOMES
from .schema import (
    AdapterResult,
    EvaluatorTask,
    EvaluatorTest,
    PublicTask,
    PublicTest,
    resolve_task_rows,
)

PROTOCOL = "PBPF-RBR"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_code(source: str) -> str:
    try:
        return ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False)
    except (SyntaxError, ValueError):
        return " ".join(source.split())


def _test_signature(tests: Sequence[Mapping[str, Any]]) -> str:
    cases = [
        {
            "source": _normalize_code(str(case.get("source", ""))),
            "expected_output": case.get("expected_output"),
        }
        for case in tests
    ]
    return _digest(_canonical(sorted(cases, key=_canonical)))


def _ordered_tests(tests: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        tests,
        key=lambda case: (
            hashlib.sha256(str(case["id"]).encode("utf-8")).hexdigest(),
            str(case["id"]),
        ),
    )


def _official_split(value: Any) -> str:
    split = str(value or "").lower()
    aliases = {"dev": "development", "validation": "development", "val": "development"}
    split = aliases.get(split, split)
    if split not in {"train", "development", "test"}:
        raise ValueError("RunBugRun rows require an official train/development/test boundary")
    return split


def _signals(row: Mapping[str, Any], active_tests: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({
        "problem:" + _digest(str(row.get("problem_id") or row["task_id"])),
        "code:" + _digest(_normalize_code(str(row["buggy_code"]))),
        "code:" + _digest(_normalize_code(str(row["fixed_code"]))),
        "tests:" + _test_signature(active_tests),
    }))


def _development_half(task_id: str) -> str:
    return "development_tune" if int(_digest(task_id), 16) % 2 == 0 else "development_select"


def _cap_rows(
    rows: Sequence[
        tuple[Mapping[str, Any], list[Mapping[str, Any]], str, tuple[str, ...], str]
    ],
    caps: Mapping[str, int],
) -> tuple[
    list[tuple[Mapping[str, Any], list[Mapping[str, Any]], str, tuple[str, ...], str]],
    list[tuple[str, str]],
]:
    kept: list[
        tuple[Mapping[str, Any], list[Mapping[str, Any]], str, tuple[str, ...], str]
    ] = []
    excluded: list[tuple[str, str]] = []
    by_split: dict[
        str,
        list[tuple[Mapping[str, Any], list[Mapping[str, Any]], str, tuple[str, ...], str]],
    ] = defaultdict(list)
    for item in rows:
        by_split[item[2]].append(item)
    for split in ("train", "development", "test"):
        components: dict[
            str,
            list[tuple[Mapping[str, Any], list[Mapping[str, Any]], str, tuple[str, ...], str]],
        ] = defaultdict(list)
        for item in by_split.get(split, ()):
            components[item[4]].append(item)
        ranked = sorted(components.items(), key=lambda item: (item[0], tuple(str(row[0]["task_id"]) for row in item[1])))
        cap = caps[split]
        count = 0
        for _, component in ranked:
            if count + len(component) <= cap:
                kept.extend(component)
                count += len(component)
            else:
                excluded.extend(
                    (str(item[0]["task_id"]), "deterministic_cap") for item in component
                )
    return kept, excluded


def adapt_runbugrun(
    rows: Iterable[Mapping[str, Any]],
    *,
    caps: Mapping[str, int] | None = None,
) -> AdapterResult:
    """Adapt local immutable RunBugRun rows without reading an upstream service."""

    raw_count, resolved_rows, exclusions = resolve_task_rows(rows)
    eligible: list[tuple[Mapping[str, Any], list[Mapping[str, Any]], str, tuple[str, ...]]] = []
    for row in resolved_rows:
        task_id = str(row["task_id"])
        if str(row.get("language", "")).lower() not in {"python", "py", "python3"}:
            exclusions.append((task_id, "non_python"))
            continue
        tests = row.get("tests")
        if not isinstance(tests, list):
            exclusions.append((task_id, "insufficient_active_tests"))
            continue
        active = [case for case in tests if isinstance(case, Mapping) and case.get("active", True)]
        if len(active) < 10:
            exclusions.append((task_id, "insufficient_active_tests"))
            continue
        if any(case.get("fixed_outcome") != "PASS" for case in active):
            exclusions.append((task_id, "fixed_fails_active_test"))
            continue
        buggy_outcomes = [case.get("buggy_outcome") for case in active]
        if any(outcome not in OUTCOMES for outcome in buggy_outcomes):
            exclusions.append((task_id, "invalid_buggy_outcomes"))
            continue
        if all(outcome == "PASS" for outcome in buggy_outcomes):
            exclusions.append((task_id, "buggy_passes_all"))
            continue
        token_count = row.get("actor_token_count")
        if not isinstance(token_count, int) or token_count < 0:
            exclusions.append((task_id, "unverified_actor_token_count"))
            continue
        if token_count > 2048:
            exclusions.append((task_id, "over_2048_actor_tokens"))
            continue
        try:
            if any(
                not isinstance(case.get("id"), (str, int)) or str(case["id"]) == ""
                for case in active
            ) or len({str(case["id"]) for case in active}) != len(active):
                raise ValueError("invalid active test ids")
            if row.get("hidden_outcomes") is not None and not isinstance(
                row.get("hidden_outcomes"), Mapping
            ):
                raise ValueError("invalid hidden outcomes")
            split = _official_split(row.get("official_split"))
            signals = _signals(row, active)
        except (KeyError, TypeError, ValueError):
            exclusions.append((task_id, "malformed_record"))
            continue
        eligible.append((row, active, split, signals))

    parents = list(range(len(eligible)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    first_by_signal: dict[str, int] = {}
    for index, (_, _, _, signals) in enumerate(eligible):
        for signal in signals:
            if signal in first_by_signal:
                union(index, first_by_signal[signal])
            else:
                first_by_signal[signal] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(eligible)):
        components[find(index)].append(index)
    safe: list[
        tuple[Mapping[str, Any], list[Mapping[str, Any]], str, tuple[str, ...], str]
    ] = []
    trusted_metadata: list[Mapping[str, Any]] = []
    for indices in components.values():
        boundaries = {eligible[index][2] for index in indices}
        task_ids = sorted(str(eligible[index][0]["task_id"]) for index in indices)
        component_key = _digest(_canonical(task_ids))
        crosses_boundary = len(boundaries) > 1
        trusted_metadata.extend(
            {
                "task_id": str(eligible[index][0]["task_id"]),
                "official_split": eligible[index][2],
                "component_id": component_key,
                "private_signatures": list(eligible[index][3]),
                "excluded_cross_boundary": crosses_boundary,
            }
            for index in indices
        )
        if crosses_boundary:
            exclusions.extend(
                (str(eligible[index][0]["task_id"]), "cross_boundary_leakage")
                for index in indices
            )
        else:
            safe.extend((*eligible[index], component_key) for index in indices)

    selected, capped = _cap_rows(
        safe, caps or {"train": 12_000, "development": 2_000, "test": 2_000}
    )
    exclusions.extend(capped)
    public: list[PublicTask] = []
    evaluator: list[EvaluatorTask] = []
    for row, active, boundary, signals, component_key in sorted(
        selected, key=lambda item: str(item[0]["task_id"])
    ):
        ordered = _ordered_tests(active)
        split = _development_half(component_key) if boundary == "development" else boundary
        group_ids = (f"rbr:{boundary}:component:{component_key}",)
        public.append(
            PublicTask(
                task_id=str(row["task_id"]),
                protocol=PROTOCOL,
                task_text=str(row.get("task_text") or row.get("prompt") or ""),
                candidate_code=str(row["buggy_code"]),
                visible_tests=tuple(
                    PublicTest(
                        str(case["id"]),
                        case.get("source") if case.get("public_source") is True else None,
                    )
                    for case in ordered[:4]
                ),
                split=split,
                group_ids=group_ids,
            )
        )
        hidden_outcomes = row.get("hidden_outcomes") or {}
        evaluator.append(
            EvaluatorTask(
                task_id=str(row["task_id"]),
                protocol=PROTOCOL,
                tests=tuple(
                    EvaluatorTest(
                        test_id=str(case["id"]),
                        source=case.get("source"),
                        expected_output=case.get("expected_output"),
                        hidden=index >= 4,
                    )
                    for index, case in enumerate(ordered)
                ),
                gold_code=str(row["fixed_code"]),
                gold_patch=None if row.get("gold_patch") is None else str(row["gold_patch"]),
                hidden_outcomes=tuple(sorted((str(key), str(value)) for key, value in hidden_outcomes.items())),
            )
        )
    return AdapterResult(
        PROTOCOL,
        tuple(public),
        tuple(evaluator),
        raw_count,
        tuple(sorted(exclusions)),
        tuple(sorted(trusted_metadata, key=lambda item: str(item["task_id"]))),
        {
            "caps": dict(
                sorted(
                    (caps or {"train": 12_000, "development": 2_000, "test": 2_000}).items()
                )
            ),
            "visible_tests": 4,
            "maximum_actor_tokens": 2048,
        },
    )
