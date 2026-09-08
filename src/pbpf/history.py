from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np

_IDENTITY_FIELDS = ("task_id", "candidate_hash", "test_id")
_OUTCOME_DERIVED_FIELDS = {
    "visible_feedback",
    "output_diff",
    "exception_trace",
    "test_source",
    "stdout",
    "stderr",
}


def _record(item: Any) -> dict[str, Any]:
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, Mapping):
        return dict(item)
    raise TypeError("history entries must be dataclasses or mappings")


def history_view(
    history: Sequence[Any],
    mode: str,
    k: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    wrong_task_history: Sequence[Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    rows = tuple(_record(item) for item in history)
    if mode == "last":
        return rows[-1:]
    if mode == "window":
        if k is None or k <= 0:
            raise ValueError("window history requires positive k")
        return rows[-k:]
    if mode == "full":
        return rows
    if mode == "orderless":
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    str(row.get("test_id", "")),
                    str(row.get("outcome", "")),
                    str(row.get("task_id", "")),
                ),
            )
        )
    if mode == "shuffled":
        if rng is None:
            raise ValueError("shuffled history requires an explicit RNG")
        order = rng.permutation(len(rows))
        return tuple(rows[int(index)] for index in order)
    if mode == "wrong_task":
        if wrong_task_history is None:
            raise ValueError("wrong_task requires wrong_task_history")
        other = tuple(_record(item) for item in wrong_task_history)
        if rows and other and rows[0].get("task_id") == other[0].get("task_id"):
            raise ValueError("wrong_task history must come from a different task")
        return other
    if mode in {"outcome_masked", "masked"}:
        return tuple(
            {
                **{key: row[key] for key in _IDENTITY_FIELDS if key in row},
                "outcome": "MASKED",
            }
            for row in rows
        )
    raise ValueError(f"unknown history mode: {mode}")


def feedback_view(
    history: Sequence[Any], mode: str
) -> tuple[dict[str, Any], ...]:
    rows = tuple(_record(item) for item in history)
    allowed_by_mode = {
        "status_only": set(),
        "status_output_diff": {"output_diff"},
        "exception_trace": {"exception_trace"},
        "full_test_source": {"output_diff", "exception_trace", "test_source"},
    }
    if mode not in allowed_by_mode:
        raise ValueError(f"unknown feedback mode: {mode}")
    result = []
    for row in rows:
        visible = {key: row[key] for key in _IDENTITY_FIELDS if key in row}
        if "outcome" in row:
            visible["outcome"] = row["outcome"]
        for key in allowed_by_mode[mode]:
            if key in row:
                visible[key] = row[key]
        result.append(visible)
    return tuple(result)
