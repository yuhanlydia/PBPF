from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class PublicTest:
    """A test descriptor that is safe to mount in the generator process."""

    test_id: str
    source: Any | None = None
    invocation_index: int | None = None

    def __post_init__(self) -> None:
        if not self.test_id:
            raise ValueError("public test_id is required")


@dataclass(frozen=True)
class PublicTask:
    """The complete, intentionally small generator-side task schema."""

    task_id: str
    protocol: str
    task_text: str
    candidate_code: str
    visible_tests: tuple[PublicTest, ...]
    split: str
    group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.protocol or not self.split:
            raise ValueError("public task identity, protocol, and split are required")
        identifiers = [test.test_id for test in self.visible_tests]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("visible test identifiers must be unique")


@dataclass(frozen=True)
class EvaluatorTest:
    test_id: str
    source: Any
    expected_output: Any | None
    hidden: bool
    invocation_index: int | None = None
    payload: Any | None = None

    def __post_init__(self) -> None:
        if not self.test_id:
            raise ValueError("evaluator test_id is required")


@dataclass(frozen=True)
class EvaluatorTask:
    task_id: str
    protocol: str
    tests: tuple[EvaluatorTest, ...]
    gold_code: str | None = None
    gold_patch: str | None = None
    hidden_outcomes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id or not self.protocol:
            raise ValueError("evaluator task identity and protocol are required")
        object.__setattr__(
            self,
            "hidden_outcomes",
            tuple((str(key), str(value)) for key, value in self.hidden_outcomes),
        )
        identifiers = [test.test_id for test in self.tests]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evaluator test identifiers must be unique")


@dataclass(frozen=True)
class AdapterResult:
    protocol: str
    public_tasks: tuple[PublicTask, ...]
    evaluator_tasks: tuple[EvaluatorTask, ...]
    raw_count: int
    exclusions: tuple[tuple[str, str], ...] = ()
    trusted_metadata: tuple[Mapping[str, Any], ...] = ()
    resolved_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        public_ids = [task.task_id for task in self.public_tasks]
        evaluator_ids = [task.task_id for task in self.evaluator_tasks]
        if public_ids != evaluator_ids:
            raise ValueError("public and evaluator task inventories differ")
        if self.raw_count != len(public_ids) + len(self.exclusions):
            raise ValueError("adapter raw/kept/excluded counts do not balance")

    @property
    def exclusion_reasons(self) -> dict[str, int]:
        return dict(sorted(Counter(reason for _, reason in self.exclusions).items()))


def resolve_task_rows(
    rows: Iterable[Any],
) -> tuple[int, tuple[Mapping[str, Any], ...], list[tuple[str, str]]]:
    """Resolve duplicate JSON rows before adapter-specific interpretation."""

    materialized = list(rows)
    exclusions: list[tuple[str, str]] = []
    buckets: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for row in materialized:
        if not isinstance(row, Mapping):
            exclusions.append(("<malformed>", "malformed_record"))
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            exclusions.append(("<missing>", "malformed_record"))
            continue
        try:
            payload = json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            exclusions.append((task_id, "malformed_record"))
            continue
        buckets.setdefault(task_id, []).append((payload, row))

    resolved: list[Mapping[str, Any]] = []
    for task_id, entries in sorted(buckets.items()):
        unique = {payload: row for payload, row in entries}
        if len(unique) > 1:
            exclusions.extend(
                (task_id, "conflicting_duplicate_task_id") for _ in entries
            )
            continue
        resolved.append(next(iter(unique.values())))
        exclusions.extend(
            (task_id, "exact_duplicate_task") for _ in range(len(entries) - 1)
        )
    return len(materialized), tuple(resolved), exclusions


@dataclass(frozen=True)
class PreparedSplit:
    protocol: str
    public_tasks: tuple[PublicTask, ...]
    generator_root: Path
    evaluator_root: Path
    generator_manifest_path: Path
    evaluator_manifest_path: Path
    generator_manifest_hash: str
    evaluator_manifest_hash: str
    completion_path: Path
    counts: dict[str, int]
    exclusion_reasons: dict[str, int]


@dataclass(frozen=True)
class LoadedEvaluator:
    tasks: tuple[EvaluatorTask, ...]
    candidate_hashes: tuple[str, ...]
    experiment_fingerprint: str
