from __future__ import annotations

import hashlib
import json
import random
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .registry import OUTCOMES

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    dataset: str
    public_prompt: str
    test_order: tuple[str, ...]
    group_ids: tuple[str, ...]
    gold_solution: str | None = None
    gold_patch: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.test_order:
            raise ValueError("task_id and a non-empty test_order are required")
        if len(set(self.test_order)) != len(self.test_order):
            raise ValueError("test_order cannot contain duplicate test identifiers")


@dataclass(frozen=True)
class CandidateVersion:
    candidate_id: str
    task_id: str
    version: int
    code: str
    content_hash: str
    parent_hash: str | None = None
    provenance: str = "unspecified"
    generation_logprob: float | None = None
    source_ref: str | None = None
    sampling_temperature: float | None = None
    sampling_top_p: float | None = None
    model_id: str | None = None
    model_revision: str | None = None
    mutation_registry_hash: str | None = None
    generation_trace_hash: str | None = None
    source_candidate_hash: str | None = None
    visible_tokens: int = 0
    generated_tokens: int = 0
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    gpu_hours: float = 0.0
    model_calls: int = 0

    def __post_init__(self) -> None:
        if self.version == 0:
            fields = {
                "candidate_id": self.candidate_id,
                "task_id": self.task_id,
                "version": 0,
                "code": self.code,
                "provenance": self.provenance,
                "generation_logprob": self.generation_logprob,
                "source_ref": self.source_ref,
                "sampling_temperature": self.sampling_temperature,
                "sampling_top_p": self.sampling_top_p,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "mutation_registry_hash": self.mutation_registry_hash,
                "generation_trace_hash": self.generation_trace_hash,
                "source_candidate_hash": self.source_candidate_hash,
                "visible_tokens": self.visible_tokens,
                "generated_tokens": self.generated_tokens,
                "cpu_seconds": self.cpu_seconds,
                "wall_seconds": self.wall_seconds,
                "gpu_hours": self.gpu_hours,
                "model_calls": self.model_calls,
            }
        else:
            fields = {
                "candidate_id": self.candidate_id,
                "task_id": self.task_id,
                "version": self.version,
                "code": self.code,
                "parent_hash": self.parent_hash,
                "provenance": self.provenance,
                "generation_logprob": self.generation_logprob,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "visible_tokens": self.visible_tokens,
                "generated_tokens": self.generated_tokens,
                "cpu_seconds": self.cpu_seconds,
                "wall_seconds": self.wall_seconds,
                "gpu_hours": self.gpu_hours,
                "model_calls": self.model_calls,
            }
        if self.content_hash != canonical_hash(fields):
            raise ValueError("candidate content_hash does not match immutable fields")

    @classmethod
    def create(
        cls,
        candidate_id: str,
        task_id: str,
        code: str,
        *,
        provenance: str = "unspecified",
        generation_logprob: float | None = None,
        source_ref: str | None = None,
        sampling_temperature: float | None = None,
        sampling_top_p: float | None = None,
        model_id: str | None = None,
        model_revision: str | None = None,
        mutation_registry_hash: str | None = None,
        generation_trace_hash: str | None = None,
        source_candidate_hash: str | None = None,
        visible_tokens: int = 0,
        generated_tokens: int = 0,
        cpu_seconds: float = 0.0,
        wall_seconds: float = 0.0,
        gpu_hours: float = 0.0,
        model_calls: int = 0,
    ) -> "CandidateVersion":
        if not candidate_id or not task_id:
            raise ValueError("candidate_id and task_id are required")
        fields = {
            "candidate_id": candidate_id,
            "task_id": task_id,
            "version": 0,
            "code": code,
            "provenance": provenance,
            "generation_logprob": generation_logprob,
            "source_ref": source_ref,
            "sampling_temperature": sampling_temperature,
            "sampling_top_p": sampling_top_p,
            "model_id": model_id,
            "model_revision": model_revision,
            "mutation_registry_hash": mutation_registry_hash,
            "generation_trace_hash": generation_trace_hash,
            "source_candidate_hash": source_candidate_hash,
            "visible_tokens": visible_tokens,
            "generated_tokens": generated_tokens,
            "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "gpu_hours": gpu_hours,
            "model_calls": model_calls,
        }
        digest = canonical_hash(fields)
        return cls(
            candidate_id=candidate_id,
            task_id=task_id,
            version=0,
            code=code,
            content_hash=digest,
            parent_hash=None,
            provenance=provenance,
            generation_logprob=generation_logprob,
            source_ref=source_ref,
            sampling_temperature=sampling_temperature,
            sampling_top_p=sampling_top_p,
            model_id=model_id,
            model_revision=model_revision,
            mutation_registry_hash=mutation_registry_hash,
            generation_trace_hash=generation_trace_hash,
            source_candidate_hash=source_candidate_hash,
            visible_tokens=visible_tokens,
            generated_tokens=generated_tokens,
            cpu_seconds=cpu_seconds,
            wall_seconds=wall_seconds,
            gpu_hours=gpu_hours,
            model_calls=model_calls,
        )

    def with_patch(
        self,
        code: str,
        *,
        generation_logprob: float | None = None,
        visible_tokens: int = 0,
        generated_tokens: int = 0,
        cpu_seconds: float = 0.0,
        wall_seconds: float = 0.0,
        gpu_hours: float = 0.0,
    ) -> "CandidateVersion":
        version = self.version + 1
        digest = canonical_hash(
            {
                "candidate_id": self.candidate_id,
                "task_id": self.task_id,
                "version": version,
                "code": code,
                "parent_hash": self.content_hash,
                "provenance": "repair",
                "generation_logprob": generation_logprob,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "visible_tokens": visible_tokens,
                "generated_tokens": generated_tokens,
                "cpu_seconds": cpu_seconds,
                "wall_seconds": wall_seconds,
                "gpu_hours": gpu_hours,
                "model_calls": 1,
            }
        )
        return CandidateVersion(
            candidate_id=self.candidate_id,
            task_id=self.task_id,
            version=version,
            code=code,
            content_hash=digest,
            parent_hash=self.content_hash,
            provenance="repair",
            generation_logprob=generation_logprob,
            source_ref=self.content_hash,
            sampling_temperature=None,
            sampling_top_p=None,
            model_id=self.model_id,
            model_revision=self.model_revision,
            mutation_registry_hash=None,
            generation_trace_hash=None,
            source_candidate_hash=None,
            visible_tokens=visible_tokens,
            generated_tokens=generated_tokens,
            cpu_seconds=cpu_seconds,
            wall_seconds=wall_seconds,
            gpu_hours=gpu_hours,
            model_calls=1,
        )


def validate_initial_candidate_bank(
    candidates: Sequence[CandidateVersion],
    mutant_registry: Mapping[str, str],
    *,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
) -> None:
    if len(candidates) != 8:
        raise ValueError("initial bank must contain six model samples and two deterministic mutants")
    if any(
        candidate.generation_logprob is None
        or not math.isfinite(float(candidate.generation_logprob))
        for candidate in candidates
    ):
        raise ValueError("every initial candidate requires a finite generation log-probability")
    for candidate in candidates:
        usage = (
            candidate.visible_tokens,
            candidate.generated_tokens,
            candidate.cpu_seconds,
            candidate.wall_seconds,
            candidate.gpu_hours,
            candidate.model_calls,
        )
        if any(type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0 for value in usage):
            raise ValueError("initial candidate usage fields must be finite and non-negative")
        if any(type(value) is not int for value in (candidate.visible_tokens, candidate.generated_tokens, candidate.model_calls)):
            raise ValueError("initial candidate token and call usage must be integers")
    provenance = [candidate.provenance for candidate in candidates]
    if provenance.count("model_sample") != 6 or provenance.count("deterministic_mutant") != 2:
        raise ValueError("initial bank must contain six model samples and two deterministic mutants")
    if any(candidate.version != 0 or candidate.parent_hash is not None for candidate in candidates):
        raise ValueError("initial candidate bank may contain only root versions")
    if len({candidate.task_id for candidate in candidates}) != 1:
        raise ValueError("initial candidates must belong to one task")
    if len({candidate.candidate_id for candidate in candidates}) != 8 or len(
        {candidate.content_hash for candidate in candidates}
    ) != 8:
        raise ValueError("initial candidates must be unique")
    if expected_model_id is not None or expected_model_revision is not None:
        if any(
            candidate.model_id != expected_model_id
            or candidate.model_revision != expected_model_revision
            for candidate in candidates
        ):
            raise ValueError("initial candidate provenance does not match the configured model")
    if any(
        candidate.provenance == "deterministic_mutant" and not candidate.source_ref
        for candidate in candidates
    ):
        raise ValueError("deterministic mutants require registered source_ref provenance")
    expected_registry_hash = canonical_hash(dict(sorted(mutant_registry.items())))
    samples_by_hash = {
        candidate.content_hash: candidate
        for candidate in candidates
        if candidate.provenance == "model_sample"
    }
    if any(
        candidate.provenance == "deterministic_mutant"
        and (
            candidate.source_ref not in mutant_registry
            or mutant_registry[candidate.source_ref] != canonical_hash({"code": candidate.code})
            or candidate.mutation_registry_hash != expected_registry_hash
            or candidate.source_candidate_hash not in samples_by_hash
            or candidate.generation_logprob
            != samples_by_hash[candidate.source_candidate_hash].generation_logprob
            or candidate.generation_trace_hash
            != samples_by_hash[candidate.source_candidate_hash].generation_trace_hash
        )
        for candidate in candidates
    ):
        raise ValueError("deterministic mutant provenance is not verified by the hashed registry")
    if any(
        candidate.provenance == "model_sample"
        and (
            candidate.model_calls != 1
            or candidate.visible_tokens <= 0
            or candidate.generated_tokens <= 0
            or candidate.sampling_temperature != 0.8
            or candidate.sampling_top_p != 0.95
            or not isinstance(candidate.model_id, str)
            or not candidate.model_id
            or not isinstance(candidate.model_revision, str)
            or not _COMMIT_RE.fullmatch(candidate.model_revision)
            or not isinstance(candidate.generation_trace_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", candidate.generation_trace_hash)
        )
        for candidate in candidates
    ):
        raise ValueError(
            "each model sample must include temperature=0.8, top_p=0.95, a model id, resolved model revision, and initial generation usage"
        )


@dataclass(frozen=True)
class Observation:
    candidate_hash: str
    test_id: str
    outcome: str
    visible_feedback: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"invalid categorical outcome: {self.outcome}")


def trainer_view(record: TaskRecord | Mapping[str, Any]) -> dict[str, Any]:
    raw = asdict(record) if isinstance(record, TaskRecord) else dict(record)
    allowed = ("task_id", "dataset", "public_prompt", "test_order", "group_ids")
    return {key: raw[key] for key in allowed}


def evaluator_view(record: TaskRecord | Mapping[str, Any]) -> dict[str, Any]:
    return asdict(record) if isinstance(record, TaskRecord) else dict(record)


def build_grouped_split(
    records: Sequence[TaskRecord],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, str]:
    if len(ratios) != 3 or any(value < 0 for value in ratios):
        raise ValueError("ratios must contain three non-negative values")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    normalized = tuple(value / total for value in ratios)

    parents = {record.task_id: record.task_id for record in records}

    def find(item: str) -> str:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    seen_group: dict[str, str] = {}
    for record in records:
        for group_id in record.group_ids:
            if group_id in seen_group:
                union(record.task_id, seen_group[group_id])
            else:
                seen_group[group_id] = record.task_id

    components: dict[str, list[str]] = {}
    for record in records:
        components.setdefault(find(record.task_id), []).append(record.task_id)
    groups = sorted(components.values(), key=lambda group: tuple(sorted(group)))
    random.Random(seed).shuffle(groups)

    names = ("train", "validation", "test")
    targets = [normalized[index] * len(records) for index in range(3)]
    counts = [0, 0, 0]
    result: dict[str, str] = {}
    for group in groups:
        destination = min(
            range(3),
            key=lambda index: (counts[index] - targets[index], index),
        )
        for task_id in group:
            result[task_id] = names[destination]
        counts[destination] += len(group)
    return result
