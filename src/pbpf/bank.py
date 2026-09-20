from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import (
    CandidateVersion,
    Observation,
    TaskRecord,
    canonical_hash,
    evaluator_view,
    trainer_view,
)
from .registry import OUTCOMES


def _json_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_disjoint_roots(trainer_path: str | Path, evaluator_path: str | Path) -> None:
    trainer_root = Path(trainer_path).resolve()
    evaluator_root = Path(evaluator_path).resolve()
    if (
        trainer_root == evaluator_root
        or trainer_root in evaluator_root.parents
        or evaluator_root in trainer_root.parents
    ):
        raise ValueError(
            "trainer and evaluator roots must be disjoint mount boundaries with no ancestor relation"
        )


@dataclass(frozen=True)
class EvaluatorBundle:
    bank: "TrajectoryBank"
    candidate_outcomes: tuple[tuple[str, str, str], ...]

    @property
    def future_outcomes_by_candidate(self) -> dict[tuple[str, str], str]:
        return {
            (candidate_hash, test_id): outcome
            for candidate_hash, test_id, outcome in self.candidate_outcomes
        }


@dataclass(frozen=True)
class TrajectoryBank:
    tasks: tuple[TaskRecord, ...]
    candidates: tuple[CandidateVersion, ...]
    observations: tuple[Observation, ...]
    mutant_registry: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        task_by_id = {task.task_id: task for task in self.tasks}
        if len(task_by_id) != len(self.tasks):
            raise ValueError("duplicate task id")
        candidate_by_hash = {candidate.content_hash: candidate for candidate in self.candidates}
        if len(candidate_by_hash) != len(self.candidates):
            raise ValueError("duplicate candidate content hash")
        for candidate in self.candidates:
            if candidate.task_id not in task_by_id:
                raise ValueError("candidate references unknown task")
        positions: dict[str, int] = {}
        for observation in self.observations:
            candidate = candidate_by_hash.get(observation.candidate_hash)
            if candidate is None:
                raise ValueError("observation references unknown candidate")
            task = task_by_id[candidate.task_id]
            position = positions.get(candidate.content_hash, 0)
            if position >= len(task.test_order) or task.test_order[position] != observation.test_id:
                raise ValueError("observation violates fixed test order")
            positions[candidate.content_hash] = position + 1
        registry = dict(self.mutant_registry)
        if len(registry) != len(self.mutant_registry):
            raise ValueError("duplicate mutant registry identifier")
        if any(
            not key
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for key, value in self.mutant_registry
        ):
            raise ValueError("mutant registry entries require identifiers and SHA-256 code hashes")

    @property
    def content_hash(self) -> str:
        return canonical_hash(
            {
                "tasks": [asdict(item) for item in self.tasks],
                "candidates": [asdict(item) for item in self.candidates],
                "observations": [asdict(item) for item in self.observations],
                "mutant_registry": list(self.mutant_registry),
            }
        )

    @property
    def trainer_content_hash(self) -> str:
        return canonical_hash(
            {
                "tasks": [trainer_view(item) for item in self.tasks],
                "candidates": [asdict(item) for item in self.candidates],
                "observations": [asdict(item) for item in self.observations],
                "mutant_registry": list(self.mutant_registry),
            }
        )

    def with_candidate(self, candidate: CandidateVersion) -> "TrajectoryBank":
        return TrajectoryBank(
            self.tasks,
            self.candidates + (candidate,),
            self.observations,
            self.mutant_registry,
        )

    def with_observation(self, observation: Observation) -> "TrajectoryBank":
        return TrajectoryBank(
            self.tasks,
            self.candidates,
            self.observations + (observation,),
            self.mutant_registry,
        )

    def write(
        self,
        trainer_path: str | Path,
        evaluator_path: str | Path,
        *,
        candidate_outcomes: dict[tuple[str, str], str],
    ) -> None:
        trainer_root = Path(trainer_path)
        evaluator_root = Path(evaluator_path)
        _require_disjoint_roots(trainer_root, evaluator_root)
        if trainer_root.exists() or evaluator_root.exists():
            raise FileExistsError("trajectory bank roots are create-once")
        self.write_trainer(trainer_root)
        self.write_evaluator(
            evaluator_root, trainer_root, candidate_outcomes=candidate_outcomes
        )

    def write_trainer(self, path: str | Path) -> None:
        trainer_root = Path(path)
        if any(
            observation.visible_feedback is not None
            for observation in self.observations
        ):
            raise ValueError(
                "trainer bank cannot serialize raw evaluator feedback; project it first"
            )
        if trainer_root.exists():
            raise FileExistsError(f"trainer bank is create-once: {trainer_root}")
        trainer_root.mkdir(parents=True)
        collections = {
            "tasks.jsonl": (trainer_view(item) for item in self.tasks),
            "candidates.jsonl": (asdict(item) for item in self.candidates),
            "observations.jsonl": (asdict(item) for item in self.observations),
        }
        for filename, rows in collections.items():
            text = "".join(_json_line(row) + "\n" for row in rows)
            (trainer_root / filename).write_text(text, encoding="utf-8")
        np.savez_compressed(
            trainer_root / "arrays.npz",
            candidate_hash=np.asarray([item.candidate_hash for item in self.observations]),
            test_id=np.asarray([item.test_id for item in self.observations]),
            outcome=np.asarray([item.outcome for item in self.observations]),
        )
        (trainer_root / "mutant_registry.json").write_text(
            _json_line(dict(self.mutant_registry)) + "\n", encoding="utf-8"
        )
        filenames = (*collections.keys(), "arrays.npz", "mutant_registry.json")
        manifest = {
            "format": "pbpf-trajectory-bank-v1",
            "checksums": {name: _file_hash(trainer_root / name) for name in filenames},
        }
        (trainer_root / "manifest.json").write_text(
            _json_line(manifest) + "\n", encoding="utf-8"
        )

    def write_evaluator(
        self,
        evaluator_path: str | Path,
        trainer_path: str | Path,
        *,
        candidate_outcomes: dict[tuple[str, str], str],
    ) -> None:
        evaluator_root = Path(evaluator_path)
        trainer_root = Path(trainer_path)
        _require_disjoint_roots(trainer_root, evaluator_root)
        if evaluator_root.exists():
            raise FileExistsError(f"evaluator sidecar is create-once: {evaluator_root}")
        if not (trainer_root / "manifest.json").is_file():
            raise ValueError("evaluator sidecar requires an existing trainer manifest")
        expected_keys = {
            (candidate.content_hash, test_id)
            for candidate in self.candidates
            for task in self.tasks
            if candidate.task_id == task.task_id
            for test_id in task.test_order
        }
        if set(candidate_outcomes) != expected_keys or any(
            outcome not in OUTCOMES for outcome in candidate_outcomes.values()
        ):
            raise ValueError(
                "evaluator outcomes must key every candidate/test with a categorical outcome"
            )
        evaluator_root.mkdir(parents=True)
        sidecar_path = evaluator_root / "task_sidecars.jsonl"
        sidecar_path.write_text(
            "".join(
                _json_line(
                    {
                        key: value
                        for key, value in evaluator_view(item).items()
                        if key
                        in {"task_id", "gold_solution", "gold_patch"}
                    }
                )
                + "\n"
                for item in self.tasks
            ),
            encoding="utf-8",
        )
        outcome_path = evaluator_root / "candidate_outcomes.jsonl"
        outcome_path.write_text(
            "".join(
                _json_line(
                    {
                        "candidate_hash": candidate_hash,
                        "test_id": test_id,
                        "outcome": candidate_outcomes[(candidate_hash, test_id)],
                    }
                )
                + "\n"
                for candidate_hash, test_id in sorted(candidate_outcomes)
            ),
            encoding="utf-8",
        )
        evaluator_manifest = {
            "format": "pbpf-evaluator-sidecars-v1",
            "trainer_manifest_hash": _file_hash(trainer_root / "manifest.json"),
            "checksums": {
                "task_sidecars.jsonl": _file_hash(sidecar_path),
                "candidate_outcomes.jsonl": _file_hash(outcome_path),
            },
        }
        (evaluator_root / "manifest.json").write_text(
            _json_line(evaluator_manifest) + "\n", encoding="utf-8"
        )

    @classmethod
    def read_trainer(cls, path: str | Path) -> "TrajectoryBank":
        source = Path(path)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "pbpf-trajectory-bank-v1":
            raise ValueError("unsupported trajectory bank format")
        actual_files = {
            file.name for file in source.iterdir() if file.is_file() and file.name != "manifest.json"
        }
        if actual_files != set(manifest.get("checksums", {})):
            raise ValueError("trainer bank file inventory mismatch")
        for filename, expected in manifest.get("checksums", {}).items():
            if _file_hash(source / filename) != expected:
                raise ValueError(f"checksum mismatch for {filename}")

        def rows(filename: str) -> list[dict[str, Any]]:
            with (source / filename).open(encoding="utf-8") as stream:
                return [json.loads(line) for line in stream]

        public_rows = rows("tasks.jsonl")
        tasks = tuple(
            TaskRecord(
                **{
                    **row,
                    "test_order": tuple(row["test_order"]),
                    "group_ids": tuple(row["group_ids"]),
                }
            )
            for row in public_rows
        )
        candidates = tuple(CandidateVersion(**row) for row in rows("candidates.jsonl"))
        observations = tuple(Observation(**row) for row in rows("observations.jsonl"))
        registry = tuple(
            sorted(
                json.loads((source / "mutant_registry.json").read_text(encoding="utf-8")).items()
            )
        )
        bank = cls(tasks, candidates, observations, registry)
        with np.load(source / "arrays.npz", allow_pickle=False) as arrays:
            if arrays["outcome"].tolist() != [item.outcome for item in observations]:
                raise ValueError("NPZ and JSONL observation data disagree")
        return bank

    @classmethod
    def read_evaluator(
        cls, trainer_path: str | Path, evaluator_path: str | Path
    ) -> EvaluatorBundle:
        trainer_root = Path(trainer_path)
        evaluator_root = Path(evaluator_path)
        _require_disjoint_roots(trainer_root, evaluator_root)
        trainer_bank = cls.read_trainer(trainer_root)
        manifest = json.loads(
            (evaluator_root / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("format") != "pbpf-evaluator-sidecars-v1":
            raise ValueError("unsupported evaluator sidecar format")
        actual_files = {
            file.name
            for file in evaluator_root.iterdir()
            if file.is_file() and file.name != "manifest.json"
        }
        if actual_files != set(manifest.get("checksums", {})):
            raise ValueError("evaluator sidecar file inventory mismatch")
        if manifest.get("trainer_manifest_hash") != _file_hash(
            trainer_root / "manifest.json"
        ):
            raise ValueError("evaluator sidecar references a different trainer bank")
        for filename, expected in manifest.get("checksums", {}).items():
            if _file_hash(evaluator_root / filename) != expected:
                raise ValueError(f"checksum mismatch for evaluator/{filename}")
        with (evaluator_root / "task_sidecars.jsonl").open(encoding="utf-8") as stream:
            sidecars = {row["task_id"]: row for row in map(json.loads, stream)}
        tasks = tuple(
            TaskRecord(
                **{
                    **trainer_view(task),
                    **sidecars.get(task.task_id, {}),
                    "test_order": task.test_order,
                    "group_ids": task.group_ids,
                }
            )
            for task in trainer_bank.tasks
        )
        bank = cls(
            tasks,
            trainer_bank.candidates,
            trainer_bank.observations,
            trainer_bank.mutant_registry,
        )
        with (evaluator_root / "candidate_outcomes.jsonl").open(encoding="utf-8") as stream:
            candidate_outcomes = tuple(
                (row["candidate_hash"], row["test_id"], row["outcome"])
                for row in map(json.loads, stream)
            )
        expected_keys = {
            (candidate.content_hash, test_id)
            for candidate in bank.candidates
            for task in bank.tasks
            if candidate.task_id == task.task_id
            for test_id in task.test_order
        }
        if {(left, right) for left, right, _ in candidate_outcomes} != expected_keys:
            raise ValueError("evaluator candidate outcome keys do not match trainer bank")
        if any(outcome not in OUTCOMES for _, _, outcome in candidate_outcomes):
            raise ValueError("evaluator candidate outcome is invalid")
        return EvaluatorBundle(bank, candidate_outcomes)
