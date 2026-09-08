from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .bank import TrajectoryBank
from .history import feedback_view
from .manifest import trainer_view
from .metrics import brier_score, future_nll
from .registry import OUTCOMES


@dataclass(frozen=True)
class PredictionReport:
    task_count: int
    future_nll: float
    brier: float
    predictions: tuple[tuple[tuple[float, ...], ...], ...]
    keyed_predictions: tuple["KeyedPrediction", ...]
    task_scores: tuple["TaskPredictionScore", ...]
    bank_hash: str | None = None


@dataclass(frozen=True)
class KeyedPrediction:
    task_id: str
    candidate_hash: str
    test_id: str
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class TaskPredictionScore:
    task_id: str
    source_id: str
    future_nll: float
    brier: float


def run_prediction(
    bank: TrajectoryBank,
    predictor: Any,
    *,
    prefix_cutoffs: Mapping[str, int],
    future_outcomes_by_candidate: Mapping[tuple[str, str], str],
    feedback_mode: str = "status_only",
) -> PredictionReport:
    if set(prefix_cutoffs) != {task.task_id for task in bank.tasks}:
        raise ValueError("prefix_cutoffs must key every task exactly once")
    expected_truth_keys = {
        (candidate.content_hash, test_id)
        for candidate in bank.candidates
        for task in bank.tasks
        if candidate.task_id == task.task_id
        for test_id in task.test_order
    }
    if set(future_outcomes_by_candidate) != expected_truth_keys:
        raise ValueError(
            "evaluator truth must key every candidate version and test exactly once"
        )
    if any(outcome not in OUTCOMES for outcome in future_outcomes_by_candidate.values()):
        raise ValueError("evaluator truth contains an invalid categorical outcome")
    legacy_predictions: list[np.ndarray] = []
    keyed: list[KeyedPrediction] = []
    task_scores: list[TaskPredictionScore] = []
    for task in bank.tasks:
        cutoff = prefix_cutoffs[task.task_id]
        if type(cutoff) is not int or cutoff < 0 or cutoff >= len(task.test_order):
            raise ValueError("each prefix cutoff must leave at least one future test")
        prefix_tests = task.test_order[:cutoff]
        future_tests = task.test_order[cutoff:]
        public_task = trainer_view(task)
        task_probabilities: list[np.ndarray] = []
        task_outcomes: list[int] = []
        for candidate in bank.candidates:
            if candidate.task_id != task.task_id:
                continue
            history = tuple(
                observation
                for observation in bank.observations
                if observation.candidate_hash == candidate.content_hash
            )
            history_ids = tuple(observation.test_id for observation in history)
            if any(test_id in future_tests for test_id in history_ids):
                raise ValueError("history/evaluation overlap for candidate version and test")
            if history_ids != prefix_tests:
                raise ValueError("candidate history must exactly match the declared prefix cutoff")
            prediction = predictor.predict(
                public_task, candidate, feedback_view(history, feedback_mode)
            )
            if not isinstance(prediction, Mapping) or set(prediction) != set(future_tests):
                raise ValueError("predictor must key one categorical row per future test")
            candidate_rows = []
            for test_id in future_tests:
                outcome = future_outcomes_by_candidate[(candidate.content_hash, test_id)]
                row = np.asarray(prediction[test_id], dtype=np.float64)
                if row.shape != (len(OUTCOMES),):
                    raise ValueError("each keyed prediction must contain five outcomes")
                if np.any(row < 0) or not np.isfinite(row).all() or not np.isclose(row.sum(), 1.0):
                    raise ValueError("predictor outputs must be finite probability rows")
                candidate_rows.append(row)
                task_probabilities.append(row)
                task_outcomes.append(OUTCOMES.index(outcome))
                keyed.append(
                    KeyedPrediction(
                        task.task_id,
                        candidate.content_hash,
                        test_id,
                        tuple(float(value) for value in row),
                    )
                )
            legacy_predictions.append(np.asarray(candidate_rows))
        if not task_probabilities:
            raise ValueError(f"task {task.task_id} has no candidates")
        stacked = np.asarray(task_probabilities)
        source_id = task.group_ids[0] if task.group_ids else task.task_id
        task_scores.append(
            TaskPredictionScore(
                task.task_id,
                source_id,
                future_nll(stacked, task_outcomes),
                brier_score(stacked, task_outcomes),
            )
        )
    sources = tuple(dict.fromkeys(score.source_id for score in task_scores))
    macro_nll = float(
        np.mean(
            [
                np.mean([score.future_nll for score in task_scores if score.source_id == source])
                for source in sources
            ]
        )
    )
    macro_brier = float(
        np.mean(
            [
                np.mean([score.brier for score in task_scores if score.source_id == source])
                for source in sources
            ]
        )
    )
    return PredictionReport(
        len(bank.tasks),
        macro_nll,
        macro_brier,
        tuple(
            tuple(tuple(float(value) for value in row) for row in candidate)
            for candidate in legacy_predictions
        ),
        tuple(keyed),
        tuple(task_scores),
        bank.trainer_content_hash,
    )
