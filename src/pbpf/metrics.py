from __future__ import annotations

import numpy as np


def _validated_predictions(
    probabilities: np.ndarray, outcomes: list[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(outcomes, dtype=np.int64)
    if predicted.ndim != 2 or len(predicted) != len(truth) or predicted.shape[1] < 2:
        raise ValueError("probabilities must be [examples, outcomes] and match outcomes")
    if not np.isfinite(predicted).all() or np.any(predicted < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(predicted.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("probability rows must sum to one")
    if np.any(truth < 0) or np.any(truth >= predicted.shape[1]):
        raise ValueError("outcome index out of range")
    return predicted, truth


def future_nll(probabilities: np.ndarray, outcomes: list[int] | np.ndarray) -> float:
    predicted, truth = _validated_predictions(probabilities, outcomes)
    selected = predicted[np.arange(len(truth)), truth]
    if np.any(selected <= 0):
        raise ValueError("observed outcomes must have positive predicted probability")
    return float(-np.log(selected).mean())


def brier_score(probabilities: np.ndarray, outcomes: list[int] | np.ndarray) -> float:
    predicted, truth = _validated_predictions(probabilities, outcomes)
    targets = np.zeros_like(predicted)
    targets[np.arange(len(truth)), truth] = 1.0
    return float(np.square(predicted - targets).sum(axis=1).mean())


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: list[int] | np.ndarray, bins: int = 10
) -> float:
    predicted, truth = _validated_predictions(probabilities, outcomes)
    if bins <= 0:
        raise ValueError("bins must be positive")
    confidence = predicted.max(axis=1)
    correct = predicted.argmax(axis=1) == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(truth)
    value = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if upper_inclusive
            else confidence < edges[index + 1]
        )
        if mask.any():
            value += float(mask.sum() / total) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return value


def posterior_kl(reference: np.ndarray, approximation: np.ndarray) -> float:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(approximation, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("posterior vectors must have equal one-dimensional shape")
    if np.any(left < 0) or np.any(right <= 0):
        raise ValueError("reference must be non-negative and approximation positive")
    mask = left > 0
    return float(np.sum(left[mask] * (np.log(left[mask]) - np.log(right[mask]))))


def randomized_hpd_coverage(
    posterior: np.ndarray, true_index: int, *, mass: float = 0.9, rng: np.random.Generator
) -> bool:
    values = np.asarray(posterior, dtype=np.float64)
    if not 0 < mass <= 1 or true_index < 0 or true_index >= len(values):
        raise ValueError("invalid HPD arguments")
    jitter = rng.random(len(values))
    order = np.lexsort((jitter, -values))
    cumulative = np.cumsum(values[order])
    cutoff = int(np.searchsorted(cumulative, mass, side="left"))
    return bool(true_index in order[: cutoff + 1])
