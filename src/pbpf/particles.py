from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_vector(values: np.ndarray | list[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError(f"{name} contains invalid values")
    return array


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if np.isneginf(maximum):
        raise ValueError("at least one log weight must be finite")
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def normalize_log_weights(log_weights: np.ndarray | list[float]) -> np.ndarray:
    values = _as_vector(log_weights, "log_weights")
    return values - _logsumexp(values)


def effective_sample_size(log_weights: np.ndarray | list[float]) -> float:
    normalized = normalize_log_weights(log_weights)
    weights = np.exp(normalized)
    return float(1.0 / np.square(weights).sum())


def systematic_resample(
    weights: np.ndarray | list[float], *, rng: np.random.Generator
) -> np.ndarray:
    values = _as_vector(weights, "weights")
    if np.any(values < 0) or not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and have positive mass")
    probabilities = values / values.sum()
    count = len(probabilities)
    positions = (float(rng.random()) + np.arange(count)) / count
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="left").astype(np.int64)


@dataclass(frozen=True)
class ParticleUpdate:
    log_weights: np.ndarray
    ancestors: np.ndarray
    ess: float
    resampled: bool


def particle_update(
    previous_log_weights: np.ndarray | list[float],
    log_likelihood: np.ndarray | list[float],
    log_transition: np.ndarray | list[float],
    log_proposal: np.ndarray | list[float],
    *,
    rng: np.random.Generator,
    ess_fraction: float = 0.5,
) -> ParticleUpdate:
    if not 0.0 <= ess_fraction <= 1.0:
        raise ValueError("ess_fraction must be in [0, 1]")
    terms = [
        _as_vector(previous_log_weights, "previous_log_weights"),
        _as_vector(log_likelihood, "log_likelihood"),
        _as_vector(log_transition, "log_transition"),
        _as_vector(log_proposal, "log_proposal"),
    ]
    if len({len(term) for term in terms}) != 1:
        raise ValueError("particle update vectors must have identical shape")
    if np.isneginf(terms[3]).any():
        raise ValueError("proposal probabilities must be positive")
    corrected = terms[0] + terms[1] + terms[2] - terms[3]
    normalized = normalize_log_weights(corrected)
    ess = effective_sample_size(normalized)
    count = len(normalized)
    if ess < ess_fraction * count:
        ancestors = systematic_resample(np.exp(normalized), rng=rng)
        return ParticleUpdate(
            np.full(count, -np.log(count), dtype=np.float64), ancestors, ess, True
        )
    return ParticleUpdate(normalized, np.arange(count, dtype=np.int64), ess, False)


def sequence_mixture_logprob(
    component_token_logprobs: np.ndarray, log_weights: np.ndarray | list[float]
) -> float:
    tokens = np.asarray(component_token_logprobs, dtype=np.float64)
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
        raise ValueError("component_token_logprobs must have shape [particles, tokens]")
    if np.isnan(tokens).any() or np.isposinf(tokens).any():
        raise ValueError("component token log probabilities contain invalid values")
    weights = normalize_log_weights(log_weights)
    if tokens.shape[0] != len(weights):
        raise ValueError("particle count differs between components and weights")
    return _logsumexp(weights + tokens.sum(axis=1))


def tokenwise_mixture_logprob(
    component_token_logprobs: np.ndarray, log_weights: np.ndarray | list[float]
) -> float:
    """Named fault ablation: re-mix particles independently at each token."""
    tokens = np.asarray(component_token_logprobs, dtype=np.float64)
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
        raise ValueError("component_token_logprobs must have shape [particles, tokens]")
    weights = normalize_log_weights(log_weights)
    if tokens.shape[0] != len(weights):
        raise ValueError("particle count differs between components and weights")
    return float(sum(_logsumexp(weights + tokens[:, index]) for index in range(tokens.shape[1])))
