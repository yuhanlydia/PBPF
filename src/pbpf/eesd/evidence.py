"""Effective-evidence probability and correction-trust primitives.

The inverse-squared concentration functional is used as an adaptive shrinkage
rule. It is not claimed to estimate dependence among executions.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _vector(values, *, name: str, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    if nonnegative and (array < 0).any():
        raise ValueError(f"{name} must be nonnegative")
    return array


def normalize_relevance(weights: Iterable[float]) -> np.ndarray:
    values = _vector(weights, name="weights", nonnegative=True)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("weights must contain positive mass")
    return values / total


def effective_mass(weights_or_probabilities: Iterable[float]) -> float:
    """Return inverse squared concentration in [1,n]."""
    p = normalize_relevance(weights_or_probabilities)
    return float(1.0 / np.square(p).sum())


def mass_rescaled_weights(weights: Iterable[float], mass: float) -> np.ndarray:
    if not math.isfinite(float(mass)) or mass <= 0:
        raise ValueError("mass must be positive and finite")
    return normalize_relevance(weights) * float(mass)


def fixed_mass_weights(weights: Iterable[float]) -> np.ndarray:
    values = _vector(weights, name="weights", nonnegative=True)
    return mass_rescaled_weights(values, len(values))


def effective_evidence_weights(weights: Iterable[float]) -> np.ndarray:
    values = _vector(weights, name="weights", nonnegative=True)
    return mass_rescaled_weights(values, effective_mass(values))


def cosine_relevance_weights(
    query: Iterable[float],
    history: Iterable[Iterable[float]],
    strength: float,
) -> np.ndarray:
    """Cosine relevance with total mass equal to history length.

    Strength zero recovers uniform fixed-mass weights.
    """
    q = np.asarray(query, dtype=float)
    h = np.asarray(history, dtype=float)
    if (
        q.ndim != 1
        or h.ndim != 2
        or not len(q)
        or not len(h)
        or h.shape[1] != len(q)
        or not np.isfinite(q).all()
        or not np.isfinite(h).all()
        or not math.isfinite(float(strength))
        or strength < 0
    ):
        raise ValueError("invalid query/history/strength")
    norms = np.linalg.norm(h, axis=1) * np.linalg.norm(q)
    similarity = np.divide(h @ q, norms, out=np.zeros(len(h)), where=norms > 0)
    logits = float(strength) * np.clip(similarity, -1.0, 1.0)
    raw = np.exp(logits - logits.max())
    return fixed_mass_weights(raw)


def _category_values(outcomes: Iterable[int], classes: int) -> np.ndarray:
    values = np.asarray(list(outcomes))
    if (
        type(classes) is not int
        or classes < 2
        or values.ndim != 1
        or (values.size and not np.issubdtype(values.dtype, np.integer))
        or (values.size and ((values < 0).any() or (values >= classes).any()))
    ):
        raise ValueError("outcomes must be integer category indices")
    return values.astype(np.int64)


def dirichlet_predict(
    outcomes: Iterable[int],
    alpha: float,
    *,
    weights: Iterable[float] | None = None,
    classes: int = 5,
) -> np.ndarray:
    values = _category_values(outcomes, classes)
    if not math.isfinite(float(alpha)) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    if weights is None:
        evidence = np.ones(len(values), dtype=float)
    else:
        evidence = np.asarray(list(weights), dtype=float)
        if (
            evidence.shape != values.shape
            or not np.isfinite(evidence).all()
            or (evidence < 0).any()
        ):
            raise ValueError("weights must match outcomes and be finite nonnegative")
    posterior = np.bincount(values, weights=evidence, minlength=classes) + float(alpha)
    return posterior / posterior.sum()


def _validate_predictions(labels, probabilities) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=float)
    if (
        y.ndim != 1
        or p.ndim != 2
        or len(y) != len(p)
        or not len(y)
        or (y < 0).any()
        or (y >= p.shape[1]).any()
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("invalid labels/probabilities")
    return y, p


def score_predictions(labels, probabilities, *, ece_bins: int = 10) -> dict[str, float]:
    y, p = _validate_predictions(labels, probabilities)
    if type(ece_bins) is not int or ece_bins < 2:
        raise ValueError("ece_bins must be an integer >=2")
    chosen = p[np.arange(len(y)), y].clip(1e-12, 1.0)
    predictions = p.argmax(axis=1)
    one_hot = np.eye(p.shape[1], dtype=float)[y]
    confidence = p.max(axis=1)
    correct = (predictions == y).astype(float)
    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    ece = 0.0
    for i in range(ece_bins):
        if i == ece_bins - 1:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "nll": float(-np.log(chosen).mean()),
        "brier": float(np.square(p - one_hot).sum(axis=1).mean()),
        "accuracy": float(correct.mean()),
        "ece": float(ece),
    }


def source_cluster_bootstrap_gain(
    labels,
    proposed,
    baseline,
    clusters,
    *,
    seed: int = 314159,
    replicates: int = 10_000,
) -> dict[str, float | list[float]]:
    """Paired source-cluster bootstrap for baseline NLL minus proposed NLL."""
    y, a = _validate_predictions(labels, proposed)
    y2, b = _validate_predictions(labels, baseline)
    c = np.asarray(clusters)
    if not np.array_equal(y, y2) or c.shape != y.shape or replicates < 1:
        raise ValueError("bootstrap inputs differ or replicates invalid")
    la = -np.log(a[np.arange(len(y)), y].clip(1e-12, 1.0))
    lb = -np.log(b[np.arange(len(y)), y].clip(1e-12, 1.0))
    unique = np.unique(c)
    if not len(unique):
        raise ValueError("at least one source cluster is required")
    per_cluster = np.asarray([float((lb[c == key] - la[c == key]).mean()) for key in unique])
    point = float(per_cluster.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for i in range(replicates):
        draws[i] = float(rng.choice(per_cluster, size=len(per_cluster), replace=True).mean())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "gain": point,
        "ci95": [float(lo), float(hi)],
        "clusters": int(len(unique)),
        "replicates": int(replicates),
    }


# Transition category order used throughout EESD.
FIX, REGRESSION, PRESERVED, UNRESOLVED = range(4)


def transition_categories(before, after, *, pass_index: int = 0) -> np.ndarray:
    before = np.asarray(before, dtype=np.int64)
    after = np.asarray(after, dtype=np.int64)
    if before.ndim != 1 or before.shape != after.shape or not len(before):
        raise ValueError("before/after outcomes must be matching nonempty vectors")
    result = np.full(len(before), UNRESOLVED, dtype=np.int64)
    before_pass = before == pass_index
    after_pass = after == pass_index
    result[(~before_pass) & after_pass] = FIX
    result[before_pass & (~after_pass)] = REGRESSION
    result[before_pass & after_pass] = PRESERVED
    return result


def correction_posterior(
    before,
    after,
    relevance,
    *,
    alpha: float = 0.5,
    mass_rule: str = "effective",
    fixed_mass: float | None = None,
    pass_index: int = 0,
) -> np.ndarray:
    transitions = transition_categories(before, after, pass_index=pass_index)
    raw = _vector(relevance, name="relevance", nonnegative=True)
    if len(raw) != len(transitions):
        raise ValueError("relevance must match transition count")
    if mass_rule == "effective":
        weights = effective_evidence_weights(raw)
    elif mass_rule == "fixed":
        weights = fixed_mass_weights(raw)
    elif mass_rule == "global":
        if fixed_mass is None:
            raise ValueError("global mass rule requires fixed_mass")
        weights = mass_rescaled_weights(raw, fixed_mass)
    else:
        raise ValueError("unknown mass rule")
    if not math.isfinite(float(alpha)) or alpha <= 0:
        raise ValueError("alpha must be positive")
    return np.bincount(transitions, weights=weights, minlength=4) + float(alpha)


def dirichlet_linear_moments(parameters, utility) -> tuple[float, float]:
    a = _vector(parameters, name="parameters", nonnegative=True)
    u = _vector(utility, name="utility")
    if len(a) != len(u) or a.sum() <= 0:
        raise ValueError("Dirichlet parameters and utility must match")
    total = float(a.sum())
    mean_prob = a / total
    mean = float(mean_prob @ u)
    variance = float((mean_prob @ np.square(u) - mean * mean) / (total + 1.0))
    return mean, max(variance, 0.0)


def conservative_utility(parameters, utility, *, uncertainty_penalty: float) -> dict[str, float]:
    if not math.isfinite(float(uncertainty_penalty)) or uncertainty_penalty < 0:
        raise ValueError("uncertainty_penalty must be finite and nonnegative")
    mean, variance = dirichlet_linear_moments(parameters, utility)
    standard_deviation = math.sqrt(variance)
    value = mean - float(uncertainty_penalty) * standard_deviation
    return {
        "mean": mean,
        "variance": variance,
        "std": standard_deviation,
        "conservative": float(value),
        "positive_weight": float(max(value, 0.0)),
    }
