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


# Parameter-free three-state correction trust used by the axiomatic EESD path.
# These categories are induced directly by execution success changes; no
# hand-authored utility vector is required.
IMPROVED, UNCHANGED, REGRESSED = range(3)


def transition_benefit_categories(before, after, *, pass_index: int = 0) -> np.ndarray:
    """Map before/after execution outcomes to improve/unchanged/regress."""
    before = np.asarray(before, dtype=np.int64)
    after = np.asarray(after, dtype=np.int64)
    if before.ndim != 1 or before.shape != after.shape or not len(before):
        raise ValueError("before/after outcomes must be matching nonempty vectors")
    before_pass = before == pass_index
    after_pass = after == pass_index
    result = np.full(len(before), UNCHANGED, dtype=np.int64)
    result[(~before_pass) & after_pass] = IMPROVED
    result[before_pass & (~after_pass)] = REGRESSED
    return result


def correction_benefit_posterior(
    before,
    after,
    relevance,
    *,
    alpha: float = 0.5,
    mass_rule: str = "effective",
    fixed_mass: float | None = None,
    pass_index: int = 0,
) -> np.ndarray:
    """Dirichlet parameters over improve/unchanged/regress correction effects."""
    categories = transition_benefit_categories(before, after, pass_index=pass_index)
    raw = _vector(relevance, name="relevance", nonnegative=True)
    if len(raw) != len(categories):
        raise ValueError("relevance must match transition count")
    if raw.sum() <= 0:
        raise ValueError("relevance must contain positive mass")
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
    return np.bincount(categories, weights=weights, minlength=3) + float(alpha)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Stable continued fraction for the regularized incomplete beta function."""
    max_iterations = 256
    epsilon = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= epsilon:
            return float(h)
    raise RuntimeError("incomplete-beta continued fraction failed to converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Return I_x(a,b) without requiring SciPy."""
    x, a, b = float(x), float(a), float(b)
    if (
        not math.isfinite(x)
        or not math.isfinite(a)
        or not math.isfinite(b)
        or not 0.0 <= x <= 1.0
        or a <= 0.0
        or b <= 0.0
    ):
        raise ValueError("beta arguments require x in [0,1] and positive finite shapes")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_beta_factor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    factor = math.exp(log_beta_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        value = factor * _beta_continued_fraction(a, b, x) / a
    else:
        value = 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b
    return float(min(1.0, max(0.0, value)))


def posterior_benefit_probability(parameters) -> float:
    """P(theta_improved > theta_regressed) under a 3-state Dirichlet posterior.

    The neutral component integrates out. The normalized improve-vs-regress share
    is Beta(alpha_improved, alpha_regressed), so the probability has a closed
    one-dimensional beta-CDF form.
    """
    values = _vector(parameters, name="parameters", nonnegative=True)
    if len(values) != 3 or (values <= 0).any():
        raise ValueError("benefit posterior requires three strictly positive parameters")
    return float(1.0 - regularized_incomplete_beta(0.5, values[IMPROVED], values[REGRESSED]))


def posterior_mean_advantage(parameters) -> float:
    """Posterior mean of improve-minus-regress probability mass."""
    values = _vector(parameters, name="parameters", nonnegative=True)
    if len(values) != 3 or (values <= 0).any() or values.sum() <= 0:
        raise ValueError("benefit posterior requires three strictly positive parameters")
    return float((values[IMPROVED] - values[REGRESSED]) / values.sum())


def posterior_update_weight(parameters) -> float:
    """Bayes-optimal nonnegative update strength under execution delta reward.

    A future public execution has reward +1 for improvement, 0 for unchanged,
    and -1 for regression. Under the Dirichlet posterior, its posterior-predictive
    expected reward is E[theta_improved-theta_regressed]. Updating has that value;
    abstaining has value zero. The Bayes action therefore uses the positive part
    of the posterior mean advantage. Weak effective evidence is automatically
    shrunk toward zero by the symmetric prior; no confidence threshold or
    uncertainty-penalty hyperparameter is introduced.
    """
    return float(max(0.0, posterior_mean_advantage(parameters)))
