from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .metrics import brier_score, future_nll, posterior_kl
from .posterior import exact_discrete_posterior, posterior_predictive
from .particles import particle_update, systematic_resample


def _arm_metrics(
    posterior: np.ndarray,
    exact: np.ndarray,
    future_probabilities: np.ndarray,
    future_outcomes: Sequence[int],
) -> dict[str, float]:
    stabilized = np.maximum(posterior, 1e-12)
    stabilized /= stabilized.sum()
    predictive = posterior_predictive(stabilized, future_probabilities)
    return {
        "future_nll": future_nll(predictive, list(future_outcomes)),
        "brier": brier_score(predictive, list(future_outcomes)),
        "posterior_kl": posterior_kl(exact, stabilized),
        "posterior": stabilized.tolist(),
    }


def run_finite_audit(
    *,
    prior: np.ndarray,
    outcome_probabilities: np.ndarray,
    observed_outcomes: Sequence[int],
    future_outcomes: Sequence[int],
    particle_counts: Sequence[int] = (1, 4, 8, 16, 32),
    rng: np.random.Generator,
    log_transition_terms: np.ndarray | None = None,
    log_proposal_terms: np.ndarray | None = None,
) -> dict[str, Any]:
    prior_values = np.asarray(prior, dtype=np.float64)
    probabilities = np.asarray(outcome_probabilities, dtype=np.float64)
    prefix = len(observed_outcomes)
    if probabilities.ndim != 3 or probabilities.shape[1] != len(prior_values):
        raise ValueError("outcome_probabilities must be [tests, hypotheses, outcomes]")
    if prefix + len(future_outcomes) != probabilities.shape[0]:
        raise ValueError("observed and future outcomes must cover all tests")
    if any(count <= 0 for count in particle_counts):
        raise ValueError("particle counts must be positive")
    factors = np.asarray(
        [probabilities[index, :, outcome] for index, outcome in enumerate(observed_outcomes)]
    )
    exact = exact_discrete_posterior(prior_values, factors)
    future = probabilities[prefix:]
    arms: dict[str, dict[str, float | int]] = {}
    arms["exact_bayes"] = _arm_metrics(exact, exact, future, future_outcomes)
    map_posterior = np.zeros_like(exact)
    map_posterior[int(np.argmax(exact))] = 1.0
    arms["map"] = _arm_metrics(map_posterior, exact, future, future_outcomes)
    arms["posterior_mean"] = _arm_metrics(exact, exact, future, future_outcomes)
    uniform = np.full_like(exact, 1.0 / len(exact))
    arms["uniform"] = _arm_metrics(uniform, exact, future, future_outcomes)
    shuffled_factors = np.asarray(
        [np.roll(factor, 1 + (index % max(1, len(factor) - 1))) for index, factor in enumerate(factors)]
    )
    shuffled = exact_discrete_posterior(prior_values, shuffled_factors)
    arms["shuffled"] = _arm_metrics(shuffled, exact, future, future_outcomes)

    normalized_prior = prior_values / prior_values.sum()
    transition_terms = (
        np.zeros((prefix, len(prior_values)), dtype=np.float64)
        if log_transition_terms is None
        else np.asarray(log_transition_terms, dtype=np.float64)
    )
    proposal_terms = (
        np.zeros((prefix, len(prior_values)), dtype=np.float64)
        if log_proposal_terms is None
        else np.asarray(log_proposal_terms, dtype=np.float64)
    )
    if transition_terms.shape != (prefix, len(prior_values)) or proposal_terms.shape != transition_terms.shape:
        raise ValueError("transition/proposal terms must be [prefix, hypotheses]")
    if np.any(transition_terms != 0.0) or np.any(proposal_terms != 0.0):
        raise ValueError(
            "nonidentity finite transition/proposal terms require a proposal sampler with recorded draws"
        )
    for count in particle_counts:
        hypotheses = systematic_resample(normalized_prior, rng=rng)
        if count != len(prior_values):
            positions = (float(rng.random()) + np.arange(count)) / count
            hypotheses = np.searchsorted(np.cumsum(normalized_prior), positions).astype(np.int64)
        log_weights = np.full(count, -np.log(count), dtype=np.float64)
        lineage = np.arange(count, dtype=np.int64)
        steps: list[dict[str, Any]] = []
        for index, factor in enumerate(factors):
            likelihood = np.full(count, -np.inf, dtype=np.float64)
            positive = factor[hypotheses] > 0
            likelihood[positive] = np.log(factor[hypotheses][positive])
            update = particle_update(
                log_weights,
                likelihood,
                transition_terms[index, hypotheses],
                proposal_terms[index, hypotheses],
                rng=rng,
            )
            if update.resampled:
                hypotheses = hypotheses[update.ancestors]
                lineage = lineage[update.ancestors]
            log_weights = update.log_weights
            steps.append(
                {
                    "ess": update.ess,
                    "resampled": update.resampled,
                    "ancestors": update.ancestors.tolist(),
                    "unique_ancestors": int(len(np.unique(lineage))),
                }
            )
        weights = np.exp(log_weights)
        approximation = np.bincount(hypotheses, weights=weights, minlength=len(prior_values))
        metrics = _arm_metrics(approximation, exact, future, future_outcomes)
        metrics["ess"] = float(1.0 / np.square(weights).sum())
        metrics["unique_ancestors"] = int(len(np.unique(lineage)))
        metrics["steps"] = steps
        arms[f"particles_{count}"] = metrics

    numeric_values = [
        value
        for metrics in arms.values()
        for value in metrics.values()
        if isinstance(value, (float, int))
    ]
    largest = f"particles_{max(particle_counts)}"
    smallest = f"particles_{min(particle_counts)}"
    gate = {
        "finite_values": bool(np.isfinite(numeric_values).all()),
        "largest_particle_kl_no_worse_than_smallest": bool(
            arms[largest]["posterior_kl"] <= arms[smallest]["posterior_kl"]
        ),
    }
    return {"exact_posterior": exact.tolist(), "arms": arms, "gate": gate}
