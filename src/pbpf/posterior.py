from __future__ import annotations

import numpy as np

from .particles import normalize_log_weights


def exact_discrete_posterior(
    prior: np.ndarray | list[float], likelihood_factors: np.ndarray
) -> np.ndarray:
    prior_values = np.asarray(prior, dtype=np.float64)
    factors = np.asarray(likelihood_factors, dtype=np.float64)
    if prior_values.ndim != 1 or prior_values.size == 0:
        raise ValueError("prior must be a non-empty vector")
    if factors.ndim != 2 or factors.shape[1] != len(prior_values):
        raise ValueError("likelihood_factors must have shape [observations, hypotheses]")
    if np.any(prior_values < 0) or prior_values.sum() <= 0:
        raise ValueError("prior must have non-negative positive mass")
    if np.any(factors < 0) or not np.isfinite(factors).all():
        raise ValueError("likelihood factors must be finite and non-negative")
    log_prior = np.full_like(prior_values, -np.inf)
    positive_prior = prior_values > 0
    log_prior[positive_prior] = np.log(prior_values[positive_prior])
    log_factors = np.full_like(factors, -np.inf)
    positive_factors = factors > 0
    log_factors[positive_factors] = np.log(factors[positive_factors])
    return np.exp(normalize_log_weights(log_prior + log_factors.sum(axis=0)))


def posterior_predictive(posterior: np.ndarray, outcome_probabilities: np.ndarray) -> np.ndarray:
    beliefs = np.asarray(posterior, dtype=np.float64)
    probabilities = np.asarray(outcome_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[1] != len(beliefs):
        raise ValueError("outcome_probabilities must have shape [tests, hypotheses, outcomes]")
    predictive = np.einsum("h,tho->to", beliefs, probabilities)
    totals = predictive.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("each future test must have positive predictive mass")
    return predictive / totals
