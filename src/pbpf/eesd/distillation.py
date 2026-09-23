"""Correction-trajectory trust rules for EESD ablations.

The main EESD path has no hand-authored transition utility or uncertainty penalty.
Execution outcomes induce improve/unchanged/regress states; relevance controls the
Dirichlet evidence distribution and effective mass; posterior benefit confidence
directly determines the update weight.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np

from .evidence import (
    correction_benefit_posterior,
    posterior_benefit_probability,
    posterior_mean_advantage,
    posterior_update_weight,
)


TRAIN_RULES = (
    "no_update",
    "equal_weight",
    "final_correctness",
    "scalar_confidence",
    "fixed_mass_dirichlet",
    "eed_mean_no_uncertainty",
    "eed_no_anchor",
    "eesd_full",
)


def _as_int_vector(values: Iterable[int], *, name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.int64)
    if result.ndim != 1 or not result.size:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    return result


def score_trajectory(
    before,
    after,
    relevance,
    *,
    alpha: float,
    utility=None,
    uncertainty_penalty=None,
    pass_index: int = 0,
) -> dict:
    """Return all predeclared training weights for one correction trajectory.

    utility and uncertainty_penalty remain accepted only for compatibility with
    sealed older orchestration. They do not affect the axiomatic EESD path.
    """
    before = _as_int_vector(before, name="before")
    after = _as_int_vector(after, name="after")
    relevance = np.asarray(list(relevance), dtype=float)
    if (
        before.shape != after.shape
        or relevance.shape != before.shape
        or not np.isfinite(relevance).all()
        or (relevance < 0).any()
        or relevance.sum() <= 0
    ):
        raise ValueError("before/after/relevance must be matching valid vectors")
    if not math.isfinite(float(alpha)) or alpha <= 0:
        raise ValueError("alpha must be positive")

    fixed = correction_benefit_posterior(
        before, after, relevance, alpha=alpha, mass_rule="fixed", pass_index=pass_index
    )
    effective = correction_benefit_posterior(
        before, after, relevance, alpha=alpha, mass_rule="effective", pass_index=pass_index
    )

    fixed_probability = posterior_benefit_probability(fixed)
    effective_probability = posterior_benefit_probability(effective)
    effective_mean = posterior_mean_advantage(effective)

    after_pass = after == pass_index
    final_correctness = float(after_pass.all())
    scalar_confidence = float(after_pass.mean())

    weights = {
        "no_update": 0.0,
        "equal_weight": 1.0,
        "final_correctness": final_correctness,
        "scalar_confidence": scalar_confidence,
        "fixed_mass_dirichlet": posterior_update_weight(fixed),
        "eed_mean_no_uncertainty": float(max(effective_mean, 0.0)),
        "eed_no_anchor": posterior_update_weight(effective),
        "eesd_full": posterior_update_weight(effective),
    }
    if set(weights) != set(TRAIN_RULES):
        raise RuntimeError("training rule registry drift")
    return {
        "fixed_posterior": fixed.tolist(),
        "effective_posterior": effective.tolist(),
        "fixed_benefit_probability": fixed_probability,
        "effective_benefit_probability": effective_probability,
        "effective_mean_advantage": effective_mean,
        "fixed_update_weight": weights["fixed_mass_dirichlet"],
        "effective_update_weight": weights["eesd_full"],
        "final_correctness": final_correctness,
        "scalar_confidence": scalar_confidence,
        "deprecated_utility_argument_used": utility is not None,
        "deprecated_uncertainty_penalty_argument_used": uncertainty_penalty is not None,
        "weights": weights,
    }


def validate_scored_record(record: Mapping) -> None:
    required = {
        "trajectory_id",
        "source_component_id",
        "split",
        "prompt",
        "correction",
        "before_outcomes",
        "after_outcomes",
        "relevance",
        "training_weights",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"scored trajectory missing: {sorted(missing)}")
    if set(record["training_weights"]) != set(TRAIN_RULES):
        raise ValueError("scored trajectory training rules differ from lock")
    for name, value in record["training_weights"].items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid training weight for {name}")
