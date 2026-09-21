"""Correction-trajectory trust rules for EESD ablations."""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np

from .evidence import conservative_utility, correction_posterior


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
    utility,
    uncertainty_penalty: float,
    pass_index: int = 0,
) -> dict:
    """Return all predeclared training weights for one correction trajectory."""
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

    fixed = correction_posterior(
        before, after, relevance, alpha=alpha, mass_rule="fixed", pass_index=pass_index
    )
    effective = correction_posterior(
        before, after, relevance, alpha=alpha, mass_rule="effective", pass_index=pass_index
    )
    fixed_u = conservative_utility(
        fixed, utility, uncertainty_penalty=uncertainty_penalty
    )
    effective_mean = conservative_utility(
        effective, utility, uncertainty_penalty=0.0
    )
    effective_u = conservative_utility(
        effective, utility, uncertainty_penalty=uncertainty_penalty
    )

    after_pass = after == pass_index
    final_correctness = float(after_pass.all())
    scalar_confidence = float(after_pass.mean())

    weights = {
        "no_update": 0.0,
        "equal_weight": 1.0,
        "final_correctness": final_correctness,
        "scalar_confidence": scalar_confidence,
        "fixed_mass_dirichlet": fixed_u["positive_weight"],
        "eed_mean_no_uncertainty": effective_mean["positive_weight"],
        "eed_no_anchor": effective_u["positive_weight"],
        "eesd_full": effective_u["positive_weight"],
    }
    if set(weights) != set(TRAIN_RULES):
        raise RuntimeError("training rule registry drift")
    return {
        "fixed_posterior": fixed.tolist(),
        "effective_posterior": effective.tolist(),
        "fixed_utility": fixed_u,
        "effective_mean_utility": effective_mean,
        "effective_conservative_utility": effective_u,
        "final_correctness": final_correctness,
        "scalar_confidence": scalar_confidence,
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
