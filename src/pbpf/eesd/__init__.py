"""Effective-Evidence Self-Distillation utilities."""

from .evidence import (
    conservative_utility,
    correction_posterior,
    cosine_relevance_weights,
    dirichlet_predict,
    effective_evidence_weights,
    effective_mass,
    fixed_mass_weights,
    mass_rescaled_weights,
    score_predictions,
    source_cluster_bootstrap_gain,
    transition_categories,
)
from .shapley_relevance import (
    absolute_relevance,
    build_causal_labels,
    coalition_masks,
    edited_token_masks,
    exact_shapley_values,
    leave_one_out_values,
    subset_execution_messages,
)

__all__ = [
    "absolute_relevance",
    "build_causal_labels",
    "coalition_masks",
    "conservative_utility",
    "correction_posterior",
    "cosine_relevance_weights",
    "dirichlet_predict",
    "edited_token_masks",
    "effective_evidence_weights",
    "effective_mass",
    "exact_shapley_values",
    "fixed_mass_weights",
    "leave_one_out_values",
    "mass_rescaled_weights",
    "score_predictions",
    "source_cluster_bootstrap_gain",
    "subset_execution_messages",
    "transition_categories",
]
