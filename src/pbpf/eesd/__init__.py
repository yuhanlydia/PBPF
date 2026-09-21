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

__all__ = [
    "conservative_utility",
    "correction_posterior",
    "cosine_relevance_weights",
    "dirichlet_predict",
    "effective_evidence_weights",
    "effective_mass",
    "fixed_mass_weights",
    "mass_rescaled_weights",
    "score_predictions",
    "source_cluster_bootstrap_gain",
    "transition_categories",
]
