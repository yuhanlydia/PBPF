import numpy as np

from pbpf.eesd.evidence import (
    FIX,
    PRESERVED,
    REGRESSION,
    UNRESOLVED,
    conservative_utility,
    correction_posterior,
    dirichlet_predict,
    effective_evidence_weights,
    effective_mass,
    fixed_mass_weights,
    score_predictions,
    transition_categories,
)


def test_effective_mass_bounds_and_uniform_recovery():
    assert effective_mass([1, 1, 1, 1]) == 4.0
    assert np.isclose(effective_mass([1, 0, 0, 0]), 1.0)
    assert np.allclose(effective_evidence_weights([1, 1, 1, 1]), np.ones(4))


def test_fixed_and_effective_share_direction_but_not_mass():
    raw = np.array([0.85, 0.05, 0.05, 0.05])
    fixed = fixed_mass_weights(raw)
    effective = effective_evidence_weights(raw)
    assert np.isclose(fixed.sum(), 4.0)
    assert effective.sum() < fixed.sum()
    assert np.allclose(fixed / fixed.sum(), effective / effective.sum())


def test_symmetric_prior_preserves_argmax_under_positive_mass_scaling():
    outcomes = [0, 0, 1, 2]
    raw = [0.7, 0.1, 0.1, 0.1]
    fixed = dirichlet_predict(outcomes, 0.1, weights=fixed_mass_weights(raw), classes=5)
    effective = dirichlet_predict(outcomes, 0.1, weights=effective_evidence_weights(raw), classes=5)
    assert fixed.argmax() == effective.argmax()


def test_transition_taxonomy_and_conservative_weight():
    before = [1, 0, 0, 2]
    after = [0, 1, 0, 3]
    assert transition_categories(before, after).tolist() == [
        FIX, REGRESSION, PRESERVED, UNRESOLVED
    ]
    posterior = correction_posterior(before, after, [1, 1, 1, 1], alpha=0.5)
    value = conservative_utility(posterior, [1.0, -1.0, 0.0, -0.25], uncertainty_penalty=0.5)
    assert value["positive_weight"] >= 0
    assert value["variance"] >= 0


def test_prediction_metrics_are_finite():
    p = np.array([[0.8, 0.2], [0.25, 0.75]])
    result = score_predictions([0, 1], p, ece_bins=5)
    assert result["accuracy"] == 1.0
    assert all(np.isfinite(v) for v in result.values())


def test_distillation_rules_are_same_bank_weighting_only():
    from pbpf.eesd.distillation import TRAIN_RULES, score_trajectory

    result = score_trajectory(
        [1, 0, 1, 2],
        [0, 0, 1, 2],
        [0.7, 0.1, 0.1, 0.1],
        alpha=0.1,
        utility=[1.0, -1.0, 0.0, -0.25],
        uncertainty_penalty=0.5,
    )
    assert set(result["weights"]) == set(TRAIN_RULES)
    assert result["weights"]["no_update"] == 0.0
    assert result["weights"]["equal_weight"] == 1.0
    assert 0.0 <= result["weights"]["eesd_full"] <= 1.0
    assert result["weights"]["eed_no_anchor"] == result["weights"]["eesd_full"]
    # no-anchor changes only the training objective, not evidence trust.
