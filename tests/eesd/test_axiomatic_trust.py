import numpy as np

from pbpf.eesd.evidence import (
    IMPROVED,
    REGRESSED,
    UNCHANGED,
    correction_benefit_posterior,
    posterior_benefit_probability,
    posterior_mean_advantage,
    posterior_update_weight,
    regularized_incomplete_beta,
    transition_benefit_categories,
)


def test_transition_benefit_categories_are_automatic_execution_deltas():
    before = [1, 0, 2, 0]
    after = [0, 0, 3, 4]
    assert transition_benefit_categories(before, after).tolist() == [
        IMPROVED, UNCHANGED, UNCHANGED, REGRESSED
    ]


def test_regularized_incomplete_beta_known_closed_forms():
    assert np.isclose(regularized_incomplete_beta(0.5, 1.0, 1.0), 0.5)
    assert np.isclose(regularized_incomplete_beta(0.5, 2.0, 1.0), 0.25)
    assert np.isclose(regularized_incomplete_beta(0.5, 1.0, 2.0), 0.75)
    assert np.isclose(regularized_incomplete_beta(0.5, 4.0, 4.0), 0.5)


def test_posterior_benefit_probability_is_half_under_symmetric_improve_regress_mass():
    posterior = np.array([2.5, 7.0, 2.5])
    assert np.isclose(posterior_benefit_probability(posterior), 0.5)
    assert posterior_update_weight(posterior) == 0.0


def test_posterior_update_weight_is_parameter_free_excess_confidence():
    posterior = np.array([5.0, 1.0, 1.0])
    c = posterior_benefit_probability(posterior)
    assert c > 0.5
    assert np.isclose(posterior_update_weight(posterior), 2.0 * c - 1.0)
    assert 0.0 < posterior_update_weight(posterior) <= 1.0


def test_uniform_relevance_recovers_ordinary_three_state_counts():
    before = [1, 1, 0, 0]
    after = [0, 1, 0, 1]
    posterior = correction_benefit_posterior(
        before, after, [1, 1, 1, 1], alpha=0.5, mass_rule="effective"
    )
    # improve=1, unchanged=2, regress=1 plus symmetric alpha
    assert np.allclose(posterior, [1.5, 2.5, 1.5])


def test_effective_mass_changes_confidence_not_relative_support():
    before = [1, 1, 1, 0]
    after = [0, 0, 1, 1]
    relevance = [0.85, 0.05, 0.05, 0.05]
    fixed = correction_benefit_posterior(
        before, after, relevance, alpha=0.5, mass_rule="fixed"
    )
    effective = correction_benefit_posterior(
        before, after, relevance, alpha=0.5, mass_rule="effective"
    )
    fixed_evidence = fixed - 0.5
    effective_evidence = effective - 0.5
    assert np.allclose(
        fixed_evidence / fixed_evidence.sum(),
        effective_evidence / effective_evidence.sum(),
    )
    assert effective_evidence.sum() < fixed_evidence.sum()


def test_more_evidence_in_same_positive_direction_increases_posterior_confidence():
    before = [1, 1, 1, 0]
    after = [0, 0, 1, 1]
    relevance = [3, 3, 1, 1]
    low = correction_benefit_posterior(
        before, after, relevance, alpha=0.5, mass_rule="global", fixed_mass=1.0
    )
    high = correction_benefit_posterior(
        before, after, relevance, alpha=0.5, mass_rule="global", fixed_mass=8.0
    )
    assert posterior_benefit_probability(high) > posterior_benefit_probability(low)
    assert posterior_update_weight(high) > posterior_update_weight(low)


def test_relevance_scale_does_not_change_effective_posterior():
    before = [1, 1, 0, 0]
    after = [0, 1, 0, 1]
    a = correction_benefit_posterior(
        before, after, [8, 1, 1, 1], alpha=0.1, mass_rule="effective"
    )
    b = correction_benefit_posterior(
        before, after, [80, 10, 10, 10], alpha=0.1, mass_rule="effective"
    )
    assert np.allclose(a, b)


def test_all_unchanged_execution_evidence_produces_zero_update_weight():
    posterior = correction_benefit_posterior(
        [1, 0, 2, 0], [1, 0, 2, 0], [1, 2, 3, 4],
        alpha=0.5, mass_rule="effective"
    )
    assert np.isclose(posterior_benefit_probability(posterior), 0.5)
    assert posterior_update_weight(posterior) == 0.0


def test_posterior_mean_advantage_has_no_hand_utility_vector():
    posterior = np.array([3.0, 4.0, 1.0])
    assert np.isclose(posterior_mean_advantage(posterior), (3.0 - 1.0) / 8.0)
