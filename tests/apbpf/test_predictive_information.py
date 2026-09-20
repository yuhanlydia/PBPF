import numpy as np
import pytest

from pbpf.apbpf import acquisition


def test_target_information_ignores_latent_bits_irrelevant_to_future_tests():
    weights = np.full(4, .25)
    query_a = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    query_b = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    targets = query_a[:, None, :]
    assert acquisition.expected_information_gain(weights, query_a) == pytest.approx(np.log(2))
    assert acquisition.expected_information_gain(weights, query_b) == pytest.approx(np.log(2))
    score = acquisition.predictive_information_gain
    assert score(weights, query_a, targets) == pytest.approx(np.log(2))
    assert score(weights, query_b, targets) == pytest.approx(0.)


def test_target_information_averages_over_targets_and_handles_zero_mass():
    query = np.eye(2)
    targets = np.stack([query, np.full((2, 2), .5)], axis=1)
    score = acquisition.predictive_information_gain
    assert score([.5, .5], query, targets) == pytest.approx(np.log(2) / 2)
    assert score([1., 0.], query, targets) == pytest.approx(0.)


def test_target_information_matches_expected_entropy_reduction_after_noisy_query():
    weights = np.array([.3, .7])
    query = np.array([[.9, .1], [.2, .8]])
    target = np.array([[.8, .2], [.4, .6]])
    def entropy(p):
        return -np.sum(p * np.log(p))
    posterior_entropy = 0.
    for outcome in range(2):
        unnormalized = weights * query[:, outcome]
        evidence = unnormalized.sum()
        posterior_entropy += evidence * entropy((unnormalized / evidence) @ target)
    expected = entropy(weights @ target) - posterior_entropy
    assert acquisition.predictive_information_gain(weights, query, target[:, None]) == pytest.approx(expected)


def test_target_information_keeps_rare_correlated_events_finite():
    weights = np.array([1e-200, 1.])
    score = acquisition.predictive_information_gain(weights, np.eye(2), np.eye(2)[:, None])
    assert np.isfinite(score)
    assert score == pytest.approx(-weights[0] * np.log(weights[0]), rel=1e-12, abs=0.)


@pytest.mark.parametrize('targets', [np.ones((2, 0, 2)), np.ones((2, 1, 2)),
                                    np.full((2, 1, 2), np.nan), np.ones((3, 1, 1))])
def test_target_information_rejects_invalid_predictions(targets):
    with pytest.raises(ValueError):
        acquisition.predictive_information_gain([.5, .5], np.eye(2), targets)


def test_two_step_information_finds_complementary_queries_greedy_misses():
    # Target is XOR(A, B). A and B individually reveal nothing about XOR.
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])
    target = np.eye(2)[a ^ b]
    noisy_direct = .75 * target + .25 * (1 - target)
    queries = np.stack([noisy_direct, np.eye(2)[a], np.eye(2)[b]], axis=1)
    weights = np.full(4, .25)
    greedy = acquisition.predictive_information_scores(weights, queries, target[:, None], budget=1)
    planned = acquisition.predictive_information_scores(weights, queries, target[:, None], budget=2)
    assert np.argmax(greedy) == 0
    assert planned[0] == pytest.approx(greedy[0])
    assert planned[1] == pytest.approx(np.log(2))
    assert planned[2] == pytest.approx(np.log(2))
    assert np.argmax(planned) in (1, 2)


def test_one_step_scores_preserve_original_and_zero_probability_branches_are_skipped():
    weights = np.array([1., 0.])
    queries = np.stack([np.eye(2), np.eye(2)], axis=1)
    targets = np.eye(2)[:, None]
    expected = [acquisition.predictive_information_gain(weights, queries[:, i], targets) for i in range(2)]
    np.testing.assert_allclose(acquisition.predictive_information_scores(weights, queries, targets, budget=1), expected)
    np.testing.assert_allclose(acquisition.predictive_information_scores(weights, queries, targets, budget=2), 0.)


def test_two_step_plan_can_choose_different_second_queries_for_each_outcome():
    # S tells whether target equals A or B. Query S, then the relevant bit.
    states = np.array([[s, a, b] for s in (0, 1) for a in (0, 1) for b in (0, 1)])
    s, a, b = states.T
    targets = np.eye(2)[np.where(s == 0, a, b)][:, None]
    queries = np.stack([np.eye(2)[a], np.eye(2)[b], np.eye(2)[s]], axis=1)
    scores = acquisition.predictive_information_scores(np.full(8, .125), queries, targets, budget=2)
    np.testing.assert_allclose(scores, [np.log(2) / 2, np.log(2) / 2, np.log(2)])
    assert np.argmax(scores) == 2


@pytest.mark.parametrize('budget', [0, 3, True])
def test_predictive_planning_rejects_unsupported_budget(budget):
    with pytest.raises(ValueError):
        acquisition.predictive_information_scores([.5, .5], np.ones((2, 2, 1)), np.ones((2, 1, 1)), budget=budget)


def test_predictive_planning_cannot_repeat_its_only_query():
    with pytest.raises(ValueError):
        acquisition.predictive_information_scores([.5, .5], np.ones((2, 1, 1)), np.ones((2, 1, 1)), budget=2)


def test_pooling_posteriors_preserves_between_draw_predictive_dependence():
    # Each one-particle draw has zero MI, but their mixture has one bit.
    weights = np.ones((2, 1))
    predictions = np.eye(2)[:, None, None, :]
    w, q, t = acquisition.pool_predictive_particles(weights, predictions, predictions)
    np.testing.assert_allclose(w, [.5, .5])
    assert acquisition.predictive_information_gain(w, q[:, 0], t) == pytest.approx(np.log(2))
    for i in range(2):
        assert acquisition.predictive_information_gain(weights[i], predictions[i, :, 0], predictions[i]) == pytest.approx(0.)


def test_pooled_update_reweights_draw_masses_instead_of_resetting_them():
    predictions = np.array([[[[.9, .1]]], [[[.2, .8]]]])
    w, q, t = acquisition.pool_predictive_particles(np.ones((2, 1)), predictions, predictions)
    posterior = w * q[:, 0, 0]
    posterior /= posterior.sum()
    np.testing.assert_allclose(posterior, [9/11, 2/11])
    assert t.shape == (2, 1, 2)


def test_pooling_rejects_inconsistent_draws():
    with pytest.raises(ValueError):
        acquisition.pool_predictive_particles(np.ones((2, 1)), np.ones((1, 1, 1, 1)), np.ones((2, 1, 1, 1)))
