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
