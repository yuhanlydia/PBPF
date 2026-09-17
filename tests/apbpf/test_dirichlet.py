import numpy as np
import pytest


def test_effective_evidence_preserves_uniform_and_discounts_concentration():
    from pbpf.apbpf.dirichlet import effective_evidence_weights
    np.testing.assert_allclose(effective_evidence_weights([1, 1, 1, 1]), [1, 1, 1, 1])
    np.testing.assert_allclose(effective_evidence_weights([4, 0, 0, 0]), [1, 0, 0, 0])
    np.testing.assert_allclose(effective_evidence_weights([3, 1]), [1.2, .4])
    np.testing.assert_allclose(effective_evidence_weights([30, 10]), [1.2, .4])
    for invalid in ([], [0, 0], [-1, 2], [float('nan')], [[1, 2]]):
        with pytest.raises(ValueError):
            effective_evidence_weights(invalid)


def test_rate_and_weighted_counts_have_exact_expected_probabilities():
    from pbpf.apbpf.dirichlet import predict_rate
    np.testing.assert_allclose(predict_rate([0, 0], 1, classes=2), [.75, .25])
    np.testing.assert_allclose(predict_rate([0, 1], 1, weights=[3, 1], classes=2), [4/6, 2/6])
    np.testing.assert_allclose(predict_rate([], .1, classes=2), [.5, .5])


def test_similarity_preserves_total_evidence_and_nested_baseline():
    from pbpf.apbpf.dirichlet import similarity_weights
    history = np.array([[1., 0.], [0., 1.], [0., 0.]])
    np.testing.assert_array_equal(similarity_weights([1., 0.], history, 0), np.ones(3))
    weights = similarity_weights([1., 0.], history, 4)
    assert weights[0] > weights[1] and weights[1] == weights[2]
    assert weights.sum() == pytest.approx(3)
    np.testing.assert_allclose(similarity_weights([0., 0.], history, 4), np.ones(3))


def test_inheritance_discounts_evidence_once_without_mutating_parent():
    from pbpf.apbpf.dirichlet import inherit_counts
    parent = np.array([4., 2.])
    np.testing.assert_array_equal(inherit_counts(parent, [1], 0), [0, 1])
    np.testing.assert_array_equal(inherit_counts(parent, [1], .5), [2, 2])
    np.testing.assert_array_equal(inherit_counts(parent, [1], 1), [4, 3])
    np.testing.assert_array_equal(parent, [4, 2])


def test_allocation_observes_exact_budget_and_only_updates_selected_candidate():
    from pbpf.apbpf.dirichlet import allocate_rollouts
    calls = []
    counts = np.zeros((3, 2))
    def observe(index):
        calls.append(index)
        return 1
    choices, updated = allocate_rollouts(counts, 4, observe, seed=1701)
    assert choices == calls and len(calls) == 4
    np.testing.assert_array_equal(updated[:, 0], 0)
    np.testing.assert_array_equal(updated[:, 1], np.bincount(calls, minlength=3))
    np.testing.assert_array_equal(counts, 0)
    assert allocate_rollouts(counts, 0, observe, seed=1701)[0] == []


def test_invalid_inputs_fail_closed():
    from pbpf.apbpf.dirichlet import predict_rate, inherit_counts, similarity_weights, allocate_rollouts
    for values in ([5], [-1], [1.5]):
        with pytest.raises(ValueError):
            predict_rate(values, .1)
    for alpha in (0, -1, float('nan')):
        with pytest.raises(ValueError):
            predict_rate([0], alpha)
    with pytest.raises(ValueError):
        inherit_counts([1, 1], [0], 1.1)
    with pytest.raises(ValueError):
        similarity_weights([1, 0], [[1, 0]], -1)
    with pytest.raises(ValueError):
        allocate_rollouts(np.zeros((2, 2)), -1, lambda _: 0, seed=1)
