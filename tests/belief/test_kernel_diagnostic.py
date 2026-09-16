import numpy as np
import pytest

from pbpf.belief.kernel_diagnostic import kernel_predict


def test_query_conditioned_history_preserves_pairs_not_presentation():
    tests = np.array([[[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]]])
    outcomes = np.array([[0, 1, 1, 0]])
    kwargs = dict(visible_steps=2, alpha=.1, temperature=.1, mix=1.)
    predictions = kernel_predict(tests, outcomes[:, :2], **kwargs)
    assert predictions[0, 0, 0] > .83
    assert predictions[0, 1, 1] > .83
    np.testing.assert_allclose(predictions.sum(-1), 1.)
    perm = [1, 0, 2, 3]
    np.testing.assert_allclose(predictions, kernel_predict(tests[:, perm], outcomes[:, [1, 0]], **kwargs))
    shuffled = kernel_predict(tests, outcomes[:, [1, 0]], **kwargs)
    assert shuffled[0, 0, 0] < .05


def test_zero_mixture_is_exact_histogram_and_future_labels_are_rejected():
    tests = np.random.default_rng(1).normal(size=(2, 7, 4))
    visible = np.array([[0, 0, 1], [2, 2, 2]])
    pred = kernel_predict(tests, visible, visible_steps=3, alpha=.2, temperature=.1, mix=0.)
    for i in range(2):
        expected = (np.bincount(visible[i], minlength=5)+.2)/4
        np.testing.assert_allclose(pred[i], np.tile(expected, (4, 1)))
    with pytest.raises(ValueError, match='visible'):
        kernel_predict(tests, np.zeros((2, 7), dtype=int), visible_steps=3, alpha=.2, temperature=.1, mix=0.)


@pytest.mark.parametrize('key,value', [('alpha',0), ('temperature',0), ('mix',1.1)])
def test_invalid_kernel_parameters(key, value):
    kwargs = dict(visible_steps=2, alpha=.1, temperature=.1, mix=.5)
    kwargs[key] = value
    with pytest.raises(ValueError):
        kernel_predict(np.ones((1,4,2)), np.array([[0,1]]), **kwargs)
