import runpy
from pathlib import Path

import numpy as np
import pytest


def api():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/test_apbpf_dirichlet_combination.py'))


def test_probability_combination_preserves_endpoints_and_normalization():
    blend = api()['blend_probabilities']
    a = np.array([[.8, .2], [.4, .6]])
    b = np.array([[.2, .8], [.6, .4]])
    np.testing.assert_array_equal(blend(a, b, 0), a)
    np.testing.assert_array_equal(blend(a, b, 1), b)
    np.testing.assert_allclose(blend(a, b, .5), .5)
    for weight in (-.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            blend(a, b, weight)


def test_selection_finds_complementary_mixture_without_assessment_labels():
    choose = api()['select_weight']
    a = np.array([[.9, .1], [.2, .8]])
    b = np.array([[.2, .8], [.9, .1]])
    assert choose(a, b, np.array([0, 0])) == .5
    assert choose(a, b, np.array([0, 1])) == 0.


def test_invalid_probability_inputs_fail_closed():
    blend = api()['blend_probabilities']
    a = np.array([[.8, .2]])
    for b in (np.array([[.4, .2]]), np.array([[1.1, -.1]]), np.array([[np.nan, .2]])):
        with pytest.raises(ValueError):
            blend(a, b, .5)
