import math

import numpy as np
import pytest

from pbpf.metrics import future_nll
from pbpf.particles import (
    effective_sample_size,
    normalize_log_weights,
    particle_update,
    sequence_mixture_logprob,
    systematic_resample,
    tokenwise_mixture_logprob,
)


def test_normalization_is_stable_for_extreme_log_weights():
    normalized = normalize_log_weights(np.array([-10000.0, -10001.0]))
    assert np.exp(normalized).tolist() == pytest.approx(
        [0.7310585786300049, 0.2689414213699951]
    )
    assert np.exp(normalized).sum() == pytest.approx(1.0)


def test_particle_update_includes_transition_and_proposal_correction():
    result = particle_update(
        np.log([0.5, 0.5]),
        np.log([0.8, 0.2]),
        np.log([0.6, 0.4]),
        np.log([0.3, 0.8]),
        rng=np.random.default_rng(5),
        ess_fraction=0.0,
    )
    assert np.exp(result.log_weights).tolist() == pytest.approx(
        [0.9411764705882353, 0.0588235294117647]
    )
    assert result.resampled is False


def test_ess_and_systematic_resampling_are_deterministic_with_explicit_rng():
    assert effective_sample_size(np.log([0.5, 0.5])) == pytest.approx(2.0)
    ancestors = systematic_resample(
        np.array([0.1, 0.2, 0.7]), rng=np.random.default_rng(123)
    )
    assert ancestors.tolist() == [1, 2, 2]


def test_sequence_mixture_is_not_product_of_token_mixtures():
    component_token_logprobs = np.log([[0.9, 0.1], [0.1, 0.9]])
    weights = np.log([0.5, 0.5])
    coherent = math.exp(sequence_mixture_logprob(component_token_logprobs, weights))
    faulty = math.exp(tokenwise_mixture_logprob(component_token_logprobs, weights))
    assert coherent == pytest.approx(0.09)
    assert faulty == pytest.approx(0.25)


def test_future_nll_rejects_zero_probability_and_shape_errors():
    assert future_nll(np.array([[0.8, 0.2], [0.25, 0.75]]), [0, 1]) == pytest.approx(
        -(math.log(0.8) + math.log(0.75)) / 2
    )
    with pytest.raises(ValueError, match="positive"):
        future_nll(np.array([[1.0, 0.0]]), [1])
