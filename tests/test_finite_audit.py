import numpy as np
import pytest

from pbpf.finite import run_finite_audit
from pbpf.posterior import exact_discrete_posterior


def test_exact_discrete_posterior_matches_hand_calculation():
    posterior = exact_discrete_posterior(
        np.array([0.5, 0.5]),
        np.array([[0.8, 0.2], [0.75, 0.25]]),
    )
    assert posterior.tolist() == pytest.approx([12 / 13, 1 / 13])


def test_finite_audit_reports_preregistered_particle_and_control_arms():
    # probability[test, hypothesis, binary outcome]
    probabilities = np.array(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.8, 0.2], [0.3, 0.7]],
            [[0.7, 0.3], [0.4, 0.6]],
            [[0.65, 0.35], [0.45, 0.55]],
            [[0.6, 0.4], [0.5, 0.5]],
        ]
    )
    report = run_finite_audit(
        prior=np.array([0.5, 0.5]),
        outcome_probabilities=probabilities,
        observed_outcomes=[0, 0, 0, 0],
        future_outcomes=[0],
        particle_counts=(1, 4, 8, 16, 32),
        rng=np.random.default_rng(9),
    )
    assert set(report["arms"]) == {
        "exact_bayes",
        "map",
        "posterior_mean",
        "uniform",
        "shuffled",
        "particles_1",
        "particles_4",
        "particles_8",
        "particles_16",
        "particles_32",
    }
    assert report["arms"]["particles_32"]["posterior_kl"] < 0.08
    assert report["gate"]["finite_values"] is True
    assert len(report["arms"]["particles_32"]["steps"]) == 4
    assert all("ess" in step and "ancestors" in step for step in report["arms"]["particles_32"]["steps"])
    assert report["arms"]["shuffled"]["posterior"] != pytest.approx(
        report["exact_posterior"]
    )


def test_finite_audit_rejects_static_nonidentity_proposal_terms_without_draws():
    probabilities = np.full((2, 2, 2), 0.5)
    with pytest.raises(ValueError, match="proposal sampler"):
        run_finite_audit(
            prior=np.array([0.5, 0.5]),
            outcome_probabilities=probabilities,
            observed_outcomes=[0],
            future_outcomes=[1],
            particle_counts=(2,),
            rng=np.random.default_rng(3),
            log_transition_terms=np.array([[0.1, -0.1]]),
            log_proposal_terms=np.array([[0.0, 0.0]]),
        )
