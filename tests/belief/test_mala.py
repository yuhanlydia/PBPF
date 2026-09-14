import numpy as np
import pytest

from pbpf.belief.mala import mala_resample_move


def normal_log_density(z):
    return -0.5 * np.square(z).sum(axis=1)


def normal_gradient(z):
    return -z


def test_mala_seeded_standard_normal_moves_are_corrected_and_reproducible():
    old = np.array([[0.0], [1.0], [-1.0], [3.0]])
    results = [mala_resample_move(
        old, log_density=normal_log_density, gradient=normal_gradient,
        step_size=2.0, rng=np.random.default_rng(7),
    ) for _ in range(2)]
    # q(y|x)=N(-x,4); for N(0,1), log MH = -(y²-x²)/2.
    # Fourth y=-4.78118 has log MH=-6.92986 and is rejected.
    expected = [[0.0024603067149651485], [-0.40250892498306024],
                [0.45172428927556484], [3.0]]
    for result in results:
        np.testing.assert_allclose(result.z, expected, rtol=1e-12)
        np.testing.assert_array_equal(result.accepted, [True, True, True, False])
        assert result.acceptance_rate == 0.75
        assert result.z[3].tobytes() == old[3].tobytes()
    np.testing.assert_array_equal(results[0].z, results[1].z)
    np.testing.assert_array_equal(old, [[0], [1], [-1], [3]])
    assert not np.shares_memory(old, results[0].z)


def test_mala_forward_reverse_correction_changes_acceptance_not_just_jitter():
    # The target-only ratio rejects particle 1: u=.94865 > exp(-.084382).
    # Correct MALA ratio is -.5²/4 * .084382 = -.005274, so it accepts.
    old = np.zeros((2, 1))
    result = mala_resample_move(
        old, log_density=normal_log_density, gradient=normal_gradient,
        step_size=0.5, rng=np.random.default_rng(1),
    )
    np.testing.assert_array_equal(result.accepted, [True, True])
    np.testing.assert_allclose(result.z[:, 0], [0.172792096032393, 0.4108090717505792])


def test_mala_handles_multiple_latent_dimensions():
    result = mala_resample_move(
        np.zeros((3, 2)), log_density=normal_log_density, gradient=normal_gradient,
        step_size=0.1, rng=np.random.default_rng(9),
    )
    assert result.z.shape == (3, 2)
    assert result.accepted.shape == (3,)
    assert result.acceptance_rate == 1.0
    np.testing.assert_allclose(result.z[0], [-0.08028369359828767, 0.02428499070790021])


@pytest.mark.parametrize("kwargs", [
    {"log_density": None}, {"gradient": None},
    {"step_size": 0}, {"step_size": -1}, {"step_size": np.inf}, {"step_size": np.nan},
    {"z": []}, {"z": [1, 2]}, {"z": [[np.nan]]},
    {"log_density": lambda z: 0.0},
    {"log_density": lambda z: np.full(len(z), np.nan)},
    {"log_density": lambda z: np.full(len(z), np.inf)},
    {"log_density": lambda z: np.full(len(z), -np.inf)},
    {"gradient": lambda z: np.zeros(len(z))},
    {"gradient": lambda z: np.full_like(z, np.nan)},
])
def test_mala_rejects_invalid_inputs_instead_of_falling_back_to_jitter(kwargs):
    arguments = dict(z=np.zeros((2, 1)), log_density=normal_log_density,
                     gradient=normal_gradient, step_size=0.5, rng=np.random.default_rng(1))
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        mala_resample_move(**arguments)


def test_mala_rejects_out_of_support_proposals_and_keeps_old_particles():
    def bounded_density(z):
        return np.where(np.abs(z[:, 0]) < 0.01, 0.0, -np.inf)

    result = mala_resample_move(
        np.zeros((2, 1)), log_density=bounded_density,
        gradient=lambda z: np.zeros_like(z), step_size=2, rng=np.random.default_rng(7),
    )
    np.testing.assert_array_equal(result.accepted, [True, False])
    assert result.z[1, 0] == 0.0
    assert result.acceptance_rate == 0.5


def test_mala_preserves_particle_conditioning_when_some_proposals_leave_support():
    # Different particle-conditioned targets retain their original row indices.
    centers = np.array([[0.0], [2.0]])

    def density(z):
        if z.shape != centers.shape:
            raise ValueError("lost particle conditioning")
        delta = z - centers
        return np.where(np.abs(delta[:, 0]) < 0.01, -0.5 * delta[:, 0]**2, -np.inf)

    def gradient(z):
        if z.shape != centers.shape:
            raise ValueError("lost particle conditioning")
        return centers - z

    result = mala_resample_move(
        centers, log_density=density, gradient=gradient,
        step_size=2, rng=np.random.default_rng(7),
    )
    np.testing.assert_array_equal(result.accepted, [True, False])
    np.testing.assert_allclose(result.z, [[0.0024603067149651485], [2.0]])
