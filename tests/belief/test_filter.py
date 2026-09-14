from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pbpf.belief import BeliefStore, ParticleSet, SMCStep
from pbpf.particles import logsumexp


def store_with_roots(*, ess_fraction=0.0):
    store = BeliefStore(rng=np.random.default_rng(123), ess_fraction=ess_fraction)
    store.initialize_root("a", [[1.0], [2.0]], np.log([0.75, 0.25]))
    store.initialize_root("b", [[10.0], [20.0]], np.log([0.2, 0.8]))
    return store


def test_observing_a_preserves_every_byte_and_metadata_of_b():
    store = store_with_roots()
    before_b = store.snapshot("b")
    before_bytes = tuple(a.tobytes() for a in (
        before_b.z, before_b.log_weights, before_b.ancestors
    ))
    result = store.observe("a", np.log([0.9, 0.1]))
    after_b = store.snapshot("b")
    assert after_b == before_b
    assert tuple(a.tobytes() for a in (
        after_b.z, after_b.log_weights, after_b.ancestors
    )) == before_bytes
    assert isinstance(result, SMCStep)
    np.testing.assert_allclose(np.exp(result.particles.log_weights), [27 / 28, 1 / 28])
    np.testing.assert_array_equal(result.particles.z, [[1], [2]])


def test_repeated_observations_add_only_likelihood_and_incremental_evidence():
    store = store_with_roots()
    first = store.observe("a", np.log([0.8, 0.2]))
    second = store.observe("a", np.log([0.5, 0.25]))
    # Marginal masses: .65 followed by .3125 / .65; posterior [.3, .0125].
    assert first.log_normalizer == pytest.approx(np.log(0.65))
    assert second.log_normalizer == pytest.approx(np.log(0.3125 / 0.65))
    assert second.particles.log_evidence == pytest.approx(np.log(0.3125))
    np.testing.assert_allclose(np.exp(second.particles.log_weights), [0.96, 0.04])
    assert second.particles.step == 2


@pytest.mark.parametrize("terms", [
    {"log_transition": [0, 0]}, {"log_proposal": [0, 0]},
    {"log_transition": [0, 0], "log_proposal": [0, 0]},
])
def test_same_code_rejects_any_creation_correction_without_mutation(terms):
    store = store_with_roots()
    before = store.snapshot("a")
    with pytest.raises(ValueError, match="only legal at candidate creation"):
        store.observe("a", [0, 0], **terms)
    assert store.snapshot("a") == before


def test_children_use_selected_parent_weights_actual_samples_and_correction_once():
    store = store_with_roots()
    before_a, before_b = store.snapshot("a"), store.snapshot("b")
    first = store.spawn_child(
        "child1", "a", sampled_z=[[101], [102]], ancestors=[1, 0],
        ancestor_scheme="deterministic_enumeration",
        log_transition=np.log([0.6, 0.4]), log_proposal=np.log([0.3, 0.8]),
        log_likelihood=np.log([0.8, 0.2]),
    )
    # Indexed parent masses [.25, .75] yield unnormalized [.4, .075].
    np.testing.assert_allclose(np.exp(first.particles.log_weights), [16 / 19, 3 / 19])
    np.testing.assert_array_equal(first.particles.z, [[101], [102]])
    np.testing.assert_array_equal(first.particles.ancestors, [1, 0])
    assert first.log_normalizer == pytest.approx(np.log(0.475))
    assert first.particles.parent_hash == "a"
    assert first.particles.step == 1
    store.spawn_child(
        "child2", "a", sampled_z=[[201], [202]], ancestors=[0, 1],
        ancestor_scheme="deterministic_enumeration",
        log_transition=[0, 0], log_proposal=[0, 0], log_likelihood=[0, 0],
    )
    sibling = store.snapshot("child2")
    np.testing.assert_allclose(np.exp(sibling.log_weights), [0.75, 0.25])
    after = store.observe("child1", np.log([0.5, 0.25]))
    np.testing.assert_allclose(np.exp(after.particles.log_weights), [32 / 35, 3 / 35])
    assert store.snapshot("child2") == sibling
    assert store.snapshot("a") == before_a
    assert store.snapshot("b") == before_b


@pytest.mark.parametrize("missing", ["sampled_z", "log_transition", "log_proposal"])
def test_child_requires_sampled_latents_and_both_gaussian_densities(missing):
    store = store_with_roots()
    kwargs = dict(sampled_z=[[3], [4]], ancestors=[0, 1],
                  ancestor_scheme="deterministic_enumeration",
                  log_transition=[0, 0], log_proposal=[0, 0], log_likelihood=[0, 0])
    del kwargs[missing]
    with pytest.raises(ValueError, match=missing):
        store.spawn_child("child", "a", **kwargs)
    with pytest.raises(KeyError):
        store.snapshot("child")


def test_child_evidence_extends_parent_evidence():
    store = store_with_roots()
    store.observe("a", np.log([0.8, 0.2]))
    child = store.spawn_child(
        "child", "a", sampled_z=[[3], [4]], ancestors=[0, 1],
        ancestor_scheme="deterministic_enumeration",
        log_transition=[0, 0], log_proposal=[0, 0], log_likelihood=np.log([0.5, 0.25]),
    )
    assert child.particles.log_evidence == pytest.approx(np.log(0.3125))


def test_resampling_preserves_pre_resampling_diagnostics_and_latent_ancestry():
    store = BeliefStore(rng=np.random.default_rng(123), ess_fraction=1.0)
    store.initialize_root("root", [[10], [20], [30]])
    result = store.observe("root", np.log([0.1, 0.2, 0.7]))
    assert result.resampled is True
    assert result.ess == pytest.approx(1 / 0.54)
    assert result.log_normalizer == pytest.approx(np.log(1 / 3))
    np.testing.assert_allclose(np.exp(result.normalized_log_weights), [0.1, 0.2, 0.7])
    np.testing.assert_allclose(np.exp(result.particles.log_weights), [1 / 3] * 3)
    np.testing.assert_array_equal(result.resampling_indices, [1, 2, 2])
    np.testing.assert_array_equal(result.particles.ancestors, [1, 2, 2])
    np.testing.assert_array_equal(result.particles.z, [[20], [30], [30]])
    assert result.unique_ancestor_count == 2
    assert result.normalized_entropy == pytest.approx(0.7298466991620975)
    assert result.particles.log_evidence == pytest.approx(np.log(1 / 3))


def test_child_resampling_composes_selected_parent_ancestry():
    store = BeliefStore(rng=np.random.default_rng(123), ess_fraction=1.0)
    store.initialize_root("root", [[1], [2], [3]])
    result = store.spawn_child(
        "child", "root", sampled_z=[[10], [20], [30]], ancestors=[2, 0, 1],
        ancestor_scheme="deterministic_enumeration",
        log_transition=[0, 0, 0], log_proposal=[0, 0, 0],
        log_likelihood=np.log([0.1, 0.2, 0.7]),
    )
    np.testing.assert_array_equal(result.resampling_indices, [1, 2, 2])
    np.testing.assert_array_equal(result.particles.ancestors, [0, 1, 1])
    np.testing.assert_array_equal(result.particles.z, [[20], [30], [30]])


def test_root_proposal_correction_is_applied_only_at_creation():
    store = BeliefStore(rng=np.random.default_rng(2), ess_fraction=0.0)
    first = store.initialize_root(
        "root", [[3], [4]], log_prior=np.log([0.6, 0.4]),
        log_proposal=np.log([0.3, 0.8]), log_likelihood=np.log([0.8, 0.2]),
    )
    np.testing.assert_allclose(np.exp(first.particles.log_weights), [16 / 17, 1 / 17])
    assert first.log_normalizer == pytest.approx(np.log(0.85))
    assert first.particles.step == 1
    second = store.observe("root", np.log([0.5, 0.25]))
    np.testing.assert_allclose(np.exp(second.particles.log_weights), [32 / 33, 1 / 33])
    assert second.particles.log_evidence == pytest.approx(np.log(0.4125))


@pytest.mark.parametrize("kwargs", [
    {"log_prior": [0, 0]}, {"log_proposal": [0, 0]},
    {"log_prior": [0, 0], "log_proposal": [0, 0], "log_weights": [0, 0]},
])
def test_root_rejects_incomplete_or_ambiguous_proposal_corrections(kwargs):
    store = BeliefStore(rng=np.random.default_rng(2))
    with pytest.raises(ValueError):
        store.initialize_root("root", [[1], [2]], **kwargs)


def test_snapshots_and_results_are_deep_immutable_and_inputs_are_not_aliased():
    store = BeliefStore(rng=np.random.default_rng(2), ess_fraction=0.0)
    z, weights = np.array([[1.0], [2.0]]), np.log([0.5, 0.5])
    result = store.initialize_root("root", z, weights)
    z[:] = 99
    weights[:] = 99
    first, second = store.snapshot("root"), store.snapshot("root")
    assert first == second
    assert first is not second
    for name in ("z", "log_weights", "ancestors"):
        assert not np.shares_memory(getattr(first, name), getattr(second, name))
        with pytest.raises(ValueError):
            getattr(first, name).flat[0] = 99
        with pytest.raises(ValueError):
            getattr(first, name).setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        first.step = 99
    np.testing.assert_array_equal(result.particles.z, [[1], [2]])
    with pytest.raises(ValueError):
        result.normalized_log_weights[0] = 99


@pytest.mark.parametrize("candidate", ["a", "b"])
def test_duplicate_candidate_hash_cannot_overwrite_existing_state(candidate):
    store = store_with_roots()
    before = store.snapshot(candidate)
    with pytest.raises(ValueError, match="already exists"):
        store.initialize_root(candidate, [[0], [0]])
    with pytest.raises(ValueError, match="already exists"):
        store.spawn_child(candidate, "a", sampled_z=[[0], [0]], ancestors=[0, 1],
                          ancestor_scheme="deterministic_enumeration",
                          log_transition=[0, 0], log_proposal=[0, 0], log_likelihood=[0, 0])
    assert store.snapshot(candidate) == before


@pytest.mark.parametrize("likelihood", [[0], [[0, 0]], [np.nan, 0], [np.inf, 0], [-np.inf, -np.inf]])
def test_invalid_observation_is_atomic(likelihood):
    store = store_with_roots()
    before = store.snapshot("a")
    with pytest.raises(ValueError):
        store.observe("a", likelihood)
    assert store.snapshot("a") == before


@pytest.mark.parametrize("replacement", [
    {"sampled_z": [[1], [2], [3]]}, {"sampled_z": [[np.nan], [2]]},
    {"sampled_z": [[1, 2], [3, 4]]}, {"ancestors": [-1, 0]},
    {"ancestors": [0, 2]}, {"ancestors": [0.5, 1]}, {"ancestors": [0]},
    {"log_transition": [-np.inf, 0]}, {"log_proposal": [-np.inf, 0]},
    {"log_transition": [0]}, {"log_proposal": [np.nan, 0]},
])
def test_child_rejects_invalid_samples_ancestry_and_gaussian_densities(replacement):
    store = store_with_roots()
    kwargs = dict(sampled_z=[[3], [4]], ancestors=[0, 1], log_transition=[0, 0],
                  ancestor_scheme="deterministic_enumeration",
                  log_proposal=[0, 0], log_likelihood=[0, 0])
    kwargs.update(replacement)
    with pytest.raises(ValueError):
        store.spawn_child("child", "a", **kwargs)
    with pytest.raises(KeyError):
        store.snapshot("child")


@pytest.mark.parametrize("replacement", [
    {"candidate_hash": ""}, {"parent_hash": ""}, {"parent_hash": "root"},
    {"z": []}, {"z": [1, 2]}, {"z": [[np.inf], [2]]},
    {"log_weights": [0]}, {"log_weights": [np.nan, 0]},
    {"log_weights": [-np.inf, -np.inf]}, {"log_weights": [0, 0]},
    {"ancestors": [0]}, {"ancestors": [0.5, 1]}, {"ancestors": [-1, 0]},
    {"step": -1}, {"step": 0.5}, {"log_evidence": np.inf},
])
def test_particle_set_validates_shape_finiteness_and_metadata(replacement):
    kwargs = dict(candidate_hash="root", parent_hash=None, z=[[1], [2]],
                  log_weights=np.log([0.5, 0.5]), ancestors=[0, 1], step=0, log_evidence=0.0)
    kwargs.update(replacement)
    with pytest.raises(ValueError):
        ParticleSet(**kwargs)


def test_zero_mass_particles_and_single_particle_entropy_are_supported():
    store = BeliefStore(rng=np.random.default_rng(2), ess_fraction=0.0)
    store.initialize_root("root", [[1], [2]])
    result = store.observe("root", [0, -np.inf])
    np.testing.assert_array_equal(np.exp(result.particles.log_weights), [1, 0])
    assert result.normalized_entropy == 0.0
    single = store.initialize_root("single", [[3]])
    assert single.normalized_entropy == 0.0
    assert single.ess == 1.0


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan])
def test_invalid_ess_fraction_rejected(fraction):
    with pytest.raises(ValueError, match="ess_fraction"):
        BeliefStore(rng=np.random.default_rng(2), ess_fraction=fraction)


def test_public_logsumexp_is_stable_and_rejects_invalid_mass():
    assert logsumexp([-10000, -10001]) == pytest.approx(-9999.68673831248)
    assert logsumexp([0, -np.inf]) == 0.0
    for values in ([], [np.nan], [np.inf], [-np.inf]):
        with pytest.raises(ValueError):
            logsumexp(values)


def test_store_mala_move_persists_only_latents_for_the_resampled_candidate():
    store = BeliefStore(rng=np.random.default_rng(123), ess_fraction=1.0)
    store.initialize_root("a", [[10], [20], [30]])
    store.initialize_root("b", [[100], [200]])
    store.observe("a", np.log([0.1, 0.2, 0.7]))
    before_a, before_b = store.snapshot("a"), store.snapshot("b")
    result = store.mala_move(
        "a", log_density=lambda z: -0.5 * np.square(z).sum(axis=1),
        gradient=lambda z: -z, step_size=0.1,
    )
    after = store.snapshot("a")
    assert np.any(result.accepted)
    assert not np.array_equal(after.z, before_a.z)
    np.testing.assert_array_equal(after.z, result.z)
    np.testing.assert_array_equal(after.log_weights, before_a.log_weights)
    np.testing.assert_array_equal(after.ancestors, before_a.ancestors)
    assert after.candidate_hash == before_a.candidate_hash
    assert after.parent_hash == before_a.parent_hash
    assert after.step == before_a.step
    assert after.log_evidence == before_a.log_evidence
    assert store.snapshot("b") == before_b
    # The next observation uses the moved state; only one move step is legal.
    with pytest.raises(ValueError, match="immediately after resampling"):
        store.mala_move("a", log_density=lambda z: np.zeros(len(z)),
                        gradient=lambda z: np.zeros_like(z), step_size=0.1)
    next_step = store.observe("a", [0, 0, 0])
    np.testing.assert_array_equal(next_step.particles.z, result.z)


def test_store_mala_move_requires_resampling_and_failure_is_atomic():
    store = store_with_roots(ess_fraction=1.0)
    kwargs = dict(log_density=lambda z: -0.5 * np.square(z).sum(axis=1),
                  gradient=lambda z: -z, step_size=0.1)
    with pytest.raises(ValueError, match="immediately after resampling"):
        store.mala_move("a", **kwargs)
    store.observe("a", np.log([0.9, 0.1]))
    before = store.snapshot("a")
    with pytest.raises(ValueError, match="callable"):
        store.mala_move("a", **(kwargs | {"gradient": None}))
    assert store.snapshot("a") == before
    store.mala_move("a", **kwargs)


def test_observation_invalidates_a_pending_move_from_an_older_step():
    store = store_with_roots(ess_fraction=1.0)
    store.observe("a", np.log([0.9, 0.1]))
    store.observe("a", [0, 0])
    with pytest.raises(ValueError, match="immediately after resampling"):
        store.mala_move("a", log_density=lambda z: np.zeros(len(z)),
                        gradient=lambda z: np.zeros_like(z), step_size=0.1)


@pytest.mark.parametrize("operation", ["initialize_root", "observe", "spawn_child"])
@pytest.mark.parametrize("field", ["z", "log_weights", "ancestors"])
@pytest.mark.parametrize("header", ["shape", "dtype"])
@pytest.mark.filterwarnings("ignore:Setting the .* on a NumPy array has been deprecated:DeprecationWarning")
def test_returned_particle_header_mutation_cannot_corrupt_store(operation, field, header):
    store = store_with_roots()
    if operation == "initialize_root":
        result = store.initialize_root("target", [[3], [4]])
    elif operation == "observe":
        result = store.observe("a", np.log([0.8, 0.2]))
    else:
        result = store.spawn_child(
            "target", "a", sampled_z=[[3], [4]], ancestors=[0, 1],
            ancestor_scheme="deterministic_enumeration",
            log_transition=[0, 0], log_proposal=[0, 0], log_likelihood=[0, 0],
        )
    candidate_hash = result.particles.candidate_hash
    before = store.snapshot(candidate_hash)
    array = getattr(result.particles, field)
    if header == "shape":
        array.shape = (1, 2)
    else:
        array.dtype = np.float64 if field == "ancestors" else np.int64
    assert store.snapshot(candidate_hash) == before
    # Subsequent observation must remain valid and use the original values.
    next_step = store.observe(candidate_hash, [0, 0])
    np.testing.assert_array_equal(next_step.particles.z, before.z)
    np.testing.assert_allclose(next_step.particles.log_weights, before.log_weights)


def test_child_ancestor_scheme_is_required_instead_of_assuming_selection():
    store = store_with_roots()
    with pytest.raises(TypeError, match="ancestor_scheme"):
        store.spawn_child(
            "child", "a", sampled_z=[[3], [4]], ancestors=[0, 0],
            log_transition=[0, 0], log_proposal=[0, 0], log_likelihood=[0, 0],
        )
    with pytest.raises(KeyError):
        store.snapshot("child")


@pytest.mark.parametrize("scheme", [None, "unweighted", "weighted", "unknown"])
def test_child_rejects_unknown_ancestor_scheme(scheme):
    store = store_with_roots()
    with pytest.raises(ValueError, match="ancestor_scheme"):
        store.spawn_child(
            "child", "a", ancestor_scheme=scheme, sampled_z=[[3], [4]], ancestors=[0, 1],
            log_transition=[0, 0], log_proposal=[0, 0], log_likelihood=[0, 0],
        )
    with pytest.raises(KeyError):
        store.snapshot("child")


@pytest.mark.parametrize("indices", [[0, 0], [1, 1]])
def test_deterministic_ancestor_enumeration_requires_a_permutation(indices):
    store = store_with_roots()
    with pytest.raises(ValueError, match="permutation"):
        store.spawn_child(
            "child", "a", ancestor_scheme="deterministic_enumeration",
            sampled_z=[[3], [4]], ancestors=indices,
            log_transition=[0, 0], log_proposal=[0, 0], log_likelihood=[0, 0],
        )
    with pytest.raises(KeyError):
        store.snapshot("child")


def test_deterministic_ancestor_enumeration_prohibits_ancestor_proposal_terms():
    store = store_with_roots()
    with pytest.raises(ValueError, match="log_ancestor_proposal"):
        store.spawn_child(
            "child", "a", ancestor_scheme="deterministic_enumeration",
            log_ancestor_proposal=np.log([0.5, 0.5]),
            sampled_z=[[3], [4]], ancestors=[1, 0],
            log_transition=[0, 0], log_proposal=[0, 0], log_likelihood=[0, 0],
        )


@pytest.mark.parametrize("terms", [None, [], [0], [[0, 0]], [np.nan, 0],
                                  [np.inf, 0], [-np.inf, 0], [0.1, 0]])
def test_sampled_ancestors_require_finite_shaped_log_proposal_probabilities(terms):
    store = store_with_roots()
    with pytest.raises(ValueError, match="log_ancestor_proposal"):
        store.spawn_child(
            "child", "a", ancestor_scheme="sampled", log_ancestor_proposal=terms,
            sampled_z=[[3], [4]], ancestors=[0, 0],
            log_transition=[0, 0], log_proposal=[0, 0], log_likelihood=[0, 0],
        )
    with pytest.raises(KeyError):
        store.snapshot("child")


@pytest.mark.parametrize("indices,ancestor_probabilities,expected_evidence", [
    ([0, 0], [0.75, 0.75], 1.0),  # Weighted draws cancel parent mass, including repeats.
    ([1, 1], [0.5, 0.5], 0.5),  # Uniform draws retain the actual MC evidence estimate.
])
def test_sampled_ancestor_correction_preserves_weighted_and_uniform_evidence(
    indices, ancestor_probabilities, expected_evidence,
):
    store = store_with_roots()
    parent = store.snapshot("a")
    child = store.spawn_child(
        "child", "a", ancestor_scheme="sampled",
        log_ancestor_proposal=np.log(ancestor_probabilities),
        sampled_z=[[3], [4]], ancestors=indices,
        log_transition=[0, 0], log_proposal=[0, 0], log_likelihood=[0, 0],
    )
    np.testing.assert_allclose(np.exp(child.particles.log_weights), [0.5, 0.5])
    assert child.log_normalizer == pytest.approx(np.log(expected_evidence))
    assert child.particles.log_evidence == pytest.approx(np.log(expected_evidence))
    np.testing.assert_array_equal(child.particles.ancestors, indices)
    assert store.snapshot("a") == parent


def test_sampled_ancestor_and_gaussian_corrections_combine_without_renormalizing_base():
    store = store_with_roots()
    child = store.spawn_child(
        "child", "a", ancestor_scheme="sampled",
        log_ancestor_proposal=np.log([0.75, 0.25]),
        sampled_z=[[3], [4]], ancestors=[0, 1],
        log_transition=np.log([0.6, 0.4]), log_proposal=np.log([0.3, 0.8]),
        log_likelihood=np.log([0.8, 0.2]),
    )
    # Bases [.5, .5], child Gaussian ratios [2, .5], and likelihoods [.8, .2].
    np.testing.assert_allclose(np.exp(child.particles.log_weights), [16 / 17, 1 / 17])
    assert child.log_normalizer == pytest.approx(np.log(0.85))
    next_step = store.observe("child", np.log([0.5, 0.25]))
    np.testing.assert_allclose(np.exp(next_step.particles.log_weights), [32 / 33, 1 / 33])
    assert next_step.particles.log_evidence == pytest.approx(np.log(0.4125))
