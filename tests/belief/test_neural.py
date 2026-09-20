"""Probability, evidence, calibration and trainability contracts (no downloads)."""

import math
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pbpf.belief.features import BeliefBatch, SemanticEncoder
from pbpf.belief.model import GaussianParams, NeuralBeliefModel
from pbpf.belief.losses import fivo_future_loss, temperature_scale
from pbpf.belief import ParticleSet


def batch():
    generator = torch.Generator().manual_seed(11)
    return BeliefBatch(
        task=torch.randn(2, 3, generator=generator),
        candidate=torch.randn(2, 3, generator=generator),
        tests=torch.randn(2, 6, 3, generator=generator),
        outcomes=torch.tensor([[0, 1, 2, 3, 4, 0], [1, 0, 3, 4, 2, 1]]),
    )


def test_semantic_pooling_ignores_padding_and_rejects_empty_sequences():
    encoder = SemanticEncoder()
    hidden = torch.tensor([[[1., 3.], [5., 7.], [900., 900.]]])
    assert torch.equal(encoder(hidden, torch.tensor([[1, 1, 0]])), torch.tensor([[3., 5.]]))
    with pytest.raises(ValueError, match="non-empty"):
        encoder(hidden, torch.zeros(1, 3))


def test_gaussian_scalar_density_and_reparameterized_actual_draw():
    mean = torch.tensor([[1.]], requires_grad=True)
    gaussian = GaussianParams(mean, torch.tensor([[math.log(2.)]]))
    z = gaussian.rsample(noise=torch.tensor([[1.5]]))
    assert z.item() == 4.
    assert gaussian.log_prob(z).item() == pytest.approx(-math.log(2.) - .5 * math.log(2 * math.pi) - 1.125)
    z.sum().backward()
    assert mean.grad.item() == 1.


def test_gaussian_scales_are_bounded_and_every_head_is_trainable():
    torch.manual_seed(3)
    model = NeuralBeliefModel(feature_dim=3, latent_dim=2, hidden_dim=7)
    data = batch()
    parent = torch.randn(2, 4, 2)
    root = model.root(data.task, data.candidate)
    transition = model.transition(parent, data.task, data.candidate, data.candidate)
    proposal = model.proposal(data.task, data.candidate, data.tests[:, 0], data.outcomes[:, 0],
                              parent_z=parent, diff=data.candidate)
    root_proposal = model.proposal(data.task, data.candidate, data.tests[:, 0], data.outcomes[:, 0])
    for params in (root, transition, proposal, root_proposal):
        assert (params.std > 0).all()
        assert (params.log_std >= -8).all() and (params.log_std <= 4).all()
    extreme = GaussianParams(torch.zeros(2), torch.tensor([-100., 100.]))
    assert extreme.log_std.tolist() == [-8., 4.]
    loss = sum(p.rsample().square().mean() + p.log_prob(p.mean).mean()
               for p in (root, transition, proposal, root_proposal))
    loss.backward()
    for name, parameter in model.named_parameters():
        if "likelihood_head" not in name:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_likelihood_five_normalized_classes_and_rejects_class_weights():
    model = NeuralBeliefModel(3, 2, 7)
    data = batch()
    z = torch.randn(2, 4, 2)
    log_probs = model.likelihood(z, data.task, data.candidate, data.tests[:, 0])
    assert log_probs.shape == (2, 4, 5)
    torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(2, 4))
    with pytest.raises(ValueError, match="proper"):
        model.likelihood(z, data.task, data.candidate, data.tests[:, 0], class_weights=torch.ones(5))


def test_fivo_keeps_every_increment_and_proposal_correction_affects_loss():
    terms = torch.tensor([[-math.log(2.), -math.log(4.)]], requires_grad=True)
    result = fivo_future_loss(terms, {}, future_weight=0.)
    assert result.total.item() == pytest.approx(math.log(8.))
    changed = fivo_future_loss(terms - torch.tensor([[1., 0.]]), {}, future_weight=0.)
    assert changed.total.item() - result.total.item() == pytest.approx(1.)
    result.total.backward()
    assert terms.grad.tolist() == [[-1., -1.]]


def test_filter_evaluates_root_density_at_draw_and_reuses_latents_across_tests():
    torch.manual_seed(7)
    model = NeuralBeliefModel(3, 2, 7)
    data = batch()
    trace = model.filter(data, particles=4, ess_fraction=0., generator=torch.Generator().manual_seed(12))
    assert trace.log_normalizers.shape == (2, 4)
    q = model.proposal(data.task, data.candidate, data.tests[:, 0], data.outcomes[:, 0])
    prior = model.root(data.task, data.candidate)
    z = q.mean[:, None] + q.std[:, None] * trace.proposal_noise
    torch.testing.assert_close(trace.latents[:, 0], z)
    ll = model.likelihood(z, data.task, data.candidate, data.tests[:, 0])
    ll = ll.gather(-1, data.outcomes[:, :1, None].expand(-1, 4, 1)).squeeze(-1)
    expected = prior.log_prob(z) + ll - q.log_prob(z) - math.log(4.)
    torch.testing.assert_close(trace.log_normalizers[:, 0], torch.logsumexp(expected, -1))
    for j in range(1, 4):
        torch.testing.assert_close(trace.latents[:, j], z)
        ll = model.likelihood(z, data.task, data.candidate, data.tests[:, j])
        ll = ll.gather(-1, data.outcomes[:, j, None, None].expand(-1, 4, 1)).squeeze(-1)
        torch.testing.assert_close(trace.log_normalizers[:, j], torch.logsumexp(trace.log_weights[:, j-1] + ll, -1))


def test_future_loss_uses_mixtures_at_prefixes_1_2_4_and_prefix4_metric():
    model = NeuralBeliefModel(3, 2, 7)
    data = batch()
    trace = model.filter(data, particles=4, ess_fraction=0.)
    future = model.future_nll(data, trace)
    assert set(future) == {1, 2, 4}
    # All-zero outcome logits imply uniform five-class predictions irrespective of particles.
    for parameter in model.likelihood_head.parameters():
        torch.nn.init.zeros_(parameter)
    future = model.future_nll(data, trace)
    for k in (1, 2, 4):
        torch.testing.assert_close(future[k], torch.full((2,), (6-k) * math.log(5.)))
    loss = fivo_future_loss(trace.log_normalizers, future, future_weight=.2)
    assert loss.prefix4_nll.item() == pytest.approx(2 * math.log(5.))
    assert loss.future.item() == pytest.approx((11 / 3) * math.log(5.))
    assert loss.total.item() == pytest.approx(loss.fivo.item() + .2 * loss.future.item())


def test_child_filter_consumes_snapshots_and_corrects_sampled_ancestors():
    model = NeuralBeliefModel(3, 2, 7)
    data = batch()
    data.diff = data.candidate.clone()
    parents = [ParticleSet(str(i), None, np.array([[0., 1.], [2., 3.]]),
                           np.log([.8, .2]), np.arange(2), 4, -2.) for i in range(2)]
    with pytest.raises(ValueError, match="ancestor_scheme"):
        model.filter(data, parents=parents)
    selected = torch.tensor([[0, 0], [1, 0]])
    rho = torch.tensor([[.8, .8], [.2, .8]]).log()
    trace = model.filter(data, parents=parents, ancestor_scheme="sampled", ancestors=selected,
                         log_ancestor_proposal=rho, ess_fraction=0.)
    parent_z = torch.tensor(np.stack([p.z.copy() for p in parents]), dtype=data.task.dtype)
    parent_z = parent_z.gather(1, selected[..., None].expand(-1, -1, 2))
    z = trace.latents[:, 0]
    p = model.transition(parent_z, data.task, data.candidate, data.diff)
    q = model.proposal(data.task, data.candidate, data.tests[:, 0], data.outcomes[:, 0],
                       parent_z=parent_z, diff=data.diff)
    ll = model.likelihood(z, data.task, data.candidate, data.tests[:, 0]).gather(
        -1, data.outcomes[:, 0, None, None].expand(-1, 2, 1)).squeeze(-1)
    expected = -math.log(2.) + p.log_prob(z) + ll - q.log_prob(z)
    torch.testing.assert_close(trace.log_normalizers[:, 0], torch.logsumexp(expected, -1))
    assert parents[0].log_evidence == -2.


def test_temperature_fit_reduces_nll_and_persists_source_disjoint_hashes():
    logits = torch.tensor([[8., 0., 0., 0., 0.]] * 5)
    labels = torch.arange(5)
    kwargs = dict(split_hash="a" * 64, training_split_hash="b" * 64,
                  calibration_source_ids={"cal"}, training_source_ids={"train"})
    calibration = temperature_scale(logits, labels, **kwargs)
    assert calibration.temperature > 1.
    assert torch.nn.functional.cross_entropy(logits / calibration.temperature, labels) < torch.nn.functional.cross_entropy(logits, labels)
    model = NeuralBeliefModel(3, 2, 7)
    model.set_calibration(calibration)
    restored = NeuralBeliefModel(3, 2, 7)
    restored.load_state_dict(model.state_dict())
    assert restored.calibration_split_hash == "a" * 64
    assert restored.training_split_hash == "b" * 64
    assert restored.temperature.item() == pytest.approx(calibration.temperature)
    with pytest.raises(ValueError, match="source-disjoint"):
        temperature_scale(logits, labels, **{**kwargs, "calibration_source_ids": {"train"}})
    with pytest.raises(ValueError, match="distinct"):
        temperature_scale(logits, labels, **{**kwargs, "split_hash": "b" * 64})


def test_training_step_updates_real_parameters_with_finite_gradient():
    from pbpf.train_belief import train_belief_step
    torch.manual_seed(4)
    model = NeuralBeliefModel(3, 2, 7)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before = {name: value.clone() for name, value in model.named_parameters()}
    metrics = train_belief_step(model, batch(), optimizer, particles=4)
    assert math.isfinite(metrics["total"])
    assert metrics["prefix4_nll"] > 0.
    for name, parameter in model.named_parameters():
        if "transition_head" not in name and "child_proposal_head" not in name:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert not torch.equal(before[name], parameter), name


def test_no_ml_imports_are_lazy():
    script = "import sys; import pbpf.belief; import pbpf.conditioning; import pbpf.train_belief; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_neural_interfaces_are_available_through_lazy_belief_exports():
    from pbpf.belief import (NeuralBeliefModel as exported_model, GaussianParams as exported_gaussian,
                             BeliefBatch as exported_batch, fivo_future_loss as exported_loss,
                             temperature_scale as exported_calibration)
    assert exported_model is NeuralBeliefModel and exported_gaussian is GaussianParams
    assert exported_batch is BeliefBatch and exported_loss is fivo_future_loss
    assert exported_calibration is temperature_scale


@pytest.mark.parametrize("weights", [torch.full((2, 4), -torch.inf), torch.full((2, 4), torch.nan)])
def test_future_prediction_rejects_invalid_belief_mass(weights):
    model = NeuralBeliefModel(3, 2, 7)
    data = batch()
    with pytest.raises(ValueError, match="weights"):
        model.future_predict(data.task, data.candidate, data.tests, torch.zeros(2, 4, 2), weights)


def test_calibration_rejects_empty_source_identifiers():
    with pytest.raises(ValueError, match="source"):
        temperature_scale(torch.zeros(1, 5), torch.zeros(1, dtype=torch.long),
                          split_hash="a" * 64, training_split_hash="b" * 64,
                          calibration_source_ids={""}, training_source_ids={"train"})


def test_torch_resampling_never_selects_underflowed_zero_mass_at_boundary(monkeypatch):
    # Seeded half-precision normal draws differ between Torch versions. Build
    # leading zero-mass particles explicitly and exercise the exact CDF boundary.
    def fixed_noise(shape, *, device, dtype, generator):
        assert tuple(shape) == (2, 4, 2)
        return torch.tensor([[[1., 0.], [-1., 0.], [-1., 0.], [-1., 0.]],
                             [[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]]],
                            device=device, dtype=dtype)

    def zero_offset(shape, *, device, dtype, generator):
        return torch.zeros(shape, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", fixed_noise)
    monkeypatch.setattr(torch, "rand", zero_offset)
    model = NeuralBeliefModel(3, 2, 7).half()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.likelihood_head[0].weight[0, -2] = -100.
        # Also underflow in the float64 CDF used by the resampler.
        model.likelihood_head[2].weight[0, 0] = 1000.
        model.likelihood_head[2].weight[1, 0] = -1000.
    data = BeliefBatch(torch.zeros(2, 3, dtype=torch.float16),
                      torch.zeros(2, 3, dtype=torch.float16),
                      torch.zeros(2, 1, 3, dtype=torch.float16), torch.zeros(2, 1, dtype=torch.long))
    # Explicit draws keep this boundary case independent of Torch's RNG version.
    # Positive first coordinates have log mass near -2000, underflowing even
    # in the float64 CDF; a zero offset lands exactly on its leading plateau.
    noise = torch.tensor([[[1., 0.], [-1., 0.], [1., 0.], [-1., 0.]]],
                         dtype=torch.float16).expand(2, -1, -1)
    uniforms = torch.zeros(2, 1, dtype=torch.float64)
    plain = model.filter(data, particles=4, visible_steps=1, ess_fraction=0.,
                         proposal_noise=noise, resampling_uniforms=uniforms)
    moved = model.filter(data, particles=4, visible_steps=1, ess_fraction=1.,
                         proposal_noise=noise, resampling_uniforms=uniforms)
    assert plain.log_weights[0, 0, 0].exp() == 0.
    assert moved.resampled[0, 0]
    selected_mass = plain.log_weights[:, 0].exp().gather(1, moved.resampling_indices[:, 0])
    assert (selected_mass > 0).all()
    assert torch.equal(moved.resampling_indices[:, 0], torch.tensor([[1, 1, 3, 3]]).expand(2, -1))


@pytest.mark.parametrize("scheme, selected, rho, base_mass", [
    ("sampled", [[0, 0, 2], [2, 1, 2]], [[.2, .2, .5], [.6, .1, .6]],
     [[1., 1., 1/15], [7/18, 2/3, 7/18]]),
    ("deterministic_enumeration", [[2, 0, 1], [1, 2, 0]], None,
     [[.1, .6, .3], [.2, .7, .1]]),
])
def test_child_filter_ancestor_base_matches_actual_draws(scheme, selected, rho, base_mass):
    torch.manual_seed(34)
    model = NeuralBeliefModel(3, 2, 7).double()
    original = batch()
    data = BeliefBatch(original.task.double(), original.candidate.double(), original.tests.double(),
                      original.outcomes, original.candidate.double())
    parent_values = np.array([[0., 1.], [2., 3.], [4., 5.]])
    parents = [ParticleSet(str(index), None, parent_values, np.log(weights), np.arange(3), 4, -2.)
               for index, weights in enumerate(([.6, .3, .1], [.1, .2, .7]))]
    indices = torch.tensor(selected)
    proposal = None if rho is None else torch.tensor(rho, dtype=torch.double).log()
    trace = model.filter(data, parents=parents, ancestor_scheme=scheme, ancestors=indices,
                         log_ancestor_proposal=proposal, visible_steps=1, ess_fraction=0.,
                         generator=torch.Generator().manual_seed(42))
    selected_z = torch.tensor(parent_values)[indices]
    q = model.proposal(data.task, data.candidate, data.tests[:, 0], data.outcomes[:, 0],
                       parent_z=selected_z, diff=data.diff)
    transition = model.transition(selected_z, data.task, data.candidate, data.diff)
    actual_z = trace.latents[:, 0]
    torch.testing.assert_close(actual_z, q.mean + q.std * trace.proposal_noise)
    assert not torch.allclose(actual_z, q.mean)
    observed = model.likelihood(actual_z, data.task, data.candidate, data.tests[:, 0]).gather(
        -1, data.outcomes[:, 0, None, None].expand(-1, 3, 1)).squeeze(-1)
    # Literal masses are w[a]/(P*rho[a]) for sampled, w[a] for enumeration.
    # Sampled masses intentionally do not sum to 1 and must not be renormalized.
    corrected = torch.tensor(base_mass, dtype=torch.double).log() + transition.log_prob(actual_z) + observed - q.log_prob(actual_z)
    expected_normalizer = corrected.logsumexp(-1)
    torch.testing.assert_close(trace.log_normalizers[:, 0], expected_normalizer, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(trace.log_weights[:, 0], corrected - expected_normalizer[:, None], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("indices, proposal", [
    ([[0, 0], [0, 1]], None),
    ([[1, 0], [0, 1]], [[.5, .5], [.5, .5]]),
])
def test_torch_deterministic_enumeration_rejects_repeats_and_proposal_terms(indices, proposal):
    model = NeuralBeliefModel(3, 2, 7)
    data = batch()
    data.diff = data.candidate.clone()
    parents = [ParticleSet(str(i), None, np.array([[0., 1.], [2., 3.]]),
                           np.log([.8, .2]), np.arange(2), 4, -2.) for i in range(2)]
    rho = None if proposal is None else torch.tensor(proposal).log()
    with pytest.raises(ValueError, match="permutations and no ancestor proposal"):
        model.filter(data, parents=parents, ancestor_scheme="deterministic_enumeration",
                     ancestors=torch.tensor(indices), log_ancestor_proposal=rho)
