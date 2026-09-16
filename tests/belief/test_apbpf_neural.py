"""Opt-in factorization, paired SMC, and association-training contracts."""

import copy
import math

import pytest

torch = pytest.importorskip("torch")

from pbpf.belief.features import BeliefBatch
from pbpf.belief.model import NeuralBeliefModel
from pbpf.belief import losses
from pbpf import train_belief


def batch():
    rng = torch.Generator().manual_seed(31)
    return BeliefBatch(torch.randn(2, 3, generator=rng),
                      torch.randn(2, 3, generator=rng),
                      torch.randn(2, 6, 3, generator=rng),
                      torch.tensor([[0, 1, 0, 1, 2, 0], [0, 0, 0, 0, 1, 2]]))


@pytest.mark.parametrize("dimension", [0, -1, 4, 5, 1.5, True])
def test_factor_dimensions_require_two_nonempty_integer_factors(dimension):
    with pytest.raises(ValueError, match="difficulty"):
        NeuralBeliefModel(3, 4, 7, difficulty_dim=dimension)


def test_additive_logits_isolate_test_and_latent_factors():
    torch.manual_seed(1)
    model = NeuralBeliefModel(3, 4, 7, difficulty_dim=1)
    data = batch()
    z = torch.randn(2, 5, 4)
    difficulty, diagnosis = model.split_latent(z)
    assert difficulty.shape == (2, 5, 1)
    assert diagnosis.shape == (2, 5, 3)
    d, g = model.likelihood_components(z, data.task, data.candidate, data.tests[:, 0])
    d2, g2 = model.likelihood_components(z, data.task, data.candidate, data.tests[:, 1])
    torch.testing.assert_close(d, d2, rtol=0, atol=0)
    assert not torch.allclose(g, g2)
    changed = z.clone()
    changed[..., 1:] += 10
    d3, _ = model.likelihood_components(changed, data.task, data.candidate, data.tests[:, 0])
    torch.testing.assert_close(d, d3, rtol=0, atol=0)
    changed = z.clone()
    changed[..., :1] += 10
    _, g3 = model.likelihood_components(changed, data.task, data.candidate, data.tests[:, 0])
    torch.testing.assert_close(g, g3, rtol=0, atol=0)
    model.temperature.fill_(2.)
    actual = model.likelihood(z, data.task, data.candidate, data.tests[:, 0])
    torch.testing.assert_close(actual, torch.log_softmax((d + g) / 2., -1))
    with pytest.raises(ValueError, match="latent"):
        model.split_latent(torch.zeros(2, 3))


def loss_inputs():
    return dict(log_normalizers=torch.tensor([[-2.], [-4.]], requires_grad=True),
                future_nll={1: torch.tensor([2., 4.], requires_grad=True)},
                aligned_nll=torch.tensor([2., 100.], requires_grad=True),
                shuffled_nll=torch.tensor([2.1, -100.], requires_grad=True),
                difficulty_aligned=torch.tensor([[.8, .2], [.9, .1]], requires_grad=True),
                difficulty_shuffled=torch.tensor([[.2, .8], [.1, .9]], requires_grad=True),
                eligible=torch.tensor([True, False]))


def test_loss_uses_only_eligible_auxiliaries_but_full_population_gap():
    args = loss_inputs()
    result = losses.association_aware_belief_loss(**args, margin=.3, future_weight=.5,
                                               association_weight=2., invariance_weight=3.)
    assert result.fivo.item() == 3.
    assert result.future.item() == 3.
    assert result.association.item() == pytest.approx(.2)
    # 1/2 (KL(p||q) + KL(q||p)) = .6 log(4) for this fixture.
    assert result.invariance.item() == pytest.approx(.6 * math.log(4))
    assert result.total.item() == pytest.approx(4.5 + .4 + 1.8 * math.log(4))
    assert result.association_gap.item() == pytest.approx(-99.95)
    assert result.eligible_gap.item() == pytest.approx(.1)
    result.total.backward()
    torch.testing.assert_close(args["aligned_nll"].grad, torch.tensor([2., 0.]))
    torch.testing.assert_close(args["shuffled_nll"].grad, torch.tensor([-2., 0.]))
    assert args["difficulty_aligned"].grad[0].abs().sum() > 0
    assert args["difficulty_aligned"].grad[1].abs().sum() == 0


def test_no_eligible_rows_give_exact_differentiable_zero_auxiliaries():
    args = loss_inputs()
    args["eligible"][:] = False
    result = losses.association_aware_belief_loss(**args)
    assert result.association.item() == result.invariance.item() == 0.
    assert result.eligible_gap is None
    (result.association + result.invariance).backward()
    for name in ("aligned_nll", "shuffled_nll", "difficulty_aligned", "difficulty_shuffled"):
        assert args[name].grad is not None
        assert args[name].grad.abs().sum() == 0


def test_identical_difficulty_distributions_have_zero_invariance():
    args = loss_inputs()
    args["difficulty_shuffled"] = args["difficulty_aligned"].clone()
    assert losses.association_aware_belief_loss(**args).invariance.item() == 0.


def test_counterfactual_batch_preserves_features_future_and_constant_rows():
    data = batch()
    counterfactual, eligible = data.outcome_counterfactual(visible_steps=4, seed=7)
    assert eligible.tolist() == [True, False]
    assert counterfactual.tests is data.tests
    assert counterfactual.task is data.task and counterfactual.candidate is data.candidate
    assert counterfactual.outcomes[0, :4].tolist() != [0, 1, 0, 1]
    assert counterfactual.outcomes[0, :4].sort().values.tolist() == [0, 0, 1, 1]
    assert torch.equal(counterfactual.outcomes[:, 4:], data.outcomes[:, 4:])
    assert torch.equal(counterfactual.outcomes[1], data.outcomes[1])
    assert data.outcomes[0].tolist() == [0, 1, 0, 1, 2, 0]


def test_explicit_smc_randomness_replays_without_consuming_rng():
    model = NeuralBeliefModel(3, 4, 7, difficulty_dim=1)
    data = batch()
    noise = torch.randn(2, 5, 4)
    uniforms = torch.full((2, 4), .25, dtype=torch.double)
    options = dict(particles=5, ess_fraction=1., proposal_noise=noise,
                   resampling_uniforms=uniforms)
    state = torch.random.get_rng_state()
    trace = model.filter(data, **options)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(trace.proposal_noise, noise)
    replay = model.filter(data, **options)
    torch.testing.assert_close(trace.log_normalizers, replay.log_normalizers, rtol=0, atol=0)
    assert torch.equal(trace.resampling_indices, replay.resampling_indices)


class RecordingModel(NeuralBeliefModel):
    """Record real filter executions; no replacement inference or fake loss."""
    def filter(self, batch, **kwargs):
        trace = super().filter(batch, **kwargs)
        self.executions.append((batch, kwargs, trace))
        return trace


def test_training_pairs_randomness_keeps_base_aligned_and_checkpoints_protocol():
    torch.manual_seed(4)
    model = RecordingModel(3, 4, 7, difficulty_dim=1)
    model.executions = []
    data = batch()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    metrics = train_belief.train_apbpf_step(model, data, optimizer, particles=5,
        future_weight=.25, association_weight=.5, invariance_weight=.75, margin=.3,
        shuffle_seed=17, generator=torch.Generator().manual_seed(91))
    assert len(model.executions) == 2
    aligned, shuffled = model.executions
    assert aligned[0] is data
    assert torch.equal(aligned[1]["proposal_noise"], shuffled[1]["proposal_noise"])
    assert torch.equal(aligned[1]["resampling_uniforms"], shuffled[1]["resampling_uniforms"])
    assert metrics["fivo"] == pytest.approx(-aligned[2].log_normalizers.detach().sum(-1).mean().item())
    assert metrics["eligible_count"] == 1 and metrics["population_count"] == 2
    assert metrics["eligible_fraction"] == .5
    assert metrics["association_gap"] == pytest.approx(metrics["shuffled_nll"] - metrics["aligned_nll"], abs=1e-6)
    assert metrics["total"] == pytest.approx(metrics["fivo"] + .25 * metrics["future"]
        + .5 * metrics["association"] + .75 * metrics["invariance"])
    assert not torch.equal(before["difficulty_head.0.weight"], model.state_dict()["difficulty_head.0.weight"])
    for parameter in model.diagnosis_head.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    restored = NeuralBeliefModel(3, 4, 7, difficulty_dim=1)
    restored.load_state_dict(model.state_dict())
    assert restored.apbpf_training_state == model.apbpf_training_state
    state = restored.apbpf_training_state
    assert state["shuffle_seed"] == 17 and state["eligible_mask"] == [True, False]
    assert state["difficulty_dim"] == 1 and state["diagnosis_dim"] == 3
    assert state["association_weight"] == .5 and state["invariance_weight"] == .75
    assert len(state["config_hash"]) == 64


def test_constant_histories_report_full_population_without_auxiliary_signal():
    data = batch()
    data.outcomes[:, :4] = 0
    model = NeuralBeliefModel(3, 4, 7, difficulty_dim=1)
    metrics = train_belief.train_apbpf_step(model, data,
        torch.optim.SGD(model.parameters(), lr=0.), particles=3, shuffle_seed=2)
    assert metrics["eligible_count"] == 0 and metrics["population_count"] == 2
    assert metrics["association"] == metrics["invariance"] == metrics["association_gap"] == 0.
    assert metrics["eligible_gap"] is None
    assert metrics["future"] > 0 and metrics["prefix4_nll"] > 0


def test_legacy_constructor_and_checkpoint_remain_unfactored():
    model = NeuralBeliefModel(3, 2, 7)
    restored = NeuralBeliefModel(3, 2, 7)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert model.difficulty_dim is None
    assert "apbpf_training_state" not in model.get_extra_state()
    with pytest.raises(ValueError, match="factored"):
        train_belief.train_apbpf_step(model, batch(), torch.optim.SGD(model.parameters(), lr=.01))
