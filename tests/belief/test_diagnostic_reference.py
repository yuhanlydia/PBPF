"""Behavioral checks for causal proposals and the revised learning objective."""

import importlib
import math

import pytest

torch = pytest.importorskip("torch")
from pbpf.belief.features import BeliefBatch


@pytest.fixture
def api():
    return importlib.import_module("pbpf.belief.diagnostic")


def data():
    rng = torch.Generator().manual_seed(4)
    return BeliefBatch(torch.randn(2, 3, generator=rng),
                      torch.randn(2, 3, generator=rng),
                      torch.randn(2, 6, 3, generator=rng),
                      torch.tensor([[0, 1, 0, 1, 2, 0], [1, 0, 1, 0, 2, 1]]))


def test_final_posterior_is_invariant_to_joint_pair_order(api):
    torch.manual_seed(2)
    model = api.HistoryISBeliefModel(3, 4, 12, difficulty_dim=2)
    b = data()
    tests, outcomes = b.tests.clone(), b.outcomes.clone()
    tests[:, :4], outcomes[:, :4] = tests[:, :4].flip(1), outcomes[:, :4].flip(1)
    reordered = BeliefBatch(b.task, b.candidate, tests, outcomes)
    noise = torch.randn(2, 16, 4)
    a = model.filter(b, proposal_noise=noise, particles=16)
    s = model.filter(reordered, proposal_noise=noise, particles=16)
    torch.testing.assert_close(a.latents[:, -1], s.latents[:, -1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a.log_weights[:, -1], s.log_weights[:, -1], atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(a.log_normalizers.sum(1), s.log_normalizers.sum(1))


def test_prefix_proposal_and_posterior_never_read_suffix(api):
    model = api.HistoryISBeliefModel(3, 4, 12, difficulty_dim=2)
    b = data()
    tests, outcomes = b.tests.clone(), b.outcomes.clone()
    tests[:, 2:] += 100
    outcomes[:, 2:] = 4
    changed = BeliefBatch(b.task, b.candidate, tests, outcomes)
    noise = torch.randn(2, 5, 4)
    a = model.filter(b, proposal_noise=noise, particles=5)
    s = model.filter(changed, proposal_noise=noise, particles=5)
    torch.testing.assert_close(a.latents[:, :2], s.latents[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(a.log_weights[:, :2], s.log_weights[:, :2], rtol=0, atol=0)
    assert not torch.allclose(a.latents[:, -1], s.latents[:, -1])


def test_difficulty_proposal_uses_histogram_but_diagnosis_uses_pairs(api):
    torch.manual_seed(5)
    model = api.HistoryISBeliefModel(3, 4, 12, difficulty_dim=2)
    b = data()
    c, _ = b.outcome_counterfactual(visible_steps=4, seed=17)
    a, s = model.history_proposal(b, 4), model.history_proposal(c, 4)
    torch.testing.assert_close(a.mean[:, :2], s.mean[:, :2], rtol=0, atol=0)
    torch.testing.assert_close(a.log_std[:, :2], s.log_std[:, :2], rtol=0, atol=0)
    assert not torch.allclose(a.mean[:, 2:], s.mean[:, 2:])


def test_evidence_includes_prior_over_proposal_correction(api):
    model = api.HistoryISBeliefModel(3, 4, 12, difficulty_dim=2)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.difficulty_proposal[-1].bias[:2] = 1.
        model.diagnosis_proposal[-1].bias[:2] = 1.
    # Four coordinates at z=1: log p(z)-log q(z)=-2. Likelihood=1/5.
    trace = model.filter(data(), particles=3, proposal_noise=torch.zeros(2, 3, 4))
    torch.testing.assert_close(trace.log_normalizers.sum(1),
                              torch.full((2,), -2. - 4 * math.log(5)))
    torch.testing.assert_close(trace.log_weights, torch.full((2, 4, 3), -math.log(3)))
    assert not trace.resampled.any()


def test_latent_mmd_detects_distributions_with_identical_mean(api):
    a = torch.tensor([[[-1.], [1.]]], requires_grad=True)
    b = torch.tensor([[[-2.], [2.]]], requires_grad=True)
    w = torch.full((1, 2), -math.log(2))
    same = api.weighted_latent_mmd(a, w, a.flip(1), w.flip(1))
    torch.testing.assert_close(same, torch.zeros(1), atol=1e-7, rtol=0)
    different = api.weighted_latent_mmd(a, w, b, w)
    assert different.item() > .01
    different.sum().backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


def test_importance_weights_for_unequal_particles_match_analytic_ratio(api):
    model = api.HistoryISBeliefModel(3, 4, 12, difficulty_dim=2)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.difficulty_proposal[-1].bias[0] = 1.
    # q=N(e1,I), p=N(0,I). For z=0 and z=e1, p/q=e^.5, e^-.5.
    noise = torch.zeros(2, 2, 4)
    noise[:, 0, 0] = -1.
    trace = model.filter(data(), particles=2, proposal_noise=noise)
    expected = torch.tensor([math.e / (math.e + 1), 1 / (math.e + 1)]).expand(2, 2)
    torch.testing.assert_close(trace.log_weights[:, -1].exp(), expected)


def test_interaction_decoder_can_express_opposite_triggers_without_nuisance(api):
    model = api.InteractionHistoryISBeliefModel(3, 4, 12, difficulty_dim=2)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.test_projection.weight[0, 0] = 1.
        model.interaction_head.weight[1, 0] = 1.
    task, candidate = torch.zeros(1, 3), torch.zeros(1, 3)
    z = torch.tensor([[[0., 0., 1., 0.], [0., 0., -1., 0.]]])
    positive = torch.tensor([[1., 0., 0.]])
    negative = -positive
    lp = model.likelihood(z, task, candidate, positive)
    ln = model.likelihood(z, task, candidate, negative)
    # Same difficulty, opposite diagnoses: reverse the test, reverse the effect.
    torch.testing.assert_close(lp[:, 0], ln[:, 1])
    torch.testing.assert_close(lp[:, 1], ln[:, 0])
    assert lp[0, 0, 1] > lp[0, 1, 1]


def test_deterministic_interaction_control_uses_pairs_without_future_labels(api):
    torch.manual_seed(23)
    model = api.DeterministicInteractionPredictor(3, 12, 4)
    b = data()
    original = model(b)
    tests, outcomes = b.tests.clone(), b.outcomes.clone()
    tests[:, :4], outcomes[:, :4] = tests[:, :4].flip(1), outcomes[:, :4].flip(1)
    outcomes[:, 4:] = 4
    reordered = BeliefBatch(b.task, b.candidate, tests, outcomes)
    torch.testing.assert_close(original, model(reordered))
    shuffled, _ = b.outcome_counterfactual(visible_steps=4, seed=7)
    assert not torch.allclose(original, model(shuffled))


def test_latent_invariance_sees_difference_when_output_head_is_constant(api):
    from pbpf.belief.model import NeuralBeliefModel
    model = NeuralBeliefModel(3, 4, 12, difficulty_dim=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Let first observed outcome change difficulty proposal despite a
        # completely constant difficulty/diagnosis outcome head.
        model.root_proposal_head[0].weight[0, -4] = 1.
        model.root_proposal_head[2].weight[0, 0] = 1.
    b = data()
    metrics = api.train_diagnostic_step(model, b, torch.optim.SGD(model.parameters(), lr=0.),
        particles=32, generator=torch.Generator().manual_seed(42), shuffle_seed=2)
    assert metrics['invariance'] > .001


def test_objective_units_and_no_bad_shuffle_gradient(api):
    aligned = torch.tensor([.4, .7], requires_grad=True)
    shuffled = torch.tensor([.4, .9], requires_grad=True)
    loss = api.diagnostic_objective(
        torch.tensor([-8., -8.]), evidence_steps=4,
        future_per_test={1: torch.tensor([3., 3.]), 4: torch.tensor([1., 1.])},
        aligned_nll=aligned, shuffled_nll=shuffled,
        eligible=torch.tensor([True, False]), difficulty_mmd=torch.tensor([.2, .4]),
        association_weight=2., invariance_weight=.5, margin=.1)
    assert loss['evidence'].item() == 2.
    assert loss['future'].item() == 2.
    assert loss['total'].item() == pytest.approx(4.3)
    loss['total'].backward()
    torch.testing.assert_close(aligned.grad, torch.tensor([2., 0.]))
    assert shuffled.grad is None or shuffled.grad.abs().sum() == 0


def test_no_eligible_rows_have_zero_auxiliary_and_valid_gradients(api):
    a = torch.tensor([.4], requires_grad=True)
    loss = api.diagnostic_objective(torch.tensor([-2.]), evidence_steps=4,
        future_per_test={4: a}, aligned_nll=a, shuffled_nll=a,
        eligible=torch.tensor([False]), difficulty_mmd=a)
    assert loss['association'].item() == loss['invariance'].item() == 0.
    loss['total'].backward()
    assert torch.isfinite(a.grad).all()


def test_selection_cannot_trade_bad_predictions_for_large_shuffle_gap(api):
    rows = [dict(aligned_nll=.4, association_gap=.0, pair_order_effect=0.),
            dict(aligned_nll=.7, association_gap=1., pair_order_effect=0.),
            dict(aligned_nll=.41, association_gap=.04, pair_order_effect=.001)]
    selected = api.select_diagnostic_checkpoint(rows)
    assert selected['index'] == 2 and selected['development_gate_passed']
    failed = api.select_diagnostic_checkpoint(rows[:2])
    assert failed['index'] == 0 and not failed['development_gate_passed']


def test_selection_rejects_pair_order_effect_even_if_nll_improves(api):
    rows = [dict(aligned_nll=.4, association_gap=.04, pair_order_effect=.03)]
    assert not api.select_diagnostic_checkpoint(rows)['development_gate_passed']


@pytest.mark.parametrize('gap', [.001, -.001])
def test_predictive_selection_reports_signal_without_enforcing_old_gate(api, gap):
    rows = [dict(aligned_nll=.4, association_gap=gap, pair_order_effect=0.),
            dict(aligned_nll=.41, association_gap=.04, pair_order_effect=0.)]
    selected = api.select_diagnostic_checkpoint(rows, policy='predictive_nll')
    assert selected['index'] == 0
    assert selected['association_positive'] == (gap > 0)
    assert selected['gate_enforced'] is False
    assert selected['development_gate_passed'] is False


def test_training_updates_model_and_records_estimator(api):
    torch.manual_seed(31)
    model = api.HistoryISBeliefModel(3, 4, 12, difficulty_dim=2)
    before = model.pair_encoder[0].weight.detach().clone()
    metrics = api.train_diagnostic_step(model, data(),
        torch.optim.Adam(model.parameters(), lr=.001), particles=8)
    assert math.isfinite(metrics['total'])
    assert not torch.equal(before, model.pair_encoder[0].weight)
    assert metrics['inference'] == 'prefix_is'
    assert metrics['invariance_kind'] == 'weighted_latent_rbf_mmd'
