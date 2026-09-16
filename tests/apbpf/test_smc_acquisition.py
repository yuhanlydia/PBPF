import numpy as np
import pytest
import torch

from pbpf.apbpf.acquisition import NeuralDiagnosticAdapter, expected_information_gain
from pbpf.apbpf.smc_acquisition import acquire, acquisition_scores, filter_history, public_history
from pbpf.belief.model import NeuralBeliefModel


def fixture():
    torch.manual_seed(9)
    model = NeuralBeliefModel(16, 8, 12, difficulty_dim=2).eval()
    task, code = torch.randn(2, 16), torch.randn(2, 16)
    tests = torch.randn(2, 4, 16)
    outcomes = torch.tensor([[0, 1, 2, 3], [1, 0, 4, 2]])
    noise, uniforms = torch.randn(2, 4, 8), torch.rand(2, 4)
    return model, task, code, tests, outcomes, noise, uniforms


def test_batched_diagnostic_scores_match_shipped_adapter():
    model, task, code, tests, outcomes, noise, uniforms = fixture()
    with torch.no_grad():
        prior = model.root(task, code)
        z = prior.mean[:, None] + prior.std[:, None] * noise
        weights = torch.zeros(2, 4)
        scores = acquisition_scores(model, task, code, tests, z, weights, 'diagnostic_mi')
    for i in range(2):
        adapter = NeuralDiagnosticAdapter(model, task[i], code[i], {j: tests[i,j] for j in range(4)})
        state = adapter.state(z[i], weights[i])
        expected = [expected_information_gain(state.component_probs, adapter.component_outcomes(state,j)) for j in range(4)]
        np.testing.assert_allclose(scores[i], expected, atol=2e-7, rtol=1e-4)


def test_filter_history_never_reads_unselected_outcomes_and_rejects_hidden_indices():
    model, task, code, tests, outcomes, noise, uniforms = fixture()
    outcomes[:, 1:] = 999  # Must never reach the model's outcome validator.
    selected = torch.zeros((2,1), dtype=torch.long)
    z, weights = filter_history(model, task, code, tests, outcomes, selected, noise, uniforms)
    assert torch.isfinite(z).all() and torch.isfinite(weights).all()
    with pytest.raises(ValueError, match='distinct public'):
        public_history(task, code, tests, outcomes, torch.full((2,1),4,dtype=torch.long))
    with pytest.raises(ValueError, match='distinct public'):
        public_history(task, code, tests, outcomes, torch.zeros((2,2),dtype=torch.long))


@pytest.mark.parametrize('policy', ['fixed','random','diagnostic_mi','predictive_entropy','joint_particle_mi'])
def test_smc_acquisition_exact_budget_and_first_selection_outcome_independence(policy):
    model, task, code, tests, outcomes, noise, uniforms = fixture()
    order = torch.tensor([[3,2,0,1],[2,0,1,3]])
    options = dict(policy=policy, noise=noise, uniforms=uniforms, random_order=order)
    a = acquire(model, task, code, tests, outcomes, **options)
    b = acquire(model, task, code, tests, (outcomes+1)%5, **options)
    assert torch.equal(a[1][2], b[1][2])
    for budget, (_, _, selected) in a.items():
        assert selected.shape == (2,budget)
        assert all(len(set(row.tolist())) == budget for row in selected)
    assert torch.equal(a[4][2].sort(1).values, torch.arange(4).expand(2,-1))
