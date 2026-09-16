import sys
from pathlib import Path

import torch

from pbpf.belief.features import BeliefBatch


def model_and_batch():
    scripts = str(Path(__file__).resolve().parents[2] / 'scripts')
    sys.path.insert(0, scripts)
    try:
        from diagnose_apbpf_pooled_proposal import PooledProposalBelief
    finally:
        sys.path.remove(scripts)
    torch.manual_seed(722)
    model = PooledProposalBelief(3, 4, 7, difficulty_dim=1)
    batch = BeliefBatch(torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 6, 3),
                       torch.tensor([[0, 1, 2, 3, 4, 0], [1, 0, 3, 2, 0, 4]]))
    return model, batch


def test_pooled_filter_uses_correct_importance_density_and_is_pair_invariant():
    model, batch = model_and_batch()
    noise = torch.randn(2, 16, 4)
    q = model.pooled_proposal(batch, 4)
    z = q.mean[:, None] + q.std[:, None] * noise
    likelihood = sum(model.likelihood(z, batch.task, batch.candidate, batch.tests[:, j])
        .gather(-1, batch.outcomes[:, j, None, None].expand(-1, 16, 1)).squeeze(-1) for j in range(4))
    reference = (model.root(batch.task, batch.candidate).log_prob(z) - q.log_prob(z) + likelihood).log_softmax(-1)
    trace = model.filter(batch, particles=16, proposal_noise=noise)
    assert not trace.resampled.any()
    torch.testing.assert_close(trace.latents[:, -1], z)
    torch.testing.assert_close(trace.log_weights[:, -1], reference)
    order = torch.tensor([2, 0, 3, 1, 4, 5])
    alternate = model.filter(BeliefBatch(batch.task, batch.candidate, batch.tests[:, order], batch.outcomes[:, order]),
                             particles=16, proposal_noise=noise)
    torch.testing.assert_close(alternate.latents[:, -1], z)
    torch.testing.assert_close(alternate.log_weights[:, -1], reference)
    assert model._pooled_proposal is None


def test_pooled_proposal_uses_visible_pairs_and_excludes_future_information():
    model, batch = model_and_batch()
    original = model.pooled_proposal(batch, 4)
    tests, outcomes = batch.tests.clone(), batch.outcomes.clone()
    tests[:, 4:] = 1000
    outcomes[:, 4:] = (outcomes[:, 4:] + 1) % 5
    future_changed = model.pooled_proposal(BeliefBatch(batch.task, batch.candidate, tests, outcomes), 4)
    torch.testing.assert_close(original.mean, future_changed.mean, rtol=0, atol=0)
    torch.testing.assert_close(original.log_std, future_changed.log_std, rtol=0, atol=0)
    for j in range(4):
        changed = batch.outcomes.clone()
        changed[:, j] = (changed[:, j] + 1) % 5
        q = model.pooled_proposal(BeliefBatch(batch.task, batch.candidate, batch.tests, changed), 4)
        assert not torch.allclose(original.mean, q.mean)
