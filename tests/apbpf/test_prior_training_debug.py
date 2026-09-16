import sys
from pathlib import Path

import torch

from pbpf.belief.features import BeliefBatch
from pbpf.train_belief import train_apbpf_step


def test_prior_training_preserves_prefix_visibility_and_updates_likelihood():
    scripts = str(Path(__file__).resolve().parents[2] / 'scripts')
    sys.path.insert(0, scripts)
    try:
        from run_apbpf_prior_training_debug import PriorTrainingBelief
    finally:
        sys.path.remove(scripts)
    torch.manual_seed(74)
    model = PriorTrainingBelief(3, 4, 7, difficulty_dim=1)
    batch = BeliefBatch(torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 6, 3),
                       torch.tensor([[0, 1, 0, 1, 2, 3], [1, 0, 1, 0, 3, 2]]))
    noise = torch.randn(2, 16, 4)
    original = model.filter(batch, particles=16, proposal_noise=noise)
    changed = batch.outcomes.clone()
    changed[:, 1:] = (changed[:, 1:] + 1) % 5
    alternate = model.filter(BeliefBatch(batch.task, batch.candidate, batch.tests, changed),
                             particles=16, proposal_noise=noise)
    assert not original.resampled.any() and not alternate.resampled.any()
    torch.testing.assert_close(original.latents[:, 0], alternate.latents[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(original.log_weights[:, 0], alternate.log_weights[:, 0], rtol=0, atol=0)
    before = model.diagnosis_head[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    metrics = train_apbpf_step(model, batch, optimizer, particles=16, shuffle_seed=77)
    assert torch.isfinite(torch.tensor(metrics['total']))
    assert not torch.equal(before, model.diagnosis_head[0].weight.detach())
    assert all(p.grad is None for p in model.root_proposal_head.parameters())
