import sys
from pathlib import Path

import torch

from pbpf.belief.features import BeliefBatch


def test_prior_no_resampling_matches_static_bayes_and_preserves_pair_order():
    scripts = str(Path(__file__).resolve().parents[2] / 'scripts')
    sys.path.insert(0, scripts)
    try:
        from diagnose_apbpf_order_sensitivity import DiagnosticBelief
    finally:
        sys.path.remove(scripts)
    torch.manual_seed(319)
    model = DiagnosticBelief(3, 4, 7, difficulty_dim=1)
    model.variant = 'prior_no_resample'
    data = BeliefBatch(torch.randn(2, 3), torch.randn(2, 3),
                      torch.randn(2, 6, 3), torch.tensor([[0, 1, 2, 3, 4, 0], [1, 0, 3, 2, 0, 4]]))
    noise = torch.randn(2, 16, 4)
    prior = model.root(data.task, data.candidate)
    z = prior.mean[:, None] + prior.std[:, None] * noise
    # Static Bayes reference: prior draws weighted by the product of four likelihoods.
    joint = sum(model.likelihood(z, data.task, data.candidate, data.tests[:, j])
                .gather(-1, data.outcomes[:, j, None, None].expand(-1, 16, 1)).squeeze(-1)
                for j in range(4))
    reference = joint.log_softmax(-1)
    trace = model.filter(data, particles=16, proposal_noise=noise)
    assert not trace.resampled.any()
    torch.testing.assert_close(trace.latents[:, -1], z)
    torch.testing.assert_close(trace.log_weights[:, -1], reference)
    order = torch.tensor([3, 2, 1, 0, 4, 5])
    reverse = BeliefBatch(data.task, data.candidate, data.tests[:, order], data.outcomes[:, order])
    reversed_trace = model.filter(reverse, particles=16, proposal_noise=noise)
    torch.testing.assert_close(reversed_trace.latents[:, -1], z)
    torch.testing.assert_close(reversed_trace.log_weights[:, -1], reference)
    changed = data.outcomes.clone()
    changed[:, 0] = (changed[:, 0] + 1) % 5
    alternate = model.filter(BeliefBatch(data.task, data.candidate, data.tests, changed),
                             particles=16, proposal_noise=noise)
    torch.testing.assert_close(alternate.latents[:, -1], z)
    assert not torch.allclose(alternate.log_weights[:, -1], reference)
