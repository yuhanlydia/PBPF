import sys
from pathlib import Path

import numpy as np
import torch

from pbpf.belief.features import BeliefBatch


class KnownLikelihood:
    def __init__(self, interaction):
        self.interaction = interaction

    def likelihood(self, z, task, candidate, test):
        latent, query = z[..., 0], test[:, :1]
        value = latent*query if self.interaction else latent+query
        logits = torch.stack([value, torch.zeros_like(value), *[torch.full_like(value, -10.)]*3], -1)
        return logits.log_softmax(-1)

    def future_predict(self, task, candidate, tests, z, log_weights):
        return torch.stack([torch.logsumexp(log_weights[..., None] + self.likelihood(z, task, candidate, test), 1)
                            for test in tests.unbind(1)], 1)


def test_target_audit_distinguishes_separable_from_assignment_sensitive_likelihood():
    scripts = str(Path(__file__).resolve().parents[2] / 'scripts')
    sys.path.insert(0, scripts)
    try:
        from audit_apbpf_likelihood_association import compare_targets
    finally:
        sys.path.remove(scripts)
    batch = BeliefBatch(torch.zeros(1, 1), torch.zeros(1, 1),
                       torch.tensor([[[1.], [-1.], [2.], [-2.], [1.], [2.]]]),
                       torch.tensor([[0, 1, 0, 1, 0, 0]]))
    shuffled = torch.tensor([[1, 0, 1, 0, 0, 0]])
    z = torch.tensor([[[-2.], [-1.], [1.], [2.]]])
    _, separable = compare_targets(KnownLikelihood(False), batch, z, shuffled)
    assert np.max(separable['centered_log_likelihood_ratio_rms']) < 1e-6
    assert np.max(separable['posterior_total_variation']) < 1e-6
    _, interactive = compare_targets(KnownLikelihood(True), batch, z, shuffled)
    assert np.min(interactive['centered_log_likelihood_ratio_rms']) > 1
    assert np.min(interactive['posterior_total_variation']) > .9
    assert np.min(interactive['future_probability_max_difference']) > .5
