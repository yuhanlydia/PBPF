import sys
from pathlib import Path

import torch

from pbpf.belief.features import BeliefBatch
from pbpf.train_belief import train_apbpf_step


def test_observer_matches_actual_backward_without_updating_prior_model():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
    try:
        from audit_apbpf_training_gradients import GradientRecorder, observe_loss
        from run_apbpf_prior_training_debug import PriorTrainingBelief
    finally:
        sys.path.pop(0)
    torch.manual_seed(29)
    model = PriorTrainingBelief(4, 6, 12, difficulty_dim=2)
    batch = BeliefBatch(torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 6, 4),
                       torch.tensor([[0, 1, 0, 1, 0, 2], [1, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 0]]))
    saved = {n: p.detach().clone() for n, p in model.named_parameters()}
    recorder = GradientRecorder(model)
    with observe_loss(recorder):
        metrics = train_apbpf_step(model, batch, recorder, particles=8, invariance_weight=.1,
                                  generator=torch.Generator().manual_seed(73))
    assert metrics['eligible_count'] == 2
    assert all(torch.equal(saved[n], p) for n, p in model.named_parameters())
    assert recorder.record['backward_sum_max_absolute_error'] < 2e-6
    assert recorder.record['groups']['diagnosis_head']['component_l2']['association'] > 0
    assert recorder.record['groups']['root_proposal_head']['component_l2']['association'] == 0
