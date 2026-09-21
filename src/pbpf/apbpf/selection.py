"""Public-history success heads and source-cross-fitted comparator selection."""
import hashlib

import numpy as np
import torch

from pbpf.belief.features import BeliefBatch
from pbpf.real_gate import public_test_text
from pbpf.registry import OUTCOMES


def public_batch(rows, encoder, *, expected_is_public=False):
    """Never read any future test field or outcome into the selector input."""
    return BeliefBatch(
        torch.tensor(np.stack([encoder(r['task_text']) for r in rows])),
        torch.tensor(np.stack([encoder(r['candidate']) for r in rows])),
        torch.tensor(np.stack([[encoder(public_test_text(t, expected_is_public=expected_is_public))
                                for t in r['tests'][:4]] for r in rows])),
        torch.tensor([[OUTCOMES.index(o) for o in r['outcomes'][:4]] for r in rows]))


class SuccessHead(torch.nn.Module):
    def __init__(self, feature_dim, latent_dim, hidden_dim, arm):
        super().__init__()
        self.arm = arm
        if arm == 'particle':
            self.network = torch.nn.Sequential(torch.nn.Linear(2*feature_dim+latent_dim, hidden_dim),
                                                torch.nn.Tanh(), torch.nn.Linear(hidden_dim, 1))
        elif arm in ('pair_aware', 'deep_sets', 'no_particle_bottleneck'):
            self.pair = torch.nn.Sequential(torch.nn.Linear(feature_dim+5, hidden_dim), torch.nn.Tanh())
            width = 4*hidden_dim if arm == 'pair_aware' else hidden_dim
            bottleneck = latent_dim if arm == 'no_particle_bottleneck' else hidden_dim
            self.network = torch.nn.Sequential(torch.nn.Linear(2*feature_dim+width, bottleneck),
                                                torch.nn.Tanh(), torch.nn.Linear(bottleneck, 1))
        else:
            raise ValueError('unknown success head')

    def forward(self, batch, particles=None, log_weights=None):
        context = torch.cat((batch.task, batch.candidate), -1)
        if self.arm == 'particle':
            values = torch.cat((context[:, None].expand(-1, particles.shape[1], -1), particles), -1)
            component = self.network(values).squeeze(-1).sigmoid()
            return (component*log_weights.softmax(-1)).sum(-1)
        pairs = self.pair(torch.cat((batch.tests,
            torch.nn.functional.one_hot(batch.outcomes, 5).to(context.dtype)), -1))
        history = pairs.flatten(1) if self.arm == 'pair_aware' else pairs.mean(1)
        return self.network(torch.cat((context, history), -1)).squeeze(-1).sigmoid()


def cross_fitted_choices(group_sources, successes, *, seed, folds=5):
    """Select a deterministic comparator on other source folds only.

    Success heads themselves are fit on independent training/validation sources.
    This fold layer chooses among their frozen selectors without self evaluation.
    """
    if len(set(group_sources)) < folds or folds < 2:
        raise ValueError('insufficient sources for cross-fitting')
    if not successes:
        raise ValueError('deterministic comparators required')
    sources = sorted(set(group_sources), key=lambda s: hashlib.sha256(f'{seed}:selection:{s}'.encode()).digest())
    assignment = {s: i % folds for i, s in enumerate(sources)}
    ids = np.array([assignment[s] for s in group_sources])
    names = sorted(successes)
    values = np.array([successes[name] for name in names], dtype=float)
    if values.shape != (len(names), len(group_sources)) or not np.isin(values, [0, 1]).all():
        raise ValueError('exact binary selected-success arrays required')
    selected = np.zeros(len(group_sources))
    choices = []
    for fold in range(folds):
        fit, evaluate = ids != fold, ids == fold
        best = int(values[:, fit].mean(1).argmax())
        selected[evaluate] = values[best, evaluate]
        choices.append({'fold': fold, 'comparator': names[best],
                        'fitting_sources': sorted(set(np.array(group_sources)[fit])),
                        'assessment_sources': sorted(set(np.array(group_sources)[evaluate]))})
    return selected, choices
