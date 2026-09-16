"""Public-test acquisition using the trained SMC filter and common noise."""
from __future__ import annotations

import torch

from pbpf.belief.features import BeliefBatch


def _entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-15).log()).sum(-1)


@torch.no_grad()
def acquisition_scores(model, task, candidate, public_tests, z, log_weights, policy):
    """Scores depend on public test features and current posterior, never labels."""
    if public_tests.shape[1] != 4:
        raise ValueError('the replay acquisition pool must contain exactly four public tests')
    weights = log_weights.softmax(-1)
    batch, particles, latent = z.shape
    if policy == 'joint_particle_mi':
        values = []
        for test in public_tests.unbind(1):
            conditional = model.likelihood(z, task, candidate, test).exp()
            conditional = conditional / conditional.sum(-1, keepdim=True)
            marginal = (weights[..., None] * conditional).sum(1)
            values.append((_entropy(marginal) - (weights * _entropy(conditional)).sum(1)).clamp_min(0))
        return torch.stack(values, 1)
    if policy not in {'diagnostic_mi', 'predictive_entropy'}:
        raise ValueError('unknown information policy')
    cut = model.difficulty_dim
    # Every diagnosis particle is crossed with the difficulty marginal.
    crossed = torch.cat((z[:, None, :, :cut].expand(-1, particles, -1, -1),
                         z[:, :, None, cut:].expand(-1, -1, particles, -1)), -1)
    crossed = crossed.reshape(batch, particles * particles, latent)
    values = []
    for test in public_tests.unbind(1):
        joint = model.likelihood(crossed, task, candidate, test).exp().reshape(batch, particles, particles, 5)
        joint = joint / joint.sum(-1, keepdim=True)
        conditional = (weights[:, None, :, None] * joint).sum(2)
        marginal = (weights[..., None] * conditional).sum(1)
        value = _entropy(marginal)
        if policy == 'diagnostic_mi':
            value = (value - (weights * _entropy(conditional)).sum(1)).clamp_min(0)
        values.append(value)
    return torch.stack(values, 1)


def public_history(task, candidate, public_tests, public_outcomes, selected):
    """The filter receives only selected public observations, in selection order."""
    if public_tests.shape[1] != 4 or public_outcomes.shape != public_tests.shape[:2]:
        raise ValueError('exactly four public features and outcome slots are required')
    if (selected.ndim != 2 or selected.shape[0] != len(task) or selected.dtype != torch.long
            or selected.shape[1] < 1 or selected.shape[1] > 4
            or ((selected < 0) | (selected >= 4)).any()
            or (selected.sort(1).values[:, 1:] == selected.sort(1).values[:, :-1]).any()):
        raise ValueError('selected tests must be distinct public indices')
    return BeliefBatch(task, candidate,
        public_tests.gather(1, selected[..., None].expand(-1, -1, public_tests.shape[-1])),
        public_outcomes.gather(1, selected))


@torch.no_grad()
def filter_history(model, task, candidate, tests, outcomes, selected, noise, uniforms):
    batch = public_history(task, candidate, tests, outcomes, selected)
    trace = model.filter(batch, particles=noise.shape[1], visible_steps=selected.shape[1],
        proposal_noise=noise, resampling_uniforms=uniforms[:, :selected.shape[1]], ess_fraction=.5)
    return trace.latents[:, -1], trace.log_weights[:, -1]


@torch.no_grad()
def acquire(model, task, candidate, public_tests, public_outcomes, *, policy, noise, uniforms, random_order,
            budgets=(1, 2, 4)):
    if not budgets or any(type(b) is not int or b not in (1, 2, 3, 4) for b in budgets):
        raise ValueError('snapshot budgets must be nonempty public prefix lengths')
    if policy not in {'fixed', 'random', 'diagnostic_mi', 'predictive_entropy', 'joint_particle_mi'}:
        raise ValueError('unknown public acquisition policy')
    if (random_order.shape != (len(task), 4) or random_order.dtype != torch.long
            or not torch.equal(random_order.sort(1).values, torch.arange(4, device=task.device).expand(len(task), -1))):
        raise ValueError('random order must be a permutation of the public pool')
    prior = model.root(task, candidate)
    z = prior.mean[:, None] + prior.std[:, None] * noise
    log_weights = task.new_full(z.shape[:2], -torch.log(task.new_tensor(noise.shape[1])).item())
    selected = torch.empty((len(task), 0), device=task.device, dtype=torch.long)
    snapshots = {}
    for step in range(4):
        if policy == 'fixed':
            choice = torch.full((len(task),), step, device=task.device, dtype=torch.long)
        elif policy == 'random':
            choice = random_order[:, step]
        else:
            scores = acquisition_scores(model, task, candidate, public_tests, z, log_weights, policy)
            scores.scatter_(1, selected, float('-inf'))
            choice = scores.argmax(1)
        selected = torch.cat((selected, choice[:, None]), 1)
        z, log_weights = filter_history(model, task, candidate, public_tests, public_outcomes,
                                        selected, noise, uniforms)
        if step + 1 in budgets:
            snapshots[step + 1] = (z, log_weights, selected.clone())
    return snapshots
