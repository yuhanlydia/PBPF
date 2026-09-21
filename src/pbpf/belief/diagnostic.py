"""Experimental causal diagnostic inference; separate from legacy SMC/FIVO."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import FilterTrace, GaussianParams, NeuralBeliefModel, _mlp


class HistoryISBeliefModel(NeuralBeliefModel):
    """Exchangeable proposals refreshed at each *visible* history prefix.

    Each prefix uses self-normalized importance sampling of the static latent.
    The evidence estimate is logmeanexp(log prior + log likelihood - log q).
    It is not an SMC/FIVO bound. No resampling or child-program transitions are
    implemented here. FilterTrace normalizers are differences of prefix log-Zs
    for compatibility with prediction utilities; ancestry fields are identities.
    """

    def __init__(self, feature_dim, latent_dim=32, hidden_dim=128, *, difficulty_dim=8):
        super().__init__(feature_dim, latent_dim, hidden_dim, difficulty_dim=difficulty_dim)
        del self.transition_head, self.root_proposal_head, self.child_proposal_head
        self.pair_encoder = _mlp(feature_dim + 5, hidden_dim, hidden_dim)
        self.difficulty_proposal = _mlp(2 * feature_dim + 6, hidden_dim, 2 * difficulty_dim)
        self.diagnosis_proposal = _mlp(2 * feature_dim + 6 + hidden_dim,
                                       hidden_dim, 2 * self.diagnosis_dim)

    def encode_pairs(self, tests, one_hot_outcomes):
        return self.pair_encoder(torch.cat([tests, one_hot_outcomes], -1)).mean(1)

    def history_proposal(self, batch, prefix):
        if type(prefix) is not int or not 1 <= prefix <= batch.tests.shape[1]:
            raise ValueError("prefix must select a nonempty visible history")
        outcomes = F.one_hot(batch.outcomes[:, :prefix], 5).to(batch.task.dtype)
        histogram = outcomes.mean(1)
        count = batch.task.new_full((len(batch.task), 1), math.log1p(prefix))
        context = torch.cat([batch.task, batch.candidate, histogram, count], -1)
        pairs = self.encode_pairs(batch.tests[:, :prefix], outcomes)
        difficulty = self._gaussian(self.difficulty_proposal, context)
        diagnosis = self._gaussian(self.diagnosis_proposal, torch.cat([context, pairs], -1))
        return GaussianParams(torch.cat([difficulty.mean, diagnosis.mean], -1),
                              torch.cat([difficulty.log_std, diagnosis.log_std], -1))

    def filter(self, batch, *, particles=8, visible_steps=4, generator=None,
               proposal_noise=None, resampling_uniforms=None, ess_fraction=.5):
        batch.validate(self.feature_dim)
        if type(particles) is not int or particles < 1:
            raise ValueError("particles must be a positive integer")
        if type(visible_steps) is not int or not 1 <= visible_steps <= batch.tests.shape[1]:
            raise ValueError("invalid visible_steps")
        if not 0 <= ess_fraction <= 1:
            raise ValueError("invalid ESS fraction")
        size, device, dtype = len(batch.task), batch.task.device, batch.task.dtype
        shape = (size, particles, self.latent_dim)
        noise = (torch.randn(shape, device=device, dtype=dtype, generator=generator)
                 if proposal_noise is None else proposal_noise)
        if noise.shape != shape or noise.device != device or noise.dtype != dtype or not torch.isfinite(noise).all():
            raise ValueError("proposal noise must match batch, particles, and latent dimensions")
        # Accepted for common-noise callers; this estimator does not resample.
        if resampling_uniforms is not None and (
                resampling_uniforms.shape != (size, visible_steps)
                or resampling_uniforms.device != device
                or not torch.isfinite(resampling_uniforms).all()
                or ((resampling_uniforms < 0) | (resampling_uniforms >= 1)).any()):
            raise ValueError("invalid resampling uniforms")
        prior = self.root(batch.task, batch.candidate)
        all_z, all_w, increments = [], [], []
        previous_log_z = batch.task.new_zeros(size)
        for prefix in range(1, visible_steps + 1):
            params = self.history_proposal(batch, prefix)
            q = GaussianParams(params.mean[:, None].expand(shape), params.log_std[:, None].expand(shape))
            z = q.rsample(noise=noise)
            log_likelihood = batch.task.new_zeros(size, particles)
            for index in range(prefix):
                lp = self.likelihood(z, batch.task, batch.candidate, batch.tests[:, index])
                targets = batch.outcomes[:, index, None, None].expand(-1, particles, 1)
                log_likelihood = log_likelihood + lp.gather(-1, targets).squeeze(-1)
            raw_weights = prior.log_prob(z) + log_likelihood - q.log_prob(z)
            log_sum = torch.logsumexp(raw_weights, -1)
            log_z = log_sum - math.log(particles)
            if not torch.isfinite(log_z).all():
                raise ValueError("nonfinite prefix importance evidence")
            increments.append(log_z - previous_log_z)
            previous_log_z = log_z
            all_z.append(z)
            all_w.append(raw_weights - log_sum[:, None])
        identity = torch.arange(particles, device=device).expand(size, -1)
        return FilterTrace(torch.stack(all_z, 1), torch.stack(all_w, 1),
            torch.stack(increments, 1), noise, identity,
            identity[:, None].expand(-1, visible_steps, -1),
            torch.zeros(size, visible_steps, device=device, dtype=torch.bool))


class InteractionHistoryISBeliefModel(HistoryISBeliefModel):
    """A separately ablated bilinear test × diagnosis residual.

    Concatenation MLPs can ignore the latent. This explicit multiplicative path
    makes opposite test-trigger hypotheses easy to express, without supervising
    the latent with bug labels or claiming that it cannot encode nuisance.
    """

    def __init__(self, feature_dim, latent_dim=32, hidden_dim=128, *, difficulty_dim=8):
        super().__init__(feature_dim, latent_dim, hidden_dim, difficulty_dim=difficulty_dim)
        self.test_projection = nn.Linear(feature_dim, self.diagnosis_dim, bias=False)
        self.interaction_head = nn.Linear(self.diagnosis_dim, 5, bias=False)

    def likelihood_components(self, z, task, candidate, test):
        difficulty_logits, diagnosis_logits = super().likelihood_components(z, task, candidate, test)
        _, diagnosis = self.split_latent(z)
        projected_test = self._expand(self.test_projection(test), z)
        return difficulty_logits, diagnosis_logits + self.interaction_head(diagnosis * projected_test)


class InteractionOnlyHistoryISBeliefModel(InteractionHistoryISBeliefModel):
    """Remove the additive diagnosis MLP shortcut; retain only test x g.

    Difficulty carries global class logits. Diagnosis vanishes if test features
    or g are zero. This does not establish identifiability: test features can
    still contain candidate-global components. The encoder is unchanged.
    """

    def __init__(self, feature_dim, latent_dim=32, hidden_dim=128, *, difficulty_dim=8):
        super().__init__(feature_dim, latent_dim, hidden_dim, difficulty_dim=difficulty_dim)
        del self.diagnosis_head

    def likelihood_components(self, z, task, candidate, test):
        _, diagnosis = self.split_latent(z)
        projected = self._expand(self.test_projection(test), z)
        return self.difficulty_logits(z, task, candidate), self.interaction_head(diagnosis * projected)


class HighGainHistoryISBeliefModel(InteractionOnlyHistoryISBeliefModel):
    """Initialization-only ablation: tenfold test x g logit scale.

    All other initialized tensors and the training objective match the
    interaction-only arm under the same seed. Weights remain freely trainable;
    this is not a likelihood temperature or a post-hoc probability adjustment.
    """

    def __init__(self, feature_dim, latent_dim=32, hidden_dim=128, *, difficulty_dim=8):
        super().__init__(feature_dim, latent_dim, hidden_dim, difficulty_dim=difficulty_dim)
        with torch.no_grad():
            self.interaction_head.weight.mul_(10.)


class BoundHistoryISBeliefModel(InteractionHistoryISBeliefModel):
    """Bind test features to outcome classes before any learned aggregation.

    A near-linear encoder of concatenated test/outcome vectors loses association
    under averaging. Outcome-conditioned feature blocks preserve correspondence
    even with a linear pair encoder. This changes encoder capacity; it is an
    architectural ablation, not a parameter-matched identifiability proof.
    """

    def __init__(self, feature_dim, latent_dim=32, hidden_dim=128, *, difficulty_dim=8):
        super().__init__(feature_dim, latent_dim, hidden_dim, difficulty_dim=difficulty_dim)
        self.pair_encoder = _mlp(5 * feature_dim, hidden_dim, hidden_dim)

    def encode_pairs(self, tests, one_hot_outcomes):
        bound = (one_hot_outcomes[..., :, None] * tests[..., None, :]).flatten(-2)
        return self.pair_encoder(bound).mean(1)


class DeterministicInteractionPredictor(nn.Module):
    """Strong non-particle control with explicit test/outcome binding.

    The five outcome-conditioned test summaries retain which tests produced each
    label. A learned history vector interacts with each future test. Only the
    first four outcome labels are read, and pair order never enters the model.
    """

    def __init__(self, feature_dim, hidden_dim=192, latent_dim=32):
        super().__init__()
        self.context_head = _mlp(7 * feature_dim + 5, hidden_dim, latent_dim)
        self.difficulty_head = _mlp(2 * feature_dim + 5, hidden_dim, 5)
        self.test_projection = nn.Linear(feature_dim, latent_dim, bias=False)
        self.interaction_head = nn.Linear(latent_dim, 5, bias=False)

    def forward(self, batch):
        labels = F.one_hot(batch.outcomes[:, :4], 5).to(batch.task.dtype)
        histogram = labels.mean(1)
        summaries = (labels[:, :, :, None] * batch.tests[:, :4, None]).mean(1).flatten(1)
        nuisance = torch.cat([batch.task, batch.candidate, histogram], -1)
        context = self.context_head(torch.cat([nuisance, summaries], -1))
        interaction = context[:, None] * self.test_projection(batch.tests[:, 4:])
        return self.difficulty_head(nuisance)[:, None] + self.interaction_head(interaction)


def weighted_latent_mmd(a, log_weights_a, b, log_weights_b):
    """Biased empirical MMD² with RBF bandwidths 0.5, 1, 2; one value per row.

    Inputs should already be scaled by a fixed/detached reference distribution.
    Unlike output-space KL this can see latent distributions with the same head
    output. It is neither latent KL nor an identifiability guarantee.
    """
    if (a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]
            or a.shape[2] != b.shape[2] or min(*a.shape, *b.shape) < 1
            or log_weights_a.shape != a.shape[:2] or log_weights_b.shape != b.shape[:2]):
        raise ValueError("MMD requires [batch,particles,dimensions] and matching weights")
    for points, weights in ((a, log_weights_a), (b, log_weights_b)):
        if not torch.isfinite(points).all() or not torch.isfinite(torch.logsumexp(weights, -1)).all():
            raise ValueError("nonfinite MMD inputs")
    wa, wb = log_weights_a.softmax(-1), log_weights_b.softmax(-1)

    def kernel(x, y):
        distances = (x[:, :, None] - y[:, None]).square().mean(-1)
        return sum(torch.exp(-distances / (2 * scale ** 2)) for scale in (.5, 1., 2.)) / 3

    aa = (wa[:, :, None] * wa[:, None] * kernel(a, a)).sum((1, 2))
    bb = (wb[:, :, None] * wb[:, None] * kernel(b, b)).sum((1, 2))
    ab = (wa[:, :, None] * wb[:, None] * kernel(a, b)).sum((1, 2))
    return (aa + bb - 2 * ab).clamp_min(0.)


def diagnostic_objective(log_evidence, *, evidence_steps, future_per_test,
                         aligned_nll, shuffled_nll, eligible, difficulty_mmd,
                         evidence_weight=1., future_weight=1., association_weight=1.,
                         invariance_weight=.1, margin=.03):
    """All likelihood losses use nats/test; shuffled NLL is stop-gradient."""
    if type(evidence_steps) is not int or evidence_steps < 1 or not future_per_test:
        raise ValueError("nonempty evidence and future targets are required")
    for value in (evidence_weight, future_weight, association_weight, invariance_weight, margin):
        if not math.isfinite(value) or value < 0:
            raise ValueError("coefficients must be finite and nonnegative")
    shape = log_evidence.shape
    vectors = (aligned_nll, shuffled_nll, difficulty_mmd, *future_per_test.values())
    if (len(shape) != 1 or shape[0] == 0 or eligible.shape != shape or eligible.dtype != torch.bool
            or any(v.shape != shape or not torch.isfinite(v).all() for v in vectors)
            or not torch.isfinite(log_evidence).all()):
        raise ValueError("finite per-candidate losses and boolean eligibility required")
    evidence = -log_evidence.mean() / evidence_steps
    future = torch.stack(list(future_per_test.values())).mean()
    association = (F.relu(margin + aligned_nll - shuffled_nll.detach())[eligible].mean()
                   if eligible.any() else aligned_nll.sum() * 0.)
    invariance = (difficulty_mmd[eligible].mean() if eligible.any() else difficulty_mmd.sum() * 0.)
    total = (evidence_weight * evidence + future_weight * future
             + association_weight * association + invariance_weight * invariance)
    return dict(total=total, evidence=evidence, future=future,
                association=association, invariance=invariance)


def train_diagnostic_step(model, batch, optimizer, *, particles=8, visible_steps=4,
                          shuffle_seed=0, generator=None, evidence_weight=1.,
                          future_weight=1., association_weight=1., invariance_weight=.1,
                          margin=.03, measure_gradients=False):
    if model.difficulty_dim is None:
        raise ValueError("diagnostic training requires a factored model")
    if type(visible_steps) is not int or not 1 <= visible_steps < batch.tests.shape[1]:
        raise ValueError("visible history must leave future targets")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    counterfactual, eligible = batch.outcome_counterfactual(visible_steps=visible_steps, seed=shuffle_seed)
    noise = torch.randn((len(batch.task), particles, model.latent_dim),
                        device=batch.task.device, dtype=batch.task.dtype, generator=generator)
    uniforms = torch.rand((len(batch.task), visible_steps), device=batch.task.device,
                          dtype=torch.double, generator=generator)
    options = dict(particles=particles, visible_steps=visible_steps,
                   proposal_noise=noise, resampling_uniforms=uniforms)
    a, s = model.filter(batch, **options), model.filter(counterfactual, **options)
    prefixes = tuple(p for p in (1, 2, 4) if p <= visible_steps)
    future = {p: value / (batch.tests.shape[1] - p)
              for p, value in model.future_nll(batch, a, prefixes=prefixes).items()}

    def nll(trace):
        predictions = model.future_predict(batch.task, batch.candidate, batch.tests[:, visible_steps:],
                                           trace.latents[:, -1], trace.log_weights[:, -1])
        return -predictions.gather(-1, batch.outcomes[:, visible_steps:, None]).squeeze(-1).mean(-1)

    aligned_nll, shuffled_nll = nll(a), nll(s)
    dim = model.difficulty_dim
    reference = model.root(batch.task, batch.candidate)
    center, scale = reference.mean[:, None, :dim].detach(), reference.std[:, None, :dim].detach()
    mmd = weighted_latent_mmd((a.latents[:, -1, :, :dim] - center) / scale, a.log_weights[:, -1],
                             (s.latents[:, -1, :, :dim] - center) / scale, s.log_weights[:, -1])
    coefficients = dict(evidence_weight=evidence_weight, future_weight=future_weight,
                        association_weight=association_weight, invariance_weight=invariance_weight, margin=margin)
    loss = diagnostic_objective(a.log_normalizers.sum(-1), evidence_steps=visible_steps,
        future_per_test=future, aligned_nll=aligned_nll, shuffled_nll=shuffled_nll,
        eligible=eligible, difficulty_mmd=mmd, **coefficients)
    gradient_norms = {}
    if measure_gradients:
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        for term in ('evidence', 'future', 'association', 'invariance'):
            grads = torch.autograd.grad(loss[term] * coefficients[term + '_weight'], parameters,
                                        retain_graph=True, allow_unused=True)
            gradient_norms[term] = math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))
    loss['total'].backward()
    optimizer.step()
    protocol = dict(inference='prefix_is' if isinstance(model, HistoryISBeliefModel) else 'legacy_smc',
                    loss_units='nats_per_test', association_gradient='aligned_only',
                    invariance_kind='weighted_latent_rbf_mmd', particles=particles,
                    visible_steps=visible_steps, **coefficients)
    model.apbpf_training_state = protocol
    gap = shuffled_nll - aligned_nll
    return {**{k: float(v.detach()) for k, v in loss.items()}, **protocol,
            'association_gap': float(gap.detach().mean()),
            'eligible_gap': float(gap[eligible].detach().mean()) if eligible.any() else None,
            'eligible_fraction': float(eligible.float().mean()),
            'aligned_nll': float(aligned_nll.detach().mean()),
            'shuffled_nll': float(shuffled_nll.detach().mean()),
            'gradient_norms': gradient_norms}


def select_diagnostic_checkpoint(rows, *, margin=.03, nll_tolerance=.02, pair_fraction=.25,
                                 policy='historical_screen'):
    """Select predictive NLL or reproduce the historical diagnostic screen."""
    if policy not in ('predictive_nll', 'historical_screen'):
        raise ValueError('unknown checkpoint policy')
    if not rows or any(not math.isfinite(v) or v < 0 for v in (margin, nll_tolerance, pair_fraction)):
        raise ValueError("nonempty validation history and nonnegative finite thresholds required")
    if any(not all(math.isfinite(r[k]) for k in ('aligned_nll', 'association_gap', 'pair_order_effect'))
           or r['pair_order_effect'] < 0 for r in rows):
        raise ValueError("invalid development metrics")
    best = min(range(len(rows)), key=lambda i: rows[i]['aligned_nll'])
    eligible = [i for i, row in enumerate(rows)
                if row['aligned_nll'] <= rows[best]['aligned_nll'] + nll_tolerance
                and row['association_gap'] >= margin and row['association_gap'] > 0
                and row['pair_order_effect'] <= pair_fraction * row['association_gap']]
    chosen = min(eligible, key=lambda i: rows[i]['aligned_nll']) if eligible and policy == 'historical_screen' else best
    return dict(index=chosen, development_gate_passed=chosen in eligible, best_nll_index=best,
                policy=policy, gate_enforced=policy == 'historical_screen',
                association_positive=rows[chosen]['association_gap'] > 0,
                margin=margin, nll_tolerance=nll_tolerance, pair_fraction=pair_fraction)
