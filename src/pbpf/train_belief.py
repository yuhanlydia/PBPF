"""Dependency-light Stage-B training entry point; no model/data downloads."""


def train_belief_step(model, batch, optimizer, *, particles=8, future_weight=1., generator=None,
                      **filter_options):
    """One optimizer step over visible SMC evidence plus teacher future targets."""
    from .belief.losses import fivo_future_loss

    model.train()
    optimizer.zero_grad(set_to_none=True)
    trace = model.filter(batch, particles=particles, generator=generator, **filter_options)
    loss = fivo_future_loss(trace.log_normalizers, model.future_nll(batch, trace), future_weight=future_weight)
    loss.total.backward()
    optimizer.step()
    return {"total": float(loss.total.detach()), "fivo": float(loss.fivo.detach()),
            "future": float(loss.future.detach()), "prefix4_nll": float(loss.prefix4_nll.detach())}


def train_apbpf_step(model, batch, optimizer, *, particles=8, visible_steps=4,
                     future_weight=1., association_weight=1., invariance_weight=1.,
                     margin=.03, shuffle_seed=0, generator=None, **filter_options):
    """One factored A-PBPF optimizer step, with common random numbers.

    Only aligned evidence/future targets enter the base loss. Counterfactuals
    alter visible outcome assignments only. Both paths score the same future
    tests, per test, for the association margin and full-population gap.
    Training metrics are diagnostics, not source-cluster confidence intervals.
    """
    import hashlib
    import json
    import torch

    from .belief.losses import association_aware_belief_loss

    if model.difficulty_dim is None:
        raise ValueError("A-PBPF training requires a factored model")
    batch.validate(model.feature_dim)
    if (isinstance(visible_steps, bool) or not isinstance(visible_steps, int)
            or not 1 <= visible_steps < batch.tests.shape[1]):
        raise ValueError("visible_steps must leave at least one untouched future target")
    if "proposal_noise" in filter_options or "resampling_uniforms" in filter_options:
        raise ValueError("A-PBPF owns the common proposal noise and resampling uniforms")
    counterfactual, eligible = batch.outcome_counterfactual(visible_steps=visible_steps, seed=shuffle_seed)
    parents = filter_options.get("parents")
    if parents is not None:
        if len(parents) != len(batch.task):
            raise ValueError("one parent snapshot per batch row is required")
        particles = parents[0].z.shape[0]
    if isinstance(particles, bool) or not isinstance(particles, int) or particles <= 0:
        raise ValueError("particles must be a positive integer")
    prefixes = tuple(prefix for prefix in (1, 2, 4) if prefix <= visible_steps)
    config = dict(feature_dim=model.feature_dim, latent_dim=model.latent_dim,
                  hidden_dim=model.hidden_dim, difficulty_dim=model.difficulty_dim,
                  diagnosis_dim=model.diagnosis_dim, particles=particles,
                  visible_steps=visible_steps, future_weight=future_weight,
                  association_weight=association_weight, invariance_weight=invariance_weight,
                  margin=margin, shuffle_seed=shuffle_seed,
                  ess_fraction=filter_options.get("ess_fraction", .5),
                  ancestor_scheme=filter_options.get("ancestor_scheme"),
                  future_prefixes=list(prefixes), association_nll_units="nats_per_future_test")
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode()).hexdigest()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    common = dict(
        proposal_noise=torch.randn((len(batch.task), particles, model.latent_dim),
            device=batch.task.device, dtype=batch.task.dtype, generator=generator),
        resampling_uniforms=torch.rand((len(batch.task), visible_steps),
            device=batch.task.device, dtype=torch.double, generator=generator))
    aligned = model.filter(batch, particles=particles, visible_steps=visible_steps,
                           **common, **filter_options)
    shuffled = model.filter(counterfactual, particles=particles, visible_steps=visible_steps,
                            **common, **filter_options)
    future = model.future_nll(batch, aligned, prefixes=prefixes)

    def paired_predictions(trace):
        z, weights = trace.latents[:, -1], trace.log_weights[:, -1]
        predictions = model.future_predict(batch.task, batch.candidate,
                                            batch.tests[:, visible_steps:], z, weights)
        nll = -predictions.gather(-1, batch.outcomes[:, visible_steps:, None]).squeeze(-1).mean(-1)
        difficulty = torch.softmax(model.difficulty_logits(z, batch.task, batch.candidate)
                                   / model.temperature, -1)
        distribution = (torch.softmax(weights, -1)[..., None] * difficulty).sum(1)
        return nll, distribution

    aligned_nll, difficulty_aligned = paired_predictions(aligned)
    shuffled_nll, difficulty_shuffled = paired_predictions(shuffled)
    loss = association_aware_belief_loss(aligned.log_normalizers, future,
        aligned_nll, shuffled_nll, difficulty_aligned, difficulty_shuffled, eligible,
        future_weight=future_weight, association_weight=association_weight,
        invariance_weight=invariance_weight, margin=margin)
    loss.total.backward()
    optimizer.step()

    model.apbpf_training_state = dict(config, config_hash=fingerprint,
                                     eligible_mask=eligible.detach().cpu().tolist())
    metrics = {name: None if (value := getattr(loss, name)) is None else float(value.detach())
               for name in ("total", "fivo", "future", "prefix4_nll", "association", "invariance",
                            "association_gap", "eligible_gap")}
    metrics.update(aligned_nll=float(aligned_nll.detach().mean()),
                   shuffled_nll=float(shuffled_nll.detach().mean()),
                   eligible_count=int(eligible.sum()), population_count=len(eligible),
                   eligible_fraction=float(eligible.float().mean()))
    return metrics
