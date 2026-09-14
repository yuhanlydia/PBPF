"""One posterior component per whole continuation, including exact serial gradients."""

from collections.abc import Callable

import torch


def _weights(log_weights):
    detached = log_weights.detach()
    if detached.dtype in (torch.float16, torch.bfloat16):
        detached = detached.float()
    if (log_weights.ndim != 2 or 0 in log_weights.shape
            or torch.isnan(log_weights).any() or torch.isposinf(log_weights).any()
            or not torch.isfinite(torch.logsumexp(detached, -1)).all()):
        raise ValueError("log weights must be [batch,components] with positive total mass")
    return detached - torch.logsumexp(detached, -1, keepdim=True)


def _sequence_sums(token_log_probs, token_mask=None):
    if (token_log_probs.ndim != 3 or 0 in token_log_probs.shape
            or torch.isnan(token_log_probs).any() or torch.isposinf(token_log_probs).any()
            or (token_log_probs > 0).any()):
        raise ValueError("token log probabilities must have non-empty shape [batch,components,tokens] and be <= 0")
    if token_mask is not None:
        if (token_mask.shape != (token_log_probs.shape[0], token_log_probs.shape[2])
                or not ((token_mask == 0) | (token_mask == 1)).all()):
            raise ValueError("token mask must have binary shape [batch,tokens]")
        token_log_probs = torch.where(token_mask[:, None].bool(), token_log_probs, 0.)
    # Long continuations overflow fp16 and lose component differences in bf16.
    # Cast before reduction, retaining the cast's backward path to input dtype.
    if token_log_probs.dtype in (torch.float16, torch.bfloat16):
        token_log_probs = token_log_probs.float()
    return token_log_probs.sum(-1)


def whole_sequence_mixture_loss(token_log_probs, log_weights, *, token_mask=None, reduction="mean"):
    """-log sum_m stopgrad(w_m) exp(sum_t log pi_m(token_t)).

    Inputs contain target-token log probabilities, not vocabulary logits. Mask
    out prompt, prefix and padding tokens. Impossible targets have infinite NLL.
    fp16/bfloat16 reductions and mixture arithmetic use float32; float64 stays
    float64. Gradients still flow back to the original token tensor dtype.
    """
    weights = _weights(log_weights)
    sequences = _sequence_sums(token_log_probs, token_mask)
    if sequences.shape != weights.shape:
        raise ValueError("component probability and weight shapes must match")
    loss = -torch.logsumexp(weights + sequences, -1)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be none, mean or sum")


def sample_components_once(log_weights, *, num_sequences=1, generator=None):
    """Systematic posterior draws [batch,repair_calls], NOT per-token draws.

    The generation caller selects one latent using each returned index before
    decoding and holds it for that entire continuation. Across calls systematic
    draws reduce allocation variance; a single call is categorical sampling.
    """
    # CDF and grid arithmetic must not round a half-precision position to 1.
    weights = _weights(log_weights.double())
    if not isinstance(num_sequences, int) or num_sequences <= 0:
        raise ValueError("num_sequences must be a positive integer")
    cdf = weights.exp().cumsum(-1)
    cdf[:, -1] = 1.
    offsets = torch.rand((len(weights), 1), device=weights.device, dtype=weights.dtype, generator=generator)
    positions = (offsets + torch.arange(num_sequences, device=weights.device)) / num_sequences
    return torch.searchsorted(cdf.contiguous(), positions.contiguous(), right=True).clamp_max(weights.shape[1] - 1)


def serial_mixture_backward(component_logprob_fn: Callable, latents, log_weights, *, parameters,
                            token_mask=None):
    """Error-atomic exact mean-mixture gradients with one live component graph.

    Callback ``(detached_z[B,D], component_index) -> target logprobs[B,T]``
    must recompute the actor/prefix graph each call, without parameter updates
    or mutable model-state/gradient changes. ``parameters`` must explicitly list
    unique trainable leaf tensors (e.g. ``projector.parameters()``). Gradients
    stage privately via autograd.grad; only after every replay succeeds are they
    added to existing .grad values. On replay/gradient-computation exceptions,
    existing gradients remain unchanged, including None. Unused parameters are
    left untouched. Step the optimizer only after success. Storage is one live
    component graph plus trainable-gradient buffers, never cloned model weights.

    Belief weights/latents are detached. Dropout RNG is replayed
    exactly (CPU and the latent's CUDA device); use a single-device actor or
    disable stochastic layers for model-parallel training.

    The returned scalar is the detached true mixture loss, not the weighted
    replay surrogate. fp16/bfloat16 likelihood and responsibility arithmetic
    accumulates in float32; float64 is preserved. Parameter gradients are
    ultimately stored in their parameter dtype, as required by Torch.
    """
    parameters = tuple(parameters)
    if (not parameters or len({id(parameter) for parameter in parameters}) != len(parameters)
            or any(not isinstance(parameter, torch.Tensor) or not parameter.is_leaf
                   or not parameter.requires_grad for parameter in parameters)):
        raise ValueError("parameters must be a non-empty sequence of unique trainable leaf tensors")
    weights = _weights(log_weights)
    if latents.ndim != 3 or latents.shape[:2] != weights.shape or not torch.isfinite(latents).all():
        raise ValueError("latents must have finite shape [batch,components,latent_dim]")
    z = latents.detach()
    devices = [z.device.index if z.device.index is not None else torch.cuda.current_device()] if z.is_cuda else []
    rng_states, values = [], []
    with torch.no_grad():
        for component in range(z.shape[1]):
            rng_states.append((torch.get_rng_state(), [torch.cuda.get_rng_state(d) for d in devices]))
            tokens = component_logprob_fn(z[:, component], component)
            values.append(_sequence_sums(tokens[:, None], token_mask).squeeze(1))
        sequences = torch.stack(values, 1)
        if sequences.shape != weights.shape:
            raise ValueError("callback batch shape does not match belief weights")
        log_mixture = torch.logsumexp(weights + sequences, -1)
        if not torch.isfinite(log_mixture).all():
            raise ValueError("serial backward requires a finite mixture likelihood")
        responsibilities = (weights + sequences - log_mixture[:, None]).exp()
    staged = [None] * len(parameters)
    for component, (cpu_state, cuda_states) in enumerate(rng_states):
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_state)
            for device, state in zip(devices, cuda_states):
                torch.cuda.set_rng_state(state, device)
            tokens = component_logprob_fn(z[:, component], component)
            sequence = _sequence_sums(tokens[:, None], token_mask).squeeze(1)
            if not torch.allclose(sequence.detach(), sequences[:, component], rtol=1e-5, atol=1e-7):
                raise ValueError("component replay changed; callback must preserve model state and RNG")
            rho = responsibilities[:, component]
            safe_sequence = torch.where(rho > 0, sequence, 0.)
            surrogate = -(rho * safe_sequence).mean()
            if surrogate.requires_grad:
                gradients = torch.autograd.grad(surrogate, parameters, allow_unused=True)
                for index, gradient in enumerate(gradients):
                    if gradient is not None:
                        if gradient.dtype in (torch.float16, torch.bfloat16):
                            gradient = gradient.float()
                        if staged[index] is None:
                            # Sum backward can return a stride-zero expanded view.
                            staged[index] = gradient.clone()
                        else:
                            staged[index].add_(gradient)
                del gradients
            del tokens, sequence, safe_sequence, surrogate
    # Build every final value before publishing any .grad, preserving additive
    # microbatch accumulation without touching caller-owned buffers on failure.
    updates = []
    for parameter, gradient in zip(parameters, staged):
        if gradient is not None:
            if parameter.grad is not None:
                gradient = gradient + parameter.grad
            updates.append((parameter, gradient.to(dtype=parameter.dtype)))
    for parameter, gradient in updates:
        parameter.grad = gradient
    return -log_mixture.mean()
