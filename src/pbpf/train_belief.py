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
