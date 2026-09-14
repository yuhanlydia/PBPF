"""One Metropolis-adjusted Langevin move for already-resampled latents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .types import _immutable_array, _latents


@dataclass(frozen=True)
class MALAResult:
    z: np.ndarray
    accepted: np.ndarray
    acceptance_rate: float

    def __post_init__(self):
        object.__setattr__(self, "z", _immutable_array(self.z, np.float64))
        object.__setattr__(self, "accepted", _immutable_array(self.accepted, np.bool_))


def mala_resample_move(
    z, *, log_density: Callable, gradient: Callable, step_size: float,
    rng: np.random.Generator,
) -> MALAResult:
    """Apply one corrected move to each row of [P, d_z] latent samples.

    The caller first resamples its weighted particle set. This function does
    not resample again. ``step_size`` is the proposal standard deviation h:
    q(y|x) = N(x + h² grad(log target(x))/2, h² I).
    ``log_density`` returns [P], and ``gradient`` returns [P, d_z]. They must
    represent the same full target, including all accumulated observations.
    The normalizing constant of the target is unnecessary. A -inf proposal
    density is outside target support and is rejected. Gradient callbacks keep
    their full [P, d_z] row identity; unsupported proposals use current points
    as safe placeholders. Non-finite current targets or invalid callbacks fail
    explicitly.
    """
    if not callable(log_density) or not callable(gradient):
        raise ValueError("log_density and gradient must be callable")
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    current = _immutable_array(_latents(z), np.float64)

    def density(points, *, allow_zero_mass=False):
        values = np.asarray(log_density(points), dtype=np.float64)
        if (values.shape != (len(points),) or np.isnan(values).any()
                or np.isposinf(values).any()
                or (not allow_zero_mass and np.isneginf(values).any())):
            raise ValueError("log_density must return valid target log densities with shape [particles]")
        return values

    def grad(points):
        values = np.asarray(gradient(points), dtype=np.float64)
        if values.shape != points.shape or not np.isfinite(values).all():
            raise ValueError("gradient must return finite values with shape [particles, latent_dim]")
        return values

    current_density = density(current)
    forward_mean = current + 0.5 * step_size**2 * grad(current)
    proposed = forward_mean + step_size * rng.normal(size=current.shape)
    if not np.isfinite(proposed).all():
        raise ValueError("MALA proposal is non-finite; reduce step_size")
    proposed_density = density(proposed, allow_zero_mass=True)
    supported = np.isfinite(proposed_density)
    log_ratio = np.full(len(current), -np.inf)
    if supported.any():
        safe_proposed = np.where(supported[:, None], proposed, current)
        reverse_mean = proposed[supported] + 0.5 * step_size**2 * grad(safe_proposed)[supported]
        # Equal Gaussian covariance normalizers cancel in log q(x|y)-log q(y|x).
        forward_log_q = -0.5 * np.square(
            (proposed[supported] - forward_mean[supported]) / step_size
        ).sum(axis=1)
        reverse_log_q = -0.5 * np.square(
            (current[supported] - reverse_mean) / step_size
        ).sum(axis=1)
        log_ratio[supported] = (proposed_density[supported] - current_density[supported]
                                + reverse_log_q - forward_log_q)
        if np.isnan(log_ratio[supported]).any():
            raise ValueError("MALA acceptance ratio is undefined")
    with np.errstate(divide="ignore"):
        log_uniform = np.log(rng.random(len(current)))
    accepted = log_uniform < np.minimum(0.0, log_ratio)
    moved = np.where(accepted[:, None], proposed, current)
    return MALAResult(moved, accepted, float(accepted.mean()))
