"""Immutable candidate-local particle state and SMC diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pbpf.particles import logsumexp


def _immutable_array(values, dtype) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    # A bytes-backed array cannot have WRITEABLE re-enabled by a caller.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _latents(values, name="z") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite, non-empty shape [particles, latent_dim]")
    return array


def _ancestors(values, count: int) -> np.ndarray:
    array = np.asarray(values)
    if (array.shape != (count,) or not np.issubdtype(array.dtype, np.integer)
            or np.any(array < 0) or np.any(array >= count)):
        raise ValueError("ancestors must be integer particle indices with shape [particles]")
    return array.astype(np.int64)


@dataclass(frozen=True, eq=False)
class ParticleSet:
    """One candidate's state; z has axes [P, d_z], never [G, P, d_z].

    Weights must already be normalized. Zero mass is represented by -inf;
    latents and cumulative log evidence must be finite. ``ancestors`` indexes
    the immediately preceding particle set (the selected parent at creation).
    """

    candidate_hash: str
    parent_hash: str | None
    z: np.ndarray
    log_weights: np.ndarray
    ancestors: np.ndarray
    step: int
    log_evidence: float

    def __post_init__(self):
        if not isinstance(self.candidate_hash, str) or not self.candidate_hash:
            raise ValueError("candidate_hash must be a non-empty string")
        if self.parent_hash is not None and (
            not isinstance(self.parent_hash, str) or not self.parent_hash
            or self.parent_hash == self.candidate_hash
        ):
            raise ValueError("parent_hash must be a distinct non-empty string or None")
        z = _latents(self.z)
        weights = np.asarray(self.log_weights, dtype=np.float64)
        if weights.shape != (len(z),):
            raise ValueError("log_weights must have shape [particles]")
        if not np.isclose(logsumexp(weights), 0.0, rtol=0.0, atol=1e-8):
            raise ValueError("log_weights must be normalized")
        ancestors = _ancestors(self.ancestors, len(z))
        if (not isinstance(self.step, (int, np.integer)) or isinstance(self.step, bool)
                or self.step < 0):
            raise ValueError("step must be a non-negative integer")
        if not np.isfinite(self.log_evidence):
            raise ValueError("log_evidence must be finite")
        object.__setattr__(self, "z", _immutable_array(z, np.float64))
        object.__setattr__(self, "log_weights", _immutable_array(weights, np.float64))
        object.__setattr__(self, "ancestors", _immutable_array(ancestors, np.int64))

    def __eq__(self, other):
        if not isinstance(other, ParticleSet):
            return NotImplemented
        return (self.candidate_hash == other.candidate_hash
                and self.parent_hash == other.parent_hash
                and self.step == other.step and self.log_evidence == other.log_evidence
                and np.array_equal(self.z, other.z)
                and np.array_equal(self.log_weights, other.log_weights)
                and np.array_equal(self.ancestors, other.ancestors))

    __hash__ = None


@dataclass(frozen=True)
class SMCStep:
    """Post-step particles plus diagnostics before any uniform-weight reset.

    ``log_normalizer`` is this step's incremental log evidence;
    ``particles.log_evidence`` is cumulative along the candidate lineage.
    ``resampling_indices`` maps output particles to pre-resampling samples.
    """

    particles: ParticleSet
    normalized_log_weights: np.ndarray
    log_normalizer: float
    ess: float
    resampled: bool
    resampling_indices: np.ndarray
    normalized_entropy: float
    unique_ancestor_count: int

    def __post_init__(self):
        object.__setattr__(self, "normalized_log_weights",
                           _immutable_array(self.normalized_log_weights, np.float64))
        object.__setattr__(self, "resampling_indices",
                           _immutable_array(self.resampling_indices, np.int64))
