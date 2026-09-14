"""Candidate-keyed, piecewise-static sequential Monte Carlo.

Only candidate creation accepts importance corrections. Subsequent tests of
unchanged code update likelihoods, with no transition or proposal density.
Callers evaluate Gaussian densities at the supplied *actual draws*; this layer
does not substitute proposal means or draw a different set of latents.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pbpf.particles import (
    effective_sample_size, logsumexp, normalize_log_weights, systematic_resample,
)

from .mala import MALAResult, mala_resample_move
from .types import ParticleSet, SMCStep, _ancestors, _latents


def _vector(values, count: int, name: str, *, gaussian=False) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required")
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape [particles]")
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError(f"{name} contains invalid values")
    if gaussian and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite Gaussian log densities")
    return array


class BeliefStore:
    """Independent immutable states keyed by exact candidate hash.

    A store may hold any number G of candidates; each owns its own P particles.
    Child creation preserves its parent's P and latent dimension. Explicit
    ``ancestor_scheme`` is required: deterministic enumeration visits each
    parent once, while sampled ancestors require their categorical proposal
    log probabilities. Both schemes add the child transition/likelihood/proposal
    correction to their ancestor-specific base mass without renormalizing it.

    Resampling is systematic. Evaluation calls ``mala_move`` immediately after
    a resampled step, supplying the model's full target and gradient.
    """

    def __init__(self, *, rng: np.random.Generator, ess_fraction: float = 0.5):
        if not 0.0 <= ess_fraction <= 1.0:
            raise ValueError("ess_fraction must be in [0, 1]")
        self.rng = rng
        self.ess_fraction = ess_fraction
        self._states: dict[str, ParticleSet] = {}
        self._pending_moves: set[str] = set()

    def _new_hash(self, candidate_hash: str):
        if not isinstance(candidate_hash, str) or not candidate_hash:
            raise ValueError("candidate_hash must be a non-empty string")
        if candidate_hash in self._states:
            raise ValueError(f"candidate_hash {candidate_hash!r} already exists")

    def snapshot(self, candidate_hash: str) -> ParticleSet:
        """Return a deep, read-only copy; never expose mutable store state."""
        return replace(self._states[candidate_hash])

    def _finish(self, *, candidate_hash, parent_hash, z, corrected, ancestors,
                step, previous_evidence, resample=True) -> SMCStep:
        log_normalizer = logsumexp(corrected)
        normalized = normalize_log_weights(corrected)
        ess = effective_sample_size(normalized)
        count = len(z)
        positive = np.isfinite(normalized)
        entropy = -float(np.sum(np.exp(normalized[positive]) * normalized[positive]))
        normalized_entropy = entropy / np.log(count) if count > 1 else 0.0
        resampled = bool(resample and ess < self.ess_fraction * count)
        indices = (systematic_resample(np.exp(normalized), rng=self.rng) if resampled
                   else np.arange(count, dtype=np.int64))
        particles = ParticleSet(
            candidate_hash, parent_hash, z[indices],
            np.full(count, -np.log(count)) if resampled else normalized,
            ancestors[indices], step, previous_evidence + log_normalizer,
        )
        result = SMCStep(
            # Read-only data does not protect NumPy shape/dtype headers.
            particles=replace(particles), normalized_log_weights=normalized,
            log_normalizer=log_normalizer, ess=ess, resampled=resampled,
            resampling_indices=indices, normalized_entropy=normalized_entropy,
            unique_ancestor_count=len(np.unique(particles.ancestors)),
        )
        self._states[candidate_hash] = particles
        if resampled:
            self._pending_moves.add(candidate_hash)
        else:
            self._pending_moves.discard(candidate_hash)
        return result

    def initialize_root(
        self, candidate_hash: str, z, log_weights=None, *,
        log_prior=None, log_proposal=None, log_likelihood=None,
    ) -> SMCStep:
        """Initialize explicit samples, optionally correcting a root proposal.

        With no Gaussian correction, ``log_weights`` is normalized as an
        initial distribution (uniform by default). With a proposal, supply
        both Gaussian log densities and no initial weights: root weights are
        ``-log(P) + log_prior + first_likelihood - log_proposal``.
        Initial weighting alone contributes no evidence and triggers no move.
        """
        self._new_hash(candidate_hash)
        z = _latents(z)
        count = len(z)
        correction = log_prior is not None or log_proposal is not None
        if correction:
            if log_weights is not None:
                raise ValueError("log_weights cannot be combined with root proposal correction")
            prior = _vector(log_prior, count, "log_prior", gaussian=True)
            proposal = _vector(log_proposal, count, "log_proposal", gaussian=True)
            corrected = np.full(count, -np.log(count)) + prior - proposal
        else:
            corrected = (np.full(count, -np.log(count)) if log_weights is None
                         else normalize_log_weights(_vector(log_weights, count, "log_weights")))
        if log_likelihood is not None:
            corrected = corrected + _vector(log_likelihood, count, "log_likelihood")
        return self._finish(
            candidate_hash=candidate_hash, parent_hash=None, z=z, corrected=corrected,
            ancestors=np.arange(count), step=int(log_likelihood is not None),
            previous_evidence=0.0, resample=correction or log_likelihood is not None,
        )

    def observe(self, candidate_hash: str, log_likelihood, *,
                log_transition=None, log_proposal=None) -> SMCStep:
        """Accumulate only the next likelihood for unchanged candidate code."""
        if log_transition is not None or log_proposal is not None:
            raise ValueError("transition/proposal terms are only legal at candidate creation")
        previous = self._states[candidate_hash]
        likelihood = _vector(log_likelihood, len(previous.z), "log_likelihood")
        return self._finish(
            candidate_hash=candidate_hash, parent_hash=previous.parent_hash,
            z=previous.z, corrected=previous.log_weights + likelihood,
            ancestors=np.arange(len(previous.z)), step=previous.step + 1,
            previous_evidence=previous.log_evidence,
        )

    def spawn_child(
        self, candidate_hash: str, parent_hash: str, *, ancestor_scheme: str, sampled_z=None,
        ancestors=None, log_transition=None, log_proposal=None, log_likelihood=None,
        log_ancestor_proposal=None,
    ) -> SMCStep:
        """Create a child with explicit ancestor and latent proposal corrections.

        ``deterministic_enumeration`` requires a permutation of parent indices,
        forbids ``log_ancestor_proposal``, and uses base ``parent.log_weights[a]``.
        ``sampled`` allows repeats and requires each draw's finite categorical
        ``log_ancestor_proposal=log rho(a)``; its base is
        ``-log(P) + parent.log_weights[a] - log rho(a)``. These evaluated values
        need not sum to one across draws: repeated ancestors repeat their rho.
        The categorical proposal must cover the parent target's support.

        Both schemes then add ``log_transition + log_likelihood - log_proposal``
        evaluated at the actual sampled child latents. No base normalization is
        performed before the evidence increment is recorded.
        """
        self._new_hash(candidate_hash)
        if ancestor_scheme not in ("deterministic_enumeration", "sampled"):
            raise ValueError("ancestor_scheme must be deterministic_enumeration or sampled")
        parent = self._states[parent_hash]
        if sampled_z is None:
            raise ValueError("sampled_z is required")
        z = _latents(sampled_z, "sampled_z")
        if z.shape != parent.z.shape:
            raise ValueError("sampled_z must match the parent's [particles, latent_dim] shape")
        count = len(z)
        selected = _ancestors(ancestors, count)
        base = parent.log_weights[selected]
        if ancestor_scheme == "deterministic_enumeration":
            if len(np.unique(selected)) != count:
                raise ValueError("deterministic_enumeration ancestors must be a permutation")
            if log_ancestor_proposal is not None:
                raise ValueError("deterministic_enumeration prohibits log_ancestor_proposal")
        else:
            ancestor_proposal = _vector(log_ancestor_proposal, count, "log_ancestor_proposal")
            if not np.isfinite(ancestor_proposal).all() or np.any(ancestor_proposal > 0):
                raise ValueError("log_ancestor_proposal must contain finite log probabilities <= 0")
            base = -np.log(count) + base - ancestor_proposal
        transition = _vector(log_transition, count, "log_transition", gaussian=True)
        proposal = _vector(log_proposal, count, "log_proposal", gaussian=True)
        likelihood = _vector(log_likelihood, count, "log_likelihood")
        return self._finish(
            candidate_hash=candidate_hash, parent_hash=parent_hash, z=z,
            corrected=base + transition + likelihood - proposal,
            ancestors=selected, step=1, previous_evidence=parent.log_evidence,
        )

    def mala_move(self, candidate_hash: str, *, log_density, gradient,
                  step_size: float) -> MALAResult:
        """Persist one corrected post-resampling move without changing weights.

        The target/gradient must include the candidate's entire current target.
        This candidate-keyed operation preserves lineage, observation step,
        evidence, and ancestry; only latent positions change. A second move or
        a move based on an older observation is rejected.
        """
        previous = self._states[candidate_hash]
        if candidate_hash not in self._pending_moves:
            raise ValueError("MALA move is only legal immediately after resampling")
        result = mala_resample_move(
            previous.z, log_density=log_density, gradient=gradient,
            step_size=step_size, rng=self.rng,
        )
        self._states[candidate_hash] = replace(previous, z=result.z)
        self._pending_moves.remove(candidate_hash)
        return result
