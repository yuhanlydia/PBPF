"""Small trainable probability model and piecewise-static Torch SMC.

This optional module imports Torch; the base ``pbpf.belief`` package does not.
No tokenizer, pretrained weights, arbitrary likelihood callbacks or ID features.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import BeliefBatch
from .losses import TemperatureCalibration
from .types import ParticleSet


@dataclass(frozen=True)
class GaussianParams:
    mean: torch.Tensor
    log_std: torch.Tensor

    def __post_init__(self):
        if (self.mean.shape != self.log_std.shape or self.mean.ndim < 1
                or self.mean.shape[-1] == 0 or not torch.isfinite(self.mean).all()
                or not torch.isfinite(self.log_std).all()):
            raise ValueError("Gaussian mean/log_std must have matching finite shapes")
        object.__setattr__(self, "log_std", self.log_std.clamp(-8., 4.))

    @property
    def std(self):
        return self.log_std.exp()

    def rsample(self, *, noise=None, generator=None):
        """Reparameterized actual draw; explicit noise supports deterministic replay."""
        if noise is None:
            noise = torch.randn(self.mean.shape, device=self.mean.device, dtype=self.mean.dtype, generator=generator)
        if noise.shape != self.mean.shape or not torch.isfinite(noise).all():
            raise ValueError("noise must match the Gaussian shape and be finite")
        return self.mean + self.std * noise

    def log_prob(self, value):
        mean, log_std = self.mean, self.log_std
        # [B,D] parameters can evaluate [B,P,D] actual particle draws.
        if value.ndim == mean.ndim + 1:
            mean, log_std = mean.unsqueeze(-2), log_std.unsqueeze(-2)
        if value.shape[-1] != mean.shape[-1]:
            raise ValueError("sample latent dimension does not match Gaussian")
        return (-.5 * ((value - mean) * (-log_std).exp()).square()
                - log_std - .5 * math.log(2 * math.pi)).sum(-1)


@dataclass(frozen=True)
class FilterTrace:
    """Differentiable [B,step,P,...] state; normalizers precede resampling.

    Training detaches discrete ancestry, not continuous reparameterized paths.
    Evaluation resample--move remains the responsibility of BeliefStore/MALA.
    """

    latents: torch.Tensor
    log_weights: torch.Tensor
    log_normalizers: torch.Tensor
    proposal_noise: torch.Tensor
    ancestors: torch.Tensor
    resampling_indices: torch.Tensor
    resampled: torch.Tensor


def _mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim))


class NeuralBeliefModel(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        if min(feature_dim, latent_dim, hidden_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        self.feature_dim, self.latent_dim = feature_dim, latent_dim
        self.root_head = _mlp(2 * feature_dim, hidden_dim, 2 * latent_dim)
        self.transition_head = _mlp(3 * feature_dim + latent_dim, hidden_dim, 2 * latent_dim)
        self.root_proposal_head = _mlp(3 * feature_dim + 5, hidden_dim, 2 * latent_dim)
        self.child_proposal_head = _mlp(4 * feature_dim + latent_dim + 5, hidden_dim, 2 * latent_dim)
        self.likelihood_head = _mlp(3 * feature_dim + latent_dim, hidden_dim, 5)
        self.register_buffer("temperature", torch.tensor(1.))
        self.calibration_split_hash = None
        self.training_split_hash = None

    @staticmethod
    def _gaussian(head, inputs):
        return GaussianParams(*head(inputs).chunk(2, -1))

    @staticmethod
    def _expand(features, z):
        return features[:, None].expand(-1, z.shape[1], -1) if z.ndim == 3 else features

    def root(self, task, candidate):
        return self._gaussian(self.root_head, torch.cat([task, candidate], -1))

    def transition(self, parent_z, task, candidate, diff):
        features = self._expand(torch.cat([task, candidate, diff], -1), parent_z)
        return self._gaussian(self.transition_head, torch.cat([features, parent_z], -1))

    def proposal(self, task, candidate, test, outcome, *, parent_z=None, diff=None):
        features = torch.cat([task, candidate, test, F.one_hot(outcome, 5).to(task.dtype)], -1)
        if parent_z is None:
            if diff is not None:
                raise ValueError("root proposal cannot receive a child diff")
            return self._gaussian(self.root_proposal_head, features)
        if diff is None:
            raise ValueError("child proposal requires semantic diff features")
        features = self._expand(torch.cat([features, diff], -1), parent_z)
        return self._gaussian(self.child_proposal_head, torch.cat([features, parent_z], -1))

    def likelihood(self, z, task, candidate, test, *, class_weights=None):
        """Calibrated log probabilities, ordered as registry.OUTCOMES."""
        if class_weights is not None:
            raise ValueError("class weights are forbidden for a proper likelihood")
        features = self._expand(torch.cat([task, candidate, test], -1), z)
        logits = self.likelihood_head(torch.cat([features, z], -1))
        return F.log_softmax(logits / self.temperature, dim=-1)

    def future_predict(self, task, candidate, tests, z, log_weights):
        if (tests.ndim != 3 or z.ndim != 3 or log_weights.shape != z.shape[:2]
                or tests.shape[0] != z.shape[0] or tests.shape[1] == 0):
            raise ValueError("future tensors require tests [B,T,F], latents [B,P,D], weights [B,P]")
        normalizer = torch.logsumexp(log_weights, -1, keepdim=True)
        if torch.isnan(log_weights).any() or torch.isposinf(log_weights).any() or not torch.isfinite(normalizer).all():
            raise ValueError("future weights must have finite positive total mass")
        weights = log_weights - normalizer
        return torch.stack([torch.logsumexp(weights[..., None] + self.likelihood(
            z, task, candidate, test), dim=1) for test in tests.unbind(1)], dim=1)

    def set_calibration(self, calibration: TemperatureCalibration):
        self.temperature.fill_(calibration.temperature)
        self.calibration_split_hash = calibration.split_hash
        self.training_split_hash = calibration.training_split_hash

    def get_extra_state(self):
        return {"calibration_split_hash": self.calibration_split_hash,
                "training_split_hash": self.training_split_hash}

    def set_extra_state(self, state):
        calibration_hash = state["calibration_split_hash"]
        training_hash = state["training_split_hash"]
        if calibration_hash is not None or training_hash is not None:
            TemperatureCalibration(float(self.temperature), calibration_hash, training_hash)
        elif not torch.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.calibration_split_hash, self.training_split_hash = calibration_hash, training_hash

    def filter(self, batch: BeliefBatch, *, particles=8, visible_steps=4, ess_fraction=.5,
               parents: list[ParticleSet] | None = None, ancestor_scheme=None,
               ancestors=None, log_ancestor_proposal=None, generator=None) -> FilterTrace:
        """Training SMC with actual Gaussian draws; no transitions within a candidate.

        Child ancestors are caller-selected, exactly as in BeliefStore. Sampled
        ancestry requires evaluated categorical proposal masses, including -log P.
        Systematic resampling decisions and indices are detached (no score term).
        """
        batch.validate(self.feature_dim)
        if not 1 <= visible_steps <= batch.tests.shape[1] or not 0 <= ess_fraction <= 1:
            raise ValueError("invalid visible_steps or ESS fraction")
        task, candidate = batch.task, batch.candidate
        size, device, dtype = len(task), task.device, task.dtype
        if parents is None:
            if ancestor_scheme is not None or ancestors is not None or log_ancestor_proposal is not None:
                raise ValueError("root filtering cannot receive ancestor options")
            if not isinstance(particles, int) or particles <= 0:
                raise ValueError("particles must be a positive integer")
            q0 = self.proposal(task, candidate, batch.tests[:, 0], batch.outcomes[:, 0])
            q = GaussianParams(q0.mean[:, None].expand(-1, particles, -1),
                               q0.log_std[:, None].expand(-1, particles, -1))
            prior = self.root(task, candidate)
            base = task.new_full((size, particles), -math.log(particles))
            selected = torch.arange(particles, device=device).expand(size, -1)
        else:
            if ancestor_scheme not in {"deterministic_enumeration", "sampled"}:
                raise ValueError("child filtering requires explicit ancestor_scheme")
            if len(parents) != size or batch.diff is None:
                raise ValueError("one parent snapshot per batch row and semantic diff are required")
            shape = parents[0].z.shape
            if shape[1] != self.latent_dim or any(p.z.shape != shape for p in parents):
                raise ValueError("parent particle shapes must match the model")
            particles = shape[0]
            if (ancestors is None or ancestors.shape != (size, particles) or ancestors.dtype != torch.long
                    or ((ancestors < 0) | (ancestors >= particles)).any()):
                raise ValueError("ancestors must be valid [batch,particles] integer indices")
            selected = ancestors.detach().to(device)
            parent_z = torch.tensor(np.stack([p.z.copy() for p in parents]), device=device, dtype=dtype)
            parent_z = parent_z.gather(1, selected[..., None].expand(-1, -1, self.latent_dim))
            weights = torch.tensor(np.stack([p.log_weights.copy() for p in parents]), device=device, dtype=dtype)
            base = weights.gather(1, selected)
            if ancestor_scheme == "deterministic_enumeration":
                if log_ancestor_proposal is not None or not torch.equal(
                        selected.sort(-1).values, torch.arange(particles, device=device).expand(size, -1)):
                    raise ValueError("deterministic_enumeration requires permutations and no ancestor proposal")
            else:
                if (log_ancestor_proposal is None or log_ancestor_proposal.shape != base.shape
                        or not torch.isfinite(log_ancestor_proposal).all() or (log_ancestor_proposal > 0).any()):
                    raise ValueError("sampled ancestors require finite log_ancestor_proposal <= 0")
                base = base - log_ancestor_proposal.detach().to(device=device, dtype=dtype) - math.log(particles)
            prior = self.transition(parent_z, task, candidate, batch.diff)
            q = self.proposal(task, candidate, batch.tests[:, 0], batch.outcomes[:, 0],
                              parent_z=parent_z, diff=batch.diff)
        noise = torch.randn(q.mean.shape, device=device, dtype=dtype, generator=generator)
        z = q.rsample(noise=noise)
        log_weights = base + prior.log_prob(z) - q.log_prob(z)
        all_z, all_w, all_normalizers, all_indices, all_resampled = [], [], [], [], []
        for j in range(visible_steps):
            log_probs = self.likelihood(z, task, candidate, batch.tests[:, j])
            likelihood = log_probs.gather(-1, batch.outcomes[:, j, None, None].expand(-1, particles, 1)).squeeze(-1)
            corrected = log_weights + likelihood
            normalizer = torch.logsumexp(corrected, -1)
            if not torch.isfinite(normalizer).all():
                raise ValueError("nonfinite SMC evidence")
            log_weights = corrected - normalizer[:, None]
            with torch.no_grad():
                resampled = log_weights.exp().square().sum(-1).reciprocal() < ess_fraction * particles
                indices = torch.arange(particles, device=device).expand(size, -1).clone()
                if resampled.any():
                    cdf = log_weights[resampled].double().softmax(-1).cumsum(-1)
                    cdf[:, -1] = 1.
                    offset = torch.rand((int(resampled.sum()), 1), device=device, dtype=cdf.dtype, generator=generator)
                    positions = (offset + torch.arange(particles, device=device, dtype=cdf.dtype)) / particles
                    indices[resampled] = torch.searchsorted(cdf.contiguous(), positions.contiguous(), right=True).clamp_max(particles - 1)
            z = z.gather(1, indices[..., None].expand(-1, -1, self.latent_dim))
            log_weights = torch.where(resampled[:, None], -math.log(particles), log_weights)
            all_z.append(z)
            all_w.append(log_weights)
            all_normalizers.append(normalizer)
            all_indices.append(indices)
            all_resampled.append(resampled)
        return FilterTrace(torch.stack(all_z, 1), torch.stack(all_w, 1), torch.stack(all_normalizers, 1),
                           noise, selected, torch.stack(all_indices, 1), torch.stack(all_resampled, 1))

    def future_nll(self, batch: BeliefBatch, trace: FilterTrace, *, prefixes=(1, 2, 4)):
        result = {}
        for prefix in prefixes:
            if prefix not in (1, 2, 4) or prefix > trace.latents.shape[1] or prefix >= batch.tests.shape[1]:
                raise ValueError("future prefix must be 1,2,4 with filtered state and later targets")
            prediction = self.future_predict(batch.task, batch.candidate, batch.tests[:, prefix:],
                trace.latents[:, prefix - 1], trace.log_weights[:, prefix - 1])
            result[prefix] = -prediction.gather(-1, batch.outcomes[:, prefix:, None]).squeeze(-1).sum(-1)
        return result
