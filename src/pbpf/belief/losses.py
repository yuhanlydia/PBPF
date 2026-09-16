"""Proper Stage-B FIVO/future loss and source-disjoint scalar calibration."""

from dataclasses import dataclass
import math
import re
from collections.abc import Mapping, Collection

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class BeliefLoss:
    total: torch.Tensor
    fivo: torch.Tensor
    future: torch.Tensor
    prefix4_nll: torch.Tensor | None


def fivo_future_loss(log_normalizers: torch.Tensor,
                     future_nll: Mapping[int, torch.Tensor], *, future_weight: float = 1.) -> BeliefLoss:
    """Average candidates, sum evidence increments; average sampled prefixes.

    Future entries are per-candidate sums over tests strictly after the prefix.
    Prefix four is reported separately and is never silently substituted.
    """
    if log_normalizers.ndim != 2 or 0 in log_normalizers.shape or not torch.isfinite(log_normalizers).all():
        raise ValueError("log_normalizers must be finite non-empty [batch,steps]")
    if not math.isfinite(future_weight) or future_weight < 0:
        raise ValueError("future_weight must be finite and non-negative")
    if set(future_nll) - {1, 2, 4} or (future_weight > 0 and not future_nll):
        raise ValueError("future loss requires prefixes from {1,2,4}")
    for value in future_nll.values():
        if value.shape != log_normalizers.shape[:1] or not torch.isfinite(value).all():
            raise ValueError("future NLL must have finite shape [batch]")
    fivo = -log_normalizers.sum(-1).mean()
    future = torch.stack(list(future_nll.values())).mean() if future_nll else fivo.new_zeros(())
    metric = future_nll[4].mean() if 4 in future_nll else None
    return BeliefLoss(fivo + future_weight * future, fivo, future, metric)


@dataclass(frozen=True)
class AssociationAwareBeliefLoss(BeliefLoss):
    association: torch.Tensor
    invariance: torch.Tensor
    association_gap: torch.Tensor
    eligible_gap: torch.Tensor | None


def association_aware_belief_loss(
        log_normalizers, future_nll, aligned_nll, shuffled_nll,
        difficulty_aligned, difficulty_shuffled, eligible, *, future_weight=1.,
        association_weight=1., invariance_weight=1., margin=.03):
    """Aligned FIVO/future loss plus eligible-only association and symmetric KL.

    Auxiliary NLL vectors are per-candidate *per-future-test* values evaluated
    against the same untouched future labels. Difficulty inputs are categorical
    posterior-predictive probabilities [B,C], not particle-weight entropies.
    The primary gap always averages the full batch, including constant histories.
    """
    for name, value in (("association_weight", association_weight),
                        ("invariance_weight", invariance_weight), ("margin", margin)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    base = fivo_future_loss(log_normalizers, future_nll, future_weight=future_weight)
    shape = log_normalizers.shape[:1]
    if (aligned_nll.shape != shape or shuffled_nll.shape != shape
            or eligible.shape != shape or eligible.dtype != torch.bool):
        raise ValueError("auxiliary NLL vectors and boolean eligibility must have shape [batch]")
    tensors = (aligned_nll, shuffled_nll, difficulty_aligned, difficulty_shuffled, eligible)
    if any(value.device != log_normalizers.device for value in tensors):
        raise ValueError("loss tensors must share a device")
    if not all(torch.isfinite(value).all() for value in tensors[:-1]):
        raise ValueError("auxiliary values must be finite")
    if (difficulty_aligned.ndim != 2 or difficulty_aligned.shape[0] != shape[0]
            or difficulty_aligned.shape[1] < 2 or difficulty_shuffled.shape != difficulty_aligned.shape):
        raise ValueError("difficulty probabilities must have matching [batch,classes] shapes")
    for value in (difficulty_aligned, difficulty_shuffled):
        if (not value.is_floating_point() or (value < 0).any()
                or not torch.allclose(value.sum(-1), torch.ones_like(value[:, 0]), atol=1e-5)):
            raise ValueError("difficulty probabilities must be non-negative and normalized")
    gap = shuffled_nll - aligned_nll
    if eligible.any():
        association = F.relu(margin - gap[eligible]).mean()
        # Clamp only for logarithms so exact zeros do not create 0 * log(0).
        p, q = difficulty_aligned[eligible], difficulty_shuffled[eligible]
        epsilon = torch.finfo(p.dtype).tiny
        invariance = (.5 * (p - q) * (p.clamp_min(epsilon).log()
                                     - q.clamp_min(epsilon).log())).sum(-1).mean()
        eligible_gap = gap[eligible].mean()
    else:
        association = (aligned_nll.sum() + shuffled_nll.sum()) * 0.
        invariance = (difficulty_aligned.sum() + difficulty_shuffled.sum()) * 0.
        eligible_gap = None
    return AssociationAwareBeliefLoss(
        base.total + association_weight * association + invariance_weight * invariance,
        base.fivo, base.future, base.prefix4_nll, association, invariance,
        gap.mean(), eligible_gap)


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    split_hash: str
    training_split_hash: str

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        for fingerprint in (self.split_hash, self.training_split_hash):
            if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                raise ValueError("split hashes must be lowercase SHA-256 digests")
        if self.split_hash == self.training_split_hash:
            raise ValueError("calibration and training split hashes must be distinct")


def temperature_scale(logits: torch.Tensor, targets: torch.Tensor, *,
                      split_hash: str, training_split_hash: str,
                      calibration_source_ids: Collection[str], training_source_ids: Collection[str],
                      max_iter: int = 100) -> TemperatureCalibration:
    """Fit one positive temperature to raw five-class logits using unweighted NLL.

    Source IDs must identify the loader's leakage/source clusters, not individual
    rows. Hashes name immutable manifests; differing hashes alone cannot prove
    source disjointness, so explicit source sets are checked as well.
    """
    initial = TemperatureCalibration(1., split_hash, training_split_hash)
    if (not calibration_source_ids or not training_source_ids
            or any(not isinstance(source, str) or not source.strip()
                   for source in [*calibration_source_ids, *training_source_ids])
            or set(calibration_source_ids) & set(training_source_ids)):
        raise ValueError("calibration must be source-disjoint from training")
    if (logits.ndim != 2 or logits.shape[-1] != 5 or len(logits) == 0
            or not torch.isfinite(logits).all() or targets.shape != logits.shape[:1]
            or targets.dtype != torch.long or ((targets < 0) | (targets >= 5)).any()):
        raise ValueError("calibration needs finite [examples,5] logits and integer targets")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    values = logits.detach().double()
    labels = targets.detach().to(values.device)
    log_temperature = torch.zeros((), device=values.device, dtype=values.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(values / log_temperature.clamp(-8, 8).exp(), labels)
        loss.backward()
        return loss

    baseline = F.cross_entropy(values, labels)
    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-8, 8).exp())
    if F.cross_entropy(values / temperature, labels) > baseline:
        return initial
    return TemperatureCalibration(temperature, split_hash, training_split_hash)
