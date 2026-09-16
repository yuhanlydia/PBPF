"""Association objectives and source-cluster uncertainty summaries."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pbpf.statistics import BootstrapInterval, paired_cluster_bootstrap


def _is_torch_tensor(value: Any) -> bool:
    return type(value).__module__.split(".", maxsplit=1)[0] == "torch"


def association_margin_loss(
    aligned_nll: Any, shuffled_nll: Any, eligible: Any, margin: float
) -> Any:
    """Penalize eligible examples lacking a margin over counterfactual NLL.

    Torch tensors retain their autograd graph.  When no example is eligible,
    the Torch branch returns ``aligned_nll.sum() * 0`` so the exact zero remains
    differentiable; the NumPy branch returns the exact Python float ``0.0``.
    """
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    if _is_torch_tensor(aligned_nll):
        import torch

        if not (_is_torch_tensor(shuffled_nll) and _is_torch_tensor(eligible)):
            raise TypeError("aligned_nll, shuffled_nll, and eligible must all be Torch tensors")
        if aligned_nll.shape != shuffled_nll.shape or aligned_nll.shape != eligible.shape:
            raise ValueError("NLL values and eligibility mask must have identical shapes")
        if eligible.dtype != torch.bool:
            raise TypeError("eligible must be a boolean Torch tensor")
        if not bool(torch.isfinite(aligned_nll).all()) or not bool(torch.isfinite(shuffled_nll).all()):
            raise ValueError("NLL values must be finite")
        if not bool(eligible.any()):
            return aligned_nll.sum() * 0.0
        return torch.relu(margin + aligned_nll - shuffled_nll)[eligible].mean()

    aligned = np.asarray(aligned_nll, dtype=np.float64)
    shuffled = np.asarray(shuffled_nll, dtype=np.float64)
    mask = np.asarray(eligible)
    if aligned.shape != shuffled.shape or aligned.shape != mask.shape:
        raise ValueError("NLL values and eligibility mask must have identical shapes")
    if mask.dtype != bool:
        raise TypeError("eligible must be a boolean array")
    if not np.isfinite(aligned).all() or not np.isfinite(shuffled).all():
        raise ValueError("NLL values must be finite")
    if not mask.any():
        return 0.0
    return float(np.maximum(0.0, margin + aligned - shuffled)[mask].mean())


def association_gap(
    aligned_nll: Sequence[float],
    shuffled_nll: Sequence[float],
    clusters: Sequence[object],
    *,
    seed: int = 0,
    replicates: int = 10_000,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Estimate ``NLL(shuffled) - NLL(aligned)`` with clustered bootstrap CI."""
    aligned = np.asarray(aligned_nll, dtype=np.float64)
    shuffled = np.asarray(shuffled_nll, dtype=np.float64)
    cluster_values = np.asarray(clusters, dtype=object)
    if aligned.ndim != 1 or aligned.shape != shuffled.shape or len(aligned) != len(cluster_values):
        raise ValueError("NLL values and clusters must have equal vector shape")
    if not np.isfinite(aligned).all() or not np.isfinite(shuffled).all():
        raise ValueError("NLL values must be finite")
    return paired_cluster_bootstrap(
        shuffled,
        aligned,
        clusters=cluster_values,
        rng=np.random.default_rng(seed),
        replicates=replicates,
        confidence=confidence,
    )
