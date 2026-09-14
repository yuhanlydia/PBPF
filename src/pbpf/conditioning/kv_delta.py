"""Optional learned low-rank K/V ablations; never loaded by the primary prefix."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TorchKVDelta:
    """Per-example [B,layers,KV-heads,head_dim,head_dim] linear cache deltas."""

    key: torch.Tensor
    value: torch.Tensor

    def apply_to_cache(self, past_key_values):
        """Non-mutating legacy tuple-cache adapter, including GQA head checks."""
        if not isinstance(past_key_values, (tuple, list)):
            raise TypeError("KV ablation requires an explicit legacy tuple/list cache")
        if len(past_key_values) != self.key.shape[1]:
            raise ValueError("cache layer shape mismatch")
        batch, _, heads, dim, _ = self.key.shape
        result = []
        for layer, cache in enumerate(past_key_values):
            if len(cache) < 2:
                raise ValueError("each cache layer requires key and value")
            k, v, *extras = cache
            if (k.ndim != 4 or k.shape != v.shape or k.shape[0] != batch
                    or k.shape[1] != heads or k.shape[-1] != dim):
                raise ValueError("cache shape must match batch, KV heads and head_dim")
            key = self.key[:, layer].to(device=k.device, dtype=k.dtype)
            value = self.value[:, layer].to(device=v.device, dtype=v.dtype)
            result.append((k + torch.einsum("bhqd,bhde->bhqe", k, key),
                           v + torch.einsum("bhqd,bhde->bhqe", v, value), *extras))
        return tuple(result)


class LowRankKVProjector(nn.Module):
    """Latent-conditioned left/right factors; each cache transform has rank <= r."""

    def __init__(self, latent_dim: int, layers: int, heads: int, head_dim: int, rank: int,
                 mode: str, hidden_dim: int = 128):
        super().__init__()
        if min(latent_dim, layers, heads, head_dim, rank, hidden_dim) <= 0 or rank > head_dim:
            raise ValueError("positive dimensions and rank <= head_dim required")
        if mode not in {"k_only", "v_only", "kv"}:
            raise ValueError("mode must be k_only, v_only or kv")
        self.latent_dim, self.layers, self.heads = latent_dim, layers, heads
        self.head_dim, self.rank, self.mode = head_dim, rank, mode
        channels = 2 if mode == "kv" else 1
        self.network = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, channels * layers * heads * 2 * head_dim * rank))

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.latent_dim or not torch.isfinite(z).all():
            raise ValueError("latent must have finite shape [batch,latent_dim]")
        channels = 2 if self.mode == "kv" else 1
        factors = self.network(z.detach()).reshape(
            len(z), channels, self.layers, self.heads, 2, self.head_dim, self.rank)
        matrix = factors[..., 0, :, :] @ factors[..., 1, :, :].transpose(-1, -2)
        zero = torch.zeros_like(matrix[:, 0])
        key = zero if self.mode == "v_only" else matrix[:, 0]
        value = zero if self.mode == "k_only" else matrix[:, -1]
        return TorchKVDelta(key, value)
