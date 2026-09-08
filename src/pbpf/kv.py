from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KVDelta:
    key: np.ndarray
    value: np.ndarray
    active_layer_indices: tuple[int, ...]
    rank: int
    mode: str

    @property
    def physical_bytes(self) -> int:
        return self.key.nbytes + self.value.nbytes

    def as_torch(self, *, device: str | None = None, dtype: object | None = None):
        import torch

        key = torch.as_tensor(self.key, device=device, dtype=dtype)
        value = torch.as_tensor(self.value, device=device, dtype=dtype)
        return key, value

    def apply_to_cache(self, past_key_values):
        if hasattr(past_key_values, "to_legacy_cache"):
            factory = getattr(type(past_key_values), "from_legacy_cache", None)
            if not callable(factory):
                raise TypeError(
                    "DynamicCache compatibility requires from_legacy_cache support"
                )
            legacy = past_key_values.to_legacy_cache()
            return factory(self.apply_to_cache(legacy))
        if not isinstance(past_key_values, (tuple, list)):
            raise TypeError(
                "KV cache must be a legacy tuple/list or DynamicCache-compatible object"
            )
        if len(past_key_values) != self.key.shape[0]:
            raise ValueError("KV delta layer count must match cache layer count")
        transformed = []
        for layer_index, layer_cache in enumerate(past_key_values):
            if len(layer_cache) < 2:
                raise ValueError("each cache layer requires key and value tensors")
            key_cache, value_cache, *extras = layer_cache
            key_shape = tuple(key_cache.shape)
            value_shape = tuple(value_cache.shape)
            expected_heads = self.key.shape[1]
            expected_dim = self.key.shape[-1]
            if (
                len(key_shape) != 4
                or len(value_shape) != 4
                or key_shape[-3] != expected_heads
                or value_shape[-3] != expected_heads
            ):
                raise ValueError(
                    "KV cache compatibility failure: delta heads must equal "
                    "model config num_key_value_heads for GQA"
                )
            if key_shape[-1] != expected_dim or value_shape[-1] != expected_dim:
                raise ValueError(
                    "KV cache compatibility failure: delta head_dim must equal cache head_dim"
                )
            if type(key_cache).__module__.split(".", 1)[0] == "torch":
                import torch

                key_delta = torch.as_tensor(
                    self.key[layer_index], device=key_cache.device, dtype=key_cache.dtype
                )
                value_delta = torch.as_tensor(
                    self.value[layer_index], device=value_cache.device, dtype=value_cache.dtype
                )
                new_key = key_cache + torch.einsum("bhqd,hde->bhqe", key_cache, key_delta)
                new_value = value_cache + torch.einsum(
                    "bhqd,hde->bhqe", value_cache, value_delta
                )
            else:
                key_array = np.asarray(key_cache)
                value_array = np.asarray(value_cache)
                new_key = key_array + np.einsum(
                    "bhqd,hde->bhqe", key_array, self.key[layer_index]
                )
                new_value = value_array + np.einsum(
                    "bhqd,hde->bhqe", value_array, self.value[layer_index]
                )
            transformed.append((new_key, new_value, *extras))
        return tuple(transformed)


def _layer_indices(layers: int, active_layers: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(active_layers, int):
        if active_layers < 0 or active_layers > layers:
            raise ValueError("active_layers count outside layer range")
        return tuple(range(layers - active_layers, layers))
    indices = tuple(int(index) for index in active_layers)
    if len(set(indices)) != len(indices) or any(index < 0 or index >= layers for index in indices):
        raise ValueError("active layer indices must be unique and in range")
    return indices


def build_kv_delta(
    latent: np.ndarray | Sequence[float],
    *,
    layers: int,
    heads: int,
    head_dim: int,
    rank: int,
    mode: str,
    active_layers: int | Sequence[int],
    target_norm: float | None,
    rng: np.random.Generator,
) -> KVDelta:
    if min(layers, heads, head_dim, rank) <= 0 or rank > head_dim:
        raise ValueError("positive dimensions and rank <= head_dim are required")
    if mode not in {"k_only", "v_only", "kv", "shared", "random", "noop"}:
        raise ValueError(f"unknown KV mode: {mode}")
    if target_norm is not None and target_norm < 0:
        raise ValueError("target_norm cannot be negative")
    latent_values = np.asarray(latent, dtype=np.float64)
    if latent_values.ndim != 1 or latent_values.size == 0 or not np.isfinite(latent_values).all():
        raise ValueError("latent must be a finite non-empty vector")
    indices = _layer_indices(layers, active_layers)
    shape = (layers, heads, head_dim, head_dim)
    key = np.zeros(shape, dtype=np.float64)
    value = np.zeros(shape, dtype=np.float64)
    coefficients = np.concatenate(([1.0], np.tanh(latent_values)))

    shared_key: np.ndarray | None = None
    shared_value: np.ndarray | None = None
    for layer in indices:
        for head in range(heads):
            if mode == "shared" and shared_key is not None and shared_value is not None:
                key[layer, head] = shared_key
                value[layer, head] = shared_value
                continue
            if mode == "random":
                left = rng.normal(size=(head_dim, rank))
                right = rng.normal(size=(rank, head_dim))
                left_v = rng.normal(size=(head_dim, rank))
                right_v = rng.normal(size=(rank, head_dim))
            else:
                left = np.tensordot(
                    coefficients,
                    rng.normal(size=(len(coefficients), head_dim, rank)),
                    axes=(0, 0),
                )
                right = np.tensordot(
                    coefficients,
                    rng.normal(size=(len(coefficients), rank, head_dim)),
                    axes=(0, 0),
                )
                left_v = np.tensordot(
                    coefficients,
                    rng.normal(size=(len(coefficients), head_dim, rank)),
                    axes=(0, 0),
                )
                right_v = np.tensordot(
                    coefficients,
                    rng.normal(size=(len(coefficients), rank, head_dim)),
                    axes=(0, 0),
                )
            key_matrix = left @ right
            value_matrix = left_v @ right_v
            if mode not in {"v_only", "noop"}:
                key[layer, head] = key_matrix
            if mode not in {"k_only", "noop"}:
                value[layer, head] = value_matrix
            if mode == "shared" and shared_key is None:
                shared_key, shared_value = key[layer, head].copy(), value[layer, head].copy()

    norm = float(np.sqrt(np.square(key).sum() + np.square(value).sum()))
    if target_norm is not None:
        if target_norm > 0 and norm == 0 and mode != "noop":
            raise ValueError("cannot normalize an all-zero KV delta")
        if norm > 0:
            multiplier = target_norm / norm
            key *= multiplier
            value *= multiplier
    return KVDelta(key, value, indices, rank, mode)


def make_trainable_kv_projector(
    latent_dim: int,
    *,
    layers: int,
    heads: int,
    head_dim: int,
    rank: int,
    active_layers: int | Sequence[int],
    mode: str = "kv",
    target_norm: float | None = None,
):
    """Create a lazy torch module whose low-rank KV directions depend on latent state."""
    import torch

    if min(latent_dim, layers, heads, head_dim, rank) <= 0 or rank > head_dim:
        raise ValueError("valid positive projector dimensions are required")
    indices = _layer_indices(layers, active_layers)
    if mode not in {"k_only", "v_only", "kv"}:
        raise ValueError("trainable projector mode must be k_only, v_only, or kv")

    class TrainableKVProjector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            factor_size = layers * heads * (head_dim * rank + rank * head_dim)
            self.key_factors = torch.nn.Linear(latent_dim, factor_size)
            self.value_factors = torch.nn.Linear(latent_dim, factor_size)

        def _matrices(self, factors):
            batch = factors.shape[0]
            per_matrix = head_dim * rank + rank * head_dim
            shaped = factors.reshape(batch, layers, heads, per_matrix)
            split = head_dim * rank
            left = shaped[..., :split].reshape(batch, layers, heads, head_dim, rank)
            right = shaped[..., split:].reshape(batch, layers, heads, rank, head_dim)
            return torch.matmul(left, right)

        def forward(self, latent):
            if latent.ndim == 1:
                latent = latent.unsqueeze(0)
            key = self._matrices(self.key_factors(latent))
            value = self._matrices(self.value_factors(latent))
            mask = torch.zeros(layers, device=latent.device, dtype=latent.dtype)
            mask[list(indices)] = 1
            mask = mask.reshape(1, layers, 1, 1, 1)
            key = key * mask if mode != "v_only" else torch.zeros_like(key)
            value = value * mask if mode != "k_only" else torch.zeros_like(value)
            if target_norm is not None:
                norm = torch.sqrt(
                    key.square().sum(dim=(1, 2, 3, 4), keepdim=True)
                    + value.square().sum(dim=(1, 2, 3, 4), keepdim=True)
                ).clamp_min(torch.finfo(latent.dtype).eps)
                key = key * (target_norm / norm)
                value = value * (target_norm / norm)
            return key, value

    return TrainableKVProjector()
