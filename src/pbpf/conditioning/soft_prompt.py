"""Primary frozen-actor conditioning: eight learned input-prefix embeddings."""

import math

import torch
from torch import nn


class SoftPrefixProjector(nn.Module):
    def __init__(self, latent_dim: int, d_model: int, hidden_dim: int = 128, prefix_length: int = 8,
                 maximum_token_rms: float | None = None, maximum_delta_rms: float | None = None):
        super().__init__()
        if min(latent_dim, d_model, hidden_dim, prefix_length) <= 0:
            raise ValueError("prefix dimensions must be positive")
        if maximum_token_rms is not None and (not math.isfinite(maximum_token_rms) or maximum_token_rms <= 0):
            raise ValueError("maximum token RMS must be positive and finite")
        if maximum_delta_rms is not None and (not math.isfinite(maximum_delta_rms) or maximum_delta_rms <= 0):
            raise ValueError("maximum delta RMS must be positive and finite")
        self.latent_dim, self.d_model, self.prefix_length = latent_dim, d_model, prefix_length
        self.maximum_token_rms = maximum_token_rms
        self.maximum_delta_rms = maximum_delta_rms
        self.network = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, prefix_length * d_model))

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.latent_dim or not torch.isfinite(z).all():
            raise ValueError("latent must have finite shape [batch,latent_dim]")
        detached = z.detach()
        value = self.network(detached).reshape(z.shape[0], self.prefix_length, self.d_model)
        if self.maximum_delta_rms is not None:
            null = self.network(torch.zeros_like(detached)).reshape_as(value)
            delta = value - null
            rms = (delta.float().square().mean(-1, keepdim=True) + 1e-12).sqrt()
            delta = delta * (self.maximum_delta_rms / rms).clamp(max=1).to(delta.dtype)
            value = null + delta
        if self.maximum_token_rms is not None:
            rms = (value.float().square().mean(-1, keepdim=True) + 1e-12).sqrt()
            scale = (self.maximum_token_rms / rms).clamp(max=1).to(value.dtype)
            value = value * scale
        return value

    def prepend(self, z, *, inputs_embeds, attention_mask, labels=None):
        """Return actor kwargs; prefix labels are ignored by causal-LM NLL.

        Attention is one on every prefix position, original padding is retained.
        The caller freezes actor parameters (but must not use no_grad around the
        actor forward, because gradients must reach these prefix embeddings).
        """
        if (inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != z.shape[0]
                or inputs_embeds.shape[-1] != self.d_model
                or attention_mask.shape != inputs_embeds.shape[:2]
                or (labels is not None and labels.shape != attention_mask.shape)):
            raise ValueError("embedding, attention mask and label shape mismatch")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("attention mask must be binary")
        prefix = self(z).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        result = {
            "inputs_embeds": torch.cat([prefix, inputs_embeds], 1),
            "attention_mask": torch.cat([attention_mask.new_ones((len(z), self.prefix_length)), attention_mask], 1),
        }
        if labels is not None:
            result["labels"] = torch.cat([labels.new_full((len(z), self.prefix_length), -100), labels], 1)
        return result
