"""Primary frozen-actor conditioning: eight learned input-prefix embeddings."""

import torch
from torch import nn


class SoftPrefixProjector(nn.Module):
    def __init__(self, latent_dim: int, d_model: int, hidden_dim: int = 128, prefix_length: int = 8):
        super().__init__()
        if min(latent_dim, d_model, hidden_dim, prefix_length) <= 0:
            raise ValueError("prefix dimensions must be positive")
        self.latent_dim, self.d_model, self.prefix_length = latent_dim, d_model, prefix_length
        self.network = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, prefix_length * d_model))

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.latent_dim or not torch.isfinite(z).all():
            raise ValueError("latent must have finite shape [batch,latent_dim]")
        return self.network(z.detach()).reshape(z.shape[0], self.prefix_length, self.d_model)

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
