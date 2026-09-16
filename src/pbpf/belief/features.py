"""Semantic tensor inputs; identifiers are metadata, never numeric features."""

from dataclasses import dataclass, replace

import torch
from torch import nn


class SemanticEncoder(nn.Module):
    """Masked mean pooling of caller-provided code/actor hidden states."""

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
            raise ValueError("expected hidden [batch,tokens,features] and matching mask")
        if not torch.isfinite(hidden).all() or not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("hidden states must be finite and masks binary")
        count = attention_mask.sum(-1, keepdim=True)
        if (count <= 0).any():
            raise ValueError("each semantic sequence must be non-empty")
        return (hidden * attention_mask[..., None]).sum(1) / count


@dataclass
class BeliefBatch:
    """One candidate per row; ordered visible+future tests, no ID embeddings.

    task/candidate/diff: [B,F]; tests: [B,T,F]; outcomes: [B,T].
    Stage-B training may use future outcomes; generator-side callers must only
    supply permitted visible feedback. Firewall enforcement belongs to loaders.
    """

    task: torch.Tensor
    candidate: torch.Tensor
    tests: torch.Tensor
    outcomes: torch.Tensor
    diff: torch.Tensor | None = None

    def outcome_counterfactual(self, *, visible_steps: int, seed: int):
        """Shuffle visible labels only; preserve semantics and all future targets.

        Constant visible histories are retained for base loss and reporting, but
        marked ineligible for the auxiliary association/invariance objectives.
        """
        from pbpf.apbpf.counterfactual import association_eligibility, outcome_derangement

        self.validate(self.task.shape[-1])
        outcomes = self.outcomes.detach().cpu().numpy()
        shuffled = outcome_derangement(outcomes, visible_steps, seed)
        eligible = association_eligibility(outcomes, visible_steps)
        return (replace(self, outcomes=torch.as_tensor(shuffled, device=self.outcomes.device)),
                torch.as_tensor(eligible, device=self.outcomes.device, dtype=torch.bool))

    def validate(self, feature_dim: int):
        if (self.task.ndim != 2 or self.task.shape[1] != feature_dim
                or self.candidate.shape != self.task.shape or self.task.shape[0] == 0):
            raise ValueError("task and candidate must have non-empty shape [batch,feature_dim]")
        if (self.tests.ndim != 3 or self.tests.shape[0] != self.task.shape[0]
                or self.tests.shape[2] != feature_dim or self.tests.shape[1] == 0
                or self.outcomes.shape != self.tests.shape[:2]):
            raise ValueError("tests [batch,tests,features] and outcomes [batch,tests] must match")
        if self.outcomes.dtype != torch.long or ((self.outcomes < 0) | (self.outcomes >= 5)).any():
            raise ValueError("outcomes must be integer indices for the five classes")
        features = [self.task, self.candidate, self.tests]
        if self.diff is not None:
            if self.diff.shape != self.task.shape:
                raise ValueError("diff must match task shape")
            features.append(self.diff)
        if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in features):
            raise ValueError("semantic features must be finite floating-point tensors")
        if any(value.device != self.task.device for value in [*features, self.outcomes]):
            raise ValueError("batch tensors must share a device")
        if any(value.dtype != self.task.dtype for value in features):
            raise ValueError("semantic features must share a dtype")
