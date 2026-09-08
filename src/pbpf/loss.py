from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _is_torch(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch"


def stop_gradient(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def belief_loss(logits: Any, targets: Any) -> Any:
    if _is_torch(logits):
        import torch.nn.functional as functional

        return functional.cross_entropy(logits, targets)
    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(truth):
        raise ValueError("belief logits must be [examples, outcomes]")
    shifted = values - values.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probabilities[np.arange(len(truth)), truth].mean())


def actor_loss(
    log_probabilities_or_fn: Any,
    rewards: Any,
    belief_state: Any | None = None,
) -> Any:
    if callable(log_probabilities_or_fn):
        if belief_state is None:
            raise ValueError("callable actor loss requires belief_state")
        log_probabilities = log_probabilities_or_fn(stop_gradient(belief_state))
    else:
        log_probabilities = log_probabilities_or_fn
    detached_rewards = stop_gradient(rewards)
    if _is_torch(log_probabilities):
        return -(log_probabilities * detached_rewards).mean()
    log_values = np.asarray(log_probabilities, dtype=np.float64)
    reward_values = np.asarray(detached_rewards, dtype=np.float64)
    if log_values.shape != reward_values.shape:
        raise ValueError("actor log probabilities and rewards must have equal shape")
    return float(-(log_values * reward_values).mean())
