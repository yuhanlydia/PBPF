"""Counterfactual evidence attribution for EESD relevance.

This module intentionally separates three concepts:
- signed Shapley attribution: whether an execution increases or decreases model preference for a correction;
- nonnegative relevance: absolute attribution magnitude used by EED;
- evidence mass: computed later from relevance concentration.

The exact Shapley routine is model-agnostic. Model-specific coalition scoring lives in
``scripts/score_eesd_shapley_relevance.py``.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from math import factorial
import copy
import json
from collections.abc import Mapping, Sequence

import numpy as np


def _validate_players(players: int) -> int:
    if type(players) is not int or players < 1 or players > 20:
        raise ValueError("players must be an integer in [1, 20]")
    return players


def _validate_coalition_values(values: Mapping[int, float], players: int, *, require_all: bool) -> dict[int, float]:
    players = _validate_players(players)
    result: dict[int, float] = {}
    limit = 1 << players
    for key, value in values.items():
        if type(key) is not int or not 0 <= key < limit:
            raise ValueError("coalition keys must be valid bit masks")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("coalition values must be finite")
        result[key] = value
    if require_all and set(result) != set(range(limit)):
        raise ValueError("exact Shapley requires all 2^n coalition values")
    return result


def exact_shapley_values(values: Mapping[int, float], players: int) -> np.ndarray:
    """Return exact signed Shapley values from a complete coalition-value table."""
    values = _validate_coalition_values(values, players, require_all=True)
    n = players
    denom = factorial(n)
    result = np.zeros(n, dtype=float)
    for i in range(n):
        bit = 1 << i
        for mask in range(1 << n):
            if mask & bit:
                continue
            size = mask.bit_count()
            coefficient = factorial(size) * factorial(n - size - 1) / denom
            result[i] += coefficient * (values[mask | bit] - values[mask])
    return result


def leave_one_out_values(values: Mapping[int, float], players: int) -> np.ndarray:
    """Return full-coalition minus leave-one-out influence, not Shapley."""
    values = _validate_coalition_values(values, players, require_all=False)
    full = (1 << players) - 1
    needed = {full, *(full & ~(1 << i) for i in range(players))}
    if not needed <= set(values):
        raise ValueError("leave-one-out requires full and all one-player-removed coalitions")
    return np.asarray([values[full] - values[full & ~(1 << i)] for i in range(players)], dtype=float)


def absolute_relevance(attribution: Sequence[float], *, zero_tolerance: float = 1e-12) -> np.ndarray:
    """Convert signed attribution to nonnegative EED relevance.

    Sign is retained separately by callers for diagnostics. EED relevance measures
    influence magnitude only. If the value function is invariant to every execution,
    use uniform relevance instead of amplifying numerical noise.
    """
    values = np.asarray(list(attribution), dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("attribution must be a nonempty finite vector")
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    relevance = np.abs(values)
    if float(relevance.max()) <= zero_tolerance:
        return np.ones(len(values), dtype=float)
    return relevance


def coalition_masks(players: int, mode: str) -> list[int]:
    """Return the coalition masks required by exact Shapley or LOO attribution."""
    players = _validate_players(players)
    full = (1 << players) - 1
    if mode == "exact":
        return list(range(1 << players))
    if mode == "loo":
        return [full] + [full & ~(1 << i) for i in range(players)]
    raise ValueError("mode must be 'exact' or 'loo'")


def build_causal_labels(prompt_length: int, response_ids: Sequence[int], target_mask: Sequence[bool]) -> np.ndarray:
    """Build Hugging Face causal-LM labels for selected response-token positions only."""
    if type(prompt_length) is not int or prompt_length < 1:
        raise ValueError("prompt_length must be a positive integer")
    response = list(response_ids)
    mask = np.asarray(list(target_mask), dtype=bool)
    if len(response) != len(mask) or any(type(x) is not int for x in response):
        raise ValueError("response IDs and target mask must have equal valid lengths")
    labels = np.full(prompt_length + len(response), -100, dtype=np.int64)
    if len(response):
        absolute = prompt_length + np.flatnonzero(mask)
        labels[absolute] = np.asarray(response, dtype=np.int64)[mask]
    return labels


def edited_token_masks(original_ids: Sequence[int], correction_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for token positions participating in the edit.

    Equal spans are excluded. Replaced/deleted original tokens are marked in the
    original mask; replaced/inserted correction tokens are marked in the correction
    mask. This allows the coalition value to score only behavior-changing tokens,
    avoiding domination by copied code.
    """
    original = list(original_ids)
    correction = list(correction_ids)
    if any(type(x) is not int for x in original + correction):
        raise ValueError("token IDs must be integers")
    original_mask = np.zeros(len(original), dtype=bool)
    correction_mask = np.zeros(len(correction), dtype=bool)
    matcher = SequenceMatcher(a=original, b=correction, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"replace", "delete"}:
            original_mask[i1:i2] = True
        if tag in {"replace", "insert"}:
            correction_mask[j1:j2] = True
    return original_mask, correction_mask


def subset_execution_messages(messages: Sequence[Mapping[str, str]], keep_indices: Sequence[int]) -> list[dict[str, str]]:
    """Remove public-execution rows while preserving all stored/clipped text.

    EESD generation stores a user message consisting of a fixed textual prefix,
    then a JSON object containing ``public_executions``. Filtering that serialized
    object prevents coalition scoring from changing truncation or other prompt text.
    """
    if len(messages) < 2:
        raise ValueError("expected at least system and user messages")
    indices = list(keep_indices)
    if any(type(i) is not int for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("keep_indices must be unique integers")
    result = copy.deepcopy(list(messages))
    user_indices = [i for i, message in enumerate(result) if message.get("role") == "user"]
    if len(user_indices) != 1:
        raise ValueError("expected exactly one user message")
    position = user_indices[0]
    content = result[position].get("content")
    if not isinstance(content, str) or "\n" not in content:
        raise ValueError("stored user message does not contain EESD JSON payload")
    prefix, payload = content.split("\n", 1)
    body = json.loads(payload)
    executions = body.get("public_executions")
    if not isinstance(executions, list) or not executions:
        raise ValueError("public_executions must be a nonempty list")
    if any(i < 0 or i >= len(executions) for i in indices):
        raise ValueError("keep_indices out of range")
    body["public_executions"] = [executions[i] for i in indices]
    result[position]["content"] = prefix + "\n" + json.dumps(body, sort_keys=True, ensure_ascii=False)
    return result
