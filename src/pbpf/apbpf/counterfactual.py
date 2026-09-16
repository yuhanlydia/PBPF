"""Deterministic counterfactual transformations for visible test histories."""

from __future__ import annotations

from typing import Any

import numpy as np


def _history_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 1:
        raise ValueError(f"{name} must have a history axis")
    return array


def _visible_count(visible_steps: int, history_length: int) -> int:
    if isinstance(visible_steps, bool) or not isinstance(visible_steps, (int, np.integer)):
        raise TypeError("visible_steps must be an integer")
    if not 0 <= int(visible_steps) <= history_length:
        raise ValueError("visible_steps must be within the history length")
    return int(visible_steps)


def association_eligibility(
    outcomes: Any, visible_steps: int, future_outcomes: Any | None = None
) -> np.ndarray:
    """Return the examples whose visible outcome history is non-constant.

    ``future_outcomes`` is accepted only to validate that separately supplied
    future labels retain the same batch axes.  Future labels do not determine
    association eligibility: a constant visible history is intrinsically
    uninformative for an outcome-only reassignment.
    """
    values = _history_array(outcomes, name="outcomes")
    visible = _visible_count(visible_steps, values.shape[-1])
    if future_outcomes is not None:
        future = _history_array(future_outcomes, name="future_outcomes")
        if future.shape[:-1] != values.shape[:-1]:
            raise ValueError("future_outcomes must share the outcomes batch axes")
    if visible < 2:
        return np.zeros(values.shape[:-1], dtype=bool)
    history = values[..., :visible]
    return np.any(history != history[..., :1], axis=-1)


def outcome_derangement(outcomes: Any, visible_steps: int, seed: int) -> np.ndarray:
    """Reassign visible outcomes while preserving every example's histogram.

    Every eligible sequence is rotated by a seeded offset chosen only from
    rotations that change that sequence. Rotation preserves duplicate counts;
    ineligible constant histories and all future outcomes are left untouched.
    """
    values = _history_array(outcomes, name="outcomes")
    visible = _visible_count(visible_steps, values.shape[-1])
    shuffled = values.copy()
    if visible < 2:
        return shuffled
    generator = np.random.default_rng(seed)
    flat_source = values.reshape(-1, values.shape[-1])
    flat_target = shuffled.reshape(-1, shuffled.shape[-1])
    for source, target in zip(flat_source, flat_target, strict=True):
        history = source[:visible]
        if np.any(history != history[0]):
            offsets = [
                offset
                for offset in range(1, visible)
                if not np.array_equal(np.roll(history, offset), history)
            ]
            offset = int(generator.choice(offsets))
            target[:visible] = np.roll(history, offset)
    return shuffled


def joint_permutation(
    tests: Any, outcomes: Any, visible_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Jointly permute visible test--outcome pairs without changing pairings.

    Outcomes use ``batch_axes + [tests]``. Test semantics may either have the
    same shape (IDs/scalars) or append feature axes such as ``[B,T,F]``.
    """
    test_values = _history_array(tests, name="tests")
    outcome_values = _history_array(outcomes, name="outcomes")
    if (test_values.ndim < outcome_values.ndim
            or test_values.shape[:outcome_values.ndim] != outcome_values.shape):
        raise ValueError("tests must share outcome batch/history axes, with optional feature axes")
    visible = _visible_count(visible_steps, outcome_values.shape[-1])
    permuted_tests = test_values.copy()
    permuted_outcomes = outcome_values.copy()
    if visible < 2:
        return permuted_tests, permuted_outcomes
    generator = np.random.default_rng(seed)
    history = outcome_values.shape[-1]
    feature_shape = test_values.shape[outcome_values.ndim:]
    flat_tests = permuted_tests.reshape(-1, history, *feature_shape)
    flat_outcomes = permuted_outcomes.reshape(-1, history)
    for test_row, outcome_row in zip(flat_tests, flat_outcomes, strict=True):
        order = generator.permutation(visible)
        if np.array_equal(order, np.arange(visible)):
            order = np.roll(order, 1)
        test_row[:visible] = test_row[:visible][order]
        outcome_row[:visible] = outcome_row[:visible][order]
    return permuted_tests, permuted_outcomes
