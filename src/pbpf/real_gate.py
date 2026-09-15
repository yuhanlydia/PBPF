"""Small, honest utilities for the real-data PBPF prediction gate.

The encoder hashes lexical *content* into a fixed frozen feature space.  Task
and candidate identifiers are intentionally never accepted by this interface.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping

import numpy as np


class FrozenTextEncoder:
    """Deterministic signed feature hashing over source text tokens/ngrams."""

    def __init__(self, dimension: int = 256) -> None:
        if not isinstance(dimension, int) or dimension < 16:
            raise ValueError("feature dimension must be an integer of at least 16")
        self.dimension = dimension

    def __call__(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError("semantic encoder accepts source text only")
        tokens = re.findall(r"[A-Za-z_]+|\d+(?:\.\d+)?|[^\w\s]", text.lower())
        features = np.zeros(self.dimension, dtype=np.float32)
        items = [("token", token) for token in tokens]
        items += [("bigram", left + "\0" + right) for left, right in zip(tokens, tokens[1:])]
        for kind, value in items:
            digest = hashlib.sha256((kind + "\0" + value).encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % (self.dimension - 4)
            sign = 1.0 if digest[8] & 1 else -1.0
            features[index] += sign
        scale = max(1.0, math.sqrt(len(items)))
        features[:-4] /= scale
        features[-4:] = np.asarray(
            [math.log1p(len(text)), math.log1p(len(tokens)), "\n" in text, bool(re.search(r"\d", text))],
            dtype=np.float32,
        )
        features[-4:-2] /= 16.0
        return features


def classify_execution(
    returncode: int | None,
    stdout: str,
    stderr: str,
    expected_stdout: str,
    *,
    timed_out: bool = False,
) -> str:
    """Map one isolated stdin/stdout execution to the registered outcome set."""
    if timed_out:
        return "TIMEOUT"
    if returncode is None or returncode != 0:
        return "RUNTIME_EXCEPTION"
    return "PASS" if stdout.rstrip() == expected_stdout.rstrip() else "WRONG_OUTPUT"


def select_disjoint_problem_ids(
    inventories: Mapping[str, set[str]],
    caps: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, tuple[str, ...]]:
    """Select reproducible problem-disjoint subsets from official partitions."""
    if set(inventories) != set(caps) or any(value < 0 for value in caps.values()):
        raise ValueError("inventories and nonnegative caps must have identical partitions")
    ownership: dict[str, int] = {}
    for values in inventories.values():
        for value in values:
            ownership[value] = ownership.get(value, 0) + 1
    result = {}
    for split, values in inventories.items():
        eligible = [value for value in values if ownership[value] == 1]
        ranked = sorted(
            eligible,
            key=lambda value: hashlib.sha256(f"{seed}\0{split}\0{value}".encode()).digest(),
        )
        result[split] = tuple(ranked[: caps[split]])
    return result


def compare_predictions(labels: np.ndarray, predictions: Mapping[str, np.ndarray]) -> dict:
    """Compute proper metrics and paired gains against the named baseline."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not len(labels) or "baseline" not in predictions:
        raise ValueError("nonempty labels and a baseline prediction matrix are required")
    reports: dict[str, dict[str, float]] = {}
    losses: dict[str, np.ndarray] = {}
    for name, raw in predictions.items():
        probabilities = np.asarray(raw, dtype=np.float64)
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != len(labels)
            or np.any(labels < 0)
            or np.any(labels >= probabilities.shape[1])
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(1), 1.0, atol=1e-6)
        ):
            raise ValueError("predictions must be normalized [examples,classes]")
        clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
        losses[name] = -np.log(clipped)
        one_hot = np.eye(probabilities.shape[1])[labels]
        reports[name] = {
            "nll": float(losses[name].mean()),
            "brier": float(np.square(probabilities - one_hot).sum(1).mean()),
            "accuracy": float((probabilities.argmax(1) == labels).mean()),
        }
    baseline = losses["baseline"]
    for name, row in reports.items():
        row["nll_gain_vs_baseline"] = float((baseline - losses[name]).mean())
    return reports
