from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .registry import OUTCOMES


def _stable_seed(value: Any, seed: int) -> int:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(f"{seed}:{payload}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


class StateEncoder:
    """Dependency-free matched-width state controls for CPU audits."""

    MODES = {"gru", "exchangeable", "pass_rate", "random", "last", "text_summary"}

    def __init__(self, mode: str, *, dimension: int, seed: int = 0) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unknown state encoder mode: {mode}")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.mode = mode
        self.dimension = dimension
        self.seed = seed
        generator = np.random.default_rng(seed)
        self._wz = generator.normal(scale=0.12, size=(dimension, dimension))
        self._uz = generator.normal(scale=0.12, size=(dimension, dimension))
        self._wr = generator.normal(scale=0.12, size=(dimension, dimension))
        self._ur = generator.normal(scale=0.12, size=(dimension, dimension))
        self._wh = generator.normal(scale=0.12, size=(dimension, dimension))
        self._uh = generator.normal(scale=0.12, size=(dimension, dimension))
        self._bz = np.zeros(dimension, dtype=np.float64)
        self._br = np.zeros(dimension, dtype=np.float64)
        self._bh = np.zeros(dimension, dtype=np.float64)
        self._projection = np.eye(dimension, dtype=np.float64)
        self._bias = np.zeros(dimension, dtype=np.float64)

    def _embedding(self, row: Mapping[str, Any]) -> np.ndarray:
        generator = np.random.default_rng(_stable_seed(dict(row), self.seed))
        return generator.normal(size=self.dimension)

    def _base_encode(self, history: Sequence[Mapping[str, Any]]) -> np.ndarray:
        rows = tuple(dict(row) for row in history)
        if self.mode == "random":
            generator = np.random.default_rng(_stable_seed(rows, self.seed))
            vector = generator.normal(size=self.dimension)
        elif self.mode == "pass_rate":
            passes = sum(row.get("outcome") == "PASS" for row in rows)
            rate = passes / len(rows) if rows else 0.0
            vector = np.zeros(self.dimension)
            vector[0] = rate
            if self.dimension > 1:
                vector[1] = len(rows)
        elif self.mode == "last":
            vector = self._embedding(rows[-1]) if rows else np.zeros(self.dimension)
        elif self.mode == "exchangeable":
            vector, _ = self._exchangeable_forward(rows)
        elif self.mode == "text_summary":
            summary = tuple(
                (row.get("test_id"), row.get("outcome")) for row in rows
            )
            generator = np.random.default_rng(_stable_seed(summary, self.seed))
            vector = generator.normal(size=self.dimension)
        else:
            vector, _ = self._gru_forward(rows)
        return np.asarray(vector, dtype=np.float64)

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        clipped = np.clip(value, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _gru_forward(self, rows: Sequence[Mapping[str, Any]]):
        hidden = np.zeros(self.dimension, dtype=np.float64)
        cache = []
        for row in rows:
            embedded = self._embedding(row)
            previous = hidden
            update = self._sigmoid(self._wz @ embedded + self._uz @ previous + self._bz)
            reset = self._sigmoid(self._wr @ embedded + self._ur @ previous + self._br)
            candidate = np.tanh(
                self._wh @ embedded + self._uh @ (reset * previous) + self._bh
            )
            hidden = (1.0 - update) * candidate + update * previous
            cache.append((embedded, previous, update, reset, candidate))
        return hidden, cache

    def _exchangeable_forward(self, rows: Sequence[Mapping[str, Any]]):
        item_cache = []
        item_values = []
        for row in rows:
            embedded = self._embedding(row)
            first = np.tanh(self._wz @ embedded)
            second = np.tanh(self._uz @ first)
            third = np.tanh(self._wr @ second)
            item_cache.append((embedded, first, second, third))
            item_values.append(third)
        pooled = (
            np.mean(item_values, axis=0)
            if item_values
            else np.zeros(self.dimension, dtype=np.float64)
        )
        first = np.tanh(self._ur @ pooled + self._bz)
        second = np.tanh(self._wh @ first + self._br)
        third = np.tanh(self._uh @ second + self._bh)
        return third, (item_cache, pooled, first, second, third)

    def encode(self, history: Sequence[Mapping[str, Any]]) -> np.ndarray:
        base = self._base_encode(history)
        return np.tanh(self._projection @ base + self._bias)

    @property
    def trainable_parameter_count(self) -> int:
        if self.mode not in {"gru", "exchangeable"}:
            return self._projection.size + self._bias.size
        return (
            sum(
                matrix.size
                for matrix in (self._wz, self._uz, self._wr, self._ur, self._wh, self._uh, self._projection)
            )
            + sum(vector.size for vector in (self._bz, self._br, self._bh, self._bias))
        )

    @property
    def gru_parameter_vector(self) -> np.ndarray:
        if self.mode != "gru":
            raise ValueError("GRU parameters are available only in gru mode")
        return np.concatenate(
            [
                value.ravel()
                for value in (
                    self._wz,
                    self._uz,
                    self._wr,
                    self._ur,
                    self._wh,
                    self._uh,
                    self._bz,
                    self._br,
                    self._bh,
                )
            ]
        )

    @property
    def trainable_parameter_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                value.ravel()
                for value in (
                    self._wz,
                    self._uz,
                    self._wr,
                    self._ur,
                    self._wh,
                    self._uh,
                    self._bz,
                    self._br,
                    self._bh,
                    self._projection,
                    self._bias,
                )
            ]
        )

    def fit(
        self,
        histories: Sequence[Sequence[Mapping[str, Any]]],
        targets: np.ndarray,
        *,
        epochs: int = 100,
        learning_rate: float = 0.05,
    ) -> tuple[float, ...]:
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        expected = np.asarray(targets, dtype=np.float64)
        if expected.shape != (len(histories), self.dimension):
            raise ValueError("targets must have shape [histories, dimension]")
        if self.mode == "gru":
            return self._fit_gru(histories, expected, epochs, learning_rate)
        if self.mode == "exchangeable":
            return self._fit_exchangeable(histories, expected, epochs, learning_rate)
        bases = np.asarray([self._base_encode(history) for history in histories])
        losses: list[float] = []
        for _ in range(epochs):
            activations = bases @ self._projection.T + self._bias
            predictions = np.tanh(activations)
            error = predictions - expected
            losses.append(float(np.square(error).mean()))
            gradient = (2.0 / error.size) * error * (1.0 - np.square(predictions))
            self._projection -= learning_rate * (gradient.T @ bases)
            self._bias -= learning_rate * gradient.sum(axis=0)
        return tuple(losses)

    def _fit_gru(self, histories, expected, epochs, learning_rate) -> tuple[float, ...]:
        losses: list[float] = []
        sample_count = len(histories)
        for _ in range(epochs):
            gradients = {
                name: np.zeros_like(getattr(self, name))
                for name in (
                    "_wz",
                    "_uz",
                    "_wr",
                    "_ur",
                    "_wh",
                    "_uh",
                    "_bz",
                    "_br",
                    "_bh",
                    "_projection",
                    "_bias",
                )
            }
            total_loss = 0.0
            for history, target in zip(histories, expected, strict=True):
                hidden, cache = self._gru_forward(tuple(dict(row) for row in history))
                output = np.tanh(self._projection @ hidden + self._bias)
                error = output - target
                total_loss += float(np.square(error).mean())
                output_gradient = (2.0 / (sample_count * self.dimension)) * error * (
                    1.0 - np.square(output)
                )
                gradients["_projection"] += np.outer(output_gradient, hidden)
                gradients["_bias"] += output_gradient
                hidden_gradient = self._projection.T @ output_gradient
                for embedded, previous, update, reset, candidate in reversed(cache):
                    candidate_gradient = hidden_gradient * (1.0 - update)
                    update_gradient = hidden_gradient * (previous - candidate)
                    previous_gradient = hidden_gradient * update

                    candidate_pre = candidate_gradient * (1.0 - np.square(candidate))
                    gradients["_wh"] += np.outer(candidate_pre, embedded)
                    gradients["_uh"] += np.outer(candidate_pre, reset * previous)
                    gradients["_bh"] += candidate_pre
                    reset_hidden_gradient = self._uh.T @ candidate_pre
                    reset_gradient = reset_hidden_gradient * previous
                    previous_gradient += reset_hidden_gradient * reset

                    reset_pre = reset_gradient * reset * (1.0 - reset)
                    gradients["_wr"] += np.outer(reset_pre, embedded)
                    gradients["_ur"] += np.outer(reset_pre, previous)
                    gradients["_br"] += reset_pre
                    previous_gradient += self._ur.T @ reset_pre

                    update_pre = update_gradient * update * (1.0 - update)
                    gradients["_wz"] += np.outer(update_pre, embedded)
                    gradients["_uz"] += np.outer(update_pre, previous)
                    gradients["_bz"] += update_pre
                    previous_gradient += self._uz.T @ update_pre
                    hidden_gradient = previous_gradient
            losses.append(total_loss / sample_count)
            for name, gradient in gradients.items():
                setattr(self, name, getattr(self, name) - learning_rate * gradient)
        return tuple(losses)

    def _fit_exchangeable(
        self, histories, expected, epochs, learning_rate
    ) -> tuple[float, ...]:
        losses: list[float] = []
        sample_count = len(histories)
        names = (
            "_wz",
            "_uz",
            "_wr",
            "_ur",
            "_wh",
            "_uh",
            "_bz",
            "_br",
            "_bh",
            "_projection",
            "_bias",
        )
        for _ in range(epochs):
            gradients = {name: np.zeros_like(getattr(self, name)) for name in names}
            total_loss = 0.0
            for history, target in zip(histories, expected, strict=True):
                hidden, cache = self._exchangeable_forward(
                    tuple(dict(row) for row in history)
                )
                item_cache, pooled, rho_first, rho_second, rho_third = cache
                output = np.tanh(self._projection @ hidden + self._bias)
                error = output - target
                total_loss += float(np.square(error).mean())
                gradient = (2.0 / (sample_count * self.dimension)) * error * (
                    1.0 - np.square(output)
                )
                gradients["_projection"] += np.outer(gradient, hidden)
                gradients["_bias"] += gradient

                rho_third_gradient = (self._projection.T @ gradient) * (
                    1.0 - np.square(rho_third)
                )
                gradients["_uh"] += np.outer(rho_third_gradient, rho_second)
                gradients["_bh"] += rho_third_gradient
                rho_second_gradient = (self._uh.T @ rho_third_gradient) * (
                    1.0 - np.square(rho_second)
                )
                gradients["_wh"] += np.outer(rho_second_gradient, rho_first)
                gradients["_br"] += rho_second_gradient
                rho_first_gradient = (self._wh.T @ rho_second_gradient) * (
                    1.0 - np.square(rho_first)
                )
                gradients["_ur"] += np.outer(rho_first_gradient, pooled)
                gradients["_bz"] += rho_first_gradient
                pooled_gradient = self._ur.T @ rho_first_gradient

                if item_cache:
                    for embedded, first, second, third in item_cache:
                        third_gradient = (pooled_gradient / len(item_cache)) * (
                            1.0 - np.square(third)
                        )
                        gradients["_wr"] += np.outer(third_gradient, second)
                        second_gradient = (self._wr.T @ third_gradient) * (
                            1.0 - np.square(second)
                        )
                        gradients["_uz"] += np.outer(second_gradient, first)
                        first_gradient = (self._uz.T @ second_gradient) * (
                            1.0 - np.square(first)
                        )
                        gradients["_wz"] += np.outer(first_gradient, embedded)
            losses.append(total_loss / sample_count)
            for name, gradient in gradients.items():
                setattr(self, name, getattr(self, name) - learning_rate * gradient)
        return tuple(losses)


def state_text(history: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{row.get('test_id', '?')}: {row.get('outcome', 'MASKED')}" for row in history
    )
