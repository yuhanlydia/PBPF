from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


@dataclass
class ExecutionLedger:
    executions: list[tuple[str, str]] = field(default_factory=list)

    def record(self, task_id: str, test_id: str) -> None:
        self.executions.append((task_id, test_id))

    def snapshot(self) -> tuple[int, tuple[tuple[str, str], ...]]:
        return len(self.executions), tuple(self.executions)


@dataclass
class TokenLedger:
    visible: int = 0
    generated: int = 0

    def record(self, *, visible: int, generated: int) -> None:
        if visible < 0 or generated < 0:
            raise ValueError("token counts cannot be negative")
        self.visible += visible
        self.generated += generated

    def snapshot(self) -> tuple[int, int]:
        return self.visible, self.generated


@dataclass
class ComputeLedger:
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    gpu_hours: float = 0.0

    def record(self, *, cpu_seconds: float, wall_seconds: float, gpu_hours: float) -> None:
        if any(
            not math.isfinite(float(value)) or value < 0
            for value in (cpu_seconds, wall_seconds, gpu_hours)
        ):
            raise ValueError("compute costs must be finite and non-negative")
        self.cpu_seconds += cpu_seconds
        self.wall_seconds += wall_seconds
        self.gpu_hours += gpu_hours

    def snapshot(self) -> tuple[float, float, float]:
        return self.cpu_seconds, self.wall_seconds, self.gpu_hours


def assert_equal_budgets(
    arms: Mapping[str, tuple[ExecutionLedger, TokenLedger, ComputeLedger]]
) -> None:
    snapshots = {
        arm: tuple(ledger.snapshot() for ledger in ledgers) for arm, ledgers in arms.items()
    }
    if len(set(snapshots.values())) > 1:
        raise ValueError(f"budget mismatch across arms: {snapshots}")
