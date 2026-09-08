from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Set

from .registry import OUTCOMES


@dataclass(frozen=True)
class SandboxResult:
    outcome: str | None
    feedback: str = ""
    infrastructure_failure: bool = False


class FakeSandbox:
    """A deterministic, non-executing backend for protocol integration tests."""

    def __init__(
        self,
        outcomes: Mapping[tuple[str, str], str],
        *,
        infrastructure_failures: Set[tuple[str, str]] | None = None,
    ) -> None:
        invalid = set(outcomes.values()) - set(OUTCOMES)
        if invalid:
            raise ValueError(f"invalid fake outcomes: {sorted(invalid)}")
        self._outcomes = dict(outcomes)
        self._infrastructure_failures = set(infrastructure_failures or ())

    def execute(self, code: str, test_id: str) -> SandboxResult:
        key = (code, test_id)
        if key in self._infrastructure_failures:
            return SandboxResult(None, "infrastructure failure", True)
        outcome = self._outcomes.get(key, "COMPILE_ERROR")
        return SandboxResult(outcome, outcome.lower().replace("_", " "), False)
