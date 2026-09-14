from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Set

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


class LocalPythonSandbox:
    """Execute Python against evaluator-only stdio cases for local pilot runs."""

    def __init__(
        self,
        tests: Mapping[str, Mapping[str, Any]],
        *,
        timeout_seconds: float = 2.0,
        memory_mb: int = 512,
        python_executable: str = sys.executable,
    ) -> None:
        self._tests = {str(key): dict(value) for key, value in tests.items()}
        self.timeout_seconds = float(timeout_seconds)
        self.memory_mb = int(memory_mb)
        self.python_executable = python_executable
        if self.timeout_seconds <= 0 or self.memory_mb <= 0:
            raise ValueError("sandbox limits must be positive")

    @staticmethod
    def _code(text: str) -> str:
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return match.group(1) if match else text

    @staticmethod
    def _set_limit(limit: int, values: tuple[int, int]) -> None:
        try:
            resource.setrlimit(limit, values)
        except (OSError, PermissionError, ValueError):
            pass

    @staticmethod
    def _drop_privileges() -> None:
        operations = tuple(
            getattr(os, name, None) for name in ("setgroups", "setgid", "setuid")
        )
        if not all(callable(operation) for operation in operations):
            return
        setgroups, setgid, setuid = operations
        setgroups([])
        setgid(65534)
        setuid(65534)

    def _limits(self) -> None:
        memory = self.memory_mb * 1024 * 1024
        self._set_limit(resource.RLIMIT_AS, (memory, memory))
        self._set_limit(
            resource.RLIMIT_CPU, (max(1, int(self.timeout_seconds)),) * 2
        )
        self._set_limit(resource.RLIMIT_NPROC, (16, 16))
        self._set_limit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        self._set_limit(resource.RLIMIT_CORE, (0, 0))
        getuid = getattr(os, "getuid", None)
        if getuid is not None and getuid() == 0:
            self._drop_privileges()

    def execute(self, code: str, test_id: str) -> SandboxResult:
        case = self._tests.get(test_id)
        if case is None:
            return SandboxResult(None, "unknown evaluator test", True)
        with tempfile.TemporaryDirectory(prefix="pbpf-eval-") as directory:
            root = Path(directory)
            root.chmod(0o755)
            source = root / "candidate.py"
            source.write_text(
                self._code(code) + str(case.get("harness", "")), encoding="utf-8"
            )
            source.chmod(0o444)
            try:
                result = subprocess.run(
                    [self.python_executable, "-I", str(source)],
                    input=str(case.get("input", "")),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=root,
                    timeout=self.timeout_seconds,
                    preexec_fn=self._limits,
                    env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult("TIMEOUT", "timeout", False)
            except (OSError, subprocess.SubprocessError) as error:
                return SandboxResult(None, str(error), True)
        if result.returncode != 0:
            stderr = result.stderr[-1000:]
            if "SyntaxError" in stderr:
                outcome = "COMPILE_ERROR"
            elif "AssertionError" in stderr:
                outcome = "WRONG_OUTPUT"
            else:
                outcome = "RUNTIME_EXCEPTION"
            return SandboxResult(outcome, stderr, False)
        expected = str(case.get("output", "")).rstrip()
        actual = result.stdout.rstrip()
        outcome = "PASS" if actual == expected else "WRONG_OUTPUT"
        return SandboxResult(outcome, outcome.lower().replace("_", " "), False)
