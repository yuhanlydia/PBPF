"""Bounded, isolated Python-call execution for the CodeARC replay protocol."""
from __future__ import annotations

import ast
import json
import math
from functools import partial
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import tempfile


def _limits(timeout):
    seconds = max(1, math.ceil(timeout))
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 ** 2, 8 * 1024 ** 2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _error_text(text):
    # SyntaxError embeds source line numbers, which depend on candidate length.
    value = re.sub(r"\s*\(<string>, line \d+\)$", "", text.strip())
    for function in ("min", "max"):
        value = value.replace(f"{function}() iterable argument is empty", f"{function}() arg is an empty sequence")
    return value


def score_result(actual, test):
    """Match output/exception semantics; a timeout never counts as a pass."""
    if actual.get("timed_out"):
        return "TIMEOUT"
    if actual.get("process_failure"):
        return "RUNTIME_EXCEPTION"
    if test["expected_error"]:
        if actual["errored"] and _error_text(actual["error"]) == _error_text(test["expected"]):
            return "PASS"
        if actual.get("error_type") in {"SyntaxError", "IndentationError", "TabError"}:
            return "COMPILE_ERROR"
        return "RUNTIME_EXCEPTION" if actual["errored"] else "WRONG_OUTPUT"
    if actual["errored"]:
        return "COMPILE_ERROR" if actual.get("error_type") in {"SyntaxError", "IndentationError", "TabError"} else "RUNTIME_EXCEPTION"
    return "PASS" if actual["stdout"].strip() == test["expected"].strip() else "WRONG_OUTPUT"


def _dictionary_order_only(actual, expected):
    """A diagnostic only: keep exact-output PASS/FAIL decisions unchanged."""
    if len(actual) > 65536 or len(expected) > 65536:
        return False
    actual, expected = actual.strip(), expected.strip()
    prefix = re.compile(r"^(Result \d+:)\s*(.*)$", re.S)
    left, right = prefix.match(actual), prefix.match(expected)
    if left is not None or right is not None:
        if left is None or right is None or left[1] != right[1]:
            return False
        actual, expected = left[2], right[2]
    try:
        a, b = ast.literal_eval(actual), ast.literal_eval(expected)
    except (ValueError, SyntaxError, RecursionError):
        return False
    return isinstance(a, dict) and isinstance(b, dict) and a == b


def execute_call(code, test, *, timeout=6.0):
    """Mount only one program and invocation; no dataset files or labels."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    program = code + "\n\n" + test["input"]
    wrapper = ("import contextlib, io, json\n"
               "out = io.StringIO()\n"
               "error = ''\n"
               "error_type = ''\n"
               "errored = False\n"
               "try:\n"
               "    with contextlib.redirect_stdout(out):\n"
               f"        exec({program!r}, {{'__name__': '__main__'}})\n"
               "except BaseException as exc:\n"
               "    error = str(exc)\n"
               "    error_type = type(exc).__name__\n"
               "    errored = True\n"
               "print(json.dumps({'stdout': out.getvalue(), 'error': error, 'error_type': error_type, 'errored': errored}))\n")
    with tempfile.TemporaryDirectory(prefix="pbpf-codearc-") as directory:
        directory = Path(directory)
        source = directory / "run.py"
        source.write_text(wrapper)
        command = ["bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
                   "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
                   "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
                   "--ro-bind", "/etc/alternatives", "/etc/alternatives", "--tmpfs", "/tmp",
                   "--proc", "/proc", "--dev", "/dev", "--ro-bind", str(source), "/run.py",
                   "--unshare-all", "--die-with-parent", "--chdir", "/tmp",
                   "/usr/bin/python3", "-I", "/run.py"]
        with (directory / "stdout").open("w+b") as stdout, (directory / "stderr").open("w+b") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
                     "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                start_new_session=True, preexec_fn=partial(_limits, timeout))
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Kill any descendants even when the direct child exited first.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            stdout.seek(0); stderr.seek(0)
            output, errors = stdout.read(8 * 1024 ** 2), stderr.read(64 * 1024)
        if process.returncode in (-signal.SIGXCPU, 128 + signal.SIGXCPU):
            timed_out = True
        errors = errors.decode(errors="replace")
        if errors.startswith("bwrap:"):
            raise RuntimeError("candidate sandbox infrastructure failure: " + errors[:1000])
        if timed_out:
            actual = {"timed_out": True, "errored": True, "error": "execution timeout", "stdout": ""}
        else:
            try:
                actual = json.loads(output)
                if (not isinstance(actual, dict) or not isinstance(actual.get("stdout"), str)
                        or not isinstance(actual.get("error"), str) or type(actual.get("errored")) is not bool):
                    raise ValueError("invalid execution record")
                actual["timed_out"] = False
            except (ValueError, UnicodeDecodeError):
                actual = {"process_failure": True, "timed_out": False, "errored": True,
                          "error": errors[:4096], "stdout": output[:4096].decode(errors="replace")}
        actual.update(returncode=process.returncode, stderr=errors[:4096])
        actual["outcome"] = score_result(actual, test)
        actual["expected_timeout"] = bool(test["expected_error"]
            and "timed out" in test["expected"].lower())
        actual["reference_behavior_matches"] = actual["outcome"] == "PASS" or (
            actual["expected_timeout"] and actual["timed_out"])
        actual["dictionary_order_only_mismatch"] = bool(actual["outcome"] == "WRONG_OUTPUT"
            and not actual["errored"] and not test["expected_error"]
            and _dictionary_order_only(actual["stdout"], test["expected"]))
        # Compare before truncating; evaluator evidence records truncation.
        actual["stdout_truncated"] = len(actual["stdout"]) > 4096
        actual["stdout"] = actual["stdout"][:4096]
        return actual
