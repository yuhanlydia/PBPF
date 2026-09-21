"""Direct execution profile: no namespaces, bubblewrap, or network isolation.

Explicit user authorization permits this separate profile. The root parent drops
uid/gid/supplementary groups for subprocesses; /root must be inaccessible to that
uid. Candidate directories are /tmp, mode 0700, owned by nobody. Fixed original
resource limits are retained, with a new RLIMIT_NPROC=64 shared across the uid.
This is a resource cap, NOT fork prevention or sandbox equivalence. Processes
retain ordinary host/network access as that uid. Never classify these results as
having been measured under the original isolated execution profile.

Scoring functions are imported unchanged. The two execution bodies preserve the
original stdin and CodeARC return fields/comparison/truncation semantics.
"""
from contextlib import contextmanager
from functools import partial
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import subprocess
import tempfile
from pbpf.apbpf.rbr_execution import canonical_stdin, output_matches
from pbpf.apbpf.codearc_execution import _limits, score_result, _dictionary_order_only


def _direct_limits(timeout):
    _limits(timeout)
    resource.setrlimit(resource.RLIMIT_NPROC, (64,64))


@contextmanager
def _directory():
    if os.geteuid()!=0:
        raise RuntimeError('direct profile requires root parent for explicit uid/gid drop')
    if stat.S_IMODE(Path('/root').stat().st_mode)!=0o700:
        raise RuntimeError('direct profile requires /root mode 0700')
    with tempfile.TemporaryDirectory(prefix='pbpf-direct-',dir='/tmp') as name:
        os.chmod(name,0o700);os.chown(name,65534,65534)
        yield name


def execute_stdin(code, test, *, timeout=6.):
    if not isinstance(code, str) or not isinstance(test['input'], str) or not isinstance(test['expected'], str):
        raise TypeError('program, input and expected output must be text')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive finite deadline required')
    try:
        compile(code, 'candidate.py', 'exec')
    except (SyntaxError, ValueError, TypeError) as error:
        return {'outcome': 'COMPILE_ERROR', 'stdout': '', 'stderr': str(error)[:4096],
                'returncode': None, 'timed_out': False, 'stdout_truncated': False}
    with _directory() as name:
        directory = Path(name)
        source = directory/'candidate.py'; source.write_text(code)
        input_path = directory/'stdin'; input_path.write_text(canonical_stdin(test['input']))
        source.chmod(0o444)
        command = ['/usr/bin/python3', '-I', str(source)]
        with input_path.open('rb') as stdin, (directory/'stdout').open('w+b') as stdout, (directory/'stderr').open('w+b') as stderr:
            process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr,
                env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'OMP_NUM_THREADS': '1',
                     'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                start_new_session=True, preexec_fn=partial(_direct_limits, timeout),
                cwd=directory, user=65534, group=65534, extra_groups=[], shell=False, close_fds=True)
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            stdout.seek(0); stderr.seek(0)
            actual = stdout.read(8*1024**2).decode(errors='replace')
            errors = stderr.read(65536).decode(errors='replace')
        timed_out |= process.returncode in (-signal.SIGXCPU, 128+signal.SIGXCPU)
        outcome = ('TIMEOUT' if timed_out else 'RUNTIME_EXCEPTION' if process.returncode else
                   'PASS' if output_matches(actual, test['expected']) else 'WRONG_OUTPUT')
        return {'outcome': outcome, 'stdout': actual[:4096], 'stderr': errors[:4096],
                'returncode': process.returncode, 'timed_out': timed_out,
                'stdout_truncated': len(actual) > 4096,
                'terminal_newline_added': not test['input'].endswith('\n')}


def execute_call(code, test, *, timeout=6.0):
    """Run one direct Python invocation; preserve original CodeARC scoring."""
    if not math.isfinite(timeout) or timeout <= 0:
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
    with _directory() as directory:
        directory = Path(directory)
        source = directory / "run.py"
        source.write_text(wrapper)
        source.chmod(0o444)
        command = ['/usr/bin/python3', '-I', str(source)]
        with (directory / "stdout").open("w+b") as stdout, (directory / "stderr").open("w+b") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
                     "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                start_new_session=True, preexec_fn=partial(_direct_limits, timeout),
                cwd=directory, user=65534, group=65534, extra_groups=[], shell=False, close_fds=True)
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
