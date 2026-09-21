"""Bounded stdin/stdout execution with no reference or expected-output mount."""
from functools import partial
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile

from pbpf.apbpf.codearc_execution import _limits


def canonical_stdin(value):
    """Match pinned RunBugRun test_runner.rb:348, including empty input."""
    return value if value.endswith('\n') else value+'\n'


def output_matches(actual, expected):
    if actual.rstrip() == expected.rstrip():
        return True
    left, right = actual.rstrip().splitlines(), expected.rstrip().splitlines()
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        a, b = a.split(), b.split()
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if x == y:
                continue
            try:
                if not math.isclose(float(x), float(y), rel_tol=0., abs_tol=1e-4):
                    return False
            except ValueError:
                return False
    return True


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
    with tempfile.TemporaryDirectory(prefix='pbpf-rbr-stdin-') as name:
        directory = Path(name)
        source = directory/'candidate.py'; source.write_text(code)
        input_path = directory/'stdin'; input_path.write_text(canonical_stdin(test['input']))
        command = ['bwrap', '--ro-bind', '/usr', '/usr', '--symlink', 'usr/bin', '/bin',
                   '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
                   '--ro-bind', '/etc/ld.so.cache', '/etc/ld.so.cache',
                   '--ro-bind', '/etc/alternatives', '/etc/alternatives',
                   '--tmpfs', '/tmp', '--proc', '/proc', '--dev', '/dev',
                   '--ro-bind', str(source), '/candidate.py', '--unshare-all', '--die-with-parent',
                   '--chdir', '/tmp', '/usr/bin/python3', '-I', '/candidate.py']
        with input_path.open('rb') as stdin, (directory/'stdout').open('w+b') as stdout, (directory/'stderr').open('w+b') as stderr:
            process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr,
                env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'OMP_NUM_THREADS': '1',
                     'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                start_new_session=True, preexec_fn=partial(_limits, timeout))
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
        if errors.startswith('bwrap:'):
            raise RuntimeError('candidate sandbox infrastructure failure: '+errors[:1000])
        timed_out |= process.returncode in (-signal.SIGXCPU, 128+signal.SIGXCPU)
        outcome = ('TIMEOUT' if timed_out else 'RUNTIME_EXCEPTION' if process.returncode else
                   'PASS' if output_matches(actual, test['expected']) else 'WRONG_OUTPUT')
        return {'outcome': outcome, 'stdout': actual[:4096], 'stderr': errors[:4096],
                'returncode': process.returncode, 'timed_out': timed_out,
                'stdout_truncated': len(actual) > 4096,
                'terminal_newline_added': not test['input'].endswith('\n')}
