import pytest
import importlib.util
from pathlib import Path
import sys
from pbpf.apbpf.rbr_execution import execute_stdin, output_matches, canonical_stdin


def test_numeric_and_line_contract():
    assert output_matches('1.00005\n', '1\n')
    assert not output_matches('1.01\n', '1\n')
    assert not output_matches('1 2\n', '1\n2\n')


def test_official_terminal_newline_prevents_readline_slicing_last_digit():
    assert canonical_stdin('7') == '7\n'
    assert canonical_stdin('7\n') == '7\n'
    assert canonical_stdin('') == '\n'
    result = execute_stdin('import sys\nprint(int(sys.stdin.readline()[:-1]))',
                           {'input': '7', 'expected': '7'})
    assert result['outcome'] == 'PASS' and result['terminal_newline_added'] is True


def test_legacy_prediction_preparation_uses_same_stdin_protocol():
    path = Path(__file__).resolve().parents[2] / 'scripts/run_rbr_prediction_gate.py'
    spec = importlib.util.spec_from_file_location('rbr_stdin_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = module._execute(('import sys\nprint(int(sys.stdin.readline()[:-1]))',
                               [{'input': '7', 'output': '7', 'id': 'regression'}], 2))
    assert results[0]['outcome'] == 'PASS'


def test_repair_execution_uses_same_stdin_protocol():
    path = Path(__file__).resolve().parents[2] / 'scripts/run_rbr_repair_gate.py'
    spec = importlib.util.spec_from_file_location('repair_stdin_contract', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    outcomes = module._execute('import sys\nprint(int(sys.stdin.readline()[:-1]))',
                               [{'input': '7', 'output': '7'}])
    assert outcomes == ['PASS']


@pytest.mark.parametrize('code,expected,outcome', [
    ('print(int(input()) + 1)', '3', 'PASS'),
    ('print(0)', '3', 'WRONG_OUTPUT'),
    ('raise ValueError("failed")', '3', 'RUNTIME_EXCEPTION'),
    ('if !!!', '3', 'COMPILE_ERROR'),
])
def test_real_stdin_execution_classifies(code, expected, outcome):
    assert execute_stdin(code, {'input': '2\n', 'expected': expected})['outcome'] == outcome


def test_timeout_and_external_files_are_unavailable(tmp_path):
    secret = tmp_path/'expected.txt'; secret.write_text('secret label')
    code = f'from pathlib import Path\nprint(Path({str(secret)!r}).exists())'
    assert execute_stdin(code, {'input': '', 'expected': 'False'})['outcome'] == 'PASS'
    assert execute_stdin('while True: pass', {'input': '', 'expected': ''}, timeout=.2)['outcome'] == 'TIMEOUT'
