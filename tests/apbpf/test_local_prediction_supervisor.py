import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[2] / 'scripts/run_local_codearc_prediction.py'
    spec = importlib.util.spec_from_file_location('local_codearc_prediction', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_partial_upstream_status_is_retryable_and_complete_failure_is_preserved(tmp_path):
    worker = module()
    path = tmp_path / 'status.json'
    assert worker.read_snapshot(path) is None
    path.write_text('{"status":')
    assert worker.read_snapshot(path) is None
    path.write_text('{"status":"needs_debug","returncode":1}')
    assert worker.read_snapshot(path) == {'status': 'needs_debug', 'returncode': 1}


def test_status_publication_replaces_complete_snapshot(tmp_path):
    worker = module()
    path = tmp_path / 'status.json'
    worker.atomic_write(path, {'status': 'running'})
    worker.atomic_write(path, {'status': 'complete', 'returncode': 0})
    assert worker.read_snapshot(path) == {'status': 'complete', 'returncode': 0}
    assert not path.with_suffix('.partial').exists()
