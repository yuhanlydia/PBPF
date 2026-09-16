import json
from pathlib import Path

import pytest

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.config import ROOT, digest
from pbpf.apbpf.worker_io import WorkerIO


def fixture(tmp_path, monkeypatch):
    source_files = ['src/pbpf/apbpf/worker_io.py', 'src/pbpf/apbpf/codearc_bank.py']
    config = {'site': {'working_directory': str(ROOT)},
              'source_hashes': {n: file_sha(ROOT/n) for n in source_files}}
    fingerprint = digest(config)
    run = tmp_path/fingerprint
    upstream = run/'stages/materialize/attempt-000001'; outputs = upstream/'outputs'; outputs.mkdir(parents=True)
    public = outputs/'rbr-public'; public.mkdir(); (public/'tasks.jsonl').write_text('PUBLIC')
    private = outputs/'rbr-evaluator'; private.mkdir(); secret = private/'tasks.jsonl'; secret.write_text('SECRET')
    artifacts = [{'path': str(f.relative_to(upstream)), 'sha256': file_sha(f)}
                 for f in outputs.rglob('*') if f.is_file()]
    value = {'stage': 'materialize', 'fingerprint': fingerprint, 'artifacts': artifacts}
    result = upstream/'result.json'; result.write_text(json.dumps(value))
    complete = {'files': {**{a['path']: a['sha256'] for a in artifacts}, 'result.json': file_sha(result)}}
    (upstream/'complete.json').write_text(json.dumps(complete))
    attempt = run/'stages/hard_bank_lock/attempt-000001'; (attempt/'outputs').mkdir(parents=True)
    request = {'stage': 'hard_bank_lock', 'fingerprint': fingerprint, 'config': config,
               'confirmatory': False, 'claim_status': 'exploratory-predeclared',
               'dependencies': {'materialize': digest(complete)},
               'inputs': {'materialize': {'result': str(result), 'checksum': digest(complete)}},
               'outputs_directory': str(attempt/'outputs')}
    path = attempt/'request.json'; path.write_text(json.dumps(request))
    monkeypatch.setenv('APBPF_REQUEST', str(path)); monkeypatch.setenv('APBPF_OUTPUTS', str(attempt/'outputs'))
    monkeypatch.setenv('APBPF_RESULT', str(attempt/'worker-result.json'))
    return public, secret


def test_public_dependency_verification_never_reads_hidden_artifact(tmp_path, monkeypatch):
    public, private = fixture(tmp_path, monkeypatch)
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path == private:
            raise AssertionError('hidden artifact opened by public-only stage')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    io = WorkerIO('hard_bank_lock', [])
    assert io.directory('materialize', 'rbr-public') == public


def test_untracked_file_cannot_enter_public_mount(tmp_path, monkeypatch):
    public, _ = fixture(tmp_path, monkeypatch)
    io = WorkerIO('hard_bank_lock', [])
    (public/'unexpected-hidden.json').write_text('PRIVATE')
    with pytest.raises(ValueError, match='undeclared'):
        io.directory('materialize', 'rbr-public')


def test_changed_declared_dependency_bytes_are_rejected(tmp_path, monkeypatch):
    public, _ = fixture(tmp_path, monkeypatch)
    io = WorkerIO('hard_bank_lock', [])
    (public/'tasks.jsonl').write_text('CHANGED')
    with pytest.raises(ValueError, match='intact declared'):
        io.directory('materialize', 'rbr-public')
