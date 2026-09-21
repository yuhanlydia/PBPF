"""Locked public EvalPlus generation must never consume evaluator-only fields."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_mechanism_runner import ROOT, load_script


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def receipt(tmp_path):
    entries = []
    for dataset, version, prefix in [('HumanEvalPlus', 'v0.1.10', 'HumanEval'), ('MbppPlus', 'v0.2.0', 'Mbpp')]:
        rows = [dict(task_id=f'{prefix}/{i}', prompt=f'def f{i}():\n    """Public task."""',
            entry_point=f'f{i}', canonical_solution='PRIVATE_SOLUTION', base_input=['PRIVATE_BASE'],
            plus_input=['PRIVATE_PLUS'], contract='PRIVATE_CONTRACT') for i in range(2)]
        path = tmp_path / f'{dataset}.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        entries.append(dict(dataset=dataset, version=version, path=str(path), sha256=sha(path),
            bytes=path.stat().st_size, tasks=2, task_ids=[r['task_id'] for r in rows], checksum_kind='local provenance'))
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(dict(schema='eesd-evalplus-download-v1', evalplus_version='0.3.1', datasets=entries)))
    return path


def test_materializer_redacts_private_fields_and_public_loader_never_reads_raw(receipt, tmp_path):
    materializer = load_script('materialize_eesd_evalplus')
    materializer.materialize(receipt, tmp_path / 'data')
    from pbpf.eesd.evalplus_public import load_public_dataset
    public = tmp_path / 'data/public'
    lock = sha(public / 'manifest.json')
    receipt_value = json.loads(receipt.read_text())
    for entry in receipt_value['datasets']:
        Path(entry['path']).unlink()  # Generation cannot depend on the original raw file.
    for dataset in ('humaneval', 'mbpp'):
        rows, binding = load_public_dataset(public, dataset, lock)
        assert len(rows) == 2
        assert all(set(row) == {'task_id', 'prompt', 'entry_point'} for row in rows)
        assert b'PRIVATE_' not in (public / f'{dataset}.jsonl').read_bytes()
        assert 'PRIVATE_SOLUTION' in (tmp_path / f'data/private/{dataset}.jsonl').read_text()
        assert binding['evalplus_version'] == '0.3.1'
        assert len(binding['raw_source_sha256']) == 64


@pytest.mark.parametrize('change', ['raw', 'coverage', 'version'])
def test_materializer_rejects_raw_tamper_or_incomplete_receipt(receipt, tmp_path, change):
    materializer = load_script('materialize_eesd_evalplus')
    value = json.loads(receipt.read_text())
    if change == 'raw':
        Path(value['datasets'][0]['path']).write_text('{}\n')
    elif change == 'coverage':
        value['datasets'][0]['task_ids'] = ['HumanEval/0']
        receipt.write_text(json.dumps(value))
    else:
        value['datasets'][0]['version'] = 'latest'
        receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        materializer.materialize(receipt, tmp_path / 'data')
    assert not (tmp_path / 'data').exists()


@pytest.mark.parametrize('change', ['file', 'manifest', 'private_field', 'coverage'])
def test_public_loader_rejects_tamper_private_fields_and_missing_tasks(receipt, tmp_path, change):
    materializer = load_script('materialize_eesd_evalplus')
    materializer.materialize(receipt, tmp_path / 'data')
    from pbpf.eesd.evalplus_public import load_public_dataset
    public = tmp_path / 'data/public'
    manifest_path = public / 'manifest.json'
    locked_sha = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    path = public / 'humaneval.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == 'manifest':
        manifest['evalplus_version'] = 'latest'
        manifest_path.write_text(json.dumps(manifest))
    else:
        if change == 'private_field':
            rows[0]['canonical_solution'] = 'hidden'
        else:
            rows.pop()
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        if change in {'private_field', 'coverage'}:
            manifest['datasets']['humaneval']['public_tasks_sha256'] = sha(path)
            manifest_path.write_text(json.dumps(manifest))
            locked_sha = sha(manifest_path)
    with pytest.raises(ValueError):
        load_public_dataset(public, 'humaneval', locked_sha)


def test_transfer_cli_refuses_unisolated_evaluation_before_model_load(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_eesd_evalplus_transfer.py'),
        '--dataset', 'humaneval', '--model-config', str(tmp_path / 'absent.yaml'), '--output', str(tmp_path / 'out'),
        '--public-data-root', str(tmp_path), '--public-manifest-sha256', 'a' * 64, '--evaluate'],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'isolated' in result.stderr
    assert not (tmp_path / 'out').exists()


def test_evalplus_summary_requires_complete_locked_coverage(tmp_path):
    runner = load_script('run_eesd_evalplus_transfer')
    path = tmp_path / 'eval.json'
    path.write_text(json.dumps({'eval': {'HumanEval/0': [{'base_status': 'pass', 'plus_status': 'pass'}]}}))
    with pytest.raises(ValueError, match='coverage'):
        runner.summarize_evalplus(path, expected_task_ids=['HumanEval/0', 'HumanEval/1'])


def test_transfer_cli_requires_explicit_public_lock(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_eesd_evalplus_transfer.py'),
        '--dataset', 'humaneval', '--model-config', str(tmp_path / 'model.yaml'), '--output', str(tmp_path / 'out')],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert '--public-data-root' in result.stderr
    assert '--public-manifest-sha256' in result.stderr
    assert not (tmp_path / 'out').exists()


def test_downstream_transfer_routes_public_only_and_never_requests_execution(receipt, tmp_path, monkeypatch):
    import yaml
    materializer = load_script('materialize_eesd_evalplus')
    materializer.materialize(receipt, tmp_path / 'data')
    public = tmp_path / 'data/public'
    runner = load_script('run_eesd_downstream')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'schema': 'eesd-iclr2027-v1', 'seeds': [1701],
        'distillation': {'transition_utility': [1, -1, 0, 0]}}))  # Fixture, not a scientific default.
    model = tmp_path / 'model.yaml'
    model.write_text('model_id: test/model\nrevision: ' + 'a' * 40 + '\n')
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(yaml.safe_dump({'schema': 'eesd-cache-manifest-v1', 'transfer_cells': [
        {'dataset': 'humaneval', 'model': 'qwen25_7b', 'model_config': str(model),
         'source_dataset': 'runbugrun', 'source_round': 1}]}))
    output = tmp_path / 'out'
    for rule in ('equal_weight', 'final_correctness', 'fixed_mass_dirichlet', 'eesd_full'):
        adapter = output / f'training/runbugrun/qwen25_7b/round1/{rule}/seed1701/adapter'
        adapter.mkdir(parents=True)
        (adapter / 'adapter_model.safetensors').write_bytes(b'adapter fixture')
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        def value(flag):
            return command[command.index(flag) + 1]
        from pbpf.eesd.evalplus_public import load_public_dataset
        rows, data_lock = load_public_dataset(public, 'humaneval', sha(public / 'manifest.json'))
        directory = Path(value('--output'))
        directory.mkdir(parents=True)
        samples = directory / 'samples.jsonl'
        samples.write_text(''.join(json.dumps({'task_id': row['task_id'], 'solution': 'pass'}) + '\n' for row in rows))
        report = dict(schema='eesd-evalplus-transfer-v1', dataset='humaneval', model_id='test/model', revision='a'*40,
            model_config_sha256=sha(model), generator_source_sha256=sha(ROOT / 'scripts/run_eesd_evalplus_transfer.py'),
            adapter_sha256=runner.adapter_tree_digest(Path(value('--adapter'))) if '--adapter' in command else None,
            data_lock=data_lock, tasks=len(rows), max_new_tokens=1024, samples_sha256=sha(samples),
            evaluation_status='pending-isolated-official-evaluation', claim_status='generation-only-no-efficacy-claim')
        (directory / 'report.json').write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, 'run', launch)
    monkeypatch.setattr(sys, 'argv', ['runner', '--config', str(config), '--manifest', str(manifest),
        '--output', str(output), '--stage', 'transfer', '--public-data-root', str(public),
        '--public-manifest-sha256', sha(public / 'manifest.json')])
    runner.main()
    assert len(commands) == 5
    assert all(Path(c[1]).name == 'run_eesd_evalplus_transfer.py' for c in commands)
    assert all(c[c.index('--public-data-root') + 1] == str(public) for c in commands)
    assert all(c[c.index('--public-manifest-sha256') + 1] == sha(public / 'manifest.json') for c in commands)
    assert all('--evaluate' not in c for c in commands)
    record = json.loads(next(output.glob('downstream-launches/*/launch.json')).read_text())
    assert record['status'] == 'generated-awaiting-isolated-evaluation'


def test_generation_preflight_uses_public_files_without_evalplus_data_import(receipt, tmp_path, monkeypatch):
    import builtins
    materializer = load_script('materialize_eesd_evalplus')
    materializer.materialize(receipt, tmp_path / 'data')
    public = tmp_path / 'data/public'
    for entry in json.loads(receipt.read_text())['datasets']:
        Path(entry['path']).unlink()
    for path in (tmp_path / 'data/private').iterdir():
        path.unlink()
    model = tmp_path / 'model.yaml'
    model.write_text('model_id: test/model\nrevision: ' + 'a' * 40 + '\n')
    runner = load_script('run_eesd_evalplus_transfer')
    monkeypatch.setattr(sys, 'argv', ['transfer', '--dataset', 'humaneval', '--model-config', str(model),
        '--output', str(tmp_path / 'out'), '--public-data-root', str(public),
        '--public-manifest-sha256', sha(public / 'manifest.json')])
    original_import = builtins.__import__
    def stop_at_model(name, *args, **kwargs):
        if name == 'evalplus' or name.startswith('evalplus.'):
            pytest.fail('generation must never load EvalPlus private data APIs')
        if name == 'torch':
            raise RuntimeError('model boundary reached with public-only files')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', stop_at_model)
    with pytest.raises(RuntimeError, match='public-only'):
        runner.main()
