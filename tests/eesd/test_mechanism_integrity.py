"""Resume must bind sealed banks, caches and reports to the current protocol."""
import hashlib
import json
from pathlib import Path

import pytest

from test_mechanism_runner import ROOT, load_script


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def sealed(tmp_path):
    runner = load_script('run_eesd_matrix')
    public = tmp_path / 'public'
    tasks = [{'task_id': split, 'source_component_id': split, 'split': split} for split in ('development', 'primary')]
    public.mkdir()
    (public / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    write(public / 'manifest.json', {'public_tasks_sha256': sha(public / 'tasks.jsonl')})
    cell = {'domain': 'rbr', 'family': 'qwen25_7b', 'dataset': 'runbugrun', 'model': 'qwen25_7b',
            'public_root': str(public), 'evaluator_root': str(tmp_path / 'evaluator'), 'development_components': 200}
    banks = []
    for split in ('development', 'primary'):
        bank = tmp_path / split
        run = {'schema': 'apbpf-rbr-generation-v1', 'family': 'qwen25_7b',
            'model': 'Qwen/Qwen2.5-Coder-7B-Instruct', 'revision': 'c03e6d358207e414f1eca0bb1891e29f1db0e242',
            'adapter_sha256': None, 'seed': 1701, 'split': split, 'candidates': 1, 'offset': 0,
            'components': 1, 'task_ids': [split], 'source_component_ids': [split],
            'source_sha256': sha(ROOT / 'scripts/generate_apbpf_rbr_bank.py'),
            'prompt_source_sha256': sha(ROOT / 'src/pbpf/apbpf/rbr_prompt.py'),
            'inventory_source_sha256': sha(ROOT / 'src/pbpf/apbpf/codearc_bank.py'),
            'public_manifest_sha256': sha(public / 'manifest.json'), 'public_tasks_sha256': sha(public / 'tasks.jsonl'),
            'decode_policy': 'temperature0.8-top_p0.95', 'temperature': .8, 'top_p': .95,
            'max_input_tokens': 4096, 'max_new_tokens': 1024}
        write(bank / 'run.json', run)
        write(bank / f'{split}.json', {**tasks[0 if split == 'development' else 1],
            'candidates': [{'candidate_id': split + '/candidate', 'code': 'pass'}]})
        write(bank / 'complete.json', {'run_sha256': sha(bank / 'run.json'), 'files': {f'{split}.json': sha(bank / f'{split}.json')}})
        banks.append((bank, run, sha(bank / 'complete.json')))
    evaluator = Path(cell['evaluator_root'])
    evaluator.mkdir()
    (evaluator / 'tasks.jsonl').write_text('[]\n')
    write(evaluator / 'manifest.json', {'evaluator_tasks_sha256': sha(evaluator / 'tasks.jsonl')})
    cache = tmp_path / 'cache.json'
    write(cache, {'schema': 'eesd-public-query-mechanism-cache-v1', 'dataset': 'rbr',
        'generator_identity': [banks[0][1]['model'], banks[0][1]['revision'], None, 1701],
        'bank_bindings': [{'split': run['split'], 'path': str(path), 'complete_sha256': digest, 'components': 1}
                          for path, run, digest in banks],
        'evaluator_manifest_sha256': sha(evaluator / 'manifest.json'), 'records': []})
    return runner, cell, banks, cache


@pytest.mark.parametrize('field,value', [('seed', 1702), ('model', 'wrong'), ('source_sha256', 'old'),
    ('public_tasks_sha256', 'wrong'), ('decode_policy', 'greedy')])
def test_completed_bank_rejects_wrong_current_identity(sealed, field, value):
    runner, cell, banks, _ = sealed
    bank, run, _ = banks[0]
    runner.verify_mechanism_bank(bank, cell=cell, seed=1701, split='development', root=ROOT)
    run[field] = value
    write(bank / 'run.json', run)
    complete = json.loads((bank / 'complete.json').read_text())
    complete['run_sha256'] = sha(bank / 'run.json')
    write(bank / 'complete.json', complete)
    with pytest.raises(ValueError, match='identity'):
        runner.verify_mechanism_bank(bank, cell=cell, seed=1701, split='development', root=ROOT)


def test_cache_requires_seal_and_rejects_changed_content(sealed):
    runner, cell, banks, cache = sealed
    with pytest.raises(ValueError, match='seal'):
        runner.verify_mechanism_cache(cache, cell=cell, banks=banks, root=ROOT)
    runner.verify_mechanism_cache(cache, cell=cell, banks=banks, root=ROOT, seal=True)
    runner.verify_mechanism_cache(cache, cell=cell, banks=banks, root=ROOT)
    payload = json.loads(cache.read_text())
    payload['records'] = [{'tampered': True}]
    write(cache, payload)
    with pytest.raises(ValueError, match='checksum|binding'):
        runner.verify_mechanism_cache(cache, cell=cell, banks=banks, root=ROOT)


def test_cache_rejects_seed_or_bank_binding_mismatch(sealed):
    runner, cell, banks, cache = sealed
    payload = json.loads(cache.read_text())
    payload['generator_identity'][-1] = 1702
    write(cache, payload)
    with pytest.raises(ValueError, match='identity|binding'):
        runner.verify_mechanism_cache(cache, cell=cell, banks=banks, root=ROOT, seal=True)


def test_report_reuse_requires_byte_integrity_and_current_protocol(sealed, tmp_path):
    runner, cell, _, cache = sealed
    config = ROOT / 'configs/experiments/eesd_iclr2027.yaml'
    output = tmp_path / 'report'
    report = dict(schema='eesd-evidence-matrix-v1', dataset='runbugrun', model='qwen25_7b', seed=1701,
        visible=4, validation_split='development', assessment_split='primary', cache_sha256=sha(cache),
        config_sha256=sha(config), runner_source_sha256=sha(ROOT / 'scripts/run_eesd_evidence_matrix.py'),
        evidence_source_sha256=sha(ROOT / 'src/pbpf/eesd/evidence.py'))
    write(output / 'report.json', report)
    (output / 'predictions.npz').write_bytes(b'predictions fixture')
    complete = dict(report_sha256=sha(output / 'report.json'), predictions_sha256=sha(output / 'predictions.npz'))
    write(output / 'complete.json', complete)
    kwargs = dict(cache=cache, config=config, cell=cell, seed=1701, root=ROOT)
    runner.verify_mechanism_report(output, **kwargs)
    with pytest.raises(ValueError, match='identity'):
        runner.verify_mechanism_report(output, **{**kwargs, 'seed': 1702})
    (output / 'predictions.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        runner.verify_mechanism_report(output, **kwargs)


@pytest.mark.parametrize('components,expected_tasks', [(0, ['y-task', 'a-task']), (1, ['y-task'])])
def test_codearc_generated_bank_preserves_materializer_source_order(tmp_path, monkeypatch, components, expected_tasks):
    """Exercise the real generator's inventory lock before the actor-load boundary."""
    import builtins
    import sys

    generator = load_script('generate_apbpf_codearc_bank')
    runner = load_script('run_eesd_matrix')
    public = tmp_path / 'public'
    public.mkdir()
    tasks = [dict(task_id=task, source_component_id=source, split='development',
        protocol='CodeARC-Replay', task_text='fixture',
        visible_tests=[{'id': str(i), 'input': 'x', 'expected': 'x'} for i in range(4)])
        for task, source in [('z-task', 'first-source'), ('a-task', 'second-source'), ('y-task', 'first-source')]]
    (public / 'tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    write(public / 'manifest.json', {'public_tasks_sha256': sha(public / 'tasks.jsonl')})
    bank = tmp_path / 'bank'
    monkeypatch.setattr(sys, 'argv', ['generator', '--public-root', str(public), '--output', str(bank),
        '--family', 'qwen25_7b', '--split', 'development', '--components', str(components),
        '--candidates', '1', '--seed', '1701'])
    original_import = builtins.__import__
    def stop_before_actor(name, *args, **kwargs):
        if name == 'torch':
            raise RuntimeError('actor loading boundary')
        return original_import(name, *args, **kwargs)
    with monkeypatch.context() as boundary:
        boundary.setattr(builtins, '__import__', stop_before_actor)
        with pytest.raises(RuntimeError, match='actor loading boundary'):
            generator.main()
    run = json.loads((bank / 'run.json').read_text())
    assert run['task_ids'] == expected_tasks
    files = {}
    for task, source in zip(run['task_ids'], run['source_component_ids'], strict=True):
        path = bank / f'{task}.json'
        write(path, dict(task_id=task, source_component_id=source, split='development',
            candidates=[{'candidate_id': task + '/candidate', 'code': 'pass'}]))
        files[path.name] = sha(path)
    write(bank / 'complete.json', dict(run_sha256=sha(bank / 'run.json'), files=files))
    cell = dict(domain='codearc', family='qwen25_7b', public_root=str(public), development_components=components)
    _, validated, _ = runner.verify_mechanism_bank(bank, cell=cell, seed=1701, split='development', root=ROOT)
    assert validated['task_ids'] == expected_tasks
    assert run['max_input_tokens'] == 4096
    assert run['prompt_policy'] == 'adaptively cap public fields while retaining all four example slots'
    assert run['codearc_prompt_source_sha256'] == sha(ROOT / 'src/pbpf/apbpf/codearc_prompt.py')
    for field, wrong in [('max_input_tokens', 8192), ('prompt_policy', 'truncate token tail'),
                         ('codearc_prompt_source_sha256', 'stale-prompt-source')]:
        write(bank / 'run.json', {**run, field: wrong})
        write(bank / 'complete.json', dict(run_sha256=sha(bank / 'run.json'), files=files))
        with pytest.raises(ValueError, match='identity'):
            runner.verify_mechanism_bank(bank, cell=cell, seed=1701, split='development', root=ROOT)
