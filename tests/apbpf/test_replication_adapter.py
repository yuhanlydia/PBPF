import json
from pathlib import Path

import pytest

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.config import ROOT, resolve_config
from pbpf.apbpf.stages import STAGES
from test_stage_training import module


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    return module('replication_bridge_test', 'run_apbpf_replication_bridge.py')


def cache_fixture(tmp_path, bridge, monkeypatch):
    config = resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory').config
    model = next(m for m in config['models'].values() if m['role'] == 'replication')
    outputs = tmp_path/'outputs'; outputs.mkdir()
    tasks = {}
    for kind in ('public', 'evaluator'):
        directory = outputs/f'codearc-{kind}'; directory.mkdir()
        (directory/'tasks.jsonl').write_text(kind)
        tasks[kind] = {'tasks_sha256': file_sha(directory/'tasks.jsonl')}
    bank = tmp_path/'bank'; bank.mkdir()
    (bank/'run.json').write_text(json.dumps({'temperature': .8, 'top_p': .95, 'max_new_tokens': 512}))
    payload = {'problem_counts': {'train': 212, 'development': 400, 'test': 500},
               'counts': {'train': 1696, 'development': 3200, 'test': 4000},
               'evaluation_role': 'exploratory_locked_primary_assessment'}
    cache = tmp_path/'cache.json'; cache.write_text(json.dumps(payload, sort_keys=True)+'\n')
    rebuilt = {'generator': model, 'source_bindings': [{'bank': str(bank)}], 'task_bindings': tasks}
    proof = tmp_path/'proof.json'
    proof.write_text(json.dumps({**rebuilt, 'cache_sha256': file_sha(cache),
        'builder_sha256': file_sha(ROOT/'scripts/build_apbpf_full_replay_cache.py')}))
    entry = {'cache': str(cache), 'proof': str(proof), 'cache_sha256': file_sha(cache),
             'proof_sha256': file_sha(proof), 'evaluation_root': str(tmp_path),
             'public_root': str(tmp_path), 'evaluator_root': str(tmp_path)}
    monkeypatch.setattr(bridge, 'assemble', lambda *a, **kw: (payload, rebuilt))
    return entry, outputs, config, payload


def test_root_import_rebuilds_execution_and_preserves_exact_bytes(tmp_path, bridge, monkeypatch):
    entry, outputs, config, _ = cache_fixture(tmp_path, bridge, monkeypatch)
    result = bridge.import_cache(entry, 'codearc', outputs, config)
    assert file_sha(outputs/'replication-inputs/codearc/cache.json') == entry['cache_sha256']
    assert result['proof_sha256'] == entry['proof_sha256']


@pytest.mark.parametrize('attack', ['cache', 'proof', 'reconstructed_execution', 'current_materialization', 'decoding'])
def test_root_import_rejects_unbound_or_incompatible_inputs(tmp_path, bridge, monkeypatch, attack):
    entry, outputs, config, payload = cache_fixture(tmp_path, bridge, monkeypatch)
    if attack in ('cache', 'proof'):
        Path(entry[attack]).write_text('{}')
    elif attack == 'reconstructed_execution':
        payload['counts']['test'] = 3999
    elif attack == 'current_materialization':
        (outputs/'codearc-evaluator/tasks.jsonl').write_text('another source inventory')
    else:
        (tmp_path/'bank/run.json').write_text(json.dumps({'temperature': .1, 'top_p': .95, 'max_new_tokens': 512}))
    with pytest.raises(ValueError):
        bridge.import_cache(entry, 'codearc', outputs, config)


def test_complete_site_binds_all_workers_and_both_root_manifests(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    worker = module('full_replication_queue_test', 'run_local_apbpf_full_pipeline.py')
    candidate, replica = tmp_path/'candidate.json', tmp_path/'replica.json'
    candidate.write_text('candidate inventory'); replica.write_text('replication inventory')
    site = worker.make_site(ROOT, tmp_path, candidate, replica, '0')
    assert set(site['commands']) == set(STAGES)
    for stage, path in site['worker_files'].items():
        assert site['worker_revisions'][stage] == file_sha(path)
    command = site['commands']['materialize']
    assert command[command.index('--replication-cache-sha256')+1] == file_sha(replica)
    assert command[command.index('--candidate-cache-sha256')+1] == file_sha(candidate)
    path = tmp_path/'site.json'; path.write_text(json.dumps(site))
    resolved = resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory', site=path)
    assert len(resolved.config['site']['commands']) == 21


def test_queue_retries_partial_status_without_restarting_upstream(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    worker = module('full_replication_queue_status_test', 'run_local_apbpf_full_pipeline.py')
    path = tmp_path/'status.json'; path.write_text('{"status":')
    monkeypatch.setattr(worker.time, 'sleep', lambda _: path.write_text('{"status":"running"}'))
    assert worker.read_status(path)['status'] == 'running'


@pytest.mark.parametrize('attack', ['cache', 'seeds', 'sources', 'comparator'])
def test_replication_cell_rejects_mismatched_primary_evidence(attack):
    worker = module('replication_cell_test', 'run_apbpf_replication_worker.py')
    association = {'seeds': [1701, 1702, 1703], 'source_components': 500, 'primary_candidates': 4000,
        'cache_sha256': 'fixed', 'comparisons': {'outcome_shuffled': {'mean_nll_gap': -.02, 'ci95': [-.03, -.01]}}}
    selection = {'seeds': [1701, 1702, 1703], 'primary_sources': 500, 'primary_candidates': 4000,
        'bootstrap_draws': 10000, 'cache_sha256': 'fixed', 'comparator': 'strongest_cross_fitted_deterministic',
        'absolute_selected_pass1_advantage': -.04, 'ci95': [-.05, -.03]}
    kwargs = dict(dataset='codearc_replay', family='qwen', checksum='fixed', bindings={}, scope='test fixture')
    assert worker.make_cell(association, selection, **kwargs)['selection_advantage'] == -.04
    if attack == 'cache': selection['cache_sha256'] = 'other'
    elif attack == 'seeds': selection['seeds'] = [1701]
    elif attack == 'sources': association['source_components'] = 499
    else: selection['comparator'] = 'weak chosen comparator'
    with pytest.raises(ValueError):
        worker.make_cell(association, selection, **kwargs)
