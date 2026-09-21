import copy
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from pbpf.apbpf.config import ROOT
from pbpf.apbpf.config import digest, resolve_config
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_cache import build_stage_cache
from test_stage_cache import population
from test_hard_bank_stage_workers import fixture_stage, artifact_inventory, complete


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize('kind', ['belief', 'baselines'])
def test_primary_outcomes_never_change_fitted_weights_or_public_predictions(tmp_path, kind):
    worker = module('stage_training_under_test', 'run_apbpf_train_worker.py')
    helper = module('stage_training_helper_under_test', 'run_rbr_prediction_gate.py')
    payload = build_stage_cache(*population(), domain='rbr')
    changed = copy.deepcopy(payload)
    for row in changed['records']:
        if row['split'] == 'test':
            for test in row['tests'][4:]: test['outcome'] = 'PASS'
            row['outcomes'][4:] = ['PASS']*6
    protocol = {'difficulty_dim': 8, 'diagnosis_dim': 24, 'particles': 8,
                'lambda_assoc': 1., 'lambda_inv': .1, 'association_margin': .03}
    args = dict(kind=kind, seed=1701, steps=2, batch_size=8, learning_rate=3e-4,
                feature_dim=16, hidden_dim=16, protocol=protocol, helper=helper)
    first = worker.train_models(payload, tmp_path/'first', **args)
    second = worker.train_models(changed, tmp_path/'second', **args)
    assert first == second
    for path in (tmp_path/'first').glob('*.pt'):
        a = torch.load(path, weights_only=True)['model']
        b = torch.load(tmp_path/'second'/path.name, weights_only=True)['model']
        assert all(torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k] for k in a)
    with np.load(tmp_path/'first/primary-predictions.npz') as a, np.load(tmp_path/'second/primary-predictions.npz') as b:
        assert not np.array_equal(a['labels'], b['labels'])
        assert all(np.array_equal(a[k], b[k]) for k in a.files if k != 'labels')
        assert all(np.allclose(a[k].sum(-1), 1.) for k in a.files if k != 'labels')
    if kind == 'baselines':
        assert set(first['arms']) == {'pair_aware', 'deep_sets', 'no_particle_bottleneck', 'tuned_dirichlet'}
        assert json.loads((tmp_path/'first/tuned_dirichlet.json').read_text())['alpha'] in [.01,.1,.25,.5,1.,2.,5.,10.]
    assert first['config']['primary_used_for_training_or_selection'] is False


@pytest.mark.parametrize('kind', ['belief', 'baselines'])
def test_worker_emits_bound_checkpoints_for_both_domains_and_all_three_seeds(tmp_path, kind):
    config = resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory').config
    fingerprint = digest(config); run = tmp_path/fingerprint
    execution, value = fixture_stage(run, 'execution_cache', fingerprint)
    index = {'training_caches': {}}
    for domain in ('rbr', 'codearc'):
        path = execution/'outputs'/f'{domain}-full-cache.json'
        path.write_text(json.dumps(build_stage_cache(*population(), domain=domain)))
        index['training_caches'][domain] = {'full': {'path': path.name, 'sha256': file_sha(path)}}
    (execution/'outputs/execution-index.json').write_text(json.dumps(index))
    value['artifacts'] = artifact_inventory(execution)
    execution_input = complete(execution, value)
    gate, value = fixture_stage(run, 'hard_bank_gate', fingerprint)
    (gate/'outputs/fixture.json').write_text('{"scope":"contract fixture only"}')
    value['artifacts'] = artifact_inventory(gate)
    gate_input = complete(gate, value)
    stage = 'train_' + kind
    attempt, _ = fixture_stage(run, stage, fingerprint)
    inputs = {'execution_cache': execution_input, 'hard_bank_gate': gate_input}
    request = {'stage': stage, 'fingerprint': fingerprint, 'config': config,
               'confirmatory': False, 'claim_status': 'exploratory-predeclared',
               'inputs': inputs, 'dependencies': {k: v['checksum'] for k, v in inputs.items()},
               'outputs_directory': str(attempt/'outputs')}
    path = attempt/'request.json'; path.write_text(json.dumps(request))
    env = dict(os.environ, APBPF_REQUEST=str(path), APBPF_OUTPUTS=str(attempt/'outputs'),
               APBPF_RESULT=str(attempt/'worker-result.json'), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([sys.executable, str(ROOT/'scripts/run_apbpf_train_worker.py'),
        '--kind', kind, '--steps', '1', '--batch-size', '8', '--feature-dim', '16', '--hidden-dim', '16'],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    value = json.loads((attempt/'worker-result.json').read_text())
    assert value['dependencies'] == request['dependencies']
    assert {(r['domain'], r['seed']) for r in value['summary']['runs']} == {
        (d, s) for d in ('rbr', 'codearc') for s in (1701, 1702, 1703)}
    artifacts = {r['path']: r['sha256'] for r in value['artifacts']}
    assert len([p for p in artifacts if p.endswith('.pt')]) == (6 if kind == 'belief' else 18)
    assert all(file_sha(attempt/p) == h for p, h in artifacts.items())
