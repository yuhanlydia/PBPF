"""Worker contract fixtures, not generated-bank experimental evidence."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.config import ROOT, digest, resolve_config
from pbpf.apbpf.stages import _validate_gate_decision


def complete(attempt, value):
    result = attempt/'result.json'; result.write_text(json.dumps(value))
    files = {f.relative_to(attempt).as_posix(): file_sha(f) for f in attempt.rglob('*') if f.is_file()}
    receipt = {'files': files}; (attempt/'complete.json').write_text(json.dumps(receipt))
    return {'result': str(result), 'checksum': digest(receipt)}


def fixture_stage(run, stage, fingerprint):
    attempt = run/'stages'/stage/'attempt-000001'; (attempt/'outputs').mkdir(parents=True)
    return attempt, {'stage': stage, 'fingerprint': fingerprint, 'artifacts': []}


def artifact_inventory(attempt):
    return [{'path': f.relative_to(attempt).as_posix(), 'sha256': file_sha(f)}
            for f in (attempt/'outputs').rglob('*') if f.is_file()]


def invoke(run, stage, config, inputs, script):
    attempt, _ = fixture_stage(run, stage, digest(config))
    request = {'stage': stage, 'fingerprint': digest(config), 'config': config,
               'confirmatory': False, 'claim_status': 'exploratory-predeclared',
               'inputs': inputs, 'dependencies': {k:v['checksum'] for k,v in inputs.items()},
               'outputs_directory': str(attempt/'outputs')}
    path = attempt/'request.json'; path.write_text(json.dumps(request))
    env = dict(os.environ, APBPF_REQUEST=str(path), APBPF_OUTPUTS=str(attempt/'outputs'),
               APBPF_RESULT=str(attempt/'worker-result.json'), CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, str(ROOT/'scripts'/script)], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    value = json.loads((attempt/'worker-result.json').read_text())
    return attempt, request, value


@pytest.mark.parametrize('mixed_development', [True, False])
def test_hard_bank_audit_and_gate_use_exact_locked_population(tmp_path, mixed_development):
    config = resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory').config
    fingerprint = digest(config); run = tmp_path/fingerprint
    lock_attempt, lock_value = fixture_stage(run, 'hard_bank_lock', fingerprint)
    groups = [{'group_id': f'{domain}/primary{i}', 'source_component_id': f'{domain}::primary{i}',
               'split': 'primary', 'candidate_ids': [f'{domain}/primary{i}/{j}' for j in range(8)],
               'visible_outcomes': [[1]*4] + [[0]*4 for _ in range(7)]}
              for domain in ('rbr', 'codearc') for i in range(500)]
    spec = importlib.util.spec_from_file_location('build_bank_fixture', ROOT/'scripts/build_apbpf_hard_bank.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.lock_bank(groups, provenance='test-fixture-no-scientific-claim',
                     output=lock_attempt/'outputs/population-lock.json')
    lock_value['artifacts'] = artifact_inventory(lock_attempt)
    lock_input = complete(lock_attempt, lock_value)
    execution, execution_value = fixture_stage(run, 'execution_cache', fingerprint)
    evaluations = []
    for domain in ('rbr', 'codearc'):
        for split in ('development', 'primary'):
            directory = f'{domain}-{split}'; target = execution/'outputs'/directory; target.mkdir()
            if split == 'primary':
                rows = [{'group_id': r['group_id'], 'candidate_ids': r['candidate_ids'],
                         'hidden_labels': [0, 1]+[0]*6} for r in groups if r['group_id'].startswith(domain+'/')]
            else:
                rows = [{'group_id': f'{domain}/dev{i}', 'source_component_id': f'dev{i}',
                         'split': split, 'candidate_ids': [f'{domain}/dev{i}/{j}' for j in range(8)],
                         'visible_outcomes': [[0]*4 for _ in range(8)],
                         'hidden_labels': ([0, 1]+[0]*6) if mixed_development else [0]*8}
                        for i in range(400)]
            (target/'bank_groups.json').write_text(json.dumps(rows))
            evaluations.append({'domain':domain, 'split':split, 'evaluation_directory':directory,
                                'phase': 'all' if split == 'development' else 'hidden'})
    (execution/'outputs/execution-index.json').write_text(json.dumps({
        'schema':'apbpf-generated-execution-cache-v1', 'evaluations': evaluations,
        'lock_stage_completion_sha256': lock_input['checksum']}))
    execution_value['artifacts'] = artifact_inventory(execution)
    execution_input = complete(execution, execution_value)
    audit_attempt, _, audit = invoke(run, 'hard_bank', config,
        {'hard_bank_lock':lock_input, 'execution_cache':execution_input}, 'run_apbpf_hard_bank_worker.py')
    audit_input = complete(audit_attempt, audit)
    _, request, result = invoke(run, 'hard_bank_gate', config,
        {'hard_bank_lock':lock_input, 'hard_bank':audit_input}, 'run_apbpf_hard_bank_gate_worker.py')
    assert result['gate']['passed'] is mixed_development
    assert result['gate']['metrics']['confirmatory_groups'] == 1000
    assert result['gate']['metrics']['mixed_pilot_groups'] == (800 if mixed_development else 0)
    assert result['gate']['metrics']['lock_artifact_sha256'] == file_sha(lock_attempt/'outputs/population-lock.json')
    _validate_gate_decision(result, {**request, 'backend': 'real'}, config)
