import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

from pbpf.apbpf.config import ROOT, resolve_config
from pbpf.apbpf.stages import doctor, report_run, run_pipeline


def test_partial_real_execution_keeps_exploratory_identity_and_full_verify_requirement(tmp_path):
    worker = tmp_path/'worker.py'
    worker.write_text('''import os,json,hashlib
from pathlib import Path
r=json.loads(Path(os.environ['APBPF_REQUEST']).read_text())
p=Path(os.environ['APBPF_OUTPUTS'])/'evidence.json';p.write_text('{"fixture":true}')
result={'schema':'apbpf-stage-result-v1','stage':r['stage'],'fingerprint':r['fingerprint'],
'dependencies':r['dependencies'],'summary':{'fixture':True},
'artifacts':[{'path':'outputs/evidence.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}]}
Path(os.environ['APBPF_RESULT']).write_text(json.dumps(result))
''')
    site = tmp_path/'site.yaml'
    site.write_text(yaml.safe_dump({'working_directory': str(ROOT),
        'paths': {k: str(tmp_path) for k in ('dataset_root', 'model_cache', 'artifact_cache')},
        'commands': {'materialize': [sys.executable, str(worker)]},
        'worker_files': {'materialize': str(worker)},
        'worker_revisions': {'materialize': hashlib.sha256(worker.read_bytes()).hexdigest()}}))
    resolved = resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory', site=site)
    assert doctor(resolved, through_stage='materialize')['ready']
    assert not doctor(resolved)['ready']
    run = tmp_path/resolved.fingerprint
    result = run_pipeline(resolved, run, through_stage='materialize')
    assert not result['complete'] and not result['main_table_eligible']
    assert result['claim_status'] == 'exploratory-predeclared'
    assert result['stages']['materialize']['confirmatory'] is False
    assert result['stages']['hard_bank_lock']['status'] == 'pending'
    assert run_pipeline(resolved, run, through_stage='materialize', resume=True) == result
    rerun = run_pipeline(resolved, run, through_stage='materialize', rerun_stage='materialize')
    assert rerun['stages']['materialize']['attempt'].endswith('attempt-000002')
    assert rerun['stages']['hard_bank_lock']['status'] == 'pending'
    with pytest.raises(ValueError, match='every DAG stage'):
        report_run(resolved, run, require_complete=True)


def test_explicit_prefix_does_not_bypass_failed_gate(tmp_path):
    from pbpf.apbpf.stages import GateFailure
    resolved = resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml', 'local_smoke')
    with pytest.raises(GateFailure):
        run_pipeline(resolved, tmp_path/resolved.fingerprint, through_stage='hard_bank_gate',
                     fail_smoke_gate='hard_bank_gate')
    attempt = tmp_path/resolved.fingerprint/'stages/hard_bank_gate/attempt-000001'
    failure = json.loads((attempt/'gate-failure.json').read_text())
    assert failure['retry_command'][-2:] == ['--through-stage', 'hard_bank_gate']
    assert failure['run_after_changes_command'][-2:] == ['--through-stage', 'hard_bank_gate']
