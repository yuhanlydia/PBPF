import hashlib
import json
from pathlib import Path
import subprocess
import sys

from pbpf.apbpf.codearc_bank import file_sha


def test_visible_execution_and_foreign_lock_rejected_before_private_open(tmp_path):
    root = Path(__file__).resolve().parents[2]
    public, bank = tmp_path/'public', tmp_path/'bank'
    public.mkdir(); bank.mkdir()
    row = {'task_id': 'RBR/p1', 'source_component_id': 'p1', 'split': 'primary',
           'visible_tests': [{'id': str(i), 'input': '2\n', 'expected': '2'} for i in range(4)]}
    (public/'tasks.jsonl').write_text(json.dumps(row)+'\n')
    public_hash = file_sha(public/'tasks.jsonl')
    (public/'manifest.json').write_text(json.dumps({'schema': 'apbpf-rbr-generated-materialization-v1',
                                                  'public_tasks_sha256': public_hash}))
    candidate = {k: row[k] for k in ('task_id', 'source_component_id', 'split')}
    candidate['candidates'] = [{'candidate_id': f'c{i}', 'code': 'print(input())'} for i in range(8)]
    (bank/'RBR-p1.json').write_text(json.dumps(candidate))
    (bank/'run.json').write_text(json.dumps({'schema': 'apbpf-rbr-generation-v1', 'task_ids': ['RBR/p1'],
        'source_component_ids': ['p1'], 'split': 'primary', 'candidates': 8, 'public_tasks_sha256': public_hash}))
    (bank/'complete.json').write_text(json.dumps({'run_sha256': file_sha(bank/'run.json'),
        'files': {'RBR-p1.json': file_sha(bank/'RBR-p1.json')}}))
    base = [sys.executable, str(root/'scripts/evaluate_apbpf_rbr_bank.py'), '--bank', str(bank)]
    visible = subprocess.run(base+['--public-root', str(public), '--phase', 'visible', '--workers', '2',
                                  '--output', str(tmp_path/'visible')], capture_output=True, text=True)
    assert visible.returncode == 0, visible.stderr
    result = json.loads((tmp_path/'visible/results.json').read_text())
    assert result['test_passes'] == 32 and result['all_pass'] == 8
    lock = {'schema': 'apbpf-hard-bank-lock-v2',
            'population_lock': {'provenance': 'codearc-bank-sha256:'+file_sha(bank/'complete.json')},
            'groups': json.loads((tmp_path/'visible/bank_groups.json').read_text())}
    lock['content_sha256'] = hashlib.sha256(json.dumps(lock, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (tmp_path/'wrong-lock.json').write_text(json.dumps(lock))
    hidden = subprocess.run(base+['--evaluator-root', str(tmp_path/'MUST_NOT_OPEN'), '--phase', 'hidden',
        '--population-lock', str(tmp_path/'wrong-lock.json'), '--output', str(tmp_path/'hidden')], capture_output=True, text=True)
    assert hidden.returncode != 0
    assert 'pre-hidden lock binds a different candidate bank' in hidden.stderr
    assert not (tmp_path/'hidden').exists()
