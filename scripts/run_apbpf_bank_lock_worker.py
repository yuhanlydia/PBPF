#!/usr/bin/env python3
"""Execute public primary tests and lock both complete declared populations."""
import json

from pbpf.apbpf.codearc_bank import file_sha, load_bank
from pbpf.apbpf.worker_io import WorkerIO, bank_inventory


def main():
    io = WorkerIO('hard_bank_lock', [
        'scripts/run_apbpf_bank_lock_worker.py', 'scripts/build_apbpf_hard_bank.py',
        'scripts/apbpf_visible_evaluator_sandbox.sh', 'scripts/evaluate_apbpf_codearc_bank.py',
        'scripts/evaluate_apbpf_rbr_bank.py', 'src/pbpf/apbpf/codearc_execution.py',
        'src/pbpf/apbpf/rbr_execution.py', 'src/pbpf/apbpf/hard_bank.py'])
    imported = bank_inventory(io)
    primary = [entry for entry in imported if entry['split'] == 'primary']
    if len(primary) != 2 or {entry['domain'] for entry in primary} != {'rbr', 'codearc'}:
        raise ValueError('exactly one complete primary bank per domain is required')
    combined, locks = [], {}
    python = str(io.root/'.venv/bin/python')
    for entry in sorted(primary, key=lambda x: x['domain']):
        domain = entry['domain']
        public = io.directory('materialize', f'{domain}-public')
        bank = io.directory('materialize', 'candidate-banks/'+entry['directory'])
        run, rows, checksum = load_bank(bank)
        if checksum != entry['complete_sha256'] or len(rows) != 500 or run['candidates'] != 8:
            raise ValueError('primary population differs from the declared 500-by-8 inventory')
        visible = io.outputs/f'{domain}-visible'; visible.mkdir()
        io.execute(f'{domain}-visible', ['bash', 'scripts/apbpf_visible_evaluator_sandbox.sh',
            str(public), str(bank), str(visible), python, '-u', f'scripts/evaluate_apbpf_{domain}_bank.py',
            '--public-root', '/input', '--bank', '/bank', '--phase', 'visible',
            '--output', '/output/evaluation', '--workers', '12'])
        groups = visible/'evaluation/bank_groups.json'
        lock = io.outputs/f'{domain}-population-lock.json'
        io.execute(f'{domain}-lock', [python, 'scripts/build_apbpf_hard_bank.py', 'lock',
            '--visible-bank', str(groups), '--provenance', domain+'-bank-sha256:'+checksum,
            '--output', str(lock)])
        rows = json.loads(groups.read_text())
        combined += [{**row, 'source_component_id': domain+'::'+row['source_component_id']} for row in rows]
        locks[domain] = {'bank_directory': entry['directory'], 'bank_complete_sha256': checksum,
                         'lock': lock.name, 'lock_sha256': file_sha(lock), 'groups': 500}
    combined_path = io.outputs/'combined-visible-bank.json'
    combined_path.write_text(json.dumps(combined, indent=2)+'\n')
    io.execute('combined-lock', [python, 'scripts/build_apbpf_hard_bank.py', 'lock',
        '--visible-bank', str(combined_path), '--provenance', 'apbpf-stage:'+io.request['fingerprint'],
        '--output', str(io.outputs/'population-lock.json')])
    (io.outputs/'locks.json').write_text(json.dumps(locks, indent=2)+'\n')
    io.finish({'actual_public_execution': True, 'hidden_execution': False,
               'domains': locks, 'primary_groups': 1000, 'candidates': 8000,
               'scope': 'exploratory full-population lock; no hidden-quality or efficacy claim'})


if __name__ == '__main__':
    main()
