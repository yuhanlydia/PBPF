#!/usr/bin/env python3
"""Execute declared train/development banks and prelocked primary hidden calls."""
import json

from pbpf.apbpf.codearc_bank import file_sha, load_bank, verify_hidden_lock
from pbpf.apbpf.worker_io import WorkerIO, bank_inventory


def main():
    io = WorkerIO('execution_cache', [
        'scripts/run_apbpf_execution_cache_worker.py', 'scripts/evaluate_apbpf_codearc_bank.py',
        'scripts/evaluate_apbpf_rbr_bank.py', 'src/pbpf/apbpf/codearc_execution.py',
        'src/pbpf/apbpf/rbr_execution.py'])
    imported = bank_inventory(io)
    locks = json.loads(io.artifact('hard_bank_lock', 'locks.json').read_text())
    if set(locks) != {'rbr', 'codearc'}:
        raise ValueError('both domain locks required before any private task read')
    lock_paths = {}
    # Complete both checks before opening even one evaluator task file.
    for domain, item in locks.items():
        bank = io.directory('materialize', 'candidate-banks/'+item['bank_directory'])
        run, rows, checksum = load_bank(bank)
        lock = io.artifact('hard_bank_lock', item['lock'])
        actual = verify_hidden_lock(lock, rows, checksum, provenance_prefix=domain+'-bank-sha256:')
        if (actual != item['lock_sha256'] or checksum != item['bank_complete_sha256']
                or run['split'] != 'primary' or len(rows) != 500):
            raise ValueError('primary lock and declared bank identity differ')
        lock_paths[domain] = lock
    python = str(io.root/'.venv/bin/python')
    evaluations = []
    for entry in imported:
        domain, name = entry['domain'], entry['directory']
        bank = io.directory('materialize', 'candidate-banks/'+name)
        _, _, checksum = load_bank(bank)
        if checksum != entry['complete_sha256']:
            raise ValueError('imported candidate inventory changed')
        evaluator = io.directory('materialize', f'{domain}-evaluator')
        primary = entry['split'] == 'primary'
        output = io.outputs/name
        command = [python, '-u', f'scripts/evaluate_apbpf_{domain}_bank.py', '--evaluator-root',
                   str(evaluator), '--bank', str(bank), '--phase', 'hidden' if primary else 'all',
                   '--output', str(output), '--workers', '12']
        if primary:
            if name != locks[domain]['bank_directory']:
                raise ValueError('primary bank was not part of the exact lock stage')
            command += ['--population-lock', str(lock_paths[domain])]
        io.execute(name, command)
        result = json.loads((output/'results.json').read_text())
        evaluations.append({**entry, 'evaluation_directory': name,
            'results_sha256': file_sha(output/'results.json'),
            'phase': result['phase'], 'tests': result['tests'], 'test_passes': result['test_passes']})
    (io.outputs/'execution-index.json').write_text(json.dumps({
        'schema': 'apbpf-generated-execution-cache-v1', 'evaluations': evaluations,
        'lock_stage_completion_sha256': io.request['dependencies']['hard_bank_lock'],
        'pre_hidden_locks': {d: file_sha(p) for d, p in lock_paths.items()}}, indent=2)+'\n')
    io.finish({'actual_candidate_execution': True, 'evaluation_runs': len(evaluations),
               'tests': sum(e['tests'] for e in evaluations),
               'hidden_access_after_both_primary_locks': True,
               'scope': 'exploratory execution evidence; no hard-bank gate decision'})


if __name__ == '__main__':
    main()
