#!/usr/bin/env python3
"""Execute declared train/development banks and prelocked primary hidden calls."""
import json

from pbpf.apbpf.codearc_bank import file_sha, load_bank, verify_hidden_lock
from pbpf.apbpf.worker_io import WorkerIO, bank_inventory
from pbpf.apbpf.stage_cache import build_stage_cache, join_primary_executions
from pbpf.apbpf.development import development_cache
from pbpf.apbpf.repair_materials import build_repair_materials


def main():
    io = WorkerIO('execution_cache', [
        'scripts/run_apbpf_execution_cache_worker.py', 'scripts/evaluate_apbpf_codearc_bank.py',
        'scripts/evaluate_apbpf_rbr_bank.py', 'src/pbpf/apbpf/codearc_execution.py',
        'src/pbpf/apbpf/rbr_execution.py', 'src/pbpf/apbpf/stage_cache.py',
        'src/pbpf/apbpf/development.py', 'src/pbpf/apbpf/codearc_prompt.py', 'src/pbpf/real_gate.py',
        'src/pbpf/apbpf/repair_materials.py'])
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
    cache_inputs = {d: {'banks': [], 'executions': [], 'bindings': []} for d in locks}
    for entry in imported:
        domain, name = entry['domain'], entry['directory']
        bank = io.directory('materialize', 'candidate-banks/'+name)
        run, rows, checksum = load_bank(bank)
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
        if (result['bank_complete_sha256'] != checksum or result['phase'] != ('hidden' if primary else 'all')
                or result['split'] != run['split']
                or result['task_manifest_sha256'] != file_sha(evaluator/'manifest.json')
                or result['reference_control']
                or (domain == 'rbr' and result.get('stdin_policy') != 'official-terminal-newline-if-missing')):
            raise ValueError('execution result identity/protocol mismatch')
        measured = result['records']
        binding = {'bank_complete_sha256': checksum, 'result_sha256': file_sha(output/'results.json')}
        if primary:
            if result['population_lock_sha256'] != file_sha(lock_paths[domain]):
                raise ValueError('primary hidden execution used another population lock')
            visible_path = io.artifact('hard_bank_lock', f'{domain}-visible/evaluation/results.json')
            visible = json.loads(visible_path.read_text())
            public = io.directory('materialize', f'{domain}-public')
            if (visible['phase'] != 'visible' or visible['reference_control']
                    or visible['bank_complete_sha256'] != checksum
                    or visible['task_manifest_sha256'] != file_sha(public/'manifest.json')):
                raise ValueError('primary visible execution identity mismatch')
            measured = join_primary_executions(visible['records'], measured)
            binding.update(visible_result_sha256=file_sha(visible_path), lock_sha256=file_sha(lock_paths[domain]))
        cache_inputs[domain]['banks'] += rows
        cache_inputs[domain]['executions'] += measured
        cache_inputs[domain]['bindings'].append(binding)
        evaluations.append({**entry, 'evaluation_directory': name,
            'results_sha256': file_sha(output/'results.json'),
            'phase': result['phase'], 'tests': result['tests'], 'test_passes': result['test_passes']})
    caches, repair = {}, {}
    for domain, inputs in cache_inputs.items():
        task_sets = []
        for kind in ('public', 'evaluator'):
            directory = io.directory('materialize', f'{domain}-{kind}')
            manifest = json.loads((directory/'manifest.json').read_text())
            if file_sha(directory/'tasks.jsonl') != manifest[f'{kind}_tasks_sha256']:
                raise ValueError('materialized task checksum mismatch')
            with (directory/'tasks.jsonl').open() as stream:
                task_sets.append({r['task_id']: r for r in map(json.loads, stream)})
        full = build_stage_cache(inputs['banks'], inputs['executions'], *task_sets, domain=domain)
        full.update(execution_bindings=inputs['bindings'],
                    pre_hidden_lock_sha256=file_sha(lock_paths[domain]),
                    lock_stage_completion_sha256=io.request['dependencies']['hard_bank_lock'])
        materials = build_repair_materials(full, *task_sets, domain=domain)
        repair[domain] = {}
        directory = io.outputs/'repair-materials'/domain
        directory.mkdir(parents=True)
        for role, payload in zip(('training-targets', 'evaluator-tests', 'public-context'), materials, strict=True):
            path = directory/f'{role}.json'
            with path.open('x') as stream:
                json.dump(payload, stream, sort_keys=True); stream.write('\n')
            repair[domain][role] = {'path': str(path.relative_to(io.outputs)), 'sha256': file_sha(path),
                                     'sources': len(payload['records']), 'usage': payload['usage']}
        development = development_cache(full)
        development['derivation'] = 'primary excluded from source-bound full stage cache'
        caches[domain] = {}
        for role, payload in [('full', full), ('development-only', development)]:
            path = io.outputs/f'{domain}-{role}-cache.json'
            with path.open('x') as stream:
                json.dump(payload, stream, sort_keys=True); stream.write('\n')
            caches[domain][role] = {'path': path.name, 'sha256': file_sha(path),
                                    'counts': payload['counts'], 'problem_counts': payload['problem_counts'],
                                    'evaluation_role': payload['evaluation_role']}
    (io.outputs/'execution-index.json').write_text(json.dumps({
        'schema': 'apbpf-generated-execution-cache-v1', 'evaluations': evaluations,
        'training_caches': caches,
        'repair_materials': repair,
        'lock_stage_completion_sha256': io.request['dependencies']['hard_bank_lock'],
        'pre_hidden_locks': {d: file_sha(p) for d, p in lock_paths.items()}}, indent=2)+'\n')
    io.finish({'actual_candidate_execution': True, 'evaluation_runs': len(evaluations),
               'training_caches': caches,
               'repair_materials': repair,
               'tests': sum(e['tests'] for e in evaluations),
               'hidden_access_after_both_primary_locks': True,
               'scope': 'exploratory execution evidence; no hard-bank gate decision'})


if __name__ == '__main__':
    main()
