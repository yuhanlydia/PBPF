#!/usr/bin/env python3
"""Wait for one full domain/family bank, then fit and score all fixed seeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from pbpf.apbpf.codearc_bank import file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-root', type=Path, required=True)
    p.add_argument('--public-root', type=Path, required=True)
    p.add_argument('--evaluator-root', type=Path, required=True)
    p.add_argument('--domain', choices=['rbr', 'codearc'], required=True)
    p.add_argument('--family', choices=['qwen', 'deepseek'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve(), *sorted((root/'src/pbpf').rglob('*.py'))]
    files += [root/'scripts'/name for name in ['build_apbpf_full_replay_cache.py',
        'run_local_stage_training_diagnostic.py', 'run_apbpf_train_worker.py',
        'run_apbpf_association_worker.py', 'run_rbr_prediction_gate.py', 'run_local_selection_diagnostic.py']]
    hashes = {str(path.relative_to(root)): file_sha(path) for path in files}
    state = {'status': 'waiting_for_full_execution', 'pid': os.getpid(), 'domain': args.domain,
             'family': args.family, 'source_sha256': hashes, 'runs': {},
             'seeds': [1701, 1702, 1703], 'steps': 1000, 'primary_sources': 500,
             'scope': 'standalone full-population replication cell; no sealed-stage or scientific-gate pass implied'}

    def save():
        path = output/'status.partial'
        path.write_text(json.dumps(state, indent=2)+'\n')
        path.replace(output/'status.json')

    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    python = str(root/'.venv/bin/python')

    def execute(name, command):
        if any(file_sha(root/n) != h for n, h in hashes.items()):
            raise ValueError('replication source changed while waiting; new attempt required')
        state.update(status='running', current=name)
        state['runs'][name] = {'status': 'running', 'command': command}
        save()
        with (output/f'{name}.log').open('x') as log:
            code = subprocess.call(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        state['runs'][name].update(status='complete' if code == 0 else 'execution_failed', returncode=code)
        save()
        if code:
            raise RuntimeError(f'{name} exited {code}')

    save()
    try:
        while True:
            upstream = json.loads((args.evaluation_root/'status.json').read_text())
            if upstream['status'] == 'execution_complete_check_scientific_gates':
                break
            if upstream['status'] == 'needs_debug' or not Path(f"/proc/{upstream['pid']}").exists():
                raise RuntimeError('bank execution failed or exited incomplete')
            time.sleep(20)
        execute('cache', [python, 'scripts/build_apbpf_full_replay_cache.py', '--evaluation-root', str(args.evaluation_root.resolve()),
            '--public-root', str(args.public_root.resolve()), '--evaluator-root', str(args.evaluator_root.resolve()),
            '--domain', args.domain, '--family', args.family, '--output', str(output/'cache')])
        execute('training', [python, '-u', 'scripts/run_local_stage_training_diagnostic.py',
            '--cache', str(output/'cache/cache.json'), '--cache-proof', str(output/'cache/proof.json'),
            '--output', str(output/'training')])
        execute('selection', [python, '-u', 'scripts/run_local_selection_diagnostic.py',
            '--cache', str(output/'cache/cache.json'), '--training-root', str(output/'training'),
            '--output', str(output/'selection')])
        association = json.loads((output/'training/comparison.json').read_text())
        selection = json.loads((output/'selection/results.json').read_text())
        proof = json.loads((output/'cache/proof.json').read_text())
        checksum = file_sha(output/'cache/cache.json')
        if any(value['cache_sha256'] != checksum for value in (association, selection, proof)):
            raise ValueError('replication cell reports do not share the exact cache')
        cell = {'domain': association['dataset'], 'family': args.family,
                'seeds': state['seeds'], 'primary_sources': association['source_components'],
                'primary_candidates': association['primary_candidates'], 'bootstrap_draws': selection['bootstrap_draws'],
                'association_gap': association['comparisons']['outcome_shuffled']['mean_nll_gap'],
                'association_ci95': association['comparisons']['outcome_shuffled']['ci95'],
                'selection_advantage': selection['absolute_selected_pass1_advantage'],
                'selection_ci95': selection['ci95'], 'cache_sha256': checksum,
                'bindings': {name: file_sha(output/path) for name, path in {
                    'cache_proof': 'cache/proof.json', 'association': 'training/comparison.json',
                    'selection': 'selection/results.json'}.items()}, 'scope': state['scope']}
        (output/'cell.json').write_text(json.dumps(cell, indent=2)+'\n')
        state.update(status='complete', cell_sha256=file_sha(output/'cell.json'))
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
