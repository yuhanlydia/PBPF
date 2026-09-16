#!/usr/bin/env python3
"""Run development-only CodeARC prediction after local execution and GPU handoff.

This supervisor produces standalone exploratory evidence, not sealed DAG stages.
It tolerates partial status reads from older non-atomic status writers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read_snapshot(path):
    try:
        value = json.loads(Path(path).read_bytes())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        raise ValueError(f'status must be an object: {path}')
    return value


def atomic_write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--gpu', default='2')
    args = p.parse_args()
    root, run = args.workspace.resolve(), args.run_root.resolve()
    status = run / 'codearc_prediction_status.json'
    if status.exists():
        raise FileExistsError('existing supervisor state must be inspected and archived first')
    sources = ['scripts/run_local_codearc_prediction.py',
               'scripts/build_codearc_development_cache.py', 'scripts/run_rbr_prediction_gate.py',
               'src/pbpf/apbpf/codearc_cache.py', 'src/pbpf/apbpf/codearc_bank.py',
               'src/pbpf/apbpf/development.py', 'src/pbpf/real_gate.py',
               'src/pbpf/train_belief.py', 'src/pbpf/belief/model.py']
    hashes = {s: hashlib.sha256((root / s).read_bytes()).hexdigest() for s in sources}
    state = {'status': 'waiting_for_development_execution_and_gpu', 'pid': os.getpid(),
             'seeds': [1701, 1702, 1703], 'steps': 1000, 'gpu': args.gpu,
             'evaluation_role': 'development_assessment_only', 'expected_is_public': False,
             'source_sha256': hashes, 'runs': {},
             'scope': 'exploratory standalone prediction; original primary tasks excluded'}
    pairs = [('codearc-generation-train212-extracted-v2', 'codearc-train212-evaluation-v5'),
             ('codearc-generation-pilot16-extracted-v2', 'codearc-pilot16-evaluation-v5'),
             ('codearc-generation-development-bounded384-extracted-v2', 'codearc-development384-evaluation-v5')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    python = str(root / '.venv/bin/python')

    def save():
        atomic_write(status, state)

    def execute(name, command):
        if any(hashlib.sha256((root / s).read_bytes()).hexdigest() != h for s, h in hashes.items()):
            raise ValueError('source changed; new supervisor attempt required')
        state.update(status='running', current=name)
        state['runs'][name] = {'status': 'running', 'command': command}
        save()
        with (run / (name + '.log')).open('x') as log:
            rc = subprocess.call(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        state['runs'][name].update(status='complete' if rc == 0 else 'execution_failed', returncode=rc)
        save()
        if rc:
            raise RuntimeError(f'{name} failed with exit {rc}')

    save()
    try:
        while True:
            semantic = read_snapshot(run / 'semantic_replication_status.json')
            pipeline = read_snapshot(run / 'codearc_pipeline_status.json')
            if semantic is None or pipeline is None:
                time.sleep(5)
                continue
            ready = all((run / evaluation / 'results.json').exists() for _, evaluation in pairs)
            if ready and semantic['status'] in ('complete', 'needs_debug'):
                break
            if not ready and pipeline['status'] == 'needs_debug':
                raise RuntimeError('upstream CodeARC execution failed')
            for upstream in (semantic, pipeline):
                if upstream['status'] == 'running' and not Path(f"/proc/{upstream['pid']}").exists():
                    raise RuntimeError('upstream supervisor exited without terminal status')
            time.sleep(20)
        cache = run / 'codearc-development-cache-v1.json'
        command = [python, 'scripts/build_codearc_development_cache.py', '--public-root',
                   str(run / 'codearc-public-v1'), '--evaluator-root', str(run / 'codearc-evaluator-v1'),
                   '--output', str(cache)]
        for bank, evaluation in pairs:
            command += ['--bank', str(run / bank), '--evaluation', str(run / evaluation / 'results.json')]
        execute('codearc-development-cache-v1', command)
        for seed in state['seeds']:
            name = f'codearc-development-prediction-seed{seed}'
            execute(name, [python, '-u', 'scripts/run_rbr_prediction_gate.py', '--cache', str(cache),
                          '--output', str(run / (name + '.json')), '--apbpf', '--seed', str(seed),
                          '--steps', str(state['steps'])])
        state['status'] = 'complete'
        save()
    except BaseException as exc:
        state.update(status='needs_debug', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    main()
