#!/usr/bin/env python3
"""Fit the prespecified 32-particle semantic development follow-up, all seeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from run_apbpf_particle_diagnostic import validate_development
from pbpf.apbpf.codearc_bank import file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-root', type=Path, required=True)
    p.add_argument('--sensitivity-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream, sensitivity = args.development_root.resolve(), args.sensitivity_root.resolve()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    cache, features = upstream/'development-cache.json', upstream/'semantic-features'
    validate_development(json.loads(cache.read_text()))
    previous = json.loads((sensitivity/'status.json').read_text())
    if previous['status'] != 'complete' or previous['results_sha256'] != file_sha(sensitivity/'results.json'):
        raise ValueError('complete frozen-model sensitivity evidence required before this follow-up')
    files = [Path(__file__).resolve(), root/'scripts/run_apbpf_particle_diagnostic.py',
             root/'scripts/run_rbr_prediction_gate.py', root/'local/sandbox.sh',
             *sorted((root/'src/pbpf').rglob('*.py'))]
    hashes = {str(p.relative_to(root)): file_sha(p) for p in files}
    plan = {'schema': 'apbpf-development-particle-training-debug-v1', 'particles': 32,
            'seeds': [1701, 1702, 1703], 'steps': 1000, 'feature_dim': 512,
            'source_sha256': hashes, 'cache_sha256': file_sha(cache),
            'feature_manifest_sha256': file_sha(features/'manifest.json'),
            'sensitivity_results_sha256': previous['results_sha256'],
            'hypothesis': 'higher particle count during training may reduce approximation error in learned association',
            'design': 'change training/inference particles from8 to32; retain semantic features, three seeds, source partitions, optimizer and step budget; refit all four strong baselines',
            'selection': 'checkpoints use 42 inner-validation sources; assess all400 original development sources; original500 primary sources absent',
            'limitations': 'particle computation increases; this is not a compute-matched or confirmatory improvement claim',
            'scope': 'bounded development-only follow-up; no primary rerun or protocol/gate replacement'}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    state = {'status': 'running', 'pid': os.getpid(), 'plan_sha256': file_sha(output/'plan.json'), 'runs': {}}
    def save():
        path = output/'status.partial'; path.write_text(json.dumps(state, indent=2)+'\n'); path.replace(output/'status.json')
    def check():
        if (any(file_sha(root/n) != h for n, h in hashes.items()) or file_sha(cache) != plan['cache_sha256']
                or file_sha(features/'manifest.json') != plan['feature_manifest_sha256']):
            raise ValueError('declared source, cache or feature manifest changed')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
    save()
    try:
        for seed in plan['seeds']:
            check(); name = f'semantic-p32-seed{seed}'; path = output/f'{name}.json'
            command = ['bash', 'local/sandbox.sh', str(root/'.venv/bin/python'), '-u',
                'scripts/run_rbr_prediction_gate.py', '--cache', str(cache), '--output', str(path),
                '--apbpf', '--seed', str(seed), '--steps', '1000', '--feature-dim', '512',
                '--particles', '32', '--feature-cache', str(features)]
            state.update(current=name); state['runs'][name] = {'status': 'running', 'command': command}; save()
            with (output/f'{name}.log').open('x') as log:
                code = subprocess.call(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            if code:
                raise RuntimeError(f'{name} exited {code}')
            result = json.loads(path.read_text())
            if (result['evaluation_role'] != 'development_assessment_only' or result['config']['particles'] != 32
                    or result['records']['test'] != 3200):
                raise ValueError('unexpected follow-up population or particle budget')
            state['runs'][name].update(status='complete', results_sha256=file_sha(path),
                association=result['cluster_bootstrap']['outcome_shuffled'], gate=result['gate']); save()
        check(); state['status'] = 'complete'; save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error)); save(); raise


if __name__ == '__main__':
    main()
