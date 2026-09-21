#!/usr/bin/env python3
"""Rebuild development-only prediction evidence after the RBR stdin correction.

Wait for an independently supervised cache rebuild, then fit three fixed seeds
on CPU. Historic literal-input evidence is retained and never relabelled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from pbpf.apbpf.development import development_cache
from pbpf.real_gate import validate_rbr_cache


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def development_difference(old, new):
    """Compare only original training/development records; no held-out metrics."""
    answer = {}
    for split in ('train', 'development'):
        a = {r['task_id']: r for r in old['records'] if r['split'] == split}
        b = {r['task_id']: r for r in new['records'] if r['split'] == split}
        common = a.keys() & b.keys()
        answer[split] = {
            'old_candidates': len(a), 'new_candidates': len(b),
            'recovered_candidates': sorted(b.keys() - a.keys()),
            'newly_rejected_candidates': sorted(a.keys() - b.keys()),
            'common_candidates': len(common),
            'candidates_with_changed_outcomes': sum(a[k]['outcomes'] != b[k]['outcomes'] for k in common),
            'changed_test_outcomes': sum(x != y for k in common
                                        for x, y in zip(a[k]['outcomes'], b[k]['outcomes'], strict=True)),
        }
    return answer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rebuild-status', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--old-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), root/'scripts/run_rbr_prediction_gate.py',
               *sorted((root/'src/pbpf').rglob('*.py'))]
    hashes = {str(f.relative_to(root)): sha(f) for f in sources}
    state = {'status': 'waiting_for_corrected_cache', 'pid': os.getpid(),
             'seeds': [1701, 1702, 1703], 'steps': 1000, 'device': 'cpu',
             'expected_is_public': True, 'source_sha256': hashes, 'runs': {},
             'evaluation_role': 'development_assessment_only',
             'scope': 'standalone exploratory replay; fixed gates; old held-out excluded'}

    def save():
        write(args.output/'status.json', state)

    def check_sources():
        if any(sha(root/s) != digest for s, digest in hashes.items()):
            raise ValueError('source changed since declaration; new replay attempt required')

    save()
    try:
        while True:
            upstream = json.loads(args.rebuild_status.read_text())
            if upstream['status'] == 'complete':
                if sha(args.cache) != upstream['cache_sha256']:
                    raise ValueError('corrected cache checksum differs from rebuild receipt')
                break
            if upstream['status'] == 'needs_debug' or not Path(f"/proc/{upstream['pid']}").exists():
                raise RuntimeError('cache rebuild failed or its supervisor exited')
            time.sleep(20)
        check_sources()
        payload = json.loads(args.cache.read_text())
        validate_rbr_cache(payload)
        if payload.get('execution_protocol', {}).get('stdin_policy') != 'official-terminal-newline-if-missing':
            raise ValueError('cache lacks corrected stdin provenance')
        old = json.loads(args.old_cache.read_text())
        write(args.output/'development_protocol_difference.json', {
            'old_cache_sha256': sha(args.old_cache), 'new_cache_sha256': sha(args.cache),
            'original_timeout_unattested': True,
            'scope': 'development-only comparison; original test metrics excluded',
            'differences': development_difference(old, payload)})
        cache = development_cache(payload)
        validate_rbr_cache(cache)
        cache_path = args.output/'development_cache.json'
        cache_path.write_text(json.dumps(cache, sort_keys=True) + '\n')
        state.update(cache_sha256=sha(cache_path), source_cache_sha256=sha(args.cache),
                     counts=cache['counts'], problem_counts=cache['problem_counts'])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
        for seed in state['seeds']:
            check_sources()
            name = f'prediction-seed{seed}'
            command = ['bash', 'local/sandbox.sh', str(root/'.venv/bin/python'), '-u',
                       'scripts/run_rbr_prediction_gate.py', '--cache', str(cache_path),
                       '--output', str(args.output/f'{name}.json'), '--apbpf', '--expected-is-public',
                       '--seed', str(seed), '--steps', str(state['steps'])]
            state.update(status='running', current=name)
            state['runs'][name] = {'status': 'running', 'command': command}
            save()
            with (args.output/f'{name}.log').open('x') as log:
                code = subprocess.call(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            state['runs'][name].update(returncode=code, status='complete' if code == 0 else 'execution_failed')
            if code:
                raise RuntimeError(f'{name} exited {code}')
            report = json.loads((args.output/f'{name}.json').read_text())
            state['runs'][name].update(gate=report['gate'], association=report['cluster_bootstrap']['outcome_shuffled'])
            save()
        state['status'] = 'complete'
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
