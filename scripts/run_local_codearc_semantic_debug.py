#!/usr/bin/env python3
"""Compare lexical and frozen code-model features on development sources only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.development import development_cache
from pbpf.real_gate import public_test_text, validate_rbr_cache


def prepare_inventory(source):
    source = validate_rbr_cache(source)
    if source['dataset'] != 'codearc_replay':
        raise ValueError('CodeARC cache required')
    payload = development_cache(source)
    original_primary = {r['source_component_id'] for r in source['records'] if r['split'] == 'test'}
    if original_primary & {r['source_component_id'] for r in payload['records']}:
        raise ValueError('original primary source entered development iteration')
    texts = set()
    for row in payload['records']:
        texts.update((row['task_text'], row['candidate']))
        texts.update(public_test_text(case, expected_is_public=False) for case in row['tests'])
    inventory = sorted(({'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest()}
                        for text in texts), key=lambda row: row['text_sha256'])
    return validate_rbr_cache(payload), inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--upstream-generation-status', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve(), root/'scripts/extract_apbpf_semantic_features.py',
             root/'scripts/run_rbr_prediction_gate.py', root/'scripts/apbpf_generator_sandbox.sh',
             root/'local/sandbox.sh', *sorted((root/'src/pbpf').rglob('*.py'))]
    hashes = {str(p.relative_to(root)): file_sha(p) for p in files}
    cache_sha = file_sha(args.cache)
    payload, rows = prepare_inventory(json.loads(args.cache.read_text()))
    cache = output/'development-cache.json'
    cache.write_text(json.dumps(payload, sort_keys=True)+'\n')
    public = output/'public-texts'
    public.mkdir()
    texts = public/'texts.jsonl'
    texts.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    manifest = {'schema': 'apbpf-public-text-inventory-v1', 'texts': len(rows),
                'texts_sha256': file_sha(texts), 'source_cache_sha256': file_sha(cache),
                'source_payload_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                'evaluation_role': 'development_assessment_only', 'expected_is_public': False}
    (public/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    upstream = json.loads(args.upstream_generation_status.read_text())
    if upstream['gpu'] != 2:
        raise ValueError('this diagnostic is declared for GPU2 after CodeARC replication generation')
    upstream_pid = upstream['pid']
    proc = Path(f'/proc/{upstream_pid}')
    upstream_start = proc.joinpath('stat').read_text().split()[21] if proc.exists() else None
    state = {'status': 'running', 'pid': os.getpid(), 'source_sha256': hashes,
             'original_cache_sha256': cache_sha, 'development_cache_sha256': file_sha(cache),
             'public_manifest_sha256': file_sha(public/'manifest.json'),
             'seeds': [1701, 1702, 1703], 'steps': 1000, 'feature_dim': 512,
             'counts': payload['counts'], 'problem_counts': payload['problem_counts'],
             'upstream_pid': upstream_pid, 'upstream_start_ticks': upstream_start,
             'gpu': 2, 'runs': {}, 'original_primary_excluded': True,
             'hypothesis': 'Frozen code-model features may improve development association over dimension-matched lexical hashing',
             'scope': 'development-only exploratory method debug; three seeds and four matched strong baselines; no primary evaluation'}

    def save():
        p = output/'status.partial'
        p.write_text(json.dumps(state, indent=2)+'\n')
        p.replace(output/'status.json')

    def execute(name, command, gpu=''):
        if any(file_sha(root/n) != h for n, h in hashes.items()) or file_sha(cache) != state['development_cache_sha256']:
            raise ValueError('declared source or data changed; launch a new attempt')
        state.update(status='running', current=name)
        state['runs'][name] = {'status': 'running', 'command': command}
        save()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        with (output/f'{name}.log').open('x') as log:
            code = subprocess.call(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        state['runs'][name].update(returncode=code, status='complete' if code == 0 else 'execution_failed')
        save()
        if code:
            raise RuntimeError(f'{name} failed with exit {code}')

    python = str(root/'.venv/bin/python')
    features = output/'semantic-features'
    save()
    try:
        for kind in ('lexical', 'semantic'):
            if kind == 'semantic':
                state.update(status='waiting_for_gpu2_generation_completion')
                save()
                while True:
                    live = json.loads(args.upstream_generation_status.read_text())
                    same_process = (proc.exists() and upstream_start is not None
                                    and proc.joinpath('stat').read_text().split()[21] == upstream_start)
                    complete = all(v['status'] == 'complete' for v in live['runs'].values())
                    if complete and not same_process:
                        break
                    if live['status'] == 'needs_debug' or (not same_process and not complete):
                        raise RuntimeError('upstream generation failed before GPU handoff')
                    time.sleep(20)
                # Another job must not have claimed the device after generation ended.
                gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=uuid',
                                                    '--format=csv,noheader'], text=True).strip()
                processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                     '--format=csv,noheader'], text=True)
                if any(line.split(',')[0].strip() == gpu_uuid for line in processes.splitlines()):
                    raise RuntimeError('GPU2 is occupied after declared upstream completion')
                features.mkdir()
                execute('extract-semantic', ['bash', 'scripts/apbpf_generator_sandbox.sh', str(public),
                        str(features), python, '-u', 'scripts/extract_apbpf_semantic_features.py',
                        '--input', '/input', '--output', '/output', '--batch-size', '8'], gpu='2')
            for seed in state['seeds']:
                name = f'{kind}-seed{seed}'
                command = ['bash', 'local/sandbox.sh', python, '-u', 'scripts/run_rbr_prediction_gate.py',
                           '--cache', str(cache), '--output', str(output/f'{name}.json'), '--apbpf',
                           '--seed', str(seed), '--steps', '1000', '--feature-dim', '512']
                if kind == 'semantic':
                    command += ['--feature-cache', str(features)]
                execute(name, command)
                report = json.loads((output/f'{name}.json').read_text())
                state['runs'][name].update(results_sha256=file_sha(output/f'{name}.json'),
                    gate=report['gate'], association=report['cluster_bootstrap']['outcome_shuffled'])
                save()
        state['status'] = 'complete'
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
