#!/usr/bin/env python3
"""Build a development-only prediction cache from completed CodeARC executions."""
import argparse
import json
from pathlib import Path

from pbpf.apbpf.codearc_bank import file_sha, load_bank
from pbpf.apbpf.codearc_cache import build_cache
from pbpf.apbpf.development import development_cache
from pbpf.real_gate import validate_rbr_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public-root', type=Path, required=True)
    p.add_argument('--evaluator-root', type=Path, required=True)
    p.add_argument('--bank', type=Path, action='append', required=True)
    p.add_argument('--evaluation', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('cache is create-once')
    tasks, manifests = [], []
    hashes = {}
    for root, key in ((args.public_root, 'public_tasks_sha256'), (args.evaluator_root, 'evaluator_tasks_sha256')):
        manifest = json.loads((root / 'manifest.json').read_text())
        manifests.append(manifest)
        if file_sha(root / 'tasks.jsonl') != manifest[key]:
            raise ValueError('task checksum mismatch')
        with (root / 'tasks.jsonl').open() as stream:
            tasks.append({r['task_id']: r for r in map(json.loads, stream) if r['split'] != 'primary'})
        hashes[str(root / 'manifest.json')] = file_sha(root / 'manifest.json')
    if manifests[0]['public_tasks_sha256'] != manifests[1]['public_tasks_sha256']:
        raise ValueError('public and evaluator materializations differ')
    banks, executions = [], []
    for bank, evaluation in zip(args.bank, args.evaluation, strict=True):
        run, rows, digest = load_bank(bank)
        result = json.loads(evaluation.read_text())
        if (run['split'] == 'primary' or result['phase'] != 'all' or result['reference_control']
                or result['bank_complete_sha256'] != digest
                or result['task_manifest_sha256'] != file_sha(args.evaluator_root / 'manifest.json')
                or run['public_tasks_sha256'] != manifests[0]['public_tasks_sha256']):
            raise ValueError('only complete non-primary generated-bank executions accepted')
        hashes[str(evaluation)] = file_sha(evaluation)
        hashes[str(bank / 'complete.json')] = digest
        banks += rows; executions += result['records']
    payload = development_cache(build_cache(banks, executions, *tasks))
    payload['input_artifact_sha256'] = hashes
    validate_rbr_cache(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write('\n')
    print(json.dumps({key: value for key, value in payload.items() if key != 'records'}))


if __name__ == '__main__':
    main()
