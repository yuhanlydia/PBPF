#!/usr/bin/env python3
"""Bind full generated train/development execution evidence and redact hidden answers."""
import argparse
import json
from pathlib import Path

from pbpf.apbpf.codearc_bank import file_sha, load_bank
from pbpf.apbpf.development import development_cache
from pbpf.apbpf.generated_cache import build_generated_cache
from pbpf.real_gate import validate_rbr_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['rbr', 'codearc'], required=True)
    parser.add_argument('--public-root', type=Path, required=True)
    parser.add_argument('--evaluator-root', type=Path, required=True)
    parser.add_argument('--bank', type=Path, action='append', required=True)
    parser.add_argument('--evaluation', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('development caches are create-once')
    roots = [args.public_root, args.evaluator_root]
    manifests, task_sets, hashes = [], [], {}
    for root, key in zip(roots, ['public_tasks_sha256', 'evaluator_tasks_sha256'], strict=True):
        manifest = json.loads((root/'manifest.json').read_text())
        if file_sha(root/'tasks.jsonl') != manifest[key]:
            raise ValueError('task data checksum mismatch')
        with (root/'tasks.jsonl').open() as stream:
            task_sets.append({r['task_id']: r for r in map(json.loads, stream) if r['split'] != 'primary'})
        manifests.append(manifest); hashes[str(root/'manifest.json')] = file_sha(root/'manifest.json')
    if manifests[0]['public_tasks_sha256'] != manifests[1]['public_tasks_sha256']:
        raise ValueError('public and evaluator source data differ')
    representatives = {}
    for row in task_sets[0].values():
        representatives.setdefault(row['source_component_id'], row)
    expected_tasks = {r['task_id'] for r in representatives.values()}
    banks, executions, model = [], [], None
    for bank, evaluation in zip(args.bank, args.evaluation, strict=True):
        run, rows, checksum = load_bank(bank)
        report = json.loads(evaluation.read_text())
        current_model = {k: run[k] for k in ('model', 'revision', 'seed', 'max_new_tokens', 'temperature', 'top_p')}
        if model is not None and current_model != model:
            raise ValueError('cannot mix generator identities in one training cache')
        model = current_model
        if (run['schema'] != f'apbpf-{args.domain}-generation-v1' or run['split'] == 'primary'
                or run['candidates'] != 8 or report['phase'] != 'all' or report['reference_control']
                or report['bank_complete_sha256'] != checksum
                or report['task_manifest_sha256'] != file_sha(args.evaluator_root/'manifest.json')
                or run['public_tasks_sha256'] != manifests[0]['public_tasks_sha256']
                or (args.domain == 'rbr' and report.get('stdin_policy') != 'official-terminal-newline-if-missing')):
            raise ValueError('requires complete matching non-primary generated execution evidence')
        hashes[str(bank/'complete.json')] = checksum; hashes[str(evaluation)] = file_sha(evaluation)
        banks += rows; executions += report['records']
    if len(banks) != len(expected_tasks) or {r['task_id'] for r in banks} != expected_tasks:
        raise ValueError('every declared train/development source must occur once')
    payload = development_cache(build_generated_cache(banks, executions, *task_sets, domain=args.domain))
    payload.update(input_artifact_sha256=hashes, generator_identity=model,
                   visibility_contract='future expected values redacted before feature extraction; original primary excluded')
    validate_rbr_cache(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(payload, stream, sort_keys=True); stream.write('\n')
    print(json.dumps({k:v for k,v in payload.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
