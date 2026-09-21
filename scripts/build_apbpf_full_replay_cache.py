#!/usr/bin/env python3
"""Assemble a full source-bound cache from a completed standalone bank replay."""
import argparse
import json
from pathlib import Path

from pbpf.apbpf.codearc_bank import file_sha, load_bank, verify_hidden_lock
from pbpf.apbpf.config import resolve_config
from pbpf.apbpf.stage_cache import build_stage_cache, join_primary_executions


def assemble(evaluation, public, evaluator, *, domain, family):
    root = Path(__file__).resolve().parents[1]
    config = resolve_config(root/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory').config
    models = [m for m in config['models'].values() if m['family'] == family]
    if len(models) != 1:
        raise ValueError('exactly one pinned generator for this family required')
    model = models[0]
    status = json.loads((evaluation/'status.json').read_text())
    if status['domain'] != domain or status['status'] != 'execution_complete_check_scientific_gates':
        raise ValueError('complete standalone domain replay required')
    task_sets, task_bindings = [], {}
    for kind, directory in [('public', public), ('evaluator', evaluator)]:
        manifest_sha = file_sha(directory/'manifest.json')
        manifest = json.loads((directory/'manifest.json').read_text())
        tasks_sha = file_sha(directory/'tasks.jsonl')
        if tasks_sha != manifest[f'{kind}_tasks_sha256']:
            raise ValueError('materialized task manifest checksum mismatch')
        with (directory/'tasks.jsonl').open() as stream:
            rows = list(map(json.loads, stream))
        indexed = {row['task_id']: row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError('duplicate materialized task')
        task_sets.append(indexed)
        task_bindings[kind] = {'tasks_sha256': tasks_sha, 'manifest_sha256': manifest_sha}
    if status['public_tasks_sha256'] != task_bindings['public']['tasks_sha256']:
        raise ValueError('execution supervisor used another public inventory')
    groups, executions, bindings = [], [], []
    expected_sizes = {'train': 321 if domain == 'rbr' else 212, 'development-pilot': 16,
                      'development-remainder': 384, 'primary': 500}
    for name, count in expected_sizes.items():
        bank = evaluation/f'{name}-extracted' if domain == 'codearc' else Path(status['banks'][name])
        run, rows, checksum = load_bank(bank)
        split = 'development' if name.startswith('development-') else name
        if (run['model'] != model['id'] or run['revision'] != model['revision']
                or run['public_tasks_sha256'] != task_bindings['public']['tasks_sha256']
                or run['seed'] != 1701 or run['split'] != split or run['candidates'] != 8 or len(rows) != count):
            raise ValueError('replay generator identity, split or population mismatch')
        phases = [('all', evaluation/f'{name}-evaluation/results.json')] if name != 'primary' else [
            ('visible', evaluation/'primary-visible/evaluation/results.json'),
            ('hidden', evaluation/'primary-hidden/results.json')]
        measured, hashes = {}, {}
        for phase, path in phases:
            value = json.loads(path.read_text())
            owner = 'public' if phase == 'visible' else 'evaluator'
            if (value['phase'] != phase or value['split'] != split or value['bank_complete_sha256'] != checksum
                    or value['task_manifest_sha256'] != task_bindings[owner]['manifest_sha256']
                    or value['reference_control']
                    or (domain == 'rbr' and value.get('stdin_policy') != 'official-terminal-newline-if-missing')):
                raise ValueError('execution manifest does not bind the complete candidate bank and protocol')
            measured[phase] = value['records']
            hashes[phase] = file_sha(path)
            if phase == 'hidden':
                lock = evaluation/'primary500-lock.json'
                lock_sha = verify_hidden_lock(lock, rows, checksum, provenance_prefix=domain+'-bank-sha256:')
                if value['population_lock_sha256'] != lock_sha:
                    raise ValueError('hidden execution does not descend from this exact primary lock')
                hashes['pre_hidden_lock'] = lock_sha
        executions += (join_primary_executions(measured['visible'], measured['hidden'])
                       if name == 'primary' else measured['all'])
        groups += rows
        bindings.append({'bank': str(bank.resolve()), 'bank_complete_sha256': checksum,
                         'run_sha256': file_sha(bank/'run.json'), 'executions': hashes})
    cache = build_stage_cache(groups, executions, *task_sets, domain=domain)
    proof = {'schema': 'apbpf-full-replay-cache-proof-v1', 'domain': domain, 'family': family,
             'generator': model, 'counts': cache['counts'], 'problem_counts': cache['problem_counts'],
             'source_bindings': bindings, 'task_bindings': task_bindings,
             'evaluation_status_sha256': file_sha(evaluation/'status.json'),
             'scope': 'assembly from existing full replay; not fresh stage execution or efficacy evidence'}
    return cache, proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-root', type=Path, required=True)
    p.add_argument('--public-root', type=Path, required=True)
    p.add_argument('--evaluator-root', type=Path, required=True)
    p.add_argument('--domain', choices=['rbr', 'codearc'], required=True)
    p.add_argument('--family', choices=['qwen', 'deepseek'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cache, proof = assemble(args.evaluation_root, args.public_root, args.evaluator_root,
                            domain=args.domain, family=args.family)
    path = args.output/'cache.json'
    path.write_text(json.dumps(cache, sort_keys=True)+'\n')
    proof['cache_sha256'] = file_sha(path)
    proof['builder_sha256'] = file_sha(__file__)
    (args.output/'proof.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
