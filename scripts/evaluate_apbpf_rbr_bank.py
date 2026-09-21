#!/usr/bin/env python3
"""Score RBR banks publicly, or after an exact pre-hidden population lock."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from pbpf.apbpf.codearc_bank import load_bank, file_sha, select_tests, verify_hidden_lock
from pbpf.apbpf.rbr_execution import execute_stdin


def work(job):
    task, candidate, code, tests, timeout = job
    result = [dict(test_id=t['id'], **execute_stdin(code, t, timeout=timeout)) for t in tests]
    return {'task_id': task, 'candidate_id': candidate, 'tests': result,
            'all_pass': all(r['outcome'] == 'PASS' for r in result)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public-root', type=Path)
    p.add_argument('--evaluator-root', type=Path)
    p.add_argument('--bank', type=Path)
    p.add_argument('--reference-control', action='store_true')
    p.add_argument('--phase', choices=['visible', 'hidden', 'all'], required=True)
    p.add_argument('--population-lock', type=Path)
    p.add_argument('--split', choices=['train', 'development', 'primary'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--timeout', type=float, default=6.)
    args = p.parse_args()
    if args.reference_control == (args.bank is not None) or min(args.workers, args.timeout) <= 0:
        raise ValueError('choose exactly one generated bank or reference control, with positive limits')
    if args.output.exists():
        raise FileExistsError('evaluations are create-once')
    if args.phase == 'visible':
        if args.public_root is None or args.evaluator_root is not None or args.reference_control:
            raise ValueError('visible scoring uses only public-root and a generated bank')
        root = args.public_root
    else:
        if args.evaluator_root is None or args.public_root is not None:
            raise ValueError('hidden/all scoring requires evaluator-root only')
        root = args.evaluator_root
    bank, bank_digest, lock_digest = [], None, None
    if args.bank is not None:
        run, bank, bank_digest = load_bank(args.bank)
        if run['schema'] != 'apbpf-rbr-generation-v1' or run['candidates'] != 8:
            raise ValueError('requires a complete eight-candidate RBR bank')
        if args.split and args.split != run['split']:
            raise ValueError('split differs from bank identity')
        args.split = run['split']
    if args.reference_control and args.split not in {'train', 'development'}:
        raise ValueError('standalone reference controls are restricted to train/development')
    if args.phase == 'all' and args.split == 'primary':
        raise ValueError('primary requires public evaluation and pre-hidden lock')
    if args.phase == 'hidden':
        if not bank or args.population_lock is None:
            raise ValueError('hidden scoring requires complete bank and population lock')
        lock_digest = verify_hidden_lock(args.population_lock, bank, bank_digest, provenance_prefix='rbr-bank-sha256:')
    elif args.population_lock is not None:
        raise ValueError('population lock is only consumed by hidden scoring')
    # No evaluator file has been opened before the full-bank lock check above.
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['schema'] != 'apbpf-rbr-generated-materialization-v1':
        raise ValueError('wrong task materialization')
    key = 'public_tasks_sha256' if args.phase == 'visible' else 'evaluator_tasks_sha256'
    if file_sha(root/'tasks.jsonl') != manifest[key]:
        raise ValueError('task data checksum mismatch')
    if bank and run['public_tasks_sha256'] != manifest['public_tasks_sha256']:
        raise ValueError('generation/execution public data mismatch')
    with (root/'tasks.jsonl').open() as stream:
        tasks = {r['task_id']: r for r in map(json.loads, stream) if r['split'] == args.split}
    jobs = []
    if args.reference_control:
        for task in tasks.values():
            jobs.append((task['task_id'], 'reference', task['reference_code'], select_tests(task, args.phase), args.timeout))
    else:
        for row in bank:
            task = tasks[row['task_id']]
            if task['source_component_id'] != row['source_component_id']:
                raise ValueError('bank source differs from task source')
            for c in row['candidates']:
                jobs.append((row['task_id'], c['candidate_id'], c['code'], select_tests(task, args.phase), args.timeout))
    if not jobs:
        raise ValueError('empty execution population')
    root_dir = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), root_dir/'src/pbpf/apbpf/rbr_execution.py',
             root_dir/'src/pbpf/apbpf/codearc_execution.py', root_dir/'src/pbpf/apbpf/codearc_bank.py']
    identity = {'schema': 'apbpf-rbr-generated-execution-v1', 'phase': args.phase,
                'reference_control': args.reference_control, 'split': args.split,
                'bank_complete_sha256': bank_digest, 'population_lock_sha256': lock_digest,
                'task_manifest_sha256': file_sha(root/'manifest.json'), 'timeout': args.timeout,
                'stdin_policy': 'official-terminal-newline-if-missing',
                'candidate_count': len(jobs), 'source_sha256': {str(f.relative_to(root_dir)): file_sha(f) for f in files},
                'scope': 'exploratory standalone execution; not sealed-stage evidence'}
    args.output.mkdir(parents=True)
    (args.output/'run.json').write_text(json.dumps(identity, indent=2)+'\n')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output/'records.jsonl').open('x') as stream:
        for row in pool.map(work, jobs):
            rows.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
            if len(rows) % 16 == 0:
                print(json.dumps({'completed': len(rows), 'total': len(jobs)}), flush=True)
    lookup = {(r['task_id'], r['candidate_id']): r for r in rows}
    groups = []
    for row in bank:
        records = [lookup[row['task_id'], c['candidate_id']] for c in row['candidates']]
        group = {'group_id': row['task_id'], 'candidate_ids': [r['candidate_id'] for r in records]}
        if args.phase in {'visible', 'all'}:
            group.update(source_component_id=row['source_component_id'], split=row['split'],
                visible_outcomes=[[int(t['outcome'] == 'PASS') for t in r['tests'] if int(t['test_id']) < 4] for r in records])
        if args.phase in {'hidden', 'all'}:
            group['hidden_labels'] = [int(all(t['outcome'] == 'PASS' for t in r['tests'] if int(t['test_id']) >= 4)) for r in records]
        groups.append(group)
    (args.output/'bank_groups.json').write_text(json.dumps(groups, indent=2)+'\n')
    summary = {**identity, 'tests': sum(len(r['tests']) for r in rows),
               'test_passes': sum(t['outcome'] == 'PASS' for r in rows for t in r['tests']),
               'all_pass': sum(r['all_pass'] for r in rows),
               'scoring_policy': 'line/token match with absolute numeric tolerance1e-4, matching prior RBR; timeout never PASS'}
    (args.output/'results.json').write_text(json.dumps({**summary, 'records': rows}, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
