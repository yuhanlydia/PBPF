#!/usr/bin/env python3
"""Prepare public Replay correction opportunities from disjoint train/dev banks."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from pbpf.apbpf.codearc_bank import load_bank
from pbpf.apbpf.direct_execution import execute_stdin
from pbpf.eesd.direct_runtime import validate_execution_lock, probe_readiness, PROFILE
from pbpf.eesd.replay_admission import rename_new_directory

ROOT = Path('/root/PBPF')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', choices=['apps_replay', 'codecontests_replay'], required=True)
    p.add_argument('--family', choices=['qwen25_7b', 'deepseek_6p7b', 'gemma3_4b'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=6)
    p.add_argument('--execution-lock', type=Path, required=True)
    p.add_argument('--execution-lock-sha256', required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    lock = validate_execution_lock(args.execution_lock, args.execution_lock_sha256)
    if lock['profile'] != PROFILE or args.timeout != lock['timeout_seconds']:
        raise ValueError('direct execution profile/timeout differs')
    readiness = probe_readiness(args.execution_lock, args.execution_lock_sha256)
    train_root = ROOT / 'runs/eesd-data/replay-training-reserve-20260921' / args.domain / 'public'
    dev_root = ROOT / 'runs/eesd-data/replay-extension-locked-20260920' / args.domain / 'public'
    public = {}
    public_hashes = {}
    for split, root in [('train', train_root), ('development', dev_root)]:
        manifest_path, tasks_path = root / 'manifest.json', root / 'tasks.jsonl'
        manifest = json.loads(manifest_path.read_text())
        if (manifest['tasks']['sha256'] != sha(tasks_path)
                or manifest['counts'].get(split) != 200):
            raise ValueError('Replay public split seal/count mismatch')
        public_hashes[split] = sha(tasks_path)
        for row in map(json.loads, tasks_path.read_text().splitlines()):
            if row['split'] == split:
                if row['task_id'] in public:
                    raise ValueError('duplicate Replay public task')
                public[row['task_id']] = row
    if len(public) != 400 or len({row['source_component_id'] for row in public.values()}) != 400:
        raise ValueError('Replay correction source overlap')
    if args.family == 'gemma3_4b':
        adapted = ROOT / 'runs/eesd-direct-20260921/correction-original-banks' / args.domain
        adapted = adapted / 'gemma3_4b/round1'
        train_banks = [adapted / f'train-shard{slot:02d}-of05-adapted'
                       for slot in range(1, 6)]
        dev_bank = adapted / 'development-adapted'
    else:
        train_banks = [ROOT / 'runs/eesd-direct-20260921/replay-training-banks' / args.domain
                       / args.family / 'seed1701' / f'train-shard{slot:02d}-of05'
                       for slot in range(1, 6)]
        dev_bank = ROOT / 'runs/eesd-20260920/replay-mechanism-banks' / args.domain
        dev_bank = dev_bank / args.family / 'seed1701/development'
    banks = [*train_banks, dev_bank]
    loaded = []
    identity = None
    inputs = {}
    source_seen = set()
    for bank in banks:
        run, rows, complete_sha = load_bank(bank)
        if (run.get('schema') != 'eesd-replay-generation-v1'
                or run.get('domain') != args.domain or run.get('family') != args.family
                or run.get('seed') != 1701 or run.get('candidates') != 1
                or run.get('split') not in ('train', 'development')):
            raise ValueError('Replay correction candidate bank identity differs')
        split = run['split']
        if run['bindings']['public_tasks_sha256'] != public_hashes[split]:
            raise ValueError('candidate bank/public split checksum mismatch')
        current = (run['model'], run['revision'], run.get('adapter'))
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError('candidate banks use different model checkpoints')
        inputs[str(bank / 'complete.json')] = complete_sha
        inputs[str(bank / 'run.json')] = sha(bank / 'run.json')
        for row in rows:
            if row['source_component_id'] in source_seen:
                raise ValueError('duplicate Replay candidate source')
            source_seen.add(row['source_component_id'])
            task = public.get(row['task_id'])
            if (task is None or task['split'] != split
                    or task['source_component_id'] != row['source_component_id']):
                raise ValueError('Replay candidate/public source binding differs')
            loaded.append((row, task))
    if len(loaded) != 400 or sum(row['split'] == 'train' for row, _ in loaded) != 200:
        raise ValueError('Replay correction train/dev candidate population differs')
    prepared, excluded = [], []
    for row, task in loaded:
        candidate = row['candidates'][0]
        measured = []
        if len(task['visible_tests']) != 4:
            raise ValueError('Replay correction requires four public tests')
        for index, test in enumerate(task['visible_tests']):
            execution = execute_stdin(candidate['code'],
                {'input': test['input'], 'expected': test['output']}, timeout=args.timeout)
            measured.append({'id': str(index), 'input': test['input'],
                'expected': test['output'], 'actual': execution.get('stdout', ''),
                'stderr': execution.get('stderr', ''), 'outcome': execution['outcome']})
        outcomes = [test['outcome'] for test in measured]
        if all(value == 'PASS' for value in outcomes):
            excluded.append({'split': row['split'], 'source_component_id': row['source_component_id'],
                             'task_id': row['task_id'], 'reason': 'current_policy_passes_all_public_tests'})
            continue
        prepared.append({'schema': 'eesd-public-correction-row-v1', 'split': row['split'],
            'source_component_id': row['source_component_id'], 'problem_id': row['task_id'],
            'task_id': row['task_id'], 'task_text': task['statement'],
            'candidate': candidate['code'], 'candidate_code_sha256':
                hashlib.sha256(candidate['code'].encode()).hexdigest(),
            'tests': measured, 'outcomes': outcomes,
            'selection': 'single current-policy candidate; correction requested iff any public test fails',
            'generator': {'model': identity[0], 'revision': identity[1],
                          'adapter_sha256': identity[2], 'seed': 1701}})
    if not prepared:
        raise ValueError('Replay current policy produced no correction opportunities')
    sources = {}
    for relative in ['scripts/prepare_eesd_replay_direct_corrections.py',
                     'src/pbpf/apbpf/direct_execution.py', 'src/pbpf/apbpf/codearc_bank.py']:
        sources[relative] = sha(ROOT / relative)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    try:
        rows_path = staging / 'public-corrections.jsonl'
        rows_path.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n'
                                     for row in prepared))
        audit = {'schema': 'eesd-replay-public-correction-bank-v1', 'domain': args.domain,
            'family': args.family, 'model_identity': identity, 'rows': len(prepared),
            'split_counts': {split: sum(row['split'] == split for row in prepared)
                             for split in ('train', 'development')},
            'excluded': excluded, 'public_corrections_sha256': sha(rows_path),
            'evaluator_root_opened': False, 'execution_profile': PROFILE}
        (staging / 'audit.json').write_text(json.dumps(audit, sort_keys=True, ensure_ascii=False,
                                                       indent=2) + '\n')
        receipt = {'schema': 'eesd-direct-corrections-profile-v1', 'stage': 'prepare',
            'profile': PROFILE, 'status': 'complete', 'execution_lock_sha256':
                args.execution_lock_sha256, 'execution_lock': str(args.execution_lock.resolve()),
            'readiness': readiness, 'sources': sources, 'inputs': inputs,
            'outputs': {name: sha(staging / name) for name in ('public-corrections.jsonl', 'audit.json')},
            'output': str(output), 'replay_domain': args.domain,
            'scope': 'public Replay train/development only; direct-no-sandbox'}
        (staging / 'direct-profile.json').write_text(json.dumps(receipt, sort_keys=True,
            ensure_ascii=False, indent=2) + '\n')
        validate_execution_lock(args.execution_lock, args.execution_lock_sha256)
        if any(sha(ROOT / relative) != digest for relative, digest in sources.items()):
            raise ValueError('Replay correction preparation source changed during run')
        if any(sha(path) != digest for path, digest in inputs.items()):
            raise ValueError('Replay correction preparation input changed during run')
        rename_new_directory(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({'status': 'Replay-direct-public-corrections-complete',
                      'domain': args.domain, 'family': args.family,
                      'rows': len(prepared), 'split_counts': audit['split_counts'],
                      'output': str(output)}), flush=True)

if __name__ == '__main__':
    main()
