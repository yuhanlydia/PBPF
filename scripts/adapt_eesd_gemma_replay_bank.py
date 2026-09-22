#!/usr/bin/env python3
"""Bind sealed Gemma Replay candidates to the common correction bank schema."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from pbpf.apbpf.codearc_bank import load_bank
from pbpf.eesd.replay_admission import rename_new_directory

ROOT = Path('/root/PBPF')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', choices=['apps_replay', 'codecontests_replay'], required=True)
    p.add_argument('--split', choices=['train', 'development'], required=True)
    p.add_argument('--slot', type=int)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.split == 'train' and (args.slot is None or not 1 <= args.slot <= 5):
        p.error('train requires shard slot 1..5')
    if args.split == 'development' and args.slot is not None:
        p.error('development is one sealed union bank')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.split == 'train':
        source = ROOT / 'runs/eesd-direct-20260921/gemma-training-banks' / args.domain
        source = source / 'gemma3_4b/seed1701' / f'train-shard{args.slot:02d}-of05'
        public = ROOT / 'runs/eesd-data/replay-training-reserve-20260921' / args.domain / 'public'
        expected_schema = 'eesd-gemma-replay-train-generation-v1'
    else:
        source = ROOT / 'runs/eesd-gemma3-4b-union-20260921' / args.domain
        source = source / 'gemma3_4b/seed1701/development'
        public = ROOT / 'runs/eesd-data/replay-extension-locked-20260920' / args.domain / 'public'
        expected_schema = 'eesd-gemma3-4b-seed1701-union-v1'
    source_run = json.loads((source / 'run.json').read_text())
    source_complete = json.loads((source / 'complete.json').read_text())
    if (source_run.get('schema') != expected_schema
            or source_run.get('domain') != args.domain
            or source_run.get('split') != args.split
            or source_run.get('seed') != 1701
            or source_complete.get('run_sha256') != sha(source / 'run.json')):
        raise ValueError('Gemma Replay source bank identity differs')
    manifest_path, tasks_path = public / 'manifest.json', public / 'tasks.jsonl'
    manifest = json.loads(manifest_path.read_text())
    public_sha = sha(tasks_path)
    if manifest['tasks']['sha256'] != public_sha:
        raise ValueError('Gemma Replay public source seal differs')
    public_rows = {row['task_id']: row for row in map(json.loads, tasks_path.read_text().splitlines())
                   if row['split'] == args.split}
    records = []
    for task_id in source_run['task_ids']:
        name = task_id.replace('/', '-') + '.json'
        path = source / name
        if source_complete['files'].get(name) != sha(path):
            raise ValueError('Gemma Replay candidate checksum differs')
        row = json.loads(path.read_text())
        expected = public_rows.get(task_id)
        if (expected is None or row['source_component_id'] != expected['source_component_id']
                or row['task_id'] != task_id or row['split'] != args.split
                or len(row['candidates']) != 1):
            raise ValueError('Gemma Replay candidate/public task binding differs')
        records.append(row)
    required = 40 if args.split == 'train' else 200
    if len(records) != required or len({row['source_component_id'] for row in records}) != required:
        raise ValueError('Gemma Replay source population differs')
    run = {'schema': 'eesd-replay-generation-v1', 'domain': args.domain,
        'family': 'gemma3_4b', 'model': source_run.get('model', source_run.get('model_id')),
        'revision': source_run['revision'], 'adapter': None, 'seed': 1701,
        'split': args.split, 'candidates': 1, 'components': len(records),
        'task_ids': source_run['task_ids'],
        'source_component_ids': [row['source_component_id'] for row in records],
        'bindings': {'public_tasks_sha256': public_sha,
            'public_manifest_sha256': sha(manifest_path),
            'source_bank_complete_sha256': sha(source / 'complete.json'),
            'source_bank_run_sha256': sha(source / 'run.json')},
        'adaptation': 'identity-preserving copy of sealed Gemma candidate records'}
    if run['model'] != 'google/gemma-3-4b-it':
        raise ValueError('Gemma Replay model identity differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    try:
        for task_id in source_run['task_ids']:
            name = task_id.replace('/', '-') + '.json'
            shutil.copyfile(source / name, staging / name)
        (staging / 'run.json').write_text(json.dumps(run, sort_keys=True, ensure_ascii=False, indent=2) + '\n')
        complete = {'schema': 'eesd-replay-generation-complete-v1',
                    'run_sha256': sha(staging / 'run.json'),
                    'files': {name: sha(staging / name) for name in source_complete['files']}}
        (staging / 'complete.json').write_text(json.dumps(complete, sort_keys=True, indent=2) + '\n')
        loaded, rows, _ = load_bank(staging)
        if loaded != run or len(rows) != required:
            raise ValueError('adapted Gemma Replay bank verification failed')
        rename_new_directory(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({'status': 'Gemma-Replay-bank-adapted', 'domain': args.domain,
                      'split': args.split, 'records': required, 'output': str(output)}), flush=True)

if __name__ == '__main__':
    main()
