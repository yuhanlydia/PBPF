#!/usr/bin/env python3
"""Bind sealed Gemma development candidates to original-domain correction banks."""
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
    p.add_argument('--domain', choices=['runbugrun', 'codearc'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source = ROOT / 'runs/eesd-gemma3-4b-union-20260921' / args.domain / 'gemma3_4b/seed1701/development'
    source_run = json.loads((source / 'run.json').read_text())
    source_complete = json.loads((source / 'complete.json').read_text())
    if (source_run.get('schema') != 'eesd-gemma3-4b-seed1701-union-v1'
            or source_run['domain'] != args.domain or source_run['split'] != 'development'
            or source_run['seed'] != 1701 or source_run['components'] != 200
            or source_run['candidates'] != 1
            or source_complete.get('run_sha256') != sha(source / 'run.json')):
        raise ValueError('Gemma source union is not sealed development population')
    public = ROOT / 'runs/eesd-data' / ('rbr-public' if args.domain == 'runbugrun' else 'codearc-public')
    manifest = json.loads((public / 'manifest.json').read_text())
    public_sha = sha(public / 'tasks.jsonl')
    if manifest['public_tasks_sha256'] != public_sha:
        raise ValueError('public development inventory checksum differs')
    public_rows = {row['task_id']: row for row in map(json.loads, (public / 'tasks.jsonl').read_text().splitlines())}
    records = []
    for task_id in source_run['task_ids']:
        name = task_id.replace('/', '-') + '.json'
        path = source / name
        if source_complete['files'].get(name) != sha(path):
            raise ValueError('Gemma union candidate checksum differs: ' + name)
        row = json.loads(path.read_text())
        public_row = public_rows.get(task_id)
        if (public_row is None or public_row['split'] != 'development'
                or row['source_component_id'] != public_row['source_component_id']
                or row['task_id'] != task_id or row['split'] != 'development'
                or len(row['candidates']) != 1):
            raise ValueError('Gemma development record/public identity differs')
        records.append(row)
    if len(records) != 200 or len({row['source_component_id'] for row in records}) != 200:
        raise ValueError('Gemma development population differs')
    run = {'schema': f'apbpf-{"rbr" if args.domain == "runbugrun" else "codearc"}-generation-v1',
           'domain': args.domain, 'family': 'gemma3_4b', 'split': 'development',
           'seed': 1701, 'model': source_run['model_id'], 'revision': source_run['revision'],
           'adapter_sha256': None, 'candidates': 1, 'components': 200,
           'task_ids': source_run['task_ids'],
           'source_component_ids': [row['source_component_id'] for row in records],
           'public_tasks_sha256': public_sha, 'public_manifest_sha256': sha(public / 'manifest.json'),
           'source_union_complete_sha256': sha(source / 'complete.json'),
           'source_union_run_sha256': sha(source / 'run.json'),
           'adaptation': 'identity-preserving copy of one candidate per sealed development source'}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    try:
        for task_id in source_run['task_ids']:
            name = task_id.replace('/', '-') + '.json'
            shutil.copyfile(source / name, staging / name)
        (staging / 'run.json').write_text(json.dumps(run, sort_keys=True, ensure_ascii=False, indent=2) + '\n')
        complete = {'schema': 'apbpf-candidate-generation-complete-v1',
                    'run_sha256': sha(staging / 'run.json'),
                    'files': {name: sha(staging / name) for name in source_complete['files']}}
        (staging / 'complete.json').write_text(json.dumps(complete, sort_keys=True, indent=2) + '\n')
        loaded, rows, _ = load_bank(staging)
        if loaded != run or len(rows) != 200:
            raise ValueError('adapted Gemma bank verification failed')
        rename_new_directory(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({'status': 'Gemma-development-bank-adapted', 'domain': args.domain,
                      'records': 200, 'output': str(output)}), flush=True)

if __name__ == '__main__':
    main()
