"""Read only sealed public EvalPlus prompts; never load evaluator raw data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PUBLIC_FIELDS = {'task_id', 'prompt', 'entry_point'}
RELEASES = {'humaneval': ('HumanEvalPlus', 'v0.1.10'), 'mbpp': ('MbppPlus', 'v0.2.0')}
EVALPLUS_VERSION = '0.3.1'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_public_dataset(root: Path, dataset: str, expected_manifest_sha256: str):
    root = Path(root)
    manifest_path = root / 'manifest.json'
    if not expected_manifest_sha256 or sha(manifest_path) != expected_manifest_sha256:
        raise ValueError('public EvalPlus manifest checksum/lock mismatch')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'eesd-evalplus-public-v1'
            or manifest.get('evalplus_version') != EVALPLUS_VERSION
            or set(manifest.get('datasets', {})) != set(RELEASES) or dataset not in RELEASES):
        raise ValueError('unsupported public EvalPlus manifest/version/dataset')
    entry = manifest['datasets'][dataset]
    name, version = RELEASES[dataset]
    if entry.get('dataset') != name or entry.get('version') != version:
        raise ValueError('public EvalPlus release mismatch')
    path = root / f'{dataset}.jsonl'
    if sha(path) != entry.get('public_tasks_sha256'):
        raise ValueError('public EvalPlus task checksum mismatch')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(set(row) != PUBLIC_FIELDS or any(
            not isinstance(row[key], str) or not row[key] for key in PUBLIC_FIELDS) for row in rows):
        raise ValueError('public EvalPlus records must contain only task_id/prompt/entry_point strings')
    task_ids = [row['task_id'] for row in rows]
    if (len(set(task_ids)) != len(task_ids) or len(rows) != entry.get('tasks')
            or sorted(task_ids) != entry.get('task_ids')):
        raise ValueError('public EvalPlus task coverage mismatch')
    return rows, {
        'public_manifest_sha256': expected_manifest_sha256,
        'public_tasks_sha256': entry['public_tasks_sha256'],
        'raw_source_sha256': entry['raw_source_sha256'],
        'download_receipt_sha256': manifest['download_receipt_sha256'],
        'evalplus_version': EVALPLUS_VERSION, 'dataset_version': version,
        'tasks': len(rows), 'task_ids': sorted(task_ids),
    }
