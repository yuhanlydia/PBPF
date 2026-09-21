#!/usr/bin/env python3
"""Offline split of receipt-verified EvalPlus releases into public and private data.

Only this trusted materializer reads raw solutions/tests. Generation consumes
public/ alone; private/ retains exact original bytes for a future isolated
invocation of the official EvalPlus evaluator. No candidate code is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from pbpf.eesd.evalplus_public import EVALPLUS_VERSION, PUBLIC_FIELDS, RELEASES, sha


def materialize(receipt_path: Path, output: Path):
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt.get('schema') != 'eesd-evalplus-download-v1' or receipt.get('evalplus_version') != EVALPLUS_VERSION:
        raise ValueError('expected pinned EvalPlus 0.3.1 download receipt')
    entries = receipt.get('datasets', [])
    if len(entries) != len(RELEASES) or {e['dataset'] for e in entries} != {v[0] for v in RELEASES.values()}:
        raise ValueError('download receipt dataset coverage mismatch')
    checked = {}
    for dataset, (release, version) in RELEASES.items():
        entry = next(e for e in entries if e['dataset'] == release)
        if entry.get('version') != version:
            raise ValueError(f'expected pinned {release} {version}')
        raw = Path(entry['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256'] or len(raw) != entry['bytes']:
            raise ValueError(f'raw release checksum/size mismatch: {release}')
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        ids = [row['task_id'] for row in rows]
        if (len(rows) != entry['tasks'] or len(set(ids)) != len(ids) or sorted(ids) != sorted(entry['task_ids'])
                or len(ids) != len(entry['task_ids'])):
            raise ValueError(f'raw release task coverage mismatch: {release}')
        public_rows = [{key: row[key] for key in sorted(PUBLIC_FIELDS)} for row in rows]
        if any(not isinstance(value, str) or not value for row in public_rows for value in row.values()):
            raise ValueError('public fields must be nonempty strings')
        checked[dataset] = (entry, raw, public_rows)
    if output.exists():
        raise FileExistsError(f'materialization is create-once: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    (staging / 'public').mkdir()
    (staging / 'private').mkdir()
    manifest = {'schema': 'eesd-evalplus-public-v1', 'evalplus_version': EVALPLUS_VERSION,
        'download_receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
        'materializer_source_sha256': sha(Path(__file__)), 'datasets': {}}
    for dataset, (entry, raw, public_rows) in checked.items():
        private_path = staging / 'private' / f'{dataset}.jsonl'
        private_path.write_bytes(raw)
        public_path = staging / 'public' / f'{dataset}.jsonl'
        public_path.write_text(''.join(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n' for row in public_rows))
        manifest['datasets'][dataset] = {
            'dataset': entry['dataset'], 'version': entry['version'], 'tasks': len(public_rows),
            'task_ids': sorted(row['task_id'] for row in public_rows),
            'public_tasks_sha256': sha(public_path), 'raw_source_sha256': sha(private_path),
            'checksum_kind': entry.get('checksum_kind', 'local provenance digest'),
        }
    (staging / 'private/download-receipt.json').write_bytes(receipt_bytes)
    (staging / 'private/manifest.json').write_text(json.dumps({
        'schema': 'eesd-evalplus-private-v1', 'evalplus_version': EVALPLUS_VERSION,
        'datasets': {key: {'file': key + '.jsonl', 'version': value['version'],
                          'sha256': value['raw_source_sha256']} for key, value in manifest['datasets'].items()},
    }, sort_keys=True, indent=2) + '\n')
    (staging / 'public/manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    staging.rename(output)
    return {'public_data_root': str((output / 'public').resolve()),
        'public_manifest_sha256': sha(output / 'public/manifest.json'),
        'private_data_root': str((output / 'private').resolve()),
        'tasks': {key: value['tasks'] for key, value in manifest['datasets'].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.receipt, args.output), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
