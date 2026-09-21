#!/usr/bin/env python3
"""Project a sealed LiveCodeBench lite release into generation-only public data.

Private test encodings remain opaque in the original files. No remote builder,
pickle, reference solution, or generated candidate is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

REPO = 'livecodebench/code_generation_lite'
REVISION = '0fe84c3912ea0c4d4a78037083943e8f0c4dd505'
FILES = ('test.jsonl', *(f'test{i}.jsonl' for i in range(2, 7)))
PUBLIC_FIELDS = ('platform', 'question_id', 'question_title', 'question_content',
                 'starter_code', 'contest_date', 'difficulty')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def materialize(receipt_path, inventory_path, output, expected_receipt_sha256):
    receipt_path, inventory_path, output = map(Path, (receipt_path, inventory_path, output))
    if output.exists():
        raise FileExistsError(f'create-once output already exists: {output}')
    if sha(receipt_path) != expected_receipt_sha256:
        raise ValueError('download receipt checksum mismatch')
    receipt = json.loads(receipt_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    if (receipt.get('schema') != 'eesd-livecodebench-download-v1'
            or receipt.get('status') != 'complete' or receipt.get('release') != 'release_v6'
            or receipt.get('inventory_sha256') != sha(inventory_path)):
        raise ValueError('require complete sealed release_v6 download')
    for data in (receipt, inventory):
        if data.get('repo_id') != REPO or data.get('revision') != REVISION:
            raise ValueError('locked dataset identity mismatch')
    files = receipt['files']
    if len(files) != len(FILES) or {f['filename'] for f in files} != set(FILES):
        raise ValueError('release file coverage mismatch')
    expected = {entry['path']: entry for entry in inventory['files']}
    by_name = {entry['filename']: entry for entry in files}
    rows, ids, raw_bindings = [], set(), []
    for name in FILES:
        entry, published = by_name[name], expected[name]
        path = Path(entry['path'])
        digest = sha(path)
        if (digest != published['lfs']['sha256'] or digest != entry['sha256']
                or digest != entry['official_lfs_sha256'] or entry.get('verified') is not True
                or path.stat().st_size != published['size']
                or path.stat().st_size != entry['bytes']):
            raise ValueError(f'raw file SHA256/size mismatch: {name}')
        count = 0
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                raw = json.loads(line)
                public = {field: raw[field] for field in PUBLIC_FIELDS}
                if any(not isinstance(value, str) for value in public.values()):
                    raise ValueError('public fields must be strings')
                if any(not public[key] for key in ('platform', 'question_id', 'question_content')):
                    raise ValueError('empty task identity or problem statement')
                # JSON encoding preserves identity unambiguously even if fields contain '/'.
                task_id = json.dumps([public['platform'], public['question_id']], separators=(',', ':'))
                if task_id in ids:
                    raise ValueError(f'duplicate cross-file task identity: {task_id}')
                ids.add(task_id)
                rows.append(dict(task_id=task_id, **public))
                count += 1
        if count != entry['statistics']['rows']:
            raise ValueError('download receipt row count mismatch')
        raw_bindings.append(dict(filename=name, path=str(path.resolve()), sha256=digest,
                                 bytes=entry['bytes'], tasks=count))
    if len(rows) != receipt['statistics']['rows']:
        raise ValueError('release row count mismatch')
    # Bind the bytes read, including the large opaque evaluator payloads.
    if sha(receipt_path) != expected_receipt_sha256 or any(
            sha(Path(entry['path'])) != entry['sha256'] for entry in raw_bindings):
        raise ValueError('sealed inputs changed during projection')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.partial-', dir=output.parent))
    (staging / 'public').mkdir(); (staging / 'private').mkdir()
    public_path = staging / 'public/tasks.jsonl'
    with public_path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
    manifest = dict(schema='eesd-livecodebench-public-v1', repo_id=REPO,
        revision=REVISION, release='release_v6', test_pool='lite', tasks=len(rows),
        task_ids=sorted(ids), public_tasks_sha256=sha(public_path),
        download_receipt_sha256=expected_receipt_sha256,
        inventory_sha256=receipt['inventory_sha256'], materializer_source_sha256=sha(__file__),
        raw_files=[{k: v for k, v in entry.items() if k != 'path'} for entry in raw_bindings],
        purpose='transfer_candidate_preparation_not_admitted_mechanism_domain')
    write_json(staging / 'public/manifest.json', manifest)
    write_json(staging / 'private/manifest.json', dict(schema='eesd-livecodebench-private-v1',
        raw_files=raw_bindings, download_receipt_sha256=expected_receipt_sha256,
        storage='original verified raw files; no duplicate payload copy',
        private_payload_decoded=False))
    write_json(staging / 'complete.json', dict(public_manifest_sha256=sha(staging / 'public/manifest.json'),
        private_manifest_sha256=sha(staging / 'private/manifest.json')))
    if output.exists():
        raise FileExistsError(f'concurrent publication: {output}')
    staging.rename(output)
    return dict(output=str(output), tasks=len(rows), public_manifest_sha256=sha(output / 'public/manifest.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.receipt, args.inventory, args.output, args.receipt_sha256)))


if __name__ == '__main__':
    main()
