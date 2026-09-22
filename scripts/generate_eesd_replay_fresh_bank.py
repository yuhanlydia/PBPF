#!/usr/bin/env python3
"""Generate one adapter-backed primary Replay candidate per fixed public source."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from generate_eesd_replay_bank import (
    decode_one, load_model, load_tokenizer, write_json_once, writer_lock,
)
from pbpf.eesd.replay_generation import (
    build_expected, load_public_split, prepare_prompt, record_filename,
    sha, verify_record, verify_replay_bank,
)


def adapter_tree_sha(directory: Path) -> str:
    directory = directory.resolve()
    files = sorted(path for path in directory.rglob('*') if path.is_file())
    if not files:
        raise ValueError('fresh generation requires a nonempty LoRA adapter')
    digest = hashlib.sha256()
    for path in files:
        name = path.relative_to(directory).as_posix().encode()
        digest.update(len(name).to_bytes(4, 'big'))
        digest.update(name)
        digest.update(bytes.fromhex(sha(path)))
    return digest.hexdigest()


def shard_preflight(bundle, admission_sha256, domain, family, offset, components):
    preflight = load_public_split(bundle, admission_sha256, domain, 'primary', family)
    rows = preflight['rows']
    if (len(rows) != 500 or offset < 0 or components < 1
            or offset + components > len(rows)):
        raise ValueError('fresh Replay shards must select a nonempty primary subset of 500')
    selected = rows[offset:offset + components]
    if len({row['source_component_id'] for row in selected}) != len(selected):
        raise ValueError('duplicate source in fresh Replay shard')
    return {**preflight, 'rows': selected,
            'audits': {row['task_id']: preflight['audits'][row['task_id']]
                       for row in selected}}


def expected_run(preflight, family, seed, offset, adapter_sha):
    run = build_expected(preflight, family, seed, Path(__file__))
    run.update(schema='eesd-replay-fresh-generation-v1',
               adapter=adapter_sha, adapter_sha256=adapter_sha,
               offset=offset,
               claim_status='fresh-adapter-candidate-generation-only')
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--admission-sha256', required=True)
    p.add_argument('--domain', choices=['apps_replay', 'codecontests_replay'], required=True)
    p.add_argument('--family', choices=['qwen25_7b', 'deepseek_6p7b'], required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--offset', type=int, required=True)
    p.add_argument('--components', type=int, required=True)
    p.add_argument('--seed', type=int, choices=[1701], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()

    preflight = shard_preflight(args.bundle, args.admission_sha256, args.domain,
                                args.family, args.offset, args.components)
    adapter_sha = adapter_tree_sha(args.adapter)
    tokenizer = load_tokenizer(preflight['model'])
    prepared = {row['task_id']: prepare_prompt(tokenizer, row,
                preflight['audits'][row['task_id']], args.family)
                for row in preflight['rows']}
    run = expected_run(preflight, args.family, args.seed, args.offset, adapter_sha)
    if args.preflight_only:
        print(json.dumps({'status': 'fresh-Replay-preflight',
                          'domain': args.domain, 'family': args.family,
                          'components': len(prepared), 'adapter_sha256': adapter_sha}), flush=True)
        return

    with writer_lock(args.output):
        args.output.mkdir(exist_ok=True)
        if list(args.output.glob('*.partial')):
            raise ValueError('partial fresh Replay publication requires audit')
        run_path = args.output / 'run.json'
        if run_path.exists():
            if json.loads(run_path.read_text()) != run:
                raise ValueError('fresh Replay resume identity changed')
        elif any(args.output.iterdir()):
            raise ValueError('orphan fresh Replay artifacts')
        else:
            write_json_once(run_path, run)
        if (args.output / 'complete.json').exists():
            verify_replay_bank(args.output, run, preflight=preflight)
            print(json.dumps({'status': 'verified-complete-resume'}), flush=True)
            return
        allowed = {'run.json'}
        pending = []
        for row in preflight['rows']:
            path = args.output / record_filename(row['task_id'])
            checksum = path.with_suffix('.sha256')
            allowed.update((path.name, checksum.name))
            if path.exists():
                verify_record(path, run, row, preflight['audits'][row['task_id']])
            elif checksum.exists():
                raise ValueError('orphan fresh Replay checksum')
            else:
                pending.append(row)
        if {path.name for path in args.output.iterdir()} - allowed:
            raise ValueError('unexpected fresh Replay bank artifacts')

        if pending:
            from peft import PeftModel
            model, torch = load_model(preflight['model'])
            model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
            model.eval()
            started = time.monotonic()
            for index, row in enumerate(pending, 1):
                task = row['task_id']
                task_seed = int.from_bytes(hashlib.sha256(
                    f'{args.seed}:{task}'.encode()).digest()[:4], 'big')
                candidate = decode_one(model, tokenizer, prepared[task],
                                       args.family, task_seed, torch)
                candidate['candidate_id'] = f'{task}/{args.family}/0'
                record = {key: row[key] for key in
                          ('task_id', 'source_component_id', 'domain', 'split')}
                record.update(seed=task_seed,
                              prompt_sha256=prepared[task]['metadata']['rendered_prompt_sha256'],
                              prompt_metadata=prepared[task]['metadata'],
                              candidates=[candidate])
                path = args.output / record_filename(task)
                write_json_once(path, record)
                with path.with_suffix('.sha256').open('x') as stream:
                    stream.write(sha(path) + '\n')
                verify_record(path, run, row, preflight['audits'][task])
                print(json.dumps({'completed': index, 'total': len(pending),
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
        files = {record_filename(row['task_id']):
                 sha(args.output / record_filename(row['task_id']))
                 for row in preflight['rows']}
        write_json_once(args.output / 'complete.json',
                        {'schema': 'eesd-replay-generation-complete-v1',
                         'run_sha256': sha(run_path), 'files': files})
        verify_replay_bank(args.output, run, preflight=preflight)
        print(json.dumps({'status': 'fresh-Replay-generation-complete',
                          'components': len(files)}), flush=True)


if __name__ == '__main__':
    main()
