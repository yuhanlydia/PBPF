#!/usr/bin/env python3
"""Generate one candidate per disjoint Replay train source from public inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from pbpf.eesd import replay_generation as base
from pbpf.eesd.replay_training_generation import load_training_split
from generate_eesd_replay_bank import (
    decode_one, load_model, load_tokenizer, write_json_once, writer_lock,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--domain', choices=base.DOMAINS, required=True)
    parser.add_argument('--family', choices=['qwen25_7b', 'deepseek_6p7b'], required=True)
    parser.add_argument('--seed', type=int, choices=[1701], required=True)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--components', type=int, default=200)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.offset < 0 or args.components < 1 or args.offset + args.components > 200:
        parser.error('training shard must lie within 200 admitted sources')
    preflight = load_training_split(args.bundle, args.admission_sha256, args.domain, args.family)
    preflight['rows'] = preflight['rows'][args.offset:args.offset + args.components]
    preflight['audits'] = {row['task_id']: preflight['audits'][row['task_id']]
                           for row in preflight['rows']}
    preflight['bindings'] = {**preflight['bindings'], 'shard_offset': args.offset,
                             'shard_components': args.components}
    tokenizer = load_tokenizer(preflight['model'])
    prepared = {row['task_id']: base.prepare_prompt(tokenizer, row,
                preflight['audits'][row['task_id']], args.family) for row in preflight['rows']}
    run = base.build_expected(preflight, args.family, args.seed, Path(__file__))
    if args.preflight_only:
        print(json.dumps({'status': 'training-public-token-preflight-only',
                          'components': len(prepared), 'run': run}), flush=True)
        return
    with writer_lock(args.output):
        args.output.mkdir(exist_ok=True)
        if list(args.output.glob('*.partial')):
            raise ValueError('partial publication requires audit before resume')
        run_path = args.output / 'run.json'
        if run_path.exists():
            if json.loads(run_path.read_text()) != run:
                raise ValueError('training bank resume identity changed')
        else:
            if any(args.output.iterdir()):
                raise ValueError('orphan training bank artifacts')
            write_json_once(run_path, run)
        if (args.output / 'complete.json').exists():
            base.verify_replay_bank(args.output, run, preflight=preflight)
            print(json.dumps({'status': 'verified-complete-resume'}), flush=True)
            return
        allowed = {'run.json'}
        pending = []
        for row in preflight['rows']:
            path = args.output / base.record_filename(row['task_id'])
            checksum = path.with_suffix('.sha256')
            allowed.update((path.name, checksum.name))
            if path.exists():
                base.verify_record(path, run, row, preflight['audits'][row['task_id']])
            elif checksum.exists():
                raise ValueError('orphan training candidate checksum')
            else:
                pending.append(row)
        if {path.name for path in args.output.iterdir()} - allowed:
            raise ValueError('unexpected training bank artifacts')
        if pending:
            model, torch = load_model(preflight['model'])
            started = time.monotonic()
            for index, row in enumerate(pending, 1):
                task_id = row['task_id']
                candidate = decode_one(model, tokenizer, prepared[task_id], args.family,
                                       base.task_seed(args.seed, task_id), torch)
                candidate['candidate_id'] = f'{task_id}/{args.family}/0'
                record = {key: row[key] for key in ('task_id', 'source_component_id', 'domain', 'split')}
                record.update(seed=base.task_seed(args.seed, task_id),
                    prompt_sha256=prepared[task_id]['metadata']['rendered_prompt_sha256'],
                    prompt_metadata=prepared[task_id]['metadata'], candidates=[candidate])
                path = args.output / base.record_filename(task_id)
                write_json_once(path, record)
                with path.with_suffix('.sha256').open('x') as stream:
                    stream.write(base.sha(path) + '\n')
                print(json.dumps({'completed': index, 'total': len(pending),
                                  'elapsed_seconds': time.monotonic() - started}), flush=True)
        files = {base.record_filename(row['task_id']):
                 base.sha(args.output / base.record_filename(row['task_id']))
                 for row in preflight['rows']}
        write_json_once(args.output / 'complete.json', {
            'schema': 'eesd-replay-generation-complete-v1',
            'run_sha256': base.sha(run_path), 'files': files})
        base.verify_replay_bank(args.output, run, preflight=preflight)
        print(json.dumps({'status': 'training-bank-complete',
                          'components': len(files)}), flush=True)


if __name__ == '__main__':
    main()
