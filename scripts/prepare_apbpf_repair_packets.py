#!/usr/bin/env python3
"""Prepare all three CodeARC repair inputs from completed standalone evidence."""
import argparse
import json
import os
from pathlib import Path

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.repair_materials import build_repair_materials
from pbpf.apbpf.repair_packets import select_repair_rows, write_packet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--cache-proof', type=Path, required=True)
    p.add_argument('--public-root', type=Path, required=True)
    p.add_argument('--evaluator-root', type=Path, required=True)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--selection-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checksum = file_sha(args.cache)
    proof = json.loads(args.cache_proof.read_text())
    if proof['cache_sha256'] != checksum or proof['domain'] != 'codearc':
        raise ValueError('exact completed CodeARC cache proof required')
    selection_state = json.loads((args.selection_root/'status.json').read_text())
    training_state = json.loads((args.training_root/'status.json').read_text())
    if any(d['status'] != 'complete' or d['cache_sha256'] != checksum for d in (selection_state, training_state)):
        raise ValueError('all declared training and selection seeds must have completed')
    paths = [Path(__file__).resolve(), *sorted((root/'src/pbpf').rglob('*.py'))]
    hashes = {str(f.relative_to(root)): file_sha(f) for f in paths}
    state = {'status': 'running', 'pid': os.getpid(), 'cache_sha256': checksum,
             'source_sha256': hashes, 'cache_proof_sha256': file_sha(args.cache_proof), 'runs': {},
             'scope': 'standalone exploratory repair preparation; no repaired programs or scoring yet'}

    def save():
        p = output/'status.partial'
        p.write_text(json.dumps(state, indent=2)+'\n')
        p.replace(output/'status.json')

    save()
    try:
        payload = json.loads(args.cache.read_text())
        task_sets = []
        state['task_bindings'] = {}
        for kind, directory in [('public', args.public_root), ('evaluator', args.evaluator_root)]:
            manifest = json.loads((directory/'manifest.json').read_text())
            task_sha = file_sha(directory/'tasks.jsonl')
            if task_sha != manifest[f'{kind}_tasks_sha256']:
                raise ValueError('task data differ from materialized manifest')
            with (directory/'tasks.jsonl').open() as stream:
                rows = list(map(json.loads, stream))
            task_sets.append({r['task_id']: r for r in rows})
            if len(task_sets[-1]) != len(rows):
                raise ValueError('duplicate materialized task')
            state['task_bindings'][kind] = {'manifest_sha256': file_sha(directory/'manifest.json'),
                                          'tasks_sha256': task_sha}
        targets, evaluator, context = build_repair_materials(payload, *task_sets, domain='codearc')
        private = output/'evaluator-only'
        private.mkdir()
        for name, data in [('training-targets', targets), ('evaluator-tests', evaluator), ('public-context', context)]:
            (private/f'{name}.json').write_text(json.dumps(data, sort_keys=True)+'\n')
        save()
        for seed in (1701, 1702, 1703):
            if any(file_sha(root/n) != h for n, h in hashes.items()):
                raise ValueError('source changed during repair preparation')
            checkpoint = args.training_root/f'belief-seed{seed}/belief.pt'
            report_path = args.selection_root/f'seed{seed}/results.json'
            if file_sha(report_path) != selection_state['runs'][str(seed)]['results_sha256']:
                raise ValueError('selection report changed after completion')
            report = json.loads(report_path.read_text())
            rows = select_repair_rows(payload, targets, context, report, cache_sha256=checksum,
                                      checkpoint_sha256=file_sha(checkpoint), seed=seed)
            if sum(r['split'] == 'primary' for r in rows) != 500:
                raise ValueError('all 500 selected primary sources required')
            state['current_seed'] = seed
            state['runs'][str(seed)] = {'status': 'running'}
            save()
            manifest = write_packet(rows, checkpoint, output/f'actor-seed{seed}', seed=seed,
                binding={'cache_sha256': checksum, 'selection_report_sha256': file_sha(report_path),
                         'cache_proof_sha256': state['cache_proof_sha256'], 'task_bindings': state['task_bindings']})
            state['runs'][str(seed)].update(status='complete', counts=manifest['counts'],
                manifest_sha256=file_sha(output/f'actor-seed{seed}/manifest.json'))
            save()
        state['status'] = 'complete'
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
