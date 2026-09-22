#!/usr/bin/env python3
"""Admit disjoint seed-1701 Replay train sources after the fixed 200/500 prefix."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from pbpf.eesd.replay_admission import digest, rename_new_directory, verify_inputs
from pbpf.eesd.replay_materialization import (
    DOMAINS, project_views, select_population, select_training_reserve,
)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('joint-receipt', 'public-candidates', 'evaluator-candidates',
                 'token-receipt', 'token-results', 'assessment-admission', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    public, private, tokens, quarantine, evidence = verify_inputs(
        root, args.joint_receipt, args.public_candidates,
        args.evaluator_candidates, args.token_receipt, args.token_results)
    assessment = select_population(public, tokens, quarantine)
    reserve = select_training_reserve(public, tokens, quarantine)
    assessment_sources = {row['source_id'] for values in assessment.values() for _, row in values}
    reserve_sources = {row['source_id'] for values in reserve.values() for _, row in values}
    if len(assessment_sources) != 1400 or len(reserve_sources) != 400 or assessment_sources & reserve_sources:
        raise ValueError('training reserve overlaps or changes the assessment population')
    admitted = json.loads(args.assessment_admission.read_text())
    if admitted.get('schema') != 'eesd-replay-admission-v1':
        raise ValueError('assessment admission schema mismatch')
    assessment_root = args.assessment_admission.parent
    actual_assessment = {}
    for domain in DOMAINS:
        path = assessment_root / domain / 'public/tasks.jsonl'
        actual_assessment[domain] = [json.loads(line) for line in path.read_text().splitlines()]
        if digest(path) != admitted['outputs'][f'{domain}/public/tasks.jsonl']['sha256']:
            raise ValueError('assessment task seal mismatch')
        expected = [(split, row['task_id'], row['source_id']) for split, row in assessment[domain]]
        actual = [(row['split'], row['task_id'], row['source_component_id'])
                  for row in actual_assessment[domain]]
        if actual != expected:
            raise ValueError('assessment selection differs from the fixed hash prefix')
    public_view, private_view = project_views(reserve, private, allowed_splits=('train',))
    ids = {row['task_id'] for row in public_view}
    token_view = [row for row in tokens if row['task_id'] in ids]
    if len(public_view) != 400 or len(private_view) != 400 or len(token_view) != 400:
        raise ValueError('train reserve population mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    seals = {}
    def write(name, value):
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = value if isinstance(value, bytes) else compact(value) + b'\n'
        path.write_bytes(data)
        seals[name] = {'sha256': digest(path), 'bytes': len(data)}
    for domain in DOMAINS:
        for view, rows in (('public', public_view), ('evaluator', private_view)):
            selected = [row for row in rows if row['domain'] == domain]
            task_name = f'{domain}/{view}/tasks.jsonl'
            write(task_name, b''.join(compact(row) + b'\n' for row in selected))
            write(f'{domain}/{view}/manifest.json', {
                'schema': f'eesd-replay-{view}-train-v1', 'domain': domain,
                'task_kind': 'stdin_synthesis', 'counts': {'train': 200},
                'tasks': seals[task_name],
            })
    write('selected-token-audit.jsonl', b''.join(compact(row) + b'\n' for row in token_view))
    receipt = {'schema': 'eesd-replay-training-reserve-admission-v1',
        'status': 'training-data-admitted-not-generated-not-scored', 'seed': 1701,
        'selection': 'positions 700..899 in each domain after eesd-replay-split-v1|1701| hash ranking',
        'assessment_admission': {'path': str(args.assessment_admission.resolve()),
                                 'sha256': digest(args.assessment_admission)},
        'assessment_sources_sha256': hashlib.sha256(compact(sorted(assessment_sources))).hexdigest(),
        'train_sources_sha256': hashlib.sha256(compact(sorted(reserve_sources))).hexdigest(),
        'assessment_source_overlap': 0, 'evidence': evidence, 'outputs': seals}
    (staging / 'admission.json').write_bytes(compact(receipt) + b'\n')
    rename_new_directory(staging, output)
    print(json.dumps({'status': 'training-reserve-admitted', 'sources': 400,
                      'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
