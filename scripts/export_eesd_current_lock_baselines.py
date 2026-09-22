#!/usr/bin/env python3
"""Expose eight sealed Gemma/Replay direct baselines as paired fresh reports."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_SHA = '8f3098efea7b9a2eb38a9434963cf200d330bfb53e869f4ed3a12280b01500e2'
OUTCOMES = {'PASS', 'WRONG_OUTPUT', 'COMPILE_ERROR', 'RUNTIME_EXCEPTION', 'TIMEOUT'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evaluator_root(domain):
    if domain in ('runbugrun', 'codearc'):
        return ROOT / 'runs/eesd-data' / ('rbr-evaluator' if domain == 'runbugrun'
                                         else 'codearc-evaluator')
    return ROOT / 'runs/eesd-data/replay-extension-locked-20260920' / domain / 'evaluator'


def export(cell, output):
    source = cell['source']
    if 'cache' in source:
        raise ValueError('Qwen/DeepSeek original cells require current-lock re-execution')
    complete_path = Path(source['complete'])
    directory = complete_path.parent
    if sha(complete_path) != source['sha256']:
        raise ValueError('initial direct baseline completion seal differs')
    complete = json.loads(complete_path.read_text())
    run_path = directory / 'run.json'
    if sha(run_path) != complete['run_sha256']:
        raise ValueError('direct baseline run identity differs')
    run = json.loads(run_path.read_text())
    lock = run.get('direct_lock_sha256', run.get('direct_execution_lock_sha256'))
    if (run.get('execution_profile') != 'direct-no-sandbox' or lock != LOCK_SHA
            or run.get('seed') != 1701 or run.get('domain') != cell['domain']
            or run.get('jobs') != 700 or run.get('tests_per_job') != 10
            or len(complete['files']) != 700):
        raise ValueError('direct baseline profile/population differs')
    if run.get('family', cell['family']) != cell['family']:
        raise ValueError('direct baseline model family differs')
    allowed = {'run.json', 'complete.json'}
    rows = []
    for name, digest in complete['files'].items():
        if Path(name).name != name or not name.endswith('.json'):
            raise ValueError('invalid direct baseline record filename')
        path = directory / name
        sidecar = path.with_suffix('.sha256')
        if sha(path) != digest or not sidecar.is_file() or sidecar.read_text().strip() != digest:
            raise ValueError('direct baseline record checksum differs')
        allowed.update((name, sidecar.name))
        row = json.loads(path.read_text())
        outcomes = row.get('outcomes')
        if (row.get('split') not in ('development', 'primary')
                or not isinstance(outcomes, list) or len(outcomes) != 10
                or any(value not in OUTCOMES for value in outcomes)
                or len(row.get('tests', [])) != 10):
            raise ValueError('direct baseline record outcome protocol differs')
        rows.append(row)
    if {path.name for path in directory.iterdir()} != allowed:
        raise ValueError('unexpected direct baseline artifacts')
    primary = [row for row in rows if row['split'] == 'primary']
    development = [row for row in rows if row['split'] == 'development']
    if (len(primary) != 500 or len(development) != 200
            or len({r['source_component_id'] for r in rows}) != 700
            or len({r['problem_id'] for r in rows}) != 700):
        raise ValueError('direct baseline primary/development source coverage differs')
    evaluator = evaluator_root(cell['domain'])
    manifest_sha = sha(evaluator / 'manifest.json')
    with (evaluator / 'tasks.jsonl').open() as stream:
        fixed = {r['task_id']: r for r in (json.loads(line) for line in stream)
                 if r['split'] == 'primary'}
    if (len({r['source_component_id'] for r in fixed.values()}) != 500
            or {r['source_component_id'] for r in fixed.values()}
               != {r['source_component_id'] for r in primary}
            or not {r['problem_id'] for r in primary}.issubset(fixed)
            or any(fixed[r['problem_id']]['source_component_id'] != r['source_component_id']
                   for r in primary)):
        raise ValueError('direct baseline/evaluator primary identity differs')
    records = []
    for row in primary:
        outcomes = row['outcomes']
        records.append({'task_id': row['problem_id'],
                        'source_component_id': row['source_component_id'],
                        'candidate_id': row['task_id'],
                        'visible_all_pass': all(value == 'PASS' for value in outcomes[:4]),
                        'hidden_all_pass': all(value == 'PASS' for value in outcomes[4:]),
                        'all_tests_pass': all(value == 'PASS' for value in outcomes),
                        'outcomes': outcomes})
    passes = sum(r['all_tests_pass'] for r in records)
    if passes != cell['all_ten_pass']:
        raise ValueError('direct baseline count differs from initial score index')
    model = run.get('model', run.get('model_id'))
    report = {'schema': 'eesd-fresh-policy-eval-v1', 'domain': cell['domain'],
              'model': model, 'revision': run['revision'], 'adapter_sha256': None,
              'bank_complete_sha256': source['sha256'],
              'task_manifest_sha256': manifest_sha,
              'execution_profile': 'direct-no-sandbox',
              'execution_lock_sha256': LOCK_SHA,
              'sources': 500, 'all_tests_passes': passes,
              'hidden_passes': sum(r['hidden_all_pass'] for r in records),
              'fresh_all_tests_pass_at_1': passes / 500,
              'fresh_hidden_pass_at_1': sum(r['hidden_all_pass'] for r in records) / 500,
              'visible_all_pass_rate': sum(r['visible_all_pass'] for r in records) / 500,
              'records': records,
              'selection': 'none; one fixed initial candidate per primary source',
              'provenance': 'sealed existing direct execution; no re-execution',
              'direct_run_sha256': sha(run_path)}
    if output.exists():
        report_path = output / 'report.json'
        records_path = output / 'records.jsonl'
        if (not report_path.is_file() or json.loads(report_path.read_text()) != report
                or not records_path.is_file()
                or [json.loads(line) for line in records_path.read_text().splitlines()] != records):
            raise ValueError('existing baseline export differs from sealed source')
    else:
        output.mkdir(parents=True, exist_ok=False)
        with (output / 'records.jsonl').open('x') as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + '\n')
        (output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    return {'domain': cell['domain'], 'family': cell['family'], 'passes': passes,
            'report': str(output / 'report.json'), 'report_sha256': sha(output / 'report.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, default=ROOT /
                   'runs/eesd-direct-20260921/operations/seed1701-initial-baselines.json')
    p.add_argument('--output-root', type=Path, default=ROOT /
                   'runs/eesd-direct-20260921/fresh-baselines')
    args = p.parse_args()
    index = json.loads(args.index.read_text())
    cells = [cell for cell in index['cells'] if 'cache' not in cell['source']]
    if len(cells) != 8 or any(cell['seed'] != 1701 for cell in cells):
        raise ValueError('expected exactly eight Gemma/Replay seed-1701 baselines')
    results = []
    for cell in cells:
        output = args.output_root / cell['domain'] / cell['family'] / 'seed1701'
        results.append(export(cell, output))
        print(json.dumps(results[-1], sort_keys=True), flush=True)
    print(json.dumps({'status': 'eight-current-lock-baselines-exported',
                      'cells': len(results)}), flush=True)


if __name__ == '__main__':
    main()
