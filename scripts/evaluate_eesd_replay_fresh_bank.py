#!/usr/bin/env python3
"""Independently score sealed one-candidate Replay primary banks under the direct profile."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from pbpf.apbpf.codearc_bank import file_sha, load_bank
from pbpf.apbpf.direct_execution import execute_stdin
from pbpf.eesd.direct_runtime import probe_readiness
from pbpf.eesd.training_outcomes import tree_sha


def load_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def work(job):
    task_id, source, candidate_id, code, tests = job
    results = [dict(test_id=str(index), **execute_stdin(
        code, {'input': test['input'], 'expected': test['output']}, timeout=6.0))
        for index, test in enumerate(tests)]
    return {'task_id': task_id, 'source_component_id': source,
            'candidate_id': candidate_id, 'tests': results,
            'visible_all_pass': all(r['outcome'] == 'PASS' for r in results[:4]),
            'hidden_all_pass': all(r['outcome'] == 'PASS' for r in results[4:]),
            'all_tests_pass': all(r['outcome'] == 'PASS' for r in results)}


def public_tasks_digest(run):
    return run.get('bindings', {}).get('public_tasks_sha256') or run.get('public_tasks_sha256')


def load_banks(paths, domain, family, adapter_sha):
    loaded = [load_bank(path) for path in paths]
    base, _, _ = loaded[0]
    expected_schema = ('eesd-gemma-replay-fresh-generation-v1' if family == 'gemma3_4b'
                       else 'eesd-replay-fresh-generation-v1')
    if (base.get('schema') != expected_schema or base.get('domain') != domain
            or base.get('family') != family or base.get('split') != 'primary'
            or base.get('seed') != 1701 or base.get('candidates') != 1
            or base.get('adapter_sha256') != adapter_sha):
        raise ValueError('fresh Replay bank identity differs')
    identity = ('schema', 'domain', 'family', 'model', 'revision', 'split', 'seed',
                'candidates', 'adapter_sha256',
                'max_input_tokens', 'max_new_tokens', 'temperature', 'top_p')
    rows = []
    for run, shard, _ in loaded:
        if (any(run.get(key) != base.get(key) for key in identity)
                or public_tasks_digest(run) != public_tasks_digest(base)):
            raise ValueError('fresh Replay shards differ in model, adapter or sampling')
        rows.extend(shard)
    if len(rows) != 500 or len({r['task_id'] for r in rows}) != 500 or len({r['source_component_id'] for r in rows}) != 500:
        raise ValueError('fresh Replay primary shards must cover 500 unique sources')
    return base, rows, [digest for _, _, digest in loaded]


def verify_bank_ancestry(paths, bundle, admission_sha, domain, family,
                         adapter_sha, public):
    if family in {'qwen25_7b', 'deepseek_6p7b'}:
        from generate_eesd_replay_fresh_bank import expected_run, shard_preflight
        from pbpf.eesd.replay_generation import verify_replay_bank
        for path in paths:
            run = json.loads((path / 'run.json').read_text())
            preflight = shard_preflight(bundle, admission_sha, domain, family,
                                        run['offset'], run['components'])
            expected = expected_run(preflight, family, 1701, run['offset'], adapter_sha)
            verify_replay_bank(path, expected, preflight=preflight)
        return

    from generate_eesd_gemma_fresh_bank import PROOF, PROMPT_SOURCE, source_rows
    source_public, all_rows = source_rows(domain)
    if source_public.resolve() != public.resolve():
        raise ValueError('Gemma fresh source root differs from evaluator admission')
    proof = json.loads(PROOF.read_text())
    for path in paths:
        run = json.loads((path / 'run.json').read_text())
        offset, components = run['offset'], run['components']
        if (type(offset) is not int or type(components) is not int or offset < 0
                or components < 1 or offset + components > len(all_rows)):
            raise ValueError('Gemma fresh shard bounds differ')
        selected = all_rows[offset:offset + components]
        expected = {'task_ids': [row['task_id'] for row in selected],
                    'source_component_ids': [row['source_component_id'] for row in selected],
                    'public_manifest_sha256': file_sha(public / 'manifest.json'),
                    'public_tasks_sha256': file_sha(public / 'tasks.jsonl'),
                    'model': proof['model_id'], 'revision': proof['revision'],
                    'model_proof_sha256': file_sha(PROOF),
                    'prompt_source_sha256': file_sha(PROMPT_SOURCE),
                    'generator_source_sha256': file_sha(Path(__file__).with_name(
                        'generate_eesd_gemma_fresh_bank.py')),
                    'adapter_sha256': adapter_sha}
        if any(run.get(key) != value for key, value in expected.items()):
            raise ValueError('Gemma fresh generation ancestry differs')
        allowed = {'run.json', 'complete.json'}
        for task_id in run['task_ids']:
            name = task_id.replace('/', '-') + '.json'
            allowed.update((name, str(Path(name).with_suffix('.sha256'))))
            file = path / name
            if (not (path / Path(name).with_suffix('.sha256')).is_file()
                    or (path / Path(name).with_suffix('.sha256')).read_text().strip() != file_sha(file)):
                raise ValueError('Gemma fresh record checksum sidecar differs')
        if {item.name for item in path.iterdir()} != allowed:
            raise ValueError('unexpected Gemma fresh bank artifacts')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--admission-sha256', required=True)
    p.add_argument('--domain', choices=['apps_replay', 'codecontests_replay'], required=True)
    p.add_argument('--family', choices=['qwen25_7b', 'deepseek_6p7b', 'gemma3_4b'], required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--bank', type=Path, action='append', required=True)
    p.add_argument('--execution-lock', type=Path, required=True)
    p.add_argument('--execution-lock-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.output.exists() or not 1 <= args.workers <= 4:
        raise ValueError('fresh Replay output must be new and workers between one and four')

    adapter_sha = tree_sha(args.adapter)
    run, bank, digests = load_banks(args.bank, args.domain, args.family, adapter_sha)
    bundle = args.bundle.resolve()
    if file_sha(bundle / 'admission.json') != args.admission_sha256:
        raise ValueError('Replay admission checksum differs')
    admission = json.loads((bundle / 'admission.json').read_text())
    public = bundle / args.domain / 'public'
    evaluator = bundle / args.domain / 'evaluator'
    verify_bank_ancestry(args.bank, bundle, args.admission_sha256, args.domain,
                         args.family, adapter_sha, public)
    for view, root in (('public', public), ('evaluator', evaluator)):
        for name in ('manifest.json', 'tasks.jsonl'):
            path = root / name
            seal = admission['outputs'][f'{args.domain}/{view}/{name}']
            if file_sha(path) != seal['sha256'] or path.stat().st_size != seal['bytes']:
                raise ValueError('Replay materialization checksum differs')
    public_manifest = json.loads((public / 'manifest.json').read_text())
    if (public_tasks_digest(run) != file_sha(public / 'tasks.jsonl')
            or public_tasks_digest(run) != public_manifest['tasks']['sha256']):
        raise ValueError('fresh bank is bound to a different public population')
    public_rows = {r['task_id']: r for r in load_rows(public / 'tasks.jsonl')
                   if r['split'] == 'primary'}
    if (len(public_rows) != 500 or {r['task_id'] for r in bank} != set(public_rows)
            or any(r['source_component_id'] != public_rows[r['task_id']]['source_component_id']
                   for r in bank)):
        raise ValueError('fresh bank differs from the fixed public primary population')

    # Private tests are opened only after all complete banks and public bindings pass.
    ready = probe_readiness(args.execution_lock, args.execution_lock_sha256)
    if ready.get('profile') != 'direct-no-sandbox' or ready.get('status') != 'ready':
        raise ValueError('direct execution readiness failed')
    evaluator_manifest = json.loads((evaluator / 'manifest.json').read_text())
    if (evaluator_manifest.get('schema') != 'eesd-replay-evaluator-v1'
            or evaluator_manifest.get('domain') != args.domain
            or evaluator_manifest.get('counts') != {'development': 200, 'primary': 500}):
        raise ValueError('wrong Replay evaluator population')
    private = {r['task_id']: r for r in load_rows(evaluator / 'tasks.jsonl')
               if r['split'] == 'primary'}
    if len(private) != 500 or set(private) != set(public_rows):
        raise ValueError('Replay private primary inventory differs')
    jobs = []
    for row in bank:
        task_id = row['task_id']
        task = private[task_id]
        public_task = public_rows[task_id]
        if any(task[k] != public_task[k] for k in
               ('source_component_id', 'split', 'domain', 'statement')):
            raise ValueError('Replay public/private source identity differs')
        tests = task['tests']
        if (len(tests) != 10 or [t['id'] for t in tests] != [str(i) for i in range(10)]
                or any({'input': t['input'], 'output': t['output']} != v
                       for t, v in zip(tests[:4], public_task['visible_tests'], strict=True))):
            raise ValueError('Replay ten-test evaluator protocol differs')
        candidate = row['candidates'][0]
        jobs.append((task_id, row['source_component_id'], candidate['candidate_id'],
                     candidate['code'], tests))

    args.output.mkdir(parents=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output / 'records.jsonl').open('x') as stream:
        for index, record in enumerate(pool.map(work, jobs), 1):
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True) + '\n')
            stream.flush()
            if index % 25 == 0:
                print(json.dumps({'completed': index, 'total': len(jobs)}), flush=True)
    if tree_sha(args.adapter) != adapter_sha or file_sha(bundle / 'admission.json') != args.admission_sha256:
        raise ValueError('Replay adapter or admission changed during independent evaluation')
    if any(file_sha(path / 'complete.json') != digest for path, digest in zip(args.bank, digests, strict=True)):
        raise ValueError('Replay candidate bank changed during independent evaluation')
    report = {'schema': 'eesd-fresh-policy-eval-v1', 'domain': args.domain,
              'model': run['model'], 'revision': run['revision'],
              'adapter_sha256': adapter_sha,
              'bank_complete_sha256s': digests,
              'task_manifest_sha256': file_sha(evaluator / 'manifest.json'),
              'execution_profile': 'direct-no-sandbox',
              'execution_lock_sha256': args.execution_lock_sha256,
              'readiness': ready, 'sources': len(records),
              'all_tests_passes': sum(r['all_tests_pass'] for r in records),
              'hidden_passes': sum(r['hidden_all_pass'] for r in records),
              'fresh_all_tests_pass_at_1': sum(r['all_tests_pass'] for r in records) / len(records),
              'fresh_hidden_pass_at_1': sum(r['hidden_all_pass'] for r in records) / len(records),
              'visible_all_pass_rate': sum(r['visible_all_pass'] for r in records) / len(records),
              'records': records,
              'selection': 'none; exactly one fresh candidate per primary source'}
    (args.output / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
