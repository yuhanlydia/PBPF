#!/usr/bin/env python3
"""Lock complete repair generations, then freshly execute all six repair arms."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.codearc_execution import execute_call
from pbpf.apbpf.rbr_execution import execute_stdin

ARMS = ('no_latent', 'mean', 'map', 'random', 'sample_once', 'token_remix_fault')


def execute_one(job):
    name, report, task, domain = job
    execute = execute_call if domain == 'codearc' else execute_stdin
    tests = [{'test_id': t['id'], **execute(report['code'], t, timeout=6.)} for t in task['tests']]
    outcomes = [t['outcome'] for t in tests]
    return name, {'binding': report['binding'], 'code_sha256': report['code_sha256'], 'tests': tests,
                  'hidden_success': all(o == 'PASS' for o in outcomes[4:]),
                  'all_ten_success': all(o == 'PASS' for o in outcomes),
                  'future_pass_fraction': outcomes[4:].count('PASS')/6}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation-root', type=Path, required=True)
    p.add_argument('--packet-root', type=Path, required=True)
    p.add_argument('--evaluator-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()
    if args.workers < 1:
        raise ValueError('positive execution worker count required')
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    generation = json.loads((args.generation_root/'status.json').read_text())
    seeds, domain = generation['seeds'], generation['domain']
    sources = [Path(__file__).resolve(), *sorted((root/'src/pbpf').rglob('*.py'))]
    hashes = {str(f.relative_to(root)): file_sha(f) for f in sources}
    state = {'status': 'waiting_for_generation', 'pid': os.getpid(), 'source_sha256': hashes,
             'seeds': seeds, 'domain': domain, 'smoke_only': generation['smoke_only'], 'runs': {},
             'timeout_seconds': 6., 'scope': 'fresh isolated execution of all generated arms; exploratory supportive repair'}

    def save():
        path = output/'status.partial'
        path.write_text(json.dumps(state, indent=2)+'\n')
        path.replace(output/'status.json')

    save()
    try:
        for seed in seeds:
            directory = args.generation_root/f'seed{seed}'
            while not (directory/'generation-complete.json').exists():
                live = json.loads((args.generation_root/'status.json').read_text())
                if live['status'] == 'needs_debug' or not Path(f"/proc/{live['pid']}").exists():
                    raise RuntimeError('generation failed or exited before completing this seed')
                time.sleep(20)
            if any(file_sha(root/n) != h for n, h in hashes.items()):
                raise ValueError('execution sources changed while waiting')
            complete = json.loads((directory/'generation-complete.json').read_text())
            packet = args.packet_root/f'actor-seed{seed}'
            packet_sha = file_sha(packet/'manifest.json')
            manifest = json.loads((packet/'manifest.json').read_text())
            if (complete['identity']['seed'] != seed or complete['identity']['domain'] != domain
                    or complete['identity']['packet_manifest_sha256'] != packet_sha
                    or generation['packets'][str(seed)]['manifest_sha256'] != packet_sha
                    or file_sha(packet/'actor-rows.jsonl') != manifest['files']['actor-rows.jsonl']):
                raise ValueError('repair packet/generation identity mismatch')
            rows = [json.loads(line) for line in (packet/'actor-rows.jsonl').read_text().splitlines()]
            selected = {r['task_id']: r for r in rows if r['split'] == 'primary'}
            expected = {(candidate, arm) for candidate in selected for arm in ARMS}
            reports, observed = {}, set()
            for name, checksum in complete['files'].items():
                if Path(name).name != name or file_sha(directory/'generated'/name) != checksum:
                    raise ValueError('generated continuation differs from sealed inventory')
                report = json.loads((directory/'generated'/name).read_text())
                binding = report['binding']
                key = binding['task_id'], binding['arm']
                if key not in expected or key in observed:
                    raise ValueError('missing, duplicate or unrequested source/arm')
                row = selected[binding['task_id']]
                if (binding['seed'] != seed or binding['actor_packet_sha256'] != packet_sha
                        or binding['source_component_id'] != row['source_component_id']
                        or binding['problem_id'] != row['problem_id']
                        or binding['projector_sha256'] != file_sha(directory/'projector.pt')
                        or report['code_sha256'] != hashlib.sha256(report['code'].encode()).hexdigest()):
                    raise ValueError('repair continuation lineage/content mismatch')
                reports[name] = report
                observed.add(key)
            if observed != expected or (not generation['smoke_only'] and len(selected) != 500):
                raise ValueError('complete all-source, six-arm repair inventory required')
            target = output/f'seed{seed}'
            target.mkdir()
            lock = {'schema': 'apbpf-repair-pre-hidden-lock-v1', 'seed': seed,
                    'generation_complete_sha256': file_sha(directory/'generation-complete.json'),
                    'packet_manifest_sha256': packet_sha, 'generated_files': complete['files'],
                    'source_candidate_inventory': {k: r['source_component_id'] for k, r in selected.items()},
                    'arms': list(ARMS), 'written_before_private_tests_opened': True}
            lock_path = target/'pre-hidden-lock.json'
            with lock_path.open('x') as stream:
                json.dump(lock, stream, indent=2)
            # First private task-file access happens after this seed's full
            # continuation inventory has been sealed above.
            binding = manifest['binding']['task_bindings']['evaluator']
            if (file_sha(args.evaluator_root/'manifest.json') != binding['manifest_sha256']
                    or file_sha(args.evaluator_root/'tasks.jsonl') != binding['tasks_sha256']):
                raise ValueError('evaluator tasks differ from original packet materialization')
            with (args.evaluator_root/'tasks.jsonl').open() as stream:
                tasks = {r['task_id']: r for r in map(json.loads, stream)}
            jobs = []
            for name, report in reports.items():
                row = selected[report['binding']['task_id']]
                task = tasks[row['problem_id']]
                if (task['split'] != 'primary' or task['source_component_id'] != row['source_component_id']
                        or [t['id'] for t in task['tests']] != [str(i) for i in range(10)]
                        or any(t['hidden'] is not (i >= 4) for i, t in enumerate(task['tests']))):
                    raise ValueError('private repair execution source or protocol mismatch')
                jobs.append((name, report, task, domain))
            state.update(status='running', current_seed=seed)
            state['runs'][str(seed)] = {'status': 'running', 'completed': 0, 'total': len(jobs),
                                       'pre_hidden_lock_sha256': file_sha(lock_path)}
            save()
            result_dir = target/'executions'
            result_dir.mkdir()
            measured = []
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for name, result in pool.map(execute_one, jobs, chunksize=1):
                    (result_dir/name).write_text(json.dumps(result, sort_keys=True)+'\n')
                    measured.append(result)
                    state['runs'][str(seed)]['completed'] = len(measured)
                    if len(measured) % 16 == 0:
                        save()
            summary = {arm: {'sources': len(selected),
                'hidden_successes': sum(r['hidden_success'] for r in measured if r['binding']['arm'] == arm),
                'all_ten_successes': sum(r['all_ten_success'] for r in measured if r['binding']['arm'] == arm),
                'future_pass_fraction': sum(r['future_pass_fraction'] for r in measured if r['binding']['arm'] == arm)/len(selected)} for arm in ARMS}
            (target/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            receipt = {'pre_hidden_lock_sha256': file_sha(lock_path), 'summary': summary,
                       'files': {p.name: file_sha(p) for p in result_dir.iterdir()}}
            (target/'complete.json').write_text(json.dumps(receipt, indent=2)+'\n')
            state['runs'][str(seed)].update(status='complete', complete_sha256=file_sha(target/'complete.json'))
            state['status'] = 'waiting_for_generation'
            save()
        state['status'] = 'complete'
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
