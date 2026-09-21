#!/usr/bin/env python3
"""Evaluate complete generated RBR/CodeARC banks with pre-hidden primary locks.

This standalone exploratory supervisor preserves all sources and failed gates.
It does not provision or complete sealed DAG stages.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from pbpf.apbpf.codearc_bank import file_sha, load_bank


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate_populations(banks, *, domain):
    """Require full predeclared splits and global source disjointness."""
    sizes = {'train': 321 if domain == 'rbr' else 212,
             'development-pilot': 16, 'development-remainder': 384, 'primary': 500}
    seen = set()
    common_identity = None
    for name, (run, rows, _) in banks.items():
        identity = tuple(run[k] for k in ('model', 'revision', 'seed', 'max_new_tokens',
                                         'temperature', 'top_p', 'public_tasks_sha256'))
        if common_identity is not None and common_identity != identity:
            raise ValueError('bank model, decoding budget or source identity differs across splits')
        common_identity = identity
        split = 'development' if name.startswith('development-') else name
        sources = {r['source_component_id'] for r in rows}
        if (run['split'] != split or len(rows) != sizes[name] or len(sources) != sizes[name]
                or run['candidates'] != 8 or seen & sources):
            raise ValueError('generation inventory differs from the complete disjoint plan')
        seen |= sources
    if set(banks) != set(sizes):
        raise ValueError('all four predeclared banks are required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['rbr', 'codearc'], required=True)
    parser.add_argument('--public-root', type=Path, required=True)
    parser.add_argument('--evaluator-root', type=Path, required=True)
    parser.add_argument('--train-bank', type=Path, required=True)
    parser.add_argument('--development-pilot-bank', type=Path, required=True)
    parser.add_argument('--development-remainder-bank', type=Path, required=True)
    parser.add_argument('--primary-bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generation-status', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {'train': args.train_bank.resolve(), 'development-pilot': args.development_pilot_bank.resolve(),
             'development-remainder': args.development_remainder_bank.resolve(), 'primary': args.primary_bank.resolve()}
    public, evaluator = args.public_root.resolve(), args.evaluator_root.resolve()
    python = str(root/'.venv/bin/python')
    runner = f'scripts/evaluate_apbpf_{args.domain}_bank.py'
    scripts = [Path(__file__).resolve(), root/runner, root/'scripts/build_apbpf_hard_bank.py',
               root/'scripts/apbpf_visible_evaluator_sandbox.sh']
    modules = ['codearc_bank.py', 'hard_bank.py', 'codearc_execution.py']
    if args.domain == 'codearc':
        scripts.append(root/'scripts/reextract_apbpf_codearc_bank.py')
        modules.append('code_extraction.py')
    else:
        modules.append('rbr_execution.py')
    sources = scripts + [root/'src/pbpf/apbpf'/n for n in modules]
    hashes = {str(f.relative_to(root)): file_sha(f) for f in sources}
    state = {'status': 'waiting_for_generation', 'pid': os.getpid(), 'domain': args.domain,
             'banks': {k: str(v) for k, v in paths.items()}, 'source_sha256': hashes, 'steps': {},
             'scope': 'exploratory standalone full-population replay; no sealed stage completion',
             'public_tasks_sha256': file_sha(public/'tasks.jsonl')}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)

    def save():
        write(output/'status.json', state)

    def execute(name, command):
        if any(file_sha(root/s) != h for s, h in hashes.items()):
            raise ValueError('source changed; new supervisor attempt required')
        state.update(status='running', current=name)
        state['steps'][name] = {'status': 'running', 'command': command}
        save()
        with (output/f'{name}.log').open('x') as log:
            code = subprocess.call(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        state['steps'][name].update(returncode=code, status='complete' if code == 0 else 'execution_failed')
        save()
        if code:
            raise RuntimeError(f'{name} exited {code}')

    save()
    try:
        banks, evaluated = {}, {}
        pending = list(paths)
        while pending:
            for name in list(pending):
                path = paths[name]
                if not (path/'complete.json').exists():
                    continue
                bank = load_bank(path)
                if bank[0]['public_tasks_sha256'] != state['public_tasks_sha256']:
                    raise ValueError('generation bank has different public source data')
                if args.domain == 'codearc':
                    extracted = output/f'{name}-extracted'
                    execute(f'extract-{name}', [python, 'scripts/reextract_apbpf_codearc_bank.py',
                            '--input', str(path), '--output', str(extracted)])
                    path = extracted
                    bank = load_bank(path)
                banks[name] = bank
                evaluated[name] = path
                if name != 'primary':
                    execute(f'evaluate-{name}', [python, '-u', runner, '--evaluator-root', str(evaluator),
                            '--bank', str(path), '--phase', 'all', '--output', str(output/f'{name}-evaluation'),
                            '--workers', '12'])
                pending.remove(name)
            if pending:
                state.update(status='waiting_for_generation', pending=pending.copy())
                save()
                if args.generation_status:
                    upstream = json.loads(args.generation_status.read_text())
                    if upstream['status'] == 'needs_debug':
                        raise RuntimeError('upstream generation needs debug')
                    if upstream['status'] not in ('complete', 'needs_debug') and not Path(f"/proc/{upstream['pid']}").exists():
                        raise RuntimeError('generation supervisor exited without terminal status')
                time.sleep(20)
        validate_populations(banks, domain=args.domain)
        development = []
        for name in ('development-pilot', 'development-remainder'):
            development += json.loads((output/f'{name}-evaluation/bank_groups.json').read_text())
        dev_path = output/'development400-bank.json'
        write(dev_path, development)
        positives = [sum(r['hidden_labels']) for r in development]
        diagnostic = {'groups': 400, 'mixed_groups': sum(0 < n < 8 for n in positives),
                      'all_fail': positives.count(0), 'all_pass': positives.count(8),
                      'minimum_mixed_groups': 300, 'bank_sha256': file_sha(dev_path)}
        diagnostic['passes'] = diagnostic['mixed_groups'] >= 300
        write(output/'development400-hardness.json', diagnostic)
        pilot = output/'development-pilot-gate.json'
        if diagnostic['passes']:
            execute('development-pilot-gate', [python, 'scripts/build_apbpf_hard_bank.py', 'pilot',
                    '--development-bank', str(dev_path), '--output', str(pilot)])
        else:
            state['steps']['development-pilot-gate'] = {'status': 'scientific_gate_failed', **diagnostic}
            save()
        primary = evaluated['primary']
        visible = output/'primary-visible'
        visible.mkdir()
        execute('primary-visible', ['bash', 'scripts/apbpf_visible_evaluator_sandbox.sh', str(public),
                str(primary), str(visible), python, '-u', runner, '--public-root', '/input',
                '--bank', '/bank', '--phase', 'visible', '--output', '/output/evaluation', '--workers', '12'])
        lock = output/'primary500-lock.json'
        execute('primary-lock', [python, 'scripts/build_apbpf_hard_bank.py', 'lock', '--visible-bank',
                str(visible/'evaluation/bank_groups.json'), '--provenance',
                args.domain + '-bank-sha256:' + file_sha(primary/'complete.json'), '--output', str(lock)])
        hidden = output/'primary-hidden'
        execute('primary-hidden', [python, '-u', runner, '--evaluator-root', str(evaluator), '--bank',
                str(primary), '--phase', 'hidden', '--population-lock', str(lock),
                '--output', str(hidden), '--workers', '12'])
        audit = [python, 'scripts/build_apbpf_hard_bank.py', 'audit', '--lock', str(lock),
                 '--hidden-bank', str(hidden/'bank_groups.json'), '--output', str(output/'primary500-audit.json')]
        if diagnostic['passes']:
            audit += ['--pilot-report', str(pilot)]
        execute('primary-audit', audit)
        state.update(status='execution_complete_check_scientific_gates', pending=[])
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
