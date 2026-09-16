#!/usr/bin/env python3
"""Supervise local CodeARC replay evaluations; scientific failures remain failures.

This is an exploratory local workflow, not a substitute for the sealed stage DAG.
It waits for immutable generation manifests, scores all development groups, then
locks the full primary bank using only visible tests before hidden execution.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    args = p.parse_args()
    root, run = args.workspace.resolve(), args.run_root.resolve()
    state_path = run / 'codearc_pipeline_status.json'
    if state_path.exists():
        raise FileExistsError('supervisor state already exists; inspect before resuming')
    python = str(root / '.venv/bin/python')
    sources = [root / 'scripts' / name for name in ('evaluate_apbpf_codearc_bank.py',
               'build_apbpf_hard_bank.py', 'apbpf_visible_evaluator_sandbox.sh', 'run_local_codearc_pipeline.py', 'reextract_apbpf_codearc_bank.py')]
    sources += [root / 'src/pbpf/apbpf' / name for name in ('codearc_execution.py', 'codearc_bank.py', 'hard_bank.py', 'code_extraction.py')]
    source_hashes = {str(path): sha(path) for path in sources}
    state = {'status': 'running', 'scope': 'exploratory replay; all original gate thresholds retained',
             'started_at': datetime.datetime.now().astimezone().isoformat(), 'pid': os.getpid(),
             'source_sha256': source_hashes, 'steps': {}}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)

    def save():
        state['updated_at'] = datetime.datetime.now().astimezone().isoformat()
        write(state_path, state)

    def execute(name, command, *, allowed_gate_failure=False):
        for path, digest in source_hashes.items():
            if sha(path) != digest:
                raise ValueError('source changed during pipeline; new attempt required: ' + path)
        state['steps'][name] = {'status': 'running', 'command': command}
        save()
        with (run / (name + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        state['steps'][name].update(exit_code=result.returncode,
            status='complete' if result.returncode == 0 else 'gate_failed' if allowed_gate_failure else 'execution_failed')
        save()
        if result.returncode and not allowed_gate_failure:
            raise RuntimeError(f'{name} failed with exit {result.returncode}; see its log')
        return result.returncode

    def evaluate(bank, output):
        return [python, '-u', 'scripts/evaluate_apbpf_codearc_bank.py', '--evaluator-root',
                str(run / 'codearc-evaluator-v1'), '--bank', str(run / bank), '--phase', 'all',
                '--output', str(run / output), '--workers', '12']

    save()
    try:
        pending = {
            'codearc-pilot16-evaluation-v5': 'codearc-generation-pilot16',
            'codearc-train212-evaluation-v5': 'codearc-generation-train212',
            'codearc-development384-evaluation-v5': 'codearc-generation-development-bounded384',
        }
        while pending:
            for output, bank in list(pending.items()):
                if (run / bank / 'complete.json').exists():
                    corrected = bank + '-extracted-v2'
                    execute('reextract-' + bank, [python, 'scripts/reextract_apbpf_codearc_bank.py',
                        '--input', str(run / bank), '--output', str(run / corrected)])
                    execute(output, evaluate(corrected, output))
                    del pending[output]
            if pending:
                time.sleep(5)
        pilot_path = run / 'codearc-pilot16-evaluation-v5/results.json'
        while not pilot_path.exists():
            time.sleep(5)
        development = []
        for name in ('codearc-pilot16-evaluation-v5', 'codearc-development384-evaluation-v5'):
            development += json.loads((run / name / 'bank_groups.json').read_text())
        if len(development) != 400 or len({r['source_component_id'] for r in development}) != 400:
            raise ValueError('development inventory is not the complete disjoint 400-group population')
        dev_path = run / 'codearc-development400-bank.json'
        with dev_path.open('x') as stream:
            json.dump(development, stream, indent=2)
        positives = [sum(r['hidden_labels']) for r in development]
        diagnostic = {'groups': len(development), 'mixed_groups': sum(0 < n < 8 for n in positives),
                      'all_fail': positives.count(0), 'all_pass': positives.count(8),
                      'minimum_mixed_groups': 300,
                      'bank_sha256': sha(dev_path), 'scope': 'development only'}
        diagnostic['passes'] = diagnostic['mixed_groups'] >= 300
        write(run / 'codearc-development400-hardness.json', diagnostic)
        if diagnostic['passes']:
            execute('codearc-development-pilot-gate', [python, 'scripts/build_apbpf_hard_bank.py', 'pilot',
                    '--development-bank', str(dev_path), '--output', str(run / 'codearc-development-pilot-sealed.json')])
        else:
            state['steps']['codearc-development-pilot-gate'] = {'status': 'scientific_gate_failed', **diagnostic}
            save()
        primary = run / 'codearc-generation-primary500'
        while not (primary / 'complete.json').exists():
            time.sleep(5)
        corrected_primary = run / 'codearc-generation-primary500-extracted-v2'
        execute('reextract-primary500', [python, 'scripts/reextract_apbpf_codearc_bank.py',
            '--input', str(primary), '--output', str(corrected_primary)])
        primary = corrected_primary
        visible_root = run / 'codearc-primary-visible-v5'
        visible_root.mkdir()
        execute('codearc-primary-visible', ['bash', 'scripts/apbpf_visible_evaluator_sandbox.sh',
            str(run / 'codearc-public-v1'), str(primary), str(visible_root), python, '-u',
            'scripts/evaluate_apbpf_codearc_bank.py', '--public-root', '/input', '--bank', '/bank',
            '--phase', 'visible', '--output', '/output/evaluation', '--workers', '12'])
        lock_path = run / 'codearc-primary500-lock.json'
        execute('codearc-primary-lock', [python, 'scripts/build_apbpf_hard_bank.py', 'lock',
            '--visible-bank', str(visible_root / 'evaluation/bank_groups.json'),
            '--provenance', 'codearc-bank-sha256:' + sha(primary / 'complete.json'), '--output', str(lock_path)])
        execute('codearc-primary-hidden', [python, '-u', 'scripts/evaluate_apbpf_codearc_bank.py',
            '--evaluator-root', str(run / 'codearc-evaluator-v1'), '--bank', str(primary),
            '--phase', 'hidden', '--population-lock', str(lock_path),
            '--output', str(run / 'codearc-primary-hidden-v5'), '--workers', '12'])
        audit = [python, 'scripts/build_apbpf_hard_bank.py', 'audit', '--lock', str(lock_path),
                 '--hidden-bank', str(run / 'codearc-primary-hidden-v5/bank_groups.json'),
                 '--output', str(run / 'codearc-primary500-audit.json')]
        if diagnostic['passes']:
            audit += ['--pilot-report', str(run / 'codearc-development-pilot-sealed.json')]
        execute('codearc-primary-audit', audit)
        state['status'] = 'execution_complete_check_scientific_gates'
        save()
    except BaseException as exc:
        state.update(status='needs_debug', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    main()
