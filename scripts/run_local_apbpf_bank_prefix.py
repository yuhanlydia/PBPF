#!/usr/bin/env python3
"""Wait for public Qwen banks, then run the bank prefix and optional training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from pbpf.apbpf.codearc_bank import file_sha, load_bank
from pbpf.apbpf.config import resolve_config


WORKERS = {
    'materialize': 'run_apbpf_materialize_worker.py',
    'hard_bank_lock': 'run_apbpf_bank_lock_worker.py',
    'execution_cache': 'run_apbpf_execution_cache_worker.py',
    'hard_bank': 'run_apbpf_hard_bank_worker.py',
    'hard_bank_gate': 'run_apbpf_hard_bank_gate_worker.py',
}
BANKS = {
    'rbr': ['rbr-qwen-train321', 'rbr-qwen-development-pilot16',
            'rbr-qwen-development384', 'rbr-qwen-primary500'],
    'codearc': ['codearc-generation-train212-extracted-v2', 'codearc-generation-pilot16-extracted-v2',
                'codearc-generation-development-bounded384-extracted-v2', 'codearc-generation-primary500-extracted-v2'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through-stage', choices=['hard_bank_gate', 'train_belief', 'pair_invariance_gate', 'active_testing_gate', 'selection_gate'], default='hard_bank_gate')
    parser.add_argument('--continue-exploratory', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, output = args.run_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = [*sorted((root/'src/pbpf').rglob('*.py')), *sorted((root/'scripts').glob('*.py')),
               *sorted((root/'scripts').glob('*.sh')), *sorted((root/'configs/apbpf').glob('*.yaml'))]
    hashes = {str(p.relative_to(root)): file_sha(p) for p in sources}
    state = {'status': 'waiting_for_full_public_banks', 'pid': os.getpid(), 'source_sha256': hashes,
             'banks': BANKS, 'scope': f'real stages through {args.through_stage}; later stages remain incomplete',
             'through_stage': args.through_stage, 'continue_exploratory': args.continue_exploratory,
             'claim_status': 'exploratory-predeclared', 'fresh_generation': False,
             'fresh_public_and_hidden_execution': True}

    def save():
        path = output/'status.partial'; path.write_text(json.dumps(state, indent=2)+'\n'); path.replace(output/'status.json')

    save()
    try:
        while not all((run/name/'complete.json').exists() for names in BANKS.values() for name in names):
            upstream = json.loads((run/'rbr_qwen_generation_status.json').read_text())
            if upstream['status'] == 'needs_debug' or not Path(f"/proc/{upstream['pid']}").exists():
                raise RuntimeError('RBR generation incomplete and its supervisor failed/exited')
            time.sleep(20)
        if any(file_sha(root/name) != value for name, value in hashes.items()):
            raise ValueError('declared source changed while waiting; archive this plan and launch a new attempt')
        domains = {}
        for domain, names in BANKS.items():
            domains[domain] = []
            for name in names:
                _, _, checksum = load_bank(run/name)
                domains[domain].append({'path': str(run/name), 'complete_sha256': checksum})
        manifest = output/'candidate-cache-manifest.json'
        manifest.write_text(json.dumps({'schema': 'apbpf-public-generation-cache-v1', 'domains': domains}, indent=2)+'\n')
        python = str(root/'.venv/bin/python')
        site = {'working_directory': str(root), 'timeout_seconds': 604800,
                'paths': {'dataset_root': str(root/'local/data'), 'model_cache': str((root/'local/model-cache').resolve()),
                          'artifact_cache': str(run)},
                'commands': {stage: [python, str(root/'scripts'/name)] for stage, name in WORKERS.items()},
                'worker_files': {stage: str(root/'scripts'/name) for stage, name in WORKERS.items()},
                'worker_revisions': {stage: file_sha(root/'scripts'/name) for stage, name in WORKERS.items()}}
        site['commands']['materialize'] += ['--candidate-cache-manifest', str(manifest),
                                            '--candidate-cache-sha256', file_sha(manifest)]
        if args.through_stage != 'hard_bank_gate':
            for kind in ('baselines', 'belief'):
                stage = 'train_' + kind
                worker = root/'scripts/run_apbpf_train_worker.py'
                site['commands'][stage] = [python, str(worker), '--kind', kind]
                site['worker_files'][stage] = str(worker)
                site['worker_revisions'][stage] = file_sha(worker)
        if args.through_stage in ('pair_invariance_gate', 'active_testing_gate', 'selection_gate'):
            for stage, filename, extra in [
                ('baseline_fairness_gate', 'run_apbpf_fairness_worker.py', []),
                ('association', 'run_apbpf_association_worker.py', []),
                ('association_gate', 'run_apbpf_prediction_gate_worker.py', ['--stage', 'association_gate']),
                ('pair_invariance_gate', 'run_apbpf_prediction_gate_worker.py', ['--stage', 'pair_invariance_gate'])]:
                worker = root/'scripts'/filename
                site['commands'][stage] = [python, str(worker), *extra]
                site['worker_files'][stage] = str(worker)
                site['worker_revisions'][stage] = file_sha(worker)
        if args.through_stage in ('active_testing_gate', 'selection_gate'):
            for stage in ('oracle_headroom', 'oracle_headroom_gate', 'active_testing', 'active_testing_gate'):
                worker = root/'scripts'/('run_apbpf_query_gate_worker.py' if stage.endswith('_gate') else 'run_apbpf_query_worker.py')
                site['commands'][stage] = [python, str(worker), '--stage', stage]
                site['worker_files'][stage] = str(worker)
                site['worker_revisions'][stage] = file_sha(worker)
        if args.through_stage == 'selection_gate':
            for stage in ('selection', 'selection_gate'):
                worker = root/'scripts'/f'run_apbpf_{stage}_worker.py'
                site['commands'][stage] = [python, str(worker)]
                site['worker_files'][stage] = str(worker)
                site['worker_revisions'][stage] = file_sha(worker)
        site_path = output/'site.json'; site_path.write_text(json.dumps(site, indent=2)+'\n')
        experiment = root/'configs/experiments/apbpf_iclr2027.yaml'
        resolved = resolve_config(experiment, 'local_exploratory', site=site_path)
        command = [python, '-m', 'pbpf.apbpf.cli', 'run', '--config', str(experiment),
                   '--profile', 'local_exploratory', '--site', str(site_path),
                   '--output-root', str(output/'runs'), '--through-stage', args.through_stage]
        if args.continue_exploratory:
            command.append('--continue-exploratory')
        state.update(status='running', fingerprint=resolved.fingerprint, command=command,
                     run_directory=str(output/'runs'/resolved.fingerprint))
        save()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
        with (output/'run.log').open('x') as log:
            code = subprocess.call(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        if code not in (0, 20):
            raise RuntimeError(f'prefix execution failed with exit {code}')
        from pbpf.apbpf.stages import report_run
        report = report_run(resolved, output/'runs'/resolved.fingerprint)
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        state.update(status='prefix_complete_gate_failed' if code == 20 else 'prefix_complete',
                     returncode=code, whole_dag_complete=report['complete'])
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error)); save()
        raise


if __name__ == '__main__':
    main()
