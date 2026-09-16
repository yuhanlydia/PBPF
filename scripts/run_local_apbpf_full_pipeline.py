#!/usr/bin/env python3
"""Wait for frozen local banks and provision all 21 real exploratory stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_local_apbpf_bank_prefix import BANKS
from pbpf.apbpf.codearc_bank import file_sha, load_bank
from pbpf.apbpf.config import resolve_config
from pbpf.apbpf.stages import STAGES, report_run


def read_status(path):
    # Older bank supervisors publish in place. A partial write is not failure.
    for attempt in range(5):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            if attempt == 4:
                raise
            time.sleep(.2)


def make_site(root, run, candidate_manifest, replication_manifest, gpu):
    python = str(root/'.venv/bin/python')
    commands = {}
    def add(stage, filename, *extra):
        commands[stage] = [python, str(root/'scripts'/filename), *extra]
    for stage in ('materialize', 'execution_cache', 'association', 'association_gate'):
        add(stage, 'run_apbpf_replication_bridge.py', '--stage', stage)
    commands['materialize'] += ['--candidate-cache-manifest', str(candidate_manifest),
        '--candidate-cache-sha256', file_sha(candidate_manifest),
        '--replication-cache-manifest', str(replication_manifest),
        '--replication-cache-sha256', file_sha(replication_manifest)]
    for stage, filename in {
        'hard_bank_lock': 'run_apbpf_bank_lock_worker.py',
        'hard_bank': 'run_apbpf_hard_bank_worker.py',
        'hard_bank_gate': 'run_apbpf_hard_bank_gate_worker.py',
        'baseline_fairness_gate': 'run_apbpf_fairness_worker.py',
        'selection': 'run_apbpf_selection_worker.py',
        'selection_gate': 'run_apbpf_selection_gate_worker.py',
        'replication': 'run_apbpf_replication_worker.py',
        'replication_gate': 'run_apbpf_replication_gate_worker.py',
        'paper_tables': 'run_apbpf_paper_tables_worker.py',
    }.items():
        add(stage, filename)
    for kind in ('baselines', 'belief'):
        add('train_'+kind, 'run_apbpf_train_worker.py', '--kind', kind)
    add('pair_invariance_gate', 'run_apbpf_prediction_gate_worker.py', '--stage', 'pair_invariance_gate')
    for stage in ('oracle_headroom', 'oracle_headroom_gate', 'active_testing', 'active_testing_gate'):
        add(stage, 'run_apbpf_query_gate_worker.py' if stage.endswith('_gate') else 'run_apbpf_query_worker.py',
            '--stage', stage)
    add('repair', 'run_apbpf_repair_worker.py', '--gpu', gpu)
    if set(commands) != set(STAGES):
        raise ValueError('full pipeline must provision every real stage')
    return {'working_directory': str(root), 'timeout_seconds': 604800,
            'paths': {'dataset_root': str(root/'local/data'),
                      'model_cache': str((root/'local/model-cache').resolve()), 'artifact_cache': str(run)},
            'commands': commands, 'worker_files': {s: c[1] for s, c in commands.items()},
            'worker_revisions': {s: file_sha(c[1]) for s, c in commands.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--materialized-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', choices=['0', '1', '2'], default='0')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, output = args.run_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = [*sorted((root/'src/pbpf').rglob('*.py')), *sorted((root/'scripts').glob('*.py')),
             *sorted((root/'scripts').glob('*.sh')), *sorted((root/'configs/apbpf').glob('*.yaml')),
             root/'configs/experiments/apbpf_iclr2027.yaml']
    hashes = {str(path.relative_to(root)): file_sha(path) for path in files}
    evaluations = {'rbr': run/'rbr-deepseek-full-evaluation-v1', 'codearc': run/'codearc-deepseek-full-evaluation-v2'}
    materialized = args.materialized_root.resolve()
    state = {'status': 'waiting_for_full_banks_and_replication_replay', 'pid': os.getpid(),
             'source_sha256': hashes, 'planned_stages': list(STAGES), 'runs': {},
             'claim_status': 'exploratory-predeclared', 'fresh_candidate_generation': False,
             'fresh_primary_family_execution': True, 'fresh_replication_family_execution': False,
             'fresh_replication_fitting': True, 'gpu': args.gpu,
             'scope': 'all 21 local adapters provisioned; neither execution completion nor gate success implied'}
    def save():
        path = output/'status.partial'; path.write_text(json.dumps(state, indent=2)+'\n'); path.replace(output/'status.json')
    def check_sources():
        if any(file_sha(root/n) != sha for n, sha in hashes.items()):
            raise ValueError('declared source changed; a new attempt is required')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
    python = str(root/'.venv/bin/python')
    def execute(name, command, allowed=(0,)):
        check_sources()
        state.update(status='running', current=name)
        state['runs'][name] = {'command': command, 'status': 'running'}; save()
        with (output/f'{name}.log').open('x') as log:
            code = subprocess.call(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        state['runs'][name].update(returncode=code, status='complete' if code in allowed else 'execution_failed'); save()
        if code not in allowed:
            raise RuntimeError(f'{name} exited {code}')
        return code
    save()
    try:
        while True:
            pending = []
            if not all((run/n/'complete.json').exists() for names in BANKS.values() for n in names):
                upstream = read_status(run/'rbr_qwen_generation_status.json')
                if upstream['status'] == 'needs_debug' or not Path(f"/proc/{upstream['pid']}").exists():
                    raise RuntimeError('Qwen bank generation exited incomplete')
                pending.append('qwen_public_banks')
            for domain, path in evaluations.items():
                upstream = read_status(path/'status.json')
                if upstream['status'] != 'execution_complete_check_scientific_gates':
                    if upstream['status'] == 'needs_debug' or not Path(f"/proc/{upstream['pid']}").exists():
                        raise RuntimeError(f'{domain} replication replay exited incomplete')
                    pending.append(domain+'_deepseek_full_replay')
            state['waiting_on'] = pending; save()
            if not pending:
                break
            time.sleep(20)
        check_sources()
        domains = {}
        for domain, names in BANKS.items():
            domains[domain] = [{'path': str(run/n), 'complete_sha256': load_bank(run/n)[2]} for n in names]
        candidate = output/'candidate-cache-manifest.json'
        candidate.write_text(json.dumps({'schema': 'apbpf-public-generation-cache-v1', 'domains': domains}, indent=2)+'\n')
        replica = {}
        for domain, evaluation in evaluations.items():
            cache = output/f'{domain}-replication-input'
            public, evaluator = materialized/f'{domain}-public', materialized/f'{domain}-evaluator'
            execute(domain+'-cache', [python, 'scripts/build_apbpf_full_replay_cache.py',
                '--evaluation-root', str(evaluation), '--public-root', str(public), '--evaluator-root', str(evaluator),
                '--domain', domain, '--family', 'deepseek', '--output', str(cache)])
            replica[domain] = {'evaluation_root': str(evaluation), 'public_root': str(public),
                'evaluator_root': str(evaluator), 'cache': str(cache/'cache.json'), 'proof': str(cache/'proof.json'),
                'cache_sha256': file_sha(cache/'cache.json'), 'proof_sha256': file_sha(cache/'proof.json')}
        replication = output/'replication-cache-manifest.json'
        replication.write_text(json.dumps({'schema': 'apbpf-root-replication-cache-v1', 'domains': replica}, indent=2)+'\n')
        site = output/'site.json'
        site.write_text(json.dumps(make_site(root, run, candidate, replication, args.gpu), indent=2)+'\n')
        experiment = root/'configs/experiments/apbpf_iclr2027.yaml'
        resolved = resolve_config(experiment, 'local_exploratory', site=site)
        state.update(fingerprint=resolved.fingerprint, run_directory=str(output/'runs'/resolved.fingerprint)); save()
        code = execute('pipeline', [python, '-m', 'pbpf.apbpf.cli', 'run', '--config', str(experiment),
            '--profile', 'local_exploratory', '--site', str(site), '--output-root', str(output/'runs'),
            '--continue-exploratory'], allowed=(0, 20))
        report = report_run(resolved, output/'runs'/resolved.fingerprint)
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        state.update(status='execution_complete_check_scientific_gates' if report['complete'] else 'needs_debug',
                     returncode=code, whole_dag_complete=report['complete']); save()
        if not report['complete']:
            raise RuntimeError('full pipeline exited without completing every stage')
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error)); save(); raise


if __name__ == '__main__':
    main()
