"""Generate five fresh candidate shards per trained arm and independently score them."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from pbpf.apbpf.codearc_bank import load_bank
from pbpf.eesd.training_outcomes import NON_ESTIMABLE, tree_sha
from pbpf.eesd.replay_generation import sha

root = Path('/root/PBPF')
run = root / 'runs/eesd-direct-20260921'
ops = run / 'operations'
receipt = ops / 'seed1701-fresh-evaluations.json'
lock_path = ops / 'seed1701-fresh-evaluations.lock'
admission_lock_path = ops / 'seed1701-gpu-admission.lock'
training_receipt = ops / 'seed1701-training-queue.json'
bundle = root / 'runs/eesd-data/replay-extension-locked-20260920'
admission_sha = sha(bundle / 'admission.json')
execution_lock = Path('/root/eesd-migration-20260921/direct-execution-lock-new-host.json')
execution_lock_sha = '8f3098efea7b9a2eb38a9434963cf200d330bfb53e869f4ed3a12280b01500e2'
domains = ('runbugrun', 'codearc', 'apps_replay', 'codecontests_replay')
families = ('qwen25_7b', 'deepseek_6p7b', 'gemma3_4b')
rules = ('eesd_full', 'equal_weight', 'final_correctness', 'scalar_confidence',
         'fixed_mass_dirichlet', 'eed_mean_no_uncertainty', 'eed_no_anchor')
python = str(root / '.venv/bin/python')


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(state):
    temp = receipt.with_suffix('.partial')
    temp.write_text(json.dumps(state, sort_keys=True, indent=2) + '\n')
    temp.replace(receipt)


def active(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ', 1)[1][0] != 'Z'


def training_state():
    try:
        state = json.loads(training_receipt.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if state['status'] == 'failed':
        raise RuntimeError('seed-1701 training queue failed')
    return state


def training_ready():
    state = training_state()
    return bool(state and any(task['status'] == 'complete'
                              for task in state.get('tasks', {}).values()))


def output_for(key, slot):
    domain, family, rule = key.split('/')
    return run / 'fresh-banks' / domain / family / rule / 'seed1701' / f'shard{slot:02d}-of05'


def eval_output(key):
    domain, family, rule = key.split('/')
    return run / 'fresh-evaluations' / domain / family / rule / 'seed1701'


def generation_command(key, slot, adapter):
    domain, family, _ = key.split('/')
    output = output_for(key, slot)
    offset = (slot - 1) * 100
    if family == 'gemma3_4b':
        return [python, str(root / 'scripts/generate_eesd_gemma_fresh_bank.py'),
                '--domain', domain, '--seed', '1701', '--adapter', str(adapter),
                '--offset', str(offset), '--components', '100', '--output', str(output)]
    if domain in ('runbugrun', 'codearc'):
        script = ('generate_apbpf_rbr_bank.py' if domain == 'runbugrun'
                  else 'generate_apbpf_codearc_bank.py')
        public = root / 'runs/eesd-data' / ('rbr-public' if domain == 'runbugrun'
                                           else 'codearc-public')
        return [python, str(root / 'scripts' / script), '--public-root', str(public),
                '--output', str(output), '--family', family, '--adapter', str(adapter),
                '--split', 'primary', '--offset', str(offset), '--components', '100',
                '--candidates', '1', '--seed', '1701']
    return [python, str(root / 'scripts/generate_eesd_replay_fresh_bank.py'),
            '--bundle', str(bundle), '--admission-sha256', admission_sha,
            '--domain', domain, '--family', family, '--adapter', str(adapter),
            '--offset', str(offset), '--components', '100', '--seed', '1701',
            '--output', str(output)]


def evaluation_command(key, adapter):
    domain, _, _ = key.split('/')
    output = eval_output(key)
    banks = [output_for(key, slot) for slot in range(1, 6)]
    if domain in ('runbugrun', 'codearc'):
        evaluator = root / 'runs/eesd-data' / ('rbr-evaluator' if domain == 'runbugrun'
                                               else 'codearc-evaluator')
        command = [python, str(root / 'scripts/evaluate_eesd_fresh_bank.py'),
                   '--domain', 'rbr' if domain == 'runbugrun' else 'codearc',
                   '--evaluator-root', str(evaluator), '--output', str(output),
                   '--workers', '4', '--timeout', '6',
                   '--execution-profile', 'direct-no-sandbox',
                   '--execution-lock', str(execution_lock),
                   '--execution-lock-sha256', execution_lock_sha]
    else:
        family = key.split('/')[1]
        command = [python, str(root / 'scripts/evaluate_eesd_replay_fresh_bank.py'),
                   '--bundle', str(bundle), '--admission-sha256', admission_sha,
                   '--domain', domain, '--family', family,
                   '--adapter', str(adapter), '--output', str(output),
                   '--workers', '4', '--execution-lock', str(execution_lock),
                   '--execution-lock-sha256', execution_lock_sha]
    for bank in banks:
        command += ['--bank', str(bank)]
    return command


def initial_state():
    if file_sha(execution_lock) != execution_lock_sha:
        raise ValueError('fresh evaluation direct lock checksum differs')
    return {'schema': 'eesd-seed1701-fresh-evaluation-queue-v1',
            'status': 'running', 'started_at': time.time(),
            'baseline_manifest_sha256': file_sha(run / 'fresh-baselines/manifest.json'),
            'execution_lock_sha256': execution_lock_sha,
            'target_gpu_tasks_per_gpu': 5, 'cells': {}, 'errors': []}


def admit_completed_training(state, training):
    """Admit each sealed arm once; unfinished arms remain with the trainer queue."""
    changed = False
    for rule in rules:
        for domain in domains:
            for family in families:
                key = f'{domain}/{family}/{rule}'
                task = training['tasks'].get(key)
                if not task or task['status'] != 'complete':
                    continue
                binding_path = Path(task['task']['output']) / 'training-binding.json'
                binding = json.loads(binding_path.read_text())
                if binding.get('schema') != 'eesd-training-binding-v1':
                    raise ValueError('unbound training arm: ' + key)
                binding_sha = file_sha(binding_path)
                if key in state['cells']:
                    if state['cells'][key]['training_binding_sha256'] != binding_sha:
                        raise ValueError('admitted training binding changed: ' + key)
                    continue
                if binding['eligibility']['status'] == NON_ESTIMABLE:
                    state['cells'][key] = {'status': 'non_estimable',
                                           'training_binding_sha256': binding_sha}
                    changed = True
                    continue
                adapter = Path(task['task']['output']) / 'adapter'
                if tree_sha(adapter) != binding['adapter_tree_sha256']:
                    raise ValueError('training adapter checksum differs: ' + key)
                state['cells'][key] = {'status': 'pending', 'adapter': str(adapter),
                                       'adapter_sha256': binding['adapter_tree_sha256'],
                                       'training_binding_sha256': binding_sha,
                                       'shards': [{'status': 'pending', 'slot': slot,
                                                   'output': str(output_for(key, slot))}
                                                  for slot in range(1, 6)],
                                       'evaluation': {'status': 'waiting',
                                                      'output': str(eval_output(key))}}
                changed = True
    return changed


def gpu_state(state):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    mapping = {}
    free = {}
    for line in raw.splitlines():
        index, uuid, available = line.split(', ')
        mapping[uuid] = int(index)
        free[int(index)] = int(available)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    counts = {gpu: 0 for gpu in range(4)}
    visible = set()
    for line in apps.splitlines():
        uuid, pid = line.split(', ')
        counts[mapping[uuid]] += 1
        visible.add(int(pid))
    for key, cell in state['cells'].items():
        for shard in cell.get('shards', []):
            if shard['status'] == 'running' and active(shard['pid']):
                if shard['pid'] not in visible:
                    counts[shard['gpu']] += 1
                if (shard['pid'] not in visible
                        or time.time() - shard.get('started_at', 0) < 120):
                    free[shard['gpu']] -= 5500 if key.split('/')[1] == 'gemma3_4b' else 7500
    try:
        training = json.loads(training_receipt.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        training = {}
    for task in training.get('tasks', {}).values():
        if (task['status'] == 'running' and task['gpu'] is not None
                and active(task['pid'])):
            if task['pid'] not in visible:
                counts[task['gpu']] += 1
            # Model loading understates the later backward peak. Keep this
            # margin even after nvidia-smi begins reporting the trainer.
            free[task['gpu']] -= 6000
            if (task['pid'] not in visible
                    or time.time() - task.get('started_at', 0) < 120):
                free[task['gpu']] -= 16000 if task['family'] == 'gemma3_4b' else 22000
    return counts, free


def launch(command, path, *, gpu, log_key):
    if path.exists():
        raise FileExistsError('fresh output already has a writer: ' + str(path))
    log_path = ops / ('fresh-' + log_key.replace('/', '-') + '.log')
    log = log_path.open('x')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu) if gpu is not None else '',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
           'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(root / 'src')}
    child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT)
    return child, log, str(log_path)


def verify_shard(key, shard, adapter_sha):
    output = Path(shard['output'])
    run_data, records, complete_sha = load_bank(output)
    domain, family, _ = key.split('/')
    if (len(records) != 100 or run_data.get('seed') != 1701
            or run_data.get('split') != 'primary' or run_data.get('candidates') != 1
            or run_data.get('family') != family
            or run_data.get('adapter_sha256') != adapter_sha
            or len({r['source_component_id'] for r in records}) != 100):
        raise ValueError('fresh generation shard seal differs: ' + key)
    if run_data.get('domain', domain) not in (domain, 'rbr' if domain == 'runbugrun' else domain):
        raise ValueError('fresh generation shard domain differs: ' + key)
    shard['complete_sha256'] = complete_sha


def verify_evaluation(key, cell):
    path = Path(cell['evaluation']['output']) / 'report.json'
    report = json.loads(path.read_text())
    domain, _, _ = key.split('/')
    if (report.get('schema') != 'eesd-fresh-policy-eval-v1'
            or report.get('sources') != 500 or len(report.get('records', [])) != 500
            or report.get('adapter_sha256') != cell['adapter_sha256']
            or report.get('execution_lock_sha256') != execution_lock_sha
            or report.get('execution_profile') != 'direct-no-sandbox'
            or report.get('domain') not in (domain, 'rbr' if domain == 'runbugrun' else domain)):
        raise ValueError('fresh independent evaluation seal differs: ' + key)
    cell['evaluation']['report_sha256'] = file_sha(path)


def main():
    import sys
    if '--preflight-only' in sys.argv:
        print(json.dumps({'training_ready': training_ready(),
                          'baseline_manifest': (run / 'fresh-baselines/manifest.json').is_file(),
                          'fresh_queue_exists': receipt.exists()}))
        return
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not training_ready():
            time.sleep(20)
        if receipt.exists():
            state = json.loads(receipt.read_text())
            if (state.get('schema') != 'eesd-seed1701-fresh-evaluation-queue-v1'
                    or state.get('baseline_manifest_sha256') != file_sha(
                        run / 'fresh-baselines/manifest.json')
                    or state.get('execution_lock_sha256') != execution_lock_sha):
                raise ValueError('fresh queue resume identity differs')
        else:
            state = initial_state()
            save(state)
        children = {}
        while True:
            training = training_state()
            if training and admit_completed_training(state, training):
                save(state)
            for key, cell in state['cells'].items():
                if cell['status'] == 'non_estimable':
                    continue
                for shard in cell['shards']:
                    if shard['status'] != 'running':
                        continue
                    owned = children.get(shard['pid'])
                    rc = owned[0].poll() if owned else (None if active(shard['pid']) else 0)
                    if rc is None:
                        continue
                    if owned:
                        owned[1].close()
                    children.pop(shard['pid'], None)
                    shard['returncode'] = rc
                    try:
                        if rc:
                            raise RuntimeError('fresh candidate generator exited nonzero')
                        verify_shard(key, shard, cell['adapter_sha256'])
                        shard['status'] = 'complete'
                    except BaseException as error:
                        shard.update(status='failed', error=repr(error))
                        state['errors'].append(key + f"/shard{shard['slot']}")
                    shard['finished_at'] = time.time()
                    save(state)
                evaluation = cell['evaluation']
                if evaluation['status'] == 'running':
                    owned = children.get(evaluation['pid'])
                    rc = owned[0].poll() if owned else (None if active(evaluation['pid']) else 0)
                    if rc is not None:
                        if owned:
                            owned[1].close()
                        children.pop(evaluation['pid'], None)
                        evaluation['returncode'] = rc
                        try:
                            if rc:
                                raise RuntimeError('independent evaluator exited nonzero')
                            verify_evaluation(key, cell)
                            evaluation['status'] = cell['status'] = 'complete'
                        except BaseException as error:
                            evaluation.update(status='failed', error=repr(error))
                            cell['status'] = 'failed'
                            state['errors'].append(key + '/evaluation')
                        evaluation['finished_at'] = time.time()
                        save(state)
            if state['errors']:
                if any(shard['status'] == 'running' for cell in state['cells'].values()
                       for shard in cell.get('shards', [])) or any(
                       cell.get('evaluation', {}).get('status') == 'running'
                       for cell in state['cells'].values()):
                    time.sleep(20)
                    continue
                state['status'] = 'failed'
                state['finished_at'] = time.time()
                save(state)
                raise RuntimeError('fresh evaluation queue failed: ' + ', '.join(state['errors']))

            # Promote sealed five-shard cells to CPU evaluation, with at most four evaluators.
            active_evals = sum(cell.get('evaluation', {}).get('status') == 'running'
                               for cell in state['cells'].values())
            for key, cell in state['cells'].items():
                if active_evals >= 4:
                    break
                if (cell['status'] != 'pending'
                        or cell['evaluation']['status'] != 'waiting'
                        or not all(shard['status'] == 'complete' for shard in cell['shards'])):
                    continue
                command = evaluation_command(key, cell['adapter'])
                output = Path(cell['evaluation']['output'])
                child, log, log_path = launch(command, output, gpu=None,
                                              log_key=key + '-evaluation')
                children[child.pid] = (child, log)
                cell['evaluation'].update(status='running', pid=child.pid,
                                          command=command, log=log_path,
                                          started_at=time.time())
                active_evals += 1
                save(state)

            with admission_lock_path.open('a') as admission_lock:
                fcntl.flock(admission_lock, fcntl.LOCK_EX)
                counts, free = gpu_state(state)
                for key, cell in state['cells'].items():
                    if cell['status'] != 'pending':
                        continue
                    for shard in cell['shards']:
                        if shard['status'] != 'pending':
                            continue
                        choices = [gpu for gpu in range(4)
                                   if counts[gpu] < 5 and free[gpu] >= 9000]
                        if not choices:
                            break
                        gpu = min(choices, key=lambda g: (counts[g], -free[g]))
                        command = generation_command(key, shard['slot'], cell['adapter'])
                        output = Path(shard['output'])
                        child, log, log_path = launch(command, output, gpu=gpu,
                                                      log_key=key + f'-shard{shard["slot"]:02d}')
                        children[child.pid] = (child, log)
                        shard.update(status='running', pid=child.pid, gpu=gpu,
                                     command=command, log=log_path,
                                     started_at=time.time())
                        counts[gpu] += 1
                        free[gpu] -= 7500 if key.split('/')[1] != 'gemma3_4b' else 5500
                        save(state)
                        print(json.dumps({'cell': key, 'slot': shard['slot'],
                                          'gpu': gpu, 'pid': child.pid}), flush=True)
                    if not any(counts[g] < 5 and free[g] >= 9000 for g in range(4)):
                        break
            if (training and training['status'] == 'complete'
                    and len(training.get('tasks', {})) == 84
                    and len(state['cells']) == 84
                    and all(cell['status'] in {'complete', 'non_estimable'}
                            for cell in state['cells'].values())):
                state['status'] = 'complete'
                state['finished_at'] = time.time()
                save(state)
                return
            time.sleep(10)


if __name__ == '__main__':
    main()
