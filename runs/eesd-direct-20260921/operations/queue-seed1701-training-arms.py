"""Run all seven update arms for each seed-1701 cell after GPU generators finish."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import time

from pbpf.eesd.training_outcomes import NON_ESTIMABLE, tree_sha

root = Path('/root/PBPF')
run = root / 'runs/eesd-direct-20260921'
ops = run / 'operations'
receipt = ops / 'seed1701-training-queue.json'
lock_path = ops / 'seed1701-training-queue.lock'
admission_lock_path = ops / 'seed1701-gpu-admission.lock'
fresh_receipt = ops / 'seed1701-fresh-evaluations.json'
domains = ('runbugrun', 'codearc', 'apps_replay', 'codecontests_replay')
families = ('qwen25_7b', 'deepseek_6p7b', 'gemma3_4b')
upstream = ('per-slot-wave2.json', 'replay-training-bank-queue.json',
            'gemma-training-bank-queue.json',
            'other-original-correction-generation.json',
            'replay-correction-generations.json',
            'gemma-original-correction-generation.json',
            'gemma-replay-correction-generations.json')
check_training_binding = runpy.run_path(str(root / 'scripts/run_eesd_downstream.py'))['check_training_binding']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(state):
    temporary = receipt.with_suffix('.partial')
    temporary.write_text(json.dumps(state, sort_keys=True, indent=2) + '\n')
    temporary.replace(receipt)


def active(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ', 1)[1][0] != 'Z'


def gate():
    values = {}
    for name in upstream:
        path = ops / name
        try:
            value = json.loads(path.read_text())
            status = value['status']
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            status = 'missing'
        values[name] = status
    if any(status == 'failed' for status in values.values()):
        raise RuntimeError('GPU upstream failure: ' + json.dumps(values))
    return values


def plan_paths():
    return {(domain, family): run / 'training-budgets' / domain / family /
            'round1/training-commands.json'
            for domain in domains for family in families}


def available_plans():
    paths = plan_paths()
    plans = {}
    for cell, path in paths.items():
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        tasks = value.get('training_tasks', [])
        if (value.get('schema') != 'eesd-training-command-plan-v1'
                or len(value.get('baseline_rows', [])) != 1
                or value['baseline_rows'][0]['seed'] != 1701
                or value['baseline_rows'][0]['rule'] != 'no_update'
                or len(tasks) != 7
                or len({t['request']['rule'] for t in tasks}) != 7
                or any(t['request']['seed'] != 1701 for t in tasks)
                or any(Path(t['output']).resolve().parent.parent !=
                       (run / 'training' / cell[0] / cell[1] / 'round1').resolve()
                       for t in tasks)
                or any(Path(t['request']['scored']).resolve() !=
                       (run / 'corrections' / cell[0] / cell[1] /
                        'round1/scored-corrections.jsonl').resolve()
                       for t in tasks)):
            raise ValueError('seed-1701 plan has wrong arm population: ' + str(path))
        plans[cell] = (path, value)
    return plans


def initial_state(plans):
    task_entries = {}
    plan_seals = {}
    for (domain, family), (path, plan) in plans.items():
        plan_seals[str(path)] = sha(path)
        for task in plan['training_tasks']:
            rule = task['request']['rule']
            key = f'{domain}/{family}/{rule}'
            if key in task_entries:
                raise ValueError('duplicate seed-1701 training task')
            entry = {'status': 'pending', 'task': task, 'domain': domain,
                     'family': family, 'rule': rule,
                     'zero': task['eligibility']['status'] == NON_ESTIMABLE}
            if Path(task['output']).exists():
                entry['binding'] = check_training_binding(task)
                entry['status'] = 'complete'
            task_entries[key] = entry
    return {'schema': 'eesd-seed1701-training-queue-v1', 'status': 'running',
            'started_at': time.time(), 'plan_sha256': plan_seals,
            'max_gpu_tasks_per_gpu': 5,
            'min_free_memory_mib_by_family': {'qwen25_7b': 30000,
                                               'deepseek_6p7b': 30000,
                                               'gemma3_4b': 24000},
            'launch_cooldown_seconds': 90, 'tasks': task_entries, 'errors': []}


def add_plans(state, plans):
    for cell, (path, plan) in plans.items():
        path_name = str(path)
        digest = sha(path)
        if path_name in state['plan_sha256']:
            if state['plan_sha256'][path_name] != digest:
                raise ValueError('training plan changed after admission: ' + path_name)
            continue
        for task in plan['training_tasks']:
            rule = task['request']['rule']
            key = f'{cell[0]}/{cell[1]}/{rule}'
            if key in state['tasks']:
                raise ValueError('duplicate seed-1701 training task: ' + key)
            entry = {'status': 'pending', 'task': task, 'domain': cell[0],
                     'family': cell[1], 'rule': rule,
                     'zero': task['eligibility']['status'] == NON_ESTIMABLE}
            if Path(task['output']).exists():
                entry['binding'] = check_training_binding(task)
                entry['status'] = 'complete'
            state['tasks'][key] = entry
        state['plan_sha256'][path_name] = digest
        save(state)
    if len(state['tasks']) > 84:
        raise ValueError('training queue exceeded 84 update arms')


def ordered_keys(state):
    return sorted(state['tasks'], key=lambda key: (
        state['tasks'][key]['rule'] != 'eesd_full',
        domains.index(state['tasks'][key]['domain']),
        families.index(state['tasks'][key]['family']),
        state['tasks'][key]['rule']))


def gpu_state():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    uuids = {}
    free = {}
    for line in raw.splitlines():
        index, uuid, available = line.split(', ')
        uuids[uuid] = int(index)
        free[int(index)] = int(available)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    counts = {gpu: 0 for gpu in range(4)}
    visible = set()
    for line in apps.splitlines():
        uuid, pid = line.split(', ')
        counts[uuids[uuid]] += 1
        visible.add(int(pid))
    return counts, free, visible


def reserve_invisible(counts, free, visible, *, gpu, pid, memory_mib,
                      started_at=0):
    if not active(pid):
        return
    if pid not in visible:
        counts[gpu] += 1
    if pid not in visible or time.time() - started_at < 120:
        free[gpu] -= memory_mib


def include_fresh_reservations(counts, free, visible):
    try:
        state = json.loads(fresh_receipt.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return
    for key, cell in state.get('cells', {}).items():
        reserve = 5500 if key.split('/')[1] == 'gemma3_4b' else 7500
        for shard in cell.get('shards', []):
            if shard['status'] == 'running':
                reserve_invisible(counts, free, visible, gpu=shard['gpu'],
                                  pid=shard['pid'], memory_mib=reserve,
                                  started_at=shard.get('started_at', 0))


def checkpoint_available(entry):
    task = entry['task']
    trainer = Path(task['command'][1])
    if trainer.name != 'run_eesd_weighted_sft_checkpointed.py':
        return False
    pointer = Path(task['output']) / 'checkpoints/latest.json'
    try:
        record = json.loads(pointer.read_text())
        identity = record['identity']
        expected = task['expected_report']
        directory = pointer.parent / record['directory']
        if (identity['trainer_source_sha256'] != expected['trainer_source_sha256']
                or sha(trainer) != expected['trainer_source_sha256']
                or identity['input_sha256'] != expected['input_sha256']
                or identity['model_config_sha256'] != expected['model_config_sha256']
                or identity['rule'] != expected['rule']
                or identity['seed'] != expected['seed']
                or identity['response_token_budget'] != expected['response_token_budget']
                or sha(directory / 'state.pt') != record['state_sha256']
                or tree_sha(directory / 'adapter') != record['adapter_sha256']):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def launch(key, entry, gpu):
    task = entry['task']
    retry = entry.get('retry_count', 0)
    resume = bool(entry.get('resume'))
    if Path(task['output']).exists() and not (resume and checkpoint_available(entry)):
        raise FileExistsError('unbound training output: ' + task['output'])
    path = ops / ('train-' + key.replace('/', '-') +
                  (f'-resume{retry}' if resume else '') + '.log')
    log = path.open('x')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu) if gpu is not None else '',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'OPENBLAS_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
           'PYTHONUNBUFFERED': '1'}
    command = task['command'] + (['--resume'] if resume else [])
    child = subprocess.Popen(command, cwd=root, env=env,
                             stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT)
    entry.update(status='running', pid=child.pid, gpu=gpu,
                 started_at=time.time(), log=str(path))
    return child, log


def seal(key, entry):
    task = entry['task']
    if entry.get('returncode') not in (None, 0):
        raise RuntimeError(f'training process failed: {key} rc={entry["returncode"]}')
    binding = Path(task['output']) / 'training-binding.json'
    entry['binding'] = check_training_binding(task, seal=not binding.exists())
    entry['status'] = 'complete'
    entry['finished_at'] = time.time()


def main():
    import sys
    if '--preflight-only' in sys.argv:
        print(json.dumps({'upstream': gate(), 'plans_ready': sum(
            path.is_file() for path in plan_paths().values()),
            'plans_admissible': len(available_plans())}), flush=True)
        return
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate()
        if receipt.exists():
            state = json.loads(receipt.read_text())
            if state.get('schema') != 'eesd-seed1701-training-queue-v1':
                raise ValueError('training queue resume identity differs')
        else:
            state = initial_state({})
            save(state)
        state['min_free_memory_mib_by_family'] = {
            'qwen25_7b': 30000, 'deepseek_6p7b': 30000, 'gemma3_4b': 24000}
        state.pop('min_free_memory_mib', None)
        save(state)
        running = {}
        last_launch = {gpu: 0.0 for gpu in range(4)}
        while True:
            statuses = gate()
            add_plans(state, available_plans())
            for key, entry in state['tasks'].items():
                if entry['status'] != 'running':
                    continue
                child_log = running.get(key)
                rc = child_log[0].poll() if child_log else (None if active(entry['pid']) else 0)
                if rc is None:
                    continue
                if child_log:
                    child_log[1].close()
                entry['returncode'] = rc
                try:
                    seal(key, entry)
                except BaseException as error:
                    if (entry.get('retry_count', 0) < 2
                            and checkpoint_available(entry)):
                        entry.update(status='pending', resume=True,
                                     retry_count=entry.get('retry_count', 0) + 1,
                                     last_failure=repr(error))
                    else:
                        entry.update(status='failed', error=repr(error), finished_at=time.time())
                        state['errors'].append(key)
                running.pop(key, None)
                save(state)
            if state['errors']:
                if any(entry['status'] == 'running' for entry in state['tasks'].values()):
                    time.sleep(20)
                    continue
                state['status'] = 'failed'
                state['finished_at'] = time.time()
                save(state)
                raise RuntimeError('training queue failed: ' + ', '.join(state['errors']))
            pending = [key for key in ordered_keys(state)
                       if state['tasks'][key]['status'] == 'pending']
            if (len(state['tasks']) == 84 and not pending and not running
                    and all(entry['status'] == 'complete'
                            for entry in state['tasks'].values())
                    and all(status == 'complete' for status in statuses.values())):
                state['status'] = 'complete'
                state['finished_at'] = time.time()
                save(state)
                return
            for key in list(pending):
                entry = state['tasks'][key]
                if entry['zero']:
                    child, log = launch(key, entry, None)
                    save(state)
                    rc = child.wait()
                    log.close()
                    entry['returncode'] = rc
                    try:
                        seal(key, entry)
                    except BaseException as error:
                        entry.update(status='failed', error=repr(error), finished_at=time.time())
                        state['errors'].append(key)
                    save(state)
                    pending.remove(key)
                    if state['errors']:
                        break
            if state['errors']:
                continue
            with admission_lock_path.open('a') as admission_lock:
                fcntl.flock(admission_lock, fcntl.LOCK_EX)
                counts, free, visible = gpu_state()
                for key, entry in state['tasks'].items():
                    if entry['status'] == 'running' and entry['gpu'] is not None:
                        reserve = (16000 if entry['family'] == 'gemma3_4b' else 22000)
                        reserve_invisible(counts, free, visible, gpu=entry['gpu'],
                                          pid=entry['pid'], memory_mib=reserve,
                                          started_at=entry.get('started_at', 0))
                include_fresh_reservations(counts, free, visible)
                for key in pending:
                    entry = state['tasks'][key]
                    if entry['zero']:
                        continue
                    required_free = state['min_free_memory_mib_by_family'][entry['family']]
                    choices = [gpu for gpu in range(4)
                               if counts[gpu] < 5 and free[gpu] >= required_free
                               and time.time() - last_launch[gpu] >= 90]
                    if not choices:
                        break
                    gpu = min(choices, key=lambda g: (counts[g], -free[g]))
                    child, log = launch(key, entry, gpu)
                    running[key] = (child, log)
                    counts[gpu] += 1
                    free[gpu] -= 16000 if entry['family'] == 'gemma3_4b' else 22000
                    last_launch[gpu] = time.time()
                    save(state)
                    print(json.dumps({'task': key, 'gpu': gpu, 'pid': child.pid}), flush=True)
            time.sleep(20)


if __name__ == '__main__':
    main()
