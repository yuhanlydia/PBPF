"""Refresh the factual seed-1701 progress snapshot without model polling."""
import json
from pathlib import Path
import subprocess
import time

ops = Path('/root/PBPF/runs/eesd-direct-20260921/operations')
snapshot = ops / 'seed1701-live-snapshot.json'
queues = ('per-slot-wave2.json', 'gemma-training-bank-queue.json',
          'deferred-codearc-deepseek-shard.json',
          'other-original-correction-generation.json',
          'gemma-original-correction-generation.json',
          'replay-correction-generations.json',
          'gemma-replay-correction-generations.json',
          'seed1701-training-queue.json',
          'seed1701-fresh-evaluations.json')
logs = ('per-slot-wave2-gpu0-slot5-correction.log',
        'generate-codearc-qwen25_7b-corrections.log',
        'generate-replay-codecontests_replay-qwen25_7b.log',
        'generate-replay-apps_replay-deepseek_6p7b.log',
        'generate-replay-apps_replay-qwen25_7b.log',
        'generate-replay-codecontests_replay-deepseek_6p7b.log',
        'generate-gemma-replay-codecontests_replay.log',
        'generate-gemma-replay-apps_replay.log',
        'generate-codearc-gemma3_4b-corrections.log',
        'generate-codearc-deepseek_6p7b-corrections.log',
        'generate-runbugrun-deepseek_6p7b-corrections.log',
        'generate-runbugrun-gemma3_4b-corrections.log',
        'deferred-codearc-deepseek-shard.log')


def gpu_processes():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                   '--format=csv,noheader'], text=True)
    uuid_to_index = {line.split(', ')[1]:line.split(', ')[0]
                     for line in raw.splitlines()}
    result = {index: [] for index in uuid_to_index.values()}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader'], text=True)
    for line in raw.splitlines():
        uuid, pid = line.split(', ')
        if uuid in uuid_to_index:
            result[uuid_to_index[uuid]].append(int(pid))
    return result


def progress(path):
    if not path.is_file():
        return None
    for line in reversed(path.read_text(errors='replace').splitlines()):
        if not line.startswith('{'):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if any(key in row for key in ('generated', 'completed')):
            return {key: row[key] for key in ('generated', 'completed', 'total')
                    if key in row}
    return None


def training_progress(queue):
    result = {}
    for key, task in queue.get('tasks', {}).items():
        row = {'status': task['status'],
               'response_token_budget': task['task']['expected_report']['response_token_budget']}
        for field in ('pid', 'gpu', 'retry_count'):
            if field in task:
                row[field] = task[field]
        log = task.get('log')
        if log and Path(log).is_file():
            for line in reversed(Path(log).read_text(errors='replace').splitlines()):
                if not line.startswith('{'):
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'optimizer_step' in value:
                    row.update(optimizer_step=value['optimizer_step'],
                               response_tokens=value['response_tokens'])
                    break
                if 'optimizer_steps' in value:
                    row.update(optimizer_step=value['optimizer_steps'],
                               response_tokens=value['response_tokens'])
                    break
        result[key] = row
    return result


def refresh():
    state = json.loads(snapshot.read_text())
    gpu = gpu_processes()
    plans_ready = len(list((ops.parent / 'training-budgets').glob(
        '*/*/round1/training-commands.json')))
    state.update(captured_at_unix=time.time(), gpu_processes=gpu,
                 shortfall={index:max(0, 5-len(pids))
                            for index, pids in gpu.items()},
                 training_plans_ready=plans_ready,
                 shortfall_reason={index: (None if len(pids) >= 5 else
                     f'{len(pids)}/5 live GPU processes; {plans_ready}/12 '
                     'training cell plans sealed; see queue states for remaining '
                     'dependency and memory gates') for index, pids in gpu.items()})
    for name in queues:
        path = ops / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        state['queues'][name] = {
            'status':value.get('status'),
            'cells':{key:cell.get('status') for key, cell in
                     value.get('cells', {}).items()},
            'groups':{key:group.get('status') for key, group in
                      value.get('groups', {}).items()},
            'errors':value.get('errors', [])}
        if name == 'seed1701-training-queue.json':
            state['training_progress'] = training_progress(value)
        if name == 'seed1701-fresh-evaluations.json':
            cells = value.get('cells', {})
            state['fresh_evaluation_progress'] = {
                'admitted_arms': len(cells),
                'completed_arms': sum(cell.get('status') == 'complete'
                                      for cell in cells.values()),
                'non_estimable_arms': sum(cell.get('status') == 'non_estimable'
                                          for cell in cells.values()),
                'completed_shards': sum(shard.get('status') == 'complete'
                                        for cell in cells.values()
                                        for shard in cell.get('shards', []))}
    state['generation_progress'] = {
        name: value for name in logs
        if (value := progress(ops / name)) is not None}
    temp = snapshot.with_suffix('.partial')
    temp.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    temp.replace(snapshot)


if __name__ == '__main__':
    import sys
    while True:
        refresh()
        if '--once' in sys.argv:
            break
        time.sleep(60)
