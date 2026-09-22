"""Keep the seed-1701 schedulers alive and record actual four-GPU occupancy."""

import fcntl
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/root/PBPF')
OPS = ROOT / 'runs/eesd-direct-20260921/operations'
PYTHON = str(ROOT / '.venv/bin/python')
SERVICES = {
    'training': ('queue-seed1701-training-arms.py', 'seed1701-training-queue.json'),
    'evaluation': ('queue-seed1701-fresh-evaluations.py', 'seed1701-fresh-evaluations.json'),
    'monitor': ('monitor-seed1701.py', None),
}


def processes(script):
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
            state = (path.parent / 'stat').read_text().split(') ', 1)[1][0]
        except (OSError, IndexError):
            continue
        if state != 'Z' and str(OPS / script).encode() in args:
            found.append(int(path.parent.name))
    return found


def queue_status(filename):
    if filename is None:
        return None
    try:
        return json.loads((OPS / filename).read_text())['status']
    except (OSError, ValueError, KeyError):
        return 'missing'


def gpu_processes():
    uuid_rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    indexes = {uuid: index for index, uuid in
               (line.split(', ') for line in uuid_rows.splitlines())}
    result = {index: [] for index in indexes.values()}
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
         '--format=csv,noheader'], text=True)
    for line in rows.splitlines():
        uuid, pid = line.split(', ')
        if uuid in indexes:
            result[indexes[uuid]].append(int(pid))
    return result


def save(value):
    path = OPS / 'seed1701-supervisor.json'
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    with (OPS / 'seed1701-supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            service_state = {}
            for name, (script, receipt) in SERVICES.items():
                pids = processes(script)
                status = queue_status(receipt)
                if not pids and status not in ('complete', 'failed'):
                    log = (OPS / f'seed1701-{name}-supervised.log').open('a')
                    child = subprocess.Popen(
                        [PYTHON, str(OPS / script)], cwd=ROOT,
                        stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    log.close()
                    pids = [child.pid]
                service_state[name] = {'pids': pids, 'queue_status': status}
            try:
                gpu = gpu_processes()
                gpu_error = None
            except (OSError, subprocess.CalledProcessError, ValueError) as error:
                gpu, gpu_error = {}, repr(error)
            save({'captured_at_unix': time.time(), 'services': service_state,
                  'gpu_processes': gpu, 'gpu_error': gpu_error,
                  'target_tasks_per_gpu': 5,
                  'live_tasks_per_gpu': {index: len(pids)
                                         for index, pids in gpu.items()}})
            if all(service_state[name]['queue_status'] == 'complete'
                   for name in ('training', 'evaluation')):
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
