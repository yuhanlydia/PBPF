#!/usr/bin/env python3
"""After Qwen exits GPU0, adopt live DeepSeek work and split remaining banks."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from pbpf.apbpf.codearc_bank import file_sha, load_bank

BANKS = {
    'rbr-deepseek-development-pilot16': ('development', 16, 0),
    'rbr-deepseek-train321': ('train', 321, 0),
    'rbr-deepseek-development384': ('development', 384, 16),
    'rbr-deepseek-primary500': ('primary', 500, 0),
}


def read_json(path):
    for attempt in range(10):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            if attempt == 9:
                raise
            time.sleep(.2)


def process_info(pid):
    p = Path('/proc')/str(pid)
    try:
        stat = (p/'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': stat[0], 'ppid': int(stat[1]), 'start_ticks': stat[19],
                'command': (p/'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                'children': [int(x) for x in (p/'task'/str(pid)/'children').read_text().split()]}
    except FileNotFoundError:
        return None


def same_process(actual, expected):
    return actual is not None and all(actual[k] == expected[k] for k in ('pid', 'start_ticks', 'command'))


def park_parent(expected):
    actual = process_info(expected['pid'])
    if not same_process(actual, expected) or actual['state'] in ('T', 't', 'Z'):
        raise ValueError('original parent identity changed or already stopped')
    os.kill(expected['pid'], signal.SIGSTOP)
    try:
        for _ in range(50):
            actual = process_info(expected['pid'])
            if not same_process(actual, expected):
                raise RuntimeError('parent disappeared while parking')
            if actual['state'] in ('T', 't'):
                return actual
            time.sleep(.1)
        raise RuntimeError('parent did not enter stopped state')
    except BaseException:
        if same_process(process_info(expected['pid']), expected):
            os.kill(expected['pid'], signal.SIGCONT)
        raise


def retire_parked_parent(expected):
    """Retire only a verified parked scheduler with no nonterminal children."""
    actual = process_info(expected['pid'])
    if not same_process(actual, expected) or actual['state'] not in ('T', 't'):
        raise ValueError('refusing to retire an unverified or running parent')
    children = [process_info(pid) for pid in actual['children']]
    if any(child is not None and child['state'] != 'Z' for child in children):
        raise ValueError('refusing to retire a parent with live work')
    os.kill(expected['pid'], signal.SIGKILL)
    return {'retired_parent': expected, 'terminal_children': children,
            'reason': 'superseded parked scheduler; adopted bank verified complete; no live child'}


def gpu0_idle():
    value = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True)
    return not value.strip()


def remaining_banks(completed, adopted_name):
    # Fixed public inventory sizes only; no generated outcome or quality selection.
    return sorted((name for name in BANKS if name not in completed and name != adopted_name),
                  key=lambda name: (-BANKS[name][1], name))


def validate_bank(run, name, expected_identity=None):
    config, rows, complete_sha = load_bank(run/name)
    split, count, offset = BANKS[name]
    if (config['family'] != 'deepseek' or config['split'] != split or config['offset'] != offset
            or config['components'] != count or len(rows) != count or config['seed'] != 1701
            or config['decode_batch_size'] != 1 or config['max_new_tokens'] != 1024
            or config['max_input_tokens'] != 4096
            or any(config.get(k) != v for k, v in (expected_identity or {}).items())):
        raise ValueError('replication bank identity differs from the original schedule')
    return complete_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, output = args.run_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = read_json(run/'rbr_deepseek_generation_status.json')
    parent = process_info(original['pid'])
    if (parent is None or 'scripts/run_local_rbr_replication_generation.py' not in parent['command']
            or Path(f"/proc/{parent['pid']}/cwd").resolve() != root):
        raise ValueError('requires the original live serial DeepSeek supervisor in this checkout')
    sources = {**original['source_sha256'], str(Path(__file__).resolve().relative_to(root)): file_sha(__file__)}
    public, proof = Path(original['public_root']), run/'deepseek_verified_manifest.json'
    expected_bank_identity = {'model': 'deepseek-ai/deepseek-coder-6.7b-instruct',
        'revision': 'e5d64addd26a6a1db0f9b863abf6ee3141936807', 'candidates': 8,
        'source_sha256': sources['scripts/generate_apbpf_rbr_replication_bank.py'],
        'prompt_source_sha256': sources['src/pbpf/apbpf/rbr_prompt.py'],
        'inventory_source_sha256': sources['src/pbpf/apbpf/codearc_bank.py'],
        'model_proof_sha256': original['model_proof_sha256'],
        'public_tasks_sha256': original['public_tasks_sha256'],
        'public_manifest_sha256': file_sha(public/'manifest.json'),
        'candidate_seed_policy': 'sha256(seed:task_id:candidate_index) first32bits',
        'temperature': .8, 'top_p': .95}
    plan = {'schema': 'apbpf-rbr-two-gpu-handoff-v1', 'original_parent': parent, 'source_sha256': sources,
            'original_status_sha256': file_sha(run/'rbr_deepseek_generation_status.json'),
            'public_tasks_sha256': original['public_tasks_sha256'], 'model_proof_sha256': original['model_proof_sha256'],
            'gpus': [0, 1], 'bank_specs': BANKS, 'expected_bank_identity': expected_bank_identity,
            'scheduling_policy': 'largest remaining declared source inventory first; no outcome-based scheduling',
            'handoff': 'wait for Qwen completion and idleGPU0; park only original parent, keep its current child running; own remaining bank scheduling; retire old parent only after adopted bank validates and child is terminal',
            'scope': 'scheduling only; unchanged generator source, model, public tasks, per-candidate seeds, serial decode and token limits; disjoint output directories; GPU2/3 untouched'}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    state = {'status': 'waiting_for_qwen_and_idle_gpu0', 'pid': os.getpid(), 'plan_sha256': file_sha(output/'plan.json')}
    generation, owned, parked = None, False, False
    jobs = {}
    adopted = None
    def save():
        temporary = output/'status.partial'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(output/'status.json')
        if owned:
            temporary = run/'rbr_deepseek_generation_status.parallel.partial'
            temporary.write_text(json.dumps(generation, indent=2)+'\n')
            temporary.replace(run/'rbr_deepseek_generation_status.json')
    def check_sources():
        if (any(file_sha(root/n) != h for n, h in sources.items())
                or file_sha(public/'tasks.jsonl') != plan['public_tasks_sha256']
                or file_sha(public/'manifest.json') != expected_bank_identity['public_manifest_sha256']
                or file_sha(proof) != plan['model_proof_sha256']):
            raise ValueError('declared generation inputs changed')
    save()
    try:
        while True:
            check_sources()
            serial = read_json(run/'rbr_deepseek_generation_status.json')
            if serial['status'] == 'complete':
                for name in BANKS:
                    validate_bank(run, name, expected_bank_identity)
                state['status'] = 'not_needed_replication_already_complete'
                save()
                return
            if serial['status'] == 'needs_debug' or not same_process(process_info(parent['pid']), parent):
                raise RuntimeError('original serial generation failed or exited incomplete')
            qwen = read_json(run/'rbr_qwen_generation_status.json')
            if qwen['status'] == 'complete' and gpu0_idle():
                qconfig, qrows, _ = load_bank(run/'rbr-qwen-primary500')
                if qconfig['family'] != 'qwen' or len(qrows) != 500:
                    raise ValueError('GPU0 handoff requires completed full Qwen primary bank')
                break
            if qwen['status'] == 'needs_debug' or process_info(qwen['pid']) is None:
                if qwen['status'] != 'complete':
                    raise RuntimeError('Qwen exited incomplete')
            time.sleep(20)
        # A stopped parent cannot start a competing bank; its existing child is unaffected.
        while True:
            parked_info = park_parent(parent)
            parked = True
            serial = read_json(run/'rbr_deepseek_generation_status.json')
            current = serial.get('current')
            children = [process_info(pid) for pid in parked_info['children']]
            children = [c for c in children if c is not None and c['state'] != 'Z']
            if (current in BANKS and serial['runs'][current]['status'] == 'running' and len(children) == 1
                    and 'bwrap ' in children[0]['command']
                    and str(run/current) in children[0]['command']
                    and 'scripts/generate_apbpf_rbr_replication_bank.py' in children[0]['command']):
                adopted = {'name': current, 'process': children[0], 'gpu': 1}
                break
            os.kill(parent['pid'], signal.SIGCONT)
            parked = False
            if serial['status'] == 'complete':
                state['status'] = 'not_needed_replication_already_complete'
                save()
                return
            time.sleep(1)
        completed = [n for n in BANKS if (run/n/'complete.json').exists() and n != adopted['name']]
        for name in completed:
            validate_bank(run, name, expected_bank_identity)
        pending = remaining_banks(completed, adopted['name'])
        if not pending:
            os.kill(parent['pid'], signal.SIGCONT)
            parked = False
            state['status'] = 'not_needed_only_current_bank_remains'
            save()
            return
        if any((run/name).exists() for name in pending):
            raise ValueError('future bank directory already exists; refusing concurrent writers')
        check_sources()
        (output/'original-supervisor-status.json').write_text(json.dumps(serial, indent=2)+'\n')
        (output/'handoff.json').write_text(json.dumps({'parent': parked_info, 'adopted': adopted,
            'pending_banks': pending, 'completed_banks': completed}, indent=2)+'\n')
        generation = copy.deepcopy(serial)
        generation.update(pid=os.getpid(), status='running', current='parallel_disjoint_banks',
            source_sha256=sources, gpus=[0, 1], gpu_assignments={}, previous_supervisor_pid=parent['pid'],
            coordinator_plan_sha256=state['plan_sha256'], coordinator_root=str(output))
        owned = True
        state.update(status='running', adopted=adopted, original_parent_parked=True, pending_banks=pending)
        generation['gpu_assignments']['1'] = adopted['name']
        save()
        while pending or jobs or adopted is not None:
            check_sources()
            if adopted is not None:
                actual = process_info(adopted['process']['pid'])
                if actual is None or actual['start_ticks'] != adopted['process']['start_ticks'] or actual['state'] == 'Z':
                    checksum = validate_bank(run, adopted['name'], expected_bank_identity)
                    retired = retire_parked_parent(parent)
                    parked = False
                    (output/'retired-original-parent.json').write_text(json.dumps(retired, indent=2)+'\n')
                    generation['runs'][adopted['name']].update(status='complete', complete_sha256=checksum,
                        completion_basis='adopted child terminal and complete bank independently verified')
                    generation['gpu_assignments'].pop('1', None)
                    state['original_parent_parked'] = False
                    adopted = None
                    save()
                elif not same_process(process_info(parent['pid']), parent) or process_info(parent['pid'])['state'] not in ('T', 't'):
                    raise RuntimeError('parked original supervisor changed unexpectedly')
            for gpu, job in list(jobs.items()):
                code = job['process'].poll()
                if code is None:
                    continue
                job['log'].close()
                if code:
                    raise RuntimeError(f"{job['name']} generation exited {code}")
                checksum = validate_bank(run, job['name'], expected_bank_identity)
                generation['runs'][job['name']].update(status='complete', returncode=0, complete_sha256=checksum)
                generation['gpu_assignments'].pop(str(gpu), None)
                del jobs[gpu]
                save()
            for gpu in (0, 1):
                if not pending or gpu in jobs or (gpu == 1 and adopted is not None):
                    continue
                name = pending.pop(0)
                split, count, offset = BANKS[name]
                bank = run/name
                bank.mkdir(exist_ok=False)
                shutil.copy2(proof, bank/'model-proof.json')
                command = ['bash', 'scripts/apbpf_generator_sandbox.sh', str(public), str(bank),
                    str(root/'.venv/bin/python'), '-u', 'scripts/generate_apbpf_rbr_replication_bank.py',
                    '--public-root', '/input', '--output', '/output', '--model-proof', '/output/model-proof.json',
                    '--family', 'deepseek', '--split', split, '--components', str(count), '--offset', str(offset),
                    '--max-input-tokens', '4096', '--max-new-tokens', '1024']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                env.pop('PYTHONPATH', None)
                env.pop('PYTHONHOME', None)
                log = (output/f'{name}.log').open('x')
                process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                jobs[gpu] = {'name': name, 'process': process, 'log': log}
                generation['runs'][name] = {'status': 'running', 'command': command, 'gpu': gpu, 'child_pid': process.pid}
                generation['gpu_assignments'][str(gpu)] = name
                state['pending_banks'] = list(pending)
                save()
            time.sleep(20)
        for name in BANKS:
            validate_bank(run, name, expected_bank_identity)
        generation['status'] = 'complete'
        state['status'] = 'complete'
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        if owned:
            generation.update(status='needs_debug', error=repr(error))
        elif parked and same_process(process_info(parent['pid']), parent):
            os.kill(parent['pid'], signal.SIGCONT)
            parked = False
        save()
        # Do not terminate healthy bank workers because another bank or observer failed.
        for job in jobs.values():
            if job['process'].poll() is None:
                job['process'].wait()
        raise


if __name__ == '__main__':
    main()
