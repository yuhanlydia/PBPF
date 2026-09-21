#!/usr/bin/env python3
"""Hand GPU1 from pinned CodeARC DeepSeek generation to RBR replication."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from pbpf.apbpf.codearc_bank import file_sha, load_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--previous-pid', type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_root.resolve()
    status = run/'rbr_deepseek_generation_status.json'
    if status.exists():
        raise FileExistsError('existing generation status requires explicit inspection before a new attempt')
    primary = run/'codearc-deepseek-primary500'
    try:
        previous_start = Path(f'/proc/{args.previous_pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        previous_start = None
    qwen = json.loads((run/'rbr_qwen_generation_status.json').read_text())
    public = Path(qwen['public_root'])
    proof = run/'deepseek_verified_manifest.json'
    sources = ['scripts/run_local_rbr_replication_generation.py', 'scripts/generate_apbpf_rbr_replication_bank.py',
               'scripts/apbpf_generator_sandbox.sh', 'src/pbpf/apbpf/rbr_prompt.py', 'src/pbpf/apbpf/codearc_bank.py']
    hashes = {name: file_sha(root/name) for name in sources}
    state = {'status': 'waiting_for_gpu1_codearc_primary', 'pid': os.getpid(), 'gpu': 1,
             'previous_pid': args.previous_pid, 'previous_process_start_ticks': previous_start,
             'public_root': str(public), 'public_tasks_sha256': file_sha(public/'tasks.jsonl'),
             'model_proof_sha256': file_sha(proof), 'source_sha256': hashes, 'runs': {},
             'decode_batch_size': 1, 'candidates_per_source': 8, 'max_new_tokens': 1024,
             'scope': 'exploratory predeclared RBR DeepSeek replication; public-only fixed sources'}

    def save():
        temporary = status.with_suffix('.partial'); temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(status)

    save()
    try:
        while True:
            process_stat = Path(f'/proc/{args.previous_pid}/stat')
            try:
                same_process = process_stat.read_text().rsplit(')', 1)[1].split()[19] == previous_start
            except FileNotFoundError:
                same_process = False
            if not same_process:
                if not (primary/'complete.json').exists():
                    raise RuntimeError('prior CodeARC generation exited before completing its bank')
                previous_run, previous_rows, _ = load_bank(primary)
                if previous_run['split'] != 'primary' or len(previous_rows) != 500:
                    raise ValueError('GPU handoff requires the complete 500-source primary bank')
                break
            time.sleep(20)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
        for name, split, count, offset in [
            ('rbr-deepseek-development-pilot16', 'development', 16, 0),
            ('rbr-deepseek-train321', 'train', 321, 0),
            ('rbr-deepseek-development384', 'development', 384, 16),
            ('rbr-deepseek-primary500', 'primary', 500, 0)]:
            if (any(file_sha(root/n) != h for n,h in hashes.items())
                    or file_sha(public/'tasks.jsonl') != state['public_tasks_sha256']
                    or file_sha(proof) != state['model_proof_sha256']):
                raise ValueError('declared sources, public inventory or model proof changed')
            output = run/name; output.mkdir(exist_ok=False)
            shutil.copy2(proof, output/'model-proof.json')
            command = ['bash', 'scripts/apbpf_generator_sandbox.sh', str(public), str(output),
                       str(root/'.venv/bin/python'), '-u', 'scripts/generate_apbpf_rbr_replication_bank.py',
                       '--public-root', '/input', '--output', '/output', '--model-proof', '/output/model-proof.json',
                       '--family', 'deepseek', '--split', split, '--components', str(count), '--offset', str(offset),
                       '--max-input-tokens', '4096', '--max-new-tokens', '1024']
            state.update(status='running', current=name); state['runs'][name] = {'status':'running','command':command};save()
            with (run/f'{name}.log').open('x') as log:
                code = subprocess.call(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            state['runs'][name].update(returncode=code, status='complete' if code==0 else 'execution_failed'); save()
            if code:
                raise RuntimeError(f'{name} exited {code}')
            load_bank(output)
        state['status'] = 'complete'; save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error)); save(); raise


if __name__ == '__main__':
    main()
