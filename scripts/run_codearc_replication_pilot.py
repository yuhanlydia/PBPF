#!/usr/bin/env python3
"""Queue the locked DeepSeek development pilot after the local Qwen training bank."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    args = p.parse_args()
    root, run = args.workspace.resolve(), args.run_root.resolve()
    python = str(root / '.venv/bin/python')
    bank = run / 'codearc-deepseek-pilot16'
    status = run / 'deepseek_pilot_status.json'
    sources = [root / name for name in ('scripts/generate_apbpf_codearc_replication_bank.py',
        'src/pbpf/apbpf/codearc_prompt.py', 'scripts/evaluate_apbpf_codearc_bank.py',
        'src/pbpf/apbpf/codearc_execution.py', 'src/pbpf/apbpf/codearc_bank.py')]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    state = {'status': 'waiting_for_gpu1_training_bank', 'pid': os.getpid(),
             'source_sha256': hashes, 'scope': 'development-only cross-family replay pilot', 'steps': {}}
    if status.exists():
        raise FileExistsError('pilot status exists; inspect previous attempt')

    def save():
        temporary = status.with_suffix('.partial')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(temporary, status)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)

    def execute(name, command):
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
            raise ValueError('source changed after queued plan; create a new attempt')
        state['status'] = name
        state['steps'][name] = {'command': command}
        save()
        with (run / (name + '.log')).open('x') as stream:
            result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        state['steps'][name]['returncode'] = result.returncode
        save()
        if result.returncode:
            raise RuntimeError(name + ' failed; see retained log')

    save()
    try:
        while not (run / 'codearc-generation-train212/complete.json').exists():
            time.sleep(5)
        time.sleep(10)
        execute('deepseek-pilot-generation', ['bash', 'scripts/apbpf_generator_sandbox.sh',
            str(run / 'codearc-public-v1'), str(bank), python, '-u',
            'scripts/generate_apbpf_codearc_replication_bank.py', '--public-root', '/input',
            '--output', '/output', '--split', 'development', '--components', '16',
            '--candidates', '8', '--max-new-tokens', '512', '--model-proof', '/output/model-proof.json'])
        execute('deepseek-pilot-evaluation', [python, '-u', 'scripts/evaluate_apbpf_codearc_bank.py',
            '--evaluator-root', str(run / 'codearc-evaluator-v1'), '--bank', str(bank),
            '--output', str(run / 'codearc-deepseek-pilot16-evaluation'), '--phase', 'all', '--workers', '12'])
        state['status'] = 'complete_pilot_only_no_replication_gate_claim'
        save()
    except BaseException as exc:
        state.update(status='needs_debug', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    main()
