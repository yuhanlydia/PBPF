#!/usr/bin/env python3
"""Run isolated packet-based repair training and generation on one free GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from pbpf.apbpf.codearc_bank import file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--domain', choices=['rbr', 'codearc'], required=True)
    p.add_argument('--gpu', choices=['0', '1', '2'], required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[1701, 1702, 1703])
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--smoke-only', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.smoke_only and (args.seeds != [1701, 1702, 1703] or args.steps != 1500 or args.max_new_tokens != 512):
        raise ValueError('full repair uses the declared three seeds and fixed 1500/512 budget')
    gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', args.gpu, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if any(line.split(',')[0].strip() == gpu_uuid for line in processes.splitlines()):
        raise RuntimeError('declared GPU is occupied; never preempt another process')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__).resolve(), root/'scripts/run_apbpf_packet_actor.py',
                    root/'scripts/run_rbr_repair_gate.py', root/'scripts/apbpf_generator_sandbox.sh',
                    *sorted((root/'src/pbpf').rglob('*.py'))]
    hashes = {str(f.relative_to(root)): file_sha(f) for f in source_files}
    packets = {}
    for seed in args.seeds:
        packet = args.packet_root.resolve()/f'actor-seed{seed}'
        manifest = json.loads((packet/'manifest.json').read_text())
        if manifest['seed'] != seed or (not args.smoke_only and manifest['primary_sources'] != 500):
            raise ValueError('packet seed or full source inventory differs')
        packets[str(seed)] = {'path': str(packet), 'manifest_sha256': file_sha(packet/'manifest.json')}
    state = {'status': 'running', 'pid': os.getpid(), 'source_sha256': hashes, 'packets': packets,
             'seeds': args.seeds, 'steps': args.steps, 'max_new_tokens': args.max_new_tokens,
             'gpu': int(args.gpu), 'domain': args.domain, 'runs': {}, 'smoke_only': args.smoke_only,
             'scope': 'real-data engineering smoke only' if args.smoke_only else
                      'supportive exploratory repair after failed upstream gates; no efficacy claim before fresh evaluation'}

    def save():
        path = output/'status.partial'
        path.write_text(json.dumps(state, indent=2)+'\n')
        path.replace(output/'status.json')

    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    try:
        for mode in ('train', 'generate'):
            for seed in args.seeds:
                if any(file_sha(root/n) != h for n, h in hashes.items()):
                    raise ValueError('repair sources changed; archive and redeclare before resuming')
                packet = packets[str(seed)]
                if file_sha(Path(packet['path'])/'manifest.json') != packet['manifest_sha256']:
                    raise ValueError('actor packet manifest changed')
                directory = output/f'seed{seed}'
                directory.mkdir(exist_ok=True)
                name = f'{mode}-seed{seed}'
                command = ['bash', 'scripts/apbpf_generator_sandbox.sh', packet['path'], str(directory),
                           str(root/'.venv/bin/python'), '-u', 'scripts/run_apbpf_packet_actor.py',
                           '--input', '/input', '--output', '/output', '--domain', args.domain, '--mode', mode,
                           '--seed', str(seed), '--steps', str(args.steps), '--max-new-tokens', str(args.max_new_tokens)]
                state['current'] = name
                state['runs'][name] = {'status': 'running', 'command': command}
                save()
                with (output/f'{name}.log').open('x') as log:
                    code = subprocess.call(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                state['runs'][name].update(status='complete' if code == 0 else 'execution_failed', returncode=code)
                save()
                if code:
                    raise RuntimeError(f'{name} exited {code}')
                artifact = directory/('projector.pt' if mode == 'train' else 'generation-complete.json')
                state['runs'][name]['artifact_sha256'] = file_sha(artifact)
                save()
        state['status'] = 'complete'
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
