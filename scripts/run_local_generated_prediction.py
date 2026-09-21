#!/usr/bin/env python3
"""Fit fixed development-only prediction seeds as soon as complete banks are scored."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from pbpf.apbpf.codearc_bank import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['rbr', 'codearc'], required=True)
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--public-root', type=Path, required=True)
    parser.add_argument('--evaluator-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evaluation, output = args.evaluation_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = ['scripts/run_local_generated_prediction.py','scripts/build_generated_development_cache.py',
        'scripts/run_rbr_prediction_gate.py','src/pbpf/apbpf/generated_cache.py','src/pbpf/apbpf/codearc_cache.py',
        'src/pbpf/apbpf/codearc_prompt.py','src/pbpf/apbpf/codearc_bank.py','src/pbpf/apbpf/development.py',
        'src/pbpf/real_gate.py','src/pbpf/belief/model.py','src/pbpf/train_belief.py',
        'src/pbpf/apbpf/counterfactual.py']
    source_files += [str(p.relative_to(root)) for p in sorted((root/'src/pbpf').rglob('*.py'))]
    hashes = {s:file_sha(root/s) for s in source_files}
    state = {'status':'waiting_for_development_execution','pid':os.getpid(),'domain':args.domain,
             'seeds':[1701,1702,1703],'steps':1000,'device':'cpu','runs':{},'source_sha256':hashes,
             'evaluation_role':'development_assessment_only','original_primary_excluded':True,
             'expected_feature_visibility':'observed-four-only' if args.domain=='rbr' else 'query-input-only',
             'scope':'standalone generated-bank prediction; unchanged gates; no main-table claim'}

    def save():
        temporary=output/'status.partial';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(output/'status.json')

    python = str(root/'.venv/bin/python')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)

    def execute(name,command):
        if any(file_sha(root/s)!=h for s,h in hashes.items()):
            raise ValueError('frozen training/cache source changed; a new attempt is required')
        state.update(status='running',current=name);state['runs'][name]={'status':'running','command':command};save()
        with (output/f'{name}.log').open('x') as log:
            code=subprocess.call(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state['runs'][name].update(returncode=code,status='complete' if code==0 else 'execution_failed');save()
        if code:
            raise RuntimeError(f'{name} exited {code}')

    save()
    try:
        names=['train','development-pilot','development-remainder']
        while True:
            upstream=json.loads((evaluation/'status.json').read_text())
            if all(upstream.get('steps',{}).get(f'evaluate-{name}',{}).get('status')=='complete'
                   and (evaluation/f'{name}-evaluation/results.json').exists() for name in names):
                break
            if upstream['status']=='needs_debug' or not Path(f"/proc/{upstream['pid']}").exists():
                raise RuntimeError('upstream development execution failed or exited incomplete')
            time.sleep(20)
        cache=output/'development-cache.json'
        command=[python,'scripts/build_generated_development_cache.py','--domain',args.domain,
                 '--public-root',str(args.public_root.resolve()),'--evaluator-root',str(args.evaluator_root.resolve()),
                 '--output',str(cache)]
        for name in names:
            bank=evaluation/f'{name}-extracted' if args.domain=='codearc' else Path(upstream['banks'][name])
            command+=['--bank',str(bank),'--evaluation',str(evaluation/f'{name}-evaluation/results.json')]
        execute('build-cache',command)
        payload=json.loads(cache.read_text())
        state.update(cache_sha256=file_sha(cache),counts=payload['counts'],problem_counts=payload['problem_counts'],
                     generator_identity=payload['generator_identity']);save()
        for seed in state['seeds']:
            name=f'prediction-seed{seed}'
            command=['bash','local/sandbox.sh',python,'-u','scripts/run_rbr_prediction_gate.py','--cache',str(cache),
                     '--output',str(output/f'{name}.json'),'--apbpf','--seed',str(seed),'--steps',str(state['steps'])]
            if args.domain=='rbr':command+=['--expected-is-public']
            execute(name,command)
            report=json.loads((output/f'{name}.json').read_text())
            state['runs'][name].update(gate=report['gate'],association=report['cluster_bootstrap']['outcome_shuffled']);save()
        state['status']='complete';save()
    except BaseException as error:
        state.update(status='needs_debug',error=repr(error));save();raise


if __name__=='__main__':
    main()
