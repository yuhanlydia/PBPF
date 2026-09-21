#!/usr/bin/env python3
"""Run full-population selection with all completed, source-bound model seeds."""
import argparse
import json
import os
from pathlib import Path

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_selection import fit_selection, aggregate_selection, NEURAL_BASELINES
from pbpf.real_gate import validate_rbr_cache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--training-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    checksum=file_sha(args.cache);training=json.loads((args.training_root/'status.json').read_text())
    if training['status']!='complete' or training['cache_sha256']!=checksum:
        raise ValueError('all three model seeds must have completed on this exact cache')
    payload=validate_rbr_cache(json.loads(args.cache.read_text()));primary=[r for r in payload['records'] if r['split']=='test']
    population=[{'candidate_id':r['task_id'],'source_component_id':r['source_component_id']} for r in primary]
    if len(primary)!=4000 or len({r['source_component_id'] for r in primary})!=500:
        raise ValueError('complete500-source primary inventory required')
    files=[Path(__file__).resolve(),*sorted((root/'src/pbpf').rglob('*.py'))]
    hashes={str(f.relative_to(root)):file_sha(f) for f in files};models={}
    for seed in (1701,1702,1703):
        models[seed]={}
        for kind in ('belief','baselines'):
            directory=args.training_root/f'{kind}-seed{seed}';report=json.loads((directory/'training.json').read_text())
            if report['primary_population']!=population or report['config']['seed']!=seed:
                raise ValueError('selection model report population/seed differs')
            for arm in (('belief',) if kind=='belief' else NEURAL_BASELINES):
                path=directory/report['arms'][arm]['checkpoint'];sha=file_sha(path)
                if sha!=report['arms'][arm]['checkpoint_sha256']:raise ValueError('model checksum mismatch')
                models[seed][arm]={'path':str(path.resolve()),'sha256':sha}
    state={'status':'running','pid':os.getpid(),'cache_sha256':checksum,'models':models,'source_sha256':hashes,
           'seeds':[1701,1702,1703],'steps':1000,'runs':{},
           'scope':'standalone exploratory full-primary utility selection after failed association and active-testing gates'}
    def save():
        p=output/'status.partial';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(output/'status.json')
    save()
    try:
        reports={}
        for seed in state['seeds']:
            if (any(file_sha(root/n)!=h for n,h in hashes.items()) or file_sha(args.cache)!=checksum
                    or any(file_sha(m['path'])!=m['sha256'] for m in models[seed].values())):
                raise ValueError('selection code, data or trained model changed')
            state['current_seed']=seed;state['runs'][str(seed)]={'status':'running'};save()
            reports[seed]=fit_selection(payload,models[seed]['belief']['path'],{a:models[seed][a]['path'] for a in NEURAL_BASELINES},
                output/f'seed{seed}',seed=seed,cache_sha256=checksum,steps=state['steps'])
            state['runs'][str(seed)].update(status='complete',results_sha256=file_sha(output/f'seed{seed}/results.json'));save()
        report=aggregate_selection(reports);report.update(scope=state['scope'],cache_sha256=checksum)
        (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        state.update(status='complete',results_sha256=file_sha(output/'results.json'));save()
    except BaseException as error:
        state.update(status='needs_debug',error=repr(error));save();raise


if __name__=='__main__':
    main()
