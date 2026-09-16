#!/usr/bin/env python3
"""Run all fixed-seed oracle subset diagnostics on completed full-cache models."""
import argparse
import json
import os
from pathlib import Path

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_queries import replay_queries, summarize_queries
from pbpf.real_gate import validate_rbr_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1];output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    upstream=json.loads((args.training_root/'status.json').read_text());checksum=file_sha(args.cache)
    if upstream['status']!='complete' or upstream['cache_sha256']!=checksum:
        raise ValueError('all fixed-seed full-cache fitting must have completed')
    payload=validate_rbr_cache(json.loads(args.cache.read_text()))
    if payload['evaluation_role']!='exploratory_locked_primary_assessment':
        raise ValueError('original primary cache required')
    rows=[r for r in payload['records'] if r['split']=='test']
    population=[{'candidate_id':r['task_id'],'source_component_id':r['source_component_id']} for r in rows]
    if len(rows)!=4000 or len({r['source_component_id'] for r in rows})!=500:
        raise ValueError('full 500-by-8 source inventory required')
    files=[Path(__file__).resolve(),*sorted((root/'src/pbpf').rglob('*.py'))]
    sources={str(f.relative_to(root)):file_sha(f) for f in files}
    models={}
    for seed in (1701,1702,1703):
        directory=args.training_root/f'belief-seed{seed}';report=json.loads((directory/'training.json').read_text())
        checkpoint=directory/'belief.pt'
        if (report['primary_population']!=population or report['config']['seed']!=seed
                or file_sha(checkpoint)!=report['arms']['belief']['checkpoint_sha256']):
            raise ValueError('trained model identity differs from full population')
        models[seed]={'path':str(checkpoint.resolve()),'sha256':file_sha(checkpoint)}
    state={'status':'running','pid':os.getpid(),'cache_sha256':checksum,'models':models,'source_sha256':sources,
           'oracle_gate_budget':2,'seeds':[1701,1702,1703],'runs':{},
           'scope':'standalone exploratory full-primary oracle diagnostic after negative association/fairness; no stage completion'}
    def save():
        tmp=output/'status.partial';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(output/'status.json')
    save()
    try:
        records={}
        for seed in state['seeds']:
            if (any(file_sha(root/n)!=h for n,h in sources.items()) or file_sha(args.cache)!=checksum
                    or file_sha(models[seed]['path'])!=models[seed]['sha256']):
                raise ValueError('oracle diagnostic source/cache/model changed')
            state['current_seed']=seed;state['runs'][str(seed)]={'status':'running','completed':0};save();records[seed]=[]
            with (output/f'seed{seed}.jsonl').open('x') as stream:
                for row in replay_queries(rows,models[seed]['path'],seed=seed,mode='oracle'):
                    records[seed].append(row);stream.write(json.dumps(row)+'\n')
                    if len(records[seed])%64==0:
                        stream.flush();state['runs'][str(seed)]['completed']=len(records[seed]);save()
            state['runs'][str(seed)].update(status='complete',records_sha256=file_sha(output/f'seed{seed}.jsonl'));save()
        report=summarize_queries(records,mode='oracle')
        report.update(scope=state['scope'],oracle_gate_budget=2,cache_sha256=checksum)
        (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        state.update(status='complete',results_sha256=file_sha(output/'results.json'));save()
    except BaseException as error:
        state.update(status='needs_debug',error=repr(error));save();raise


if __name__=='__main__':
    main()
