#!/usr/bin/env python3
"""Full fixed-seed active replay with development-locked stopping, exploratory."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_queries import replay_queries, summarize_queries, paired_loss_gap
from pbpf.apbpf.stopping import lock_threshold, stopped_budgets
from pbpf.real_gate import validate_rbr_cache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--training-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    checksum=file_sha(args.cache);upstream=json.loads((args.training_root/'status.json').read_text())
    if upstream['status']!='complete' or upstream['cache_sha256']!=checksum:
        raise ValueError('requires all fixed-seed completed fitting on this exact cache')
    payload=validate_rbr_cache(json.loads(args.cache.read_text()))
    if payload['evaluation_role']!='exploratory_locked_primary_assessment':
        raise ValueError('requires original source partitions')
    primary=[r for r in payload['records'] if r['split']=='test'];development=[r for r in payload['records'] if r['split']=='development']
    population=[{'candidate_id':r['task_id'],'source_component_id':r['source_component_id']} for r in primary]
    if (len(primary)!=4000 or len({r['source_component_id'] for r in primary})!=500
            or {r['source_component_id'] for r in primary}&{r['source_component_id'] for r in development}):
        raise ValueError('requires complete primary and disjoint development sources')
    models={}
    for seed in (1701,1702,1703):
        directory=args.training_root/f'belief-seed{seed}';report=json.loads((directory/'training.json').read_text())
        path=directory/'belief.pt'
        if report['primary_population']!=population or report['config']['seed']!=seed or file_sha(path)!=report['arms']['belief']['checkpoint_sha256']:
            raise ValueError('model identity or source population mismatch')
        models[seed]={'path':str(path.resolve()),'sha256':file_sha(path)}
    files=[Path(__file__).resolve(),root/'scripts/run_apbpf_query_worker.py',*sorted((root/'src/pbpf').rglob('*.py'))]
    hashes={str(p.relative_to(root)):file_sha(p) for p in files}
    spec=importlib.util.spec_from_file_location('active_diagnostic_helpers',root/'scripts/run_apbpf_query_worker.py')
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
    state={'status':'running','pid':os.getpid(),'cache_sha256':checksum,'models':models,'source_sha256':hashes,
           'seeds':[1701,1702,1703],'runs':{},'threshold_grid':helper.THRESHOLDS,
           'scope':'standalone exploratory full-primary active replay after failed association/fairness; cached observation budgets only'}
    def save():
        p=output/'status.partial';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(output/'status.json')
    def check():
        if any(file_sha(root/n)!=h for n,h in hashes.items()) or file_sha(args.cache)!=checksum:
            raise ValueError('active diagnostic source or data changed')
    def execute(rows,seed,name):
        check();state['current']=name;state['runs'][name]={'status':'running','completed':0};save();records=[]
        with (output/f'{name}.jsonl').open('x') as stream:
            for row in replay_queries(rows,models[seed]['path'],seed=seed,mode='active'):
                records.append(row);stream.write(json.dumps(row)+'\n')
                if len(records)%64==0:
                    stream.flush();state['runs'][name]['completed']=len(records);save()
        state['runs'][name].update(status='complete',completed=len(records),sha256=file_sha(output/f'{name}.jsonl'));save()
        return records
    save()
    try:
        records,locks,curves,stopped,sources,budgets={},{},{},[],[],[]
        for seed in state['seeds']:
            if file_sha(models[seed]['path'])!=models[seed]['sha256']:
                raise ValueError('checkpoint changed')
            calibration=execute(development,seed,f'calibration-seed{seed}')
            information,nll=helper.stopping_arrays(calibration)
            lock=lock_threshold(information,nll.mean(-1),helper.THRESHOLDS)
            lock.update(calibration_sources=sorted({r['source'] for r in calibration}),checkpoint_sha256=models[seed]['sha256'])
            lock_path=output/f'threshold-lock-seed{seed}.json';lock_path.write_text(json.dumps(lock,indent=2)+'\n');locks[str(seed)]=lock
            records[seed]=execute(primary,seed,f'primary-seed{seed}')
            information,nll=helper.stopping_arrays(records[seed]);chosen=stopped_budgets(information,lock['threshold'])
            stopped.extend(nll[np.arange(len(nll)),chosen-1]);budgets.extend(chosen);sources.extend(r['source'] for r in records[seed])
            curves[str(seed)]=[]
            for threshold in [*helper.THRESHOLDS,None]:
                b=stopped_budgets(information,threshold);values=nll[np.arange(len(nll)),b-1]
                curves[str(seed)].append({'threshold':threshold,'mean_tests':float(b.mean()),'nll':float(values.mean()),
                    'selected_by_development':threshold==lock['threshold'],
                    'against_same_policy_four':paired_loss_gap(nll[:,-1],values,[r['source'] for r in records[seed]])})
        report=summarize_queries(records,mode='active');matched={}
        for reference in ('fixed','random','diagnostic_mi'):
            fixed=[r['budgets']['4'][reference]['future_nll'] for seed in state['seeds'] for r in records[seed]]
            matched[reference]=paired_loss_gap(fixed,stopped,sources)
        report.update(scope=state['scope'],cache_sha256=checksum,stopping={'locks':locks,'curves':curves,
            'mean_tests':float(np.mean(budgets)),'test_reduction':float(1-np.mean(budgets)/4),
            'matched_hidden_quality':all(x['mean_nll_gap']>=0 and x['ci95'][0]>=0 for x in matched.values()),
            'against_four_test_references':matched})
        (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        check();state.update(status='complete',results_sha256=file_sha(output/'results.json'));save()
    except BaseException as error:
        state.update(status='needs_debug',error=repr(error));save();raise


if __name__=='__main__':
    main()
