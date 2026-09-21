#!/usr/bin/env python3
"""Offline, sealed mechanism inference; preflight inventories are not results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import yaml

from pbpf.eesd.mechanism_contrasts import contrast_ids, reconstruct_contrasts, BIN_IDS
from pbpf.eesd.mechanism_inference import PairedMechanismData, bootstrap_gain, family_report, unsupported, METRICS

ROOT=Path(__file__).resolve().parents[1]
MODELS=('qwen25_7b','deepseek_6p7b','seed_coder_8b','starcoder2_15b')
DATASETS={'runbugrun':'rbr','codearc':'codearc'}
SEEDS=(1701,1702,1703)
PRIMARY='core/effective_params'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def write(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def derive_public_support(**kwargs):
    from pbpf.eesd.mechanism_contrasts import derive_public_support as derive
    return derive(**kwargs)


def public_inventory(cell):
    public=Path(cell['public_root'])
    manifest=read(public/'manifest.json')
    require(sha(public/'tasks.jsonl')==manifest['public_tasks_sha256'],'public tasks checksum mismatch')
    rows=[json.loads(line) for line in (public/'tasks.jsonl').read_text().splitlines() if line.strip()]
    require(len({r['task_id'] for r in rows})==len(rows),'duplicate public task IDs')
    inventories={}
    for split,count in [('development',200),('primary',500)]:
        tasks=[r for r in rows if r['split']==split]
        if cell['domain']=='codearc':
            grouped={}
            for row in tasks:
                key=row['source_component_id']
                if key not in grouped or row['task_id']<grouped[key]['task_id']:grouped[key]=row
            tasks=list(grouped.values())
        if split=='development':tasks=tasks[:200]
        require(len(tasks)==count and len({r['source_component_id'] for r in tasks})==count,
                f'public {split} requires exactly {count} declared sources')
        inventories[split]=[{'task_id':r['task_id'],'source_component_id':r['source_component_id']} for r in tasks]
    require(not {r['source_component_id'] for r in inventories['development']} &
            {r['source_component_id'] for r in inventories['primary']},'public split source overlap')
    return inventories


def families():
    ids=contrast_ids()
    require(len(ids)==49 and len(set(ids))==49,'exactly 49 unique reserved contrast slots required')
    primary=[];secondary=[]
    for dataset in DATASETS:
        for model in MODELS:
            for contrast in ids:
                for metric in METRICS:
                    key=f'{dataset}/{model}/{contrast}/{metric}'
                    (primary if contrast==PRIMARY and metric=='nll' else secondary).append(key)
    require(len(primary)==8 and len(secondary)==1560,'locked hypothesis count mismatch')
    return primary,secondary


def incomplete_families(reason):
    p,s=families()
    return {name:family_report({key:unsupported(reason) for key in keys},family=keys)
            for name,keys in [('primary',p),('secondary',s)]}


def verify_report_bytes(directory):
    complete=read(directory/'complete.json')
    for name,suffix in [('report','.json'),('predictions','.npz')]:
        require(sha(directory/(name+suffix))==complete.get(name+'_sha256'),'existing report checksum mismatch')


def bin_support(rows):
    memberships={}
    for seed,row in rows.items():
        mask=np.asarray(row['public_mask'])
        require(mask.dtype.kind=='b' and mask.shape==(len(row['sources']),),'public mask shape/type mismatch')
        require(len(row['query_ids'])==len(mask) and len(row['seeds'])==len(mask)
            and all(int(s)==seed for s in row['seeds']),'public support identity/seed mismatch')
        memberships[str(seed)]=sorted({str(source) for source,keep in zip(row['sources'],mask) if keep})
    common=sorted(set.intersection(*(set(v) for v in memberships.values())))
    return {'common_sources':common,'sources_by_seed':memberships,
            'source_counts_by_seed':{k:len(v) for k,v in memberships.items()},
            'excluded_sources_by_seed':{k:sorted(set(v)-set(common)) for k,v in memberships.items()},
            'ordered_sources_by_seed':{str(seed):list(map(str,row['sources'])) for seed,row in rows.items()},
            'ordered_seeds_by_seed':{str(seed):list(map(int,row['seeds'])) for seed,row in rows.items()},
            'public_masks_by_seed':{str(seed):[bool(x) for x in row['public_mask']] for seed,row in rows.items()},
            'query_ids_by_seed':{str(seed):[q if isinstance(q,str) else canonical(q) for q in row['query_ids']] for seed,row in rows.items()}}


def combine(rows, *, selected_sources=None, use_public_mask=False):
    selected=None if selected_sources is None else set(selected_sources)
    out={'labels':[],'proposed':[],'comparators':{},'sources':[],'seeds':[],'query_ids':[]}
    names=None
    for seed,row in sorted(rows.items()):
        current=set(row['comparators'])
        require(names is None or names==current,'comparator realizations differ across seeds')
        names=current
        indices=[i for i,s in enumerate(row['sources']) if (selected is None or str(s) in selected)
                 and (not use_public_mask or row['public_mask'][i])]
        require(all(int(s)==seed for s in row['seeds']),'reconstructed seed identity mismatch')
        for key in ('labels','proposed'):out[key].extend(np.asarray(row[key])[indices].tolist())
        for name,probabilities in row['comparators'].items():
            out['comparators'].setdefault(name,[]).extend(np.asarray(probabilities)[indices].tolist())
        out['sources'].extend(str(row['sources'][i]) for i in indices)
        out['seeds'].extend(int(row['seeds'][i]) for i in indices)
        out['query_ids'].extend(row['query_ids'][i] if isinstance(row['query_ids'][i],str)
                                else canonical(row['query_ids'][i]) for i in indices)
    return out


def validate_inputs(manifest_path,config_path,lock_path,lock_sha):
    require(sha(lock_path)==lock_sha,'statistical amendment lock checksum mismatch')
    lock=read(lock_path)
    require(lock.get('schema')=='eesd-statistical-amendment-lock-v1' and lock.get('primary_slots')==8
        and lock.get('secondary_reserved_slots')==1560 and lock.get('bootstrap_draws')==10000
        and lock.get('generation_seeds')==list(SEEDS) and lock.get('models')==4
        and lock.get('mechanism_domains')==2,'statistical amendment lock protocol mismatch')
    plan=(ROOT/lock['plan']).resolve()
    require(plan.is_relative_to(ROOT) and sha(plan)==lock['plan_sha256'],'locked statistical plan changed')
    manifest=yaml.safe_load(Path(manifest_path).read_text());cfg=yaml.safe_load(Path(config_path).read_text())
    require(manifest.get('schema')=='eesd-cache-manifest-v1','manifest schema mismatch')
    cells=manifest.get('mechanism_cells',[])
    require(len(cells)==8 and {(c['dataset'],c['model']) for c in cells}==
            {(d,m) for d in DATASETS for m in MODELS},'complete unique 8-cell manifest required')
    require(cfg.get('schema')=='eesd-iclr2027-v1' and cfg.get('seeds')==list(SEEDS)
        and cfg['evidence']['bootstrap_draws']==10000 and cfg['evidence']['ece_bins']==10
        and cfg['evidence']['primary_metric']=='nll' and cfg['evidence']['secondary_metrics']==['brier','ece','accuracy']
        and cfg['statistics']['holm_secondary'] is True
        and cfg['statistics']['unit']=='source_problem' and cfg['statistics']['paired_bootstrap'] is True
        and cfg.get('bootstrap_seed')==314159,'inference config mismatch')
    for cell in cells:
        require(cell['domain']==DATASETS[cell['dataset']] and cell['family']==cell['model']
            and cell.get('development_components',200)==200 and cell.get('visible',4)==4,'cell protocol mismatch')
    return cells


def inspect_cell(cell,results_root,config,verifier):
    inventory=public_inventory(cell);coverage={}
    for seed in SEEDS:
        base=Path(cell['dataset'])/cell['model']/f'seed{seed}'
        directory=results_root/'mechanism'/base;cache=results_root/'mechanism-cache'/base/'cache.json'
        banks=[];missing=[]
        for split in ('development','primary'):
            bank=results_root/'mechanism-banks'/base/split
            if (bank/'complete.json').exists():
                verified=verifier['verify_mechanism_bank'](bank,cell=cell,seed=seed,split=split,root=ROOT)
                info=verified[1]
                require(info['task_ids']==[r['task_id'] for r in inventory[split]] and
                    info['source_component_ids']==[r['source_component_id'] for r in inventory[split]],'bank differs from declared public population')
                banks.append(verified)
            else:missing.append(f'{split} bank incomplete' if bank.exists() else f'{split} bank missing')
        if cache.with_suffix('.binding.json').exists():
            require(len(banks)==2,'sealed cache depends on missing sealed banks')
            verifier['verify_mechanism_cache'](cache,cell=cell,banks=banks,root=ROOT)
        else:missing.append('cache/binding missing')
        if (directory/'complete.json').exists():
            verify_report_bytes(directory)
            require(not missing,'sealed report depends on incomplete banks/cache')
            verifier['verify_mechanism_report'](directory,cache=cache,config=config,cell=cell,seed=seed,root=ROOT)
        else:missing.append('report incomplete' if directory.exists() else 'report missing')
        coverage[str(seed)]={'status':'missing' if missing else 'verified','reasons':missing,
            'report_dir':str(directory),'cache':str(cache),'verified_banks':[
                {'path':str(p),'complete_sha256':h,'split':r['split'],'components':r['components']} for p,r,h in banks]}
    return inventory,coverage


def run_report(*,manifest,config,results_root,output,statistical_lock,statistical_lock_sha256,preflight_only=False):
    cells=validate_inputs(manifest,config,statistical_lock,statistical_lock_sha256)
    verifier=runpy.run_path(str(ROOT/'scripts/run_eesd_matrix.py'))
    coverage={};inventories={}
    for cell in cells:
        key=f"{cell['dataset']}/{cell['model']}"
        inventories[key],coverage[key]=inspect_cell(cell,results_root,config,verifier)
    complete=all(v['status']=='verified' for cell in coverage.values() for v in cell.values())
    # Preflight does no assessment inferential reconstruction. Existing cache
    # verification parses complete JSON; this is not a claim of byte-level or
    # field-loading blindness to labels. A complete inventory is not inference.
    output.mkdir(parents=True,exist_ok=False)
    lock={'schema':'eesd-mechanism-inference-input-lock-v1','manifest_sha256':sha(manifest),
          'config_sha256':sha(config),'statistical_lock_sha256':sha(statistical_lock),
          'reporter_source_sha256':sha(__file__),
          'analysis_source_sha256':{name:sha(ROOT/name) for name in (
              'src/pbpf/eesd/mechanism_inference.py','src/pbpf/eesd/mechanism_contrasts.py',
              'src/pbpf/eesd/mechanism_artifacts.py','scripts/run_eesd_matrix.py')},
          'public_materializations':{f"{c['dataset']}/{c['model']}":{
              'root':str(Path(c['public_root']).resolve()),'manifest_sha256':sha(Path(c['public_root'])/'manifest.json'),
              'tasks_sha256':sha(Path(c['public_root'])/'tasks.jsonl')} for c in cells},
          'public_inventories':inventories,'coverage':coverage}
    write(output/'input-lock.json',lock)
    if preflight_only or not complete:
        report={'schema':'eesd-mechanism-inference-report-v1','status':'preflight_only' if preflight_only else 'incomplete',
                'scientific_inference_performed':False,'coverage':coverage,
                'families':incomplete_families('preflight only' if preflight_only else 'missing required reports')}
    else:
        reconstructed={};receipts={};support={};support_receipts={}
        # Stage 1: only public-feature masks and development-selected settings.
        # Full JSON verification may parse assessment outcomes; the feature
        # stage never accesses those outcome fields or computes their metrics.
        for cell in cells:
            key=f"{cell['dataset']}/{cell['model']}";public_rows={};support[key]={};support_receipts[key]={}
            expected={r['source_component_id'] for r in inventories[key]['primary']}
            for seed in SEEDS:
                entry=coverage[key][str(seed)]
                rows,receipt=derive_public_support(report_dir=entry['report_dir'],cache=entry['cache'],config=config,
                    evaluator_root=cell['evaluator_root'],domain=cell['domain'],model=cell['model'],seed=seed,root=ROOT)
                require(set(rows)==set(BIN_IDS),'public support bin inventory mismatch')
                require(all(set(map(str,row['sources']))==expected for row in rows.values()),
                        'public support differs from declared 500 sources')
                public_rows[seed]=rows;support_receipts[key][str(seed)]=receipt
            for contrast in BIN_IDS:
                support[key][contrast]=bin_support({seed:public_rows[seed][contrast] for seed in SEEDS})
        write(output/'a9-public-support.json',support)
        write(output/'a9-public-support-receipts.json',support_receipts)
        support_sha=sha(output/'a9-public-support.json')
        write(output/'a9-public-support.complete.json',{
            'support_sha256':support_sha,'receipts_sha256':sha(output/'a9-public-support-receipts.json'),
            'input_lock_sha256':sha(output/'input-lock.json'),
            'scope':'public-feature support frozen before assessment outcome reconstruction'})
        # Stage 2: label-bearing artifact reconstruction can now begin.
        for cell in cells:
            key=f"{cell['dataset']}/{cell['model']}";reconstructed[key]={};receipts[key]={}
            expected={r['source_component_id'] for r in inventories[key]['primary']}
            for seed in SEEDS:
                entry=coverage[key][str(seed)]
                contrasts,receipt=reconstruct_contrasts(report_dir=entry['report_dir'],cache=entry['cache'],config=config,
                    evaluator_root=cell['evaluator_root'],domain=cell['domain'],model=cell['model'],seed=seed,root=ROOT)
                require(set(contrasts)==set(contrast_ids()),'reconstruction inventory mismatch')
                for row in contrasts.values():
                    if row['status']=='available':
                        require(set(map(str,row['sources']))==expected and len(expected)==500,
                                'reconstructed source population differs from declared 500 sources')
                reconstructed[key][seed]=contrasts;receipts[key][str(seed)]=receipt
            for contrast in BIN_IDS:
                rows={seed:reconstructed[key][seed][contrast] for seed in SEEDS}
                if all(r['status']=='available' for r in rows.values()):
                    require(bin_support(rows)==support[key][contrast],
                            'assessment reconstruction changed frozen public support/identities')
        require(sha(output/'a9-public-support.json')==support_sha,'frozen public support bytes changed')
        write(output/'reconstruction-receipts.json',receipts)
        results={};contrasts_report={}
        for cell in cells:
            key=f"{cell['dataset']}/{cell['model']}";contrasts_report[key]={}
            for contrast in contrast_ids():
                if contrast=='A8/n4':continue
                rows={seed:reconstructed[key][seed][contrast] for seed in SEEDS}
                if any(r['status']!='available' for r in rows.values()):
                    result=unsupported('; '.join(f'{s}: {r.get("reason","unavailable")}' for s,r in rows.items() if r['status']!='available'))
                else:
                    selected=support[key][contrast]['common_sources'] if contrast in BIN_IDS else None
                    if selected is not None and len(selected)<2:result=unsupported('fewer than two common public-bin source clusters')
                    else:
                        data=PairedMechanismData(**combine(rows,selected_sources=selected,use_public_mask=contrast in BIN_IDS))
                        population=f'mass_bin_{BIN_IDS.index(contrast)+1}' if contrast in BIN_IDS else 'all'
                        result=bootstrap_gain(data,domain=cell['domain'],model=cell['model'],population=population)
                contrasts_report[key][contrast]=result
            # Deliberate reserved alias: identical p values and effects, one computation.
            alias_rows=[reconstructed[key][s]['A8/n4'] for s in SEEDS]
            if any(r['status']!='available' for r in alias_rows):
                contrasts_report[key]['A8/n4']=unsupported('history n4 reconstruction unavailable')
            else:
                require(all(r.get('alias_of')=='core/tuned' for r in alias_rows),
                        'history n4 alias declaration missing or inconsistent')
                contrasts_report[key]['A8/n4']={**contrasts_report[key]['core/tuned'],'alias_of':'core/tuned'}
            for contrast,result in contrasts_report[key].items():
                for metric in METRICS:results[f'{key}/{contrast}/{metric}']=result['metrics'][metric] if result['status']=='supported' else result
        primary,secondary=families()
        report={'schema':'eesd-mechanism-inference-report-v1','status':'inference_computed',
                'scientific_inference_performed':True,'coverage':coverage,'contrasts':contrasts_report,
                'a9_public_support_sha256':sha(output/'a9-public-support.json'),
                'a9_public_support_seal_sha256':sha(output/'a9-public-support.complete.json'),
                'reconstruction_receipts_sha256':sha(output/'reconstruction-receipts.json'),
                'families':{name:family_report({k:results[k] for k in keys},family=keys)
                            for name,keys in [('primary',primary),('secondary',secondary)]}}
    report['input_lock_sha256']=sha(output/'input-lock.json')
    write(output/'report.json',report)
    write(output/'complete.json',{'report_sha256':sha(output/'report.json'),'input_lock_sha256':sha(output/'input-lock.json')})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','config','results-root','output','statistical-lock'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--statistical-lock-sha256',required=True)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    report=run_report(**vars(args))
    print(json.dumps({'status':report['status'],'scientific_inference_performed':report['scientific_inference_performed']}))

if __name__=='__main__':main()
