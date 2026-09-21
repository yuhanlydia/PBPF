#!/usr/bin/env python3
"""Independent Replay inference; incomplete inventories are never conclusions."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import runpy
from pbpf.eesd.mechanism_contrasts import contrast_ids,BIN_IDS
from pbpf.eesd.mechanism_inference import PairedMechanismData,bootstrap_gain,family_report,unsupported,METRICS
from pbpf.eesd import replay_generation as generation,replay_execution_cache as execution,replay_runtime as runtime

ROOT=Path(__file__).resolve().parents[1]
DOMAINS=('apps_replay','codecontests_replay')
MODELS=('qwen25_7b','deepseek_6p7b','seed_coder_8b','starcoder2_15b')
SEEDS=(1701,1702,1703)
PRIMARY='core/effective_params'
# Reuse mathematical/identity combination helpers without changing their globals.
_helpers=runpy.run_path(str(ROOT/'scripts/report_eesd_mechanism_inference.py'))
bin_support=_helpers['bin_support'];combine=_helpers['combine'];verify_report_bytes=_helpers['verify_report_bytes']
sha=_helpers['sha'];read=_helpers['read'];write=_helpers['write'];require=_helpers['require']


def families():
    from pbpf.eesd.replay_statistics import families as get
    return get()

def derive_public_support(**kwargs):
    from pbpf.eesd.replay_artifacts import derive_public_support as derive
    return derive(**kwargs)

def reconstruct_contrasts(**kwargs):
    from pbpf.eesd.replay_artifacts import reconstruct_contrasts as reconstruct
    return reconstruct(**kwargs)


def source_inventory():
    names=('scripts/report_eesd_replay_inference.py','scripts/report_eesd_mechanism_inference.py',
           'src/pbpf/eesd/replay_statistics.py','src/pbpf/eesd/replay_artifacts.py',
           'src/pbpf/eesd/mechanism_inference.py','src/pbpf/eesd/mechanism_contrasts.py',
           'src/pbpf/eesd/mechanism_artifacts.py','scripts/run_eesd_evidence_matrix.py','src/pbpf/eesd/evidence.py',
           'src/pbpf/eesd/replay_execution_cache.py','src/pbpf/eesd/replay_runtime.py',
           'src/pbpf/eesd/replay_generation.py','src/pbpf/eesd/replay_prompt.py','scripts/build_eesd_replay_mechanism_cache.py')
    return {name:sha(ROOT/name) for name in names}


def validate_inputs(*,statistical_lock,statistical_lock_sha256,config=None,cache_manifest=None):
    from pbpf.eesd.replay_statistics import validate_statistical_lock
    lock=validate_statistical_lock(statistical_lock,statistical_lock_sha256,config_path=config,cache_manifest_path=cache_manifest)
    config=Path(lock['config']['path']);matrix_path=Path(lock['cache_manifest']['path'])
    matrix=read(matrix_path)
    require(matrix.get('schema')=='eesd-replay-cache-matrix-v1' and matrix.get('status')=='planned-cache-execution-not-started','wrong Replay cache manifest')
    cells=matrix['cells']
    require(len(cells)==24 and {(c['domain'],c['family'],c['seed']) for c in cells}==set(itertools.product(DOMAINS,MODELS,SEEDS)),'complete unique 24-cell cache matrix required')
    for cell in cells:
        require(type(cell['seed']) is int,'candidate seed must be Python integer')
        require(set(cell['dependencies'])=={'development','primary'} and all(cell['dependencies'][s]['components']==n for s,n in generation.COUNTS.items()),'cache population count mismatch')
        expected=Path(matrix['output_root'])/'replay-mechanism-cache'/cell['domain']/cell['family']/f"seed{cell['seed']}"
        require(Path(cell['output']).resolve()==expected.resolve(),'cache output layout mismatch')
    runtime.validate_execution_lock(matrix['execution_lock']['path'],matrix['execution_lock']['sha256'])
    return lock,matrix,config


def verify_report_identity(directory,*,cache,config,domain,model,seed):
    report=read(Path(directory)/'report.json')
    expected=dict(schema='eesd-evidence-matrix-v1',dataset=domain,model=model,seed=seed,visible=4,
        validation_split='development',assessment_split='primary',cache_sha256=sha(cache),config_sha256=sha(config),
        runner_source_sha256=sha(ROOT/'scripts/run_eesd_evidence_matrix.py'),evidence_source_sha256=sha(ROOT/'src/pbpf/eesd/evidence.py'))
    require(all(report.get(k)==v for k,v in expected.items()),'existing report identity mismatch')


def verify_artifact_snapshot(entry):
    for path,expected in entry.get('sealed_artifacts',{}).items():
        require(sha(path)==expected,'sealed artifact changed during inference')


def inspect_cell(cell,matrix,config,statistical_lock_sha256,*,public_preflights=None):
    """Verify existing seals without accessing assessment outcome values.

    Public cache verification parses the complete JSON document. This is a
    field-access boundary, not a claim that label bytes were never loaded.
    """
    from pbpf.eesd import replay_artifacts as artifacts
    shared={} if public_preflights is None else public_preflights
    domain,model,seed=cell['domain'],cell['family'],cell['seed']
    inventory={};bank_receipts=[];missing=[]
    for split,count in generation.COUNTS.items():
        key=(domain,model,split)
        if key not in shared:shared[key]=generation.load_public_split(matrix['bundle'],matrix['admission_sha256'],domain,split,model)
        preflight=shared[key]
        identities=[{k:r[k] for k in ('task_id','source_component_id')} for r in preflight['rows']]
        require(len(identities)==count and len({r['source_component_id'] for r in identities})==count,'admitted source population count differs')
        require(execution.digest(identities)==cell['dependencies'][split]['population_sha256'],'planned population differs from admitted public sources')
        require(identities==matrix['populations'][domain][split]['ordered_identities'],'matrix public identities differ')
        inventory[split]=identities
        bank=Path(cell['dependencies'][split]['bank'])
        expected=generation.build_expected(preflight,model,seed,ROOT/'scripts/generate_eesd_replay_bank.py')
        if (bank/'complete.json').exists():
            verified=generation.verify_replay_bank(bank,expected,preflight=preflight)
            bank_receipts.append({'split':split,'path':str(bank.resolve()),'components':count,'complete_sha256':verified['complete_sha256']})
        else:
            if (bank/'run.json').exists():require(read(bank/'run.json')==expected,'partial bank run identity changed')
            missing.append(f'{split} bank incomplete' if bank.exists() else f'{split} bank missing')
    require(not {r['source_component_id'] for r in inventory['development']} & {r['source_component_id'] for r in inventory['primary']},'public source overlap')
    cache=Path(cell['output']);report=Path(matrix['output_root'])/'replay-mechanism'/domain/model/f'seed{seed}'
    context=None
    if (cache/'complete.json').exists():
        require(len(bank_receipts)==2,'sealed cache depends on missing sealed banks')
        cli=runpy.run_path(str(ROOT/'scripts/build_eesd_replay_mechanism_cache.py'))
        inputs=execution.load_replay_execution_inputs(matrix['bundle'],matrix['admission_sha256'],domain,model,seed,
            cell['dependencies']['development']['bank'],cell['dependencies']['primary']['bank'],
            generation_manifest=matrix['generation_manifest']['path'],generation_manifest_sha256=matrix['generation_manifest']['sha256'],
            extra_source_paths=cli['common_sources']())
        ready=read(cache/'binding.json')['readiness'];lock=read(matrix['execution_lock']['path'])
        cli['validate_historical_readiness'](ready,lock,matrix['execution_lock']['sha256'])
        # This verifier deliberately does not inspect outcome values.
        artifacts.verify_public_cache_projection(cache/'cache.json',inputs,ready)
        context=dict(report_dir=str(report),cache=str(cache/'cache.json'),config=config,domain=domain,model=model,seed=seed,
            expected_inputs=inputs,expected_readiness=ready,
            expected_bindings={'statistical_lock_sha256':statistical_lock_sha256,'execution_lock_sha256':matrix['execution_lock']['sha256']},root=ROOT)
    else:missing.append('cache incomplete' if cache.exists() else 'cache missing')
    if (report/'complete.json').exists():
        verify_report_bytes(report)
        require(context is not None and not missing,'sealed report depends on incomplete banks/cache')
        expected=artifacts.build_report_binding(**{k:context[k] for k in ('cache','config','expected_inputs','expected_readiness','expected_bindings','root')})
        require(read(report/'replay-binding.json')==expected,'Replay report binding mismatch')
        verify_report_identity(report,cache=cache/'cache.json',config=config,domain=domain,model=model,seed=seed)
    else:missing.append('report incomplete' if report.exists() else 'report missing')
    entry={'status':'missing' if missing else 'verified','reasons':missing,'report_dir':str(report),'cache':str(cache/'cache.json'),'verified_banks':bank_receipts}
    entry['sealed_artifacts']={}
    if (cache/'complete.json').exists():
        entry['sealed_artifacts'].update({str(cache/name):sha(cache/name) for name in ('cache.json','binding.json','complete.json')})
    if (report/'complete.json').exists():
        entry['sealed_artifacts'].update({str(report/name):sha(report/name) for name in ('report.json','predictions.npz','complete.json','replay-binding.json')})
    return inventory,entry,context


def incomplete_families(reason):
    primary,secondary=families()
    return {name:family_report({key:unsupported(reason) for key in keys},family=keys) for name,keys in [('primary',primary),('secondary',secondary)]}


def run_report(*,statistical_lock,statistical_lock_sha256,output,config=None,cache_manifest=None,preflight_only=False):
    lock,matrix,config=validate_inputs(statistical_lock=statistical_lock,statistical_lock_sha256=statistical_lock_sha256,config=config,cache_manifest=cache_manifest)
    coverage={};inventories={};contexts={};public_preflights={}
    for cell in matrix['cells']:
        key=f"{cell['domain']}/{cell['family']}"
        inventory,entry,context=inspect_cell(cell,matrix,config,statistical_lock_sha256,public_preflights=public_preflights)
        require(key not in inventories or inventory==inventories[key],'cross-seed public population changed')
        inventories[key]=inventory;coverage.setdefault(key,{})[str(cell['seed'])]=entry
        contexts.setdefault(key,{})[cell['seed']]=context
    for domain in DOMAINS:
        require(all(inventories[f'{domain}/{model}']==inventories[f'{domain}/{MODELS[0]}'] for model in MODELS),'cross-model public population changed')
    complete=all(entry['status']=='verified' for seeds in coverage.values() for entry in seeds.values())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    input_lock={'schema':'eesd-replay-inference-input-lock-v1','statistical_lock':{'path':str(Path(statistical_lock).resolve()),'sha256':statistical_lock_sha256},
        'config':lock['config'],'cache_manifest':lock['cache_manifest'],'reporter_source_sha256':sha(__file__),
        'analysis_sources':source_inventory(),'public_inventories':inventories,'coverage':coverage}
    write(output/'input-lock.json',input_lock)
    if preflight_only or not complete:
        report={'schema':'eesd-replay-inference-report-v1','status':'preflight_only' if preflight_only else 'incomplete','scientific_inference_performed':False,
            'coverage':coverage,'families':incomplete_families('preflight only' if preflight_only else 'missing required reports')}
    else:
        support={};support_receipts={};reconstructed={};receipts={}
        for domain,model in itertools.product(DOMAINS,MODELS):
            key=f'{domain}/{model}';public_rows={};support[key]={};support_receipts[key]={}
            expected={r['source_component_id'] for r in inventories[key]['primary']}
            require(len(expected)==500,'formal primary population must be 500 sources')
            for seed in SEEDS:
                verify_artifact_snapshot(coverage[key][str(seed)])
                rows,receipt=derive_public_support(**contexts[key][seed])
                verify_artifact_snapshot(coverage[key][str(seed)])
                require(set(rows)==set(BIN_IDS),'public support bin inventory mismatch')
                require(all(set(map(str,row['sources']))==expected for row in rows.values()),'public support differs from declared 500 sources')
                public_rows[seed]=rows;support_receipts[key][str(seed)]=receipt
            for contrast in BIN_IDS:support[key][contrast]=bin_support({seed:public_rows[seed][contrast] for seed in SEEDS})
        write(output/'a9-public-support.json',support);write(output/'a9-public-support-receipts.json',support_receipts)
        support_sha=sha(output/'a9-public-support.json')
        write(output/'a9-public-support.complete.json',{'support_sha256':support_sha,'receipts_sha256':sha(output/'a9-public-support-receipts.json'),
            'input_lock_sha256':sha(output/'input-lock.json'),'scope':'all 24 public supports sealed before assessment inferential reconstruction'})
        for domain,model in itertools.product(DOMAINS,MODELS):
            key=f'{domain}/{model}';reconstructed[key]={};receipts[key]={}
            expected={r['source_component_id'] for r in inventories[key]['primary']}
            for seed in SEEDS:
                verify_artifact_snapshot(coverage[key][str(seed)])
                rows,receipt=reconstruct_contrasts(**contexts[key][seed])
                verify_artifact_snapshot(coverage[key][str(seed)])
                require(set(rows)==set(contrast_ids()),'49 contrast reconstruction inventory mismatch')
                for row in rows.values():
                    if row['status']=='available':require(set(map(str,row['sources']))==expected,'reconstructed source population differs from declared 500 sources')
                reconstructed[key][seed]=rows;receipts[key][str(seed)]=receipt
            for contrast in BIN_IDS:
                rows={seed:reconstructed[key][seed][contrast] for seed in SEEDS}
                if all(row['status']=='available' for row in rows.values()):
                    require(bin_support(rows)==support[key][contrast],'assessment reconstruction changed frozen public support/identities')
        require(sha(output/'a9-public-support.json')==support_sha,'frozen support bytes changed')
        write(output/'reconstruction-receipts.json',receipts)
        results={};contrasts_report={}
        for domain,model in itertools.product(DOMAINS,MODELS):
            key=f'{domain}/{model}';contrasts_report[key]={}
            for contrast in contrast_ids():
                if contrast=='A8/n4':continue
                rows={seed:reconstructed[key][seed][contrast] for seed in SEEDS}
                if any(row['status']!='available' for row in rows.values()):
                    result=unsupported('; '.join(f'{s}: {r.get("reason","unavailable")}' for s,r in rows.items() if r['status']!='available'))
                else:
                    selected=support[key][contrast]['common_sources'] if contrast in BIN_IDS else None
                    if selected is not None and len(selected)<2:result=unsupported('fewer than two common public-bin source clusters')
                    else:
                        data=PairedMechanismData(**combine(rows,selected_sources=selected,use_public_mask=contrast in BIN_IDS))
                        result=bootstrap_gain(data,domain=domain,model=model,population=f'mass_bin_{BIN_IDS.index(contrast)+1}' if contrast in BIN_IDS else 'all')
                contrasts_report[key][contrast]=result
            aliases=[reconstructed[key][seed]['A8/n4'] for seed in SEEDS]
            if any(row['status']!='available' for row in aliases):contrasts_report[key]['A8/n4']=unsupported('history n4 reconstruction unavailable')
            else:
                require(all(row.get('alias_of')=='core/tuned' for row in aliases),'history n4 alias declaration mismatch')
                contrasts_report[key]['A8/n4']={**contrasts_report[key]['core/tuned'],'alias_of':'core/tuned'}
            for contrast,result in contrasts_report[key].items():
                for metric in METRICS:results[f'{key}/{contrast}/{metric}']=result['metrics'][metric] if result['status']=='supported' else result
        primary,secondary=families()
        report={'schema':'eesd-replay-inference-report-v1','status':'inference_computed','scientific_inference_performed':True,'coverage':coverage,'contrasts':contrasts_report,
            'a9_public_support_sha256':support_sha,'a9_public_support_seal_sha256':sha(output/'a9-public-support.complete.json'),
            'reconstruction_receipts_sha256':sha(output/'reconstruction-receipts.json'),
            'families':{name:family_report({k:results[k] for k in keys},family=keys) for name,keys in [('primary',primary),('secondary',secondary)]}}
    for entries in coverage.values():
        for entry in entries.values():verify_artifact_snapshot(entry)
    require(source_inventory()==input_lock['analysis_sources'],'analysis source changed during reporting')
    require(sha(statistical_lock)==statistical_lock_sha256 and sha(lock['config']['path'])==lock['config']['sha256']
        and sha(lock['cache_manifest']['path'])==lock['cache_manifest']['sha256'],'locked inputs changed during reporting')
    report['input_lock_sha256']=sha(output/'input-lock.json')
    report['claim_scope']='Replay-only primary/secondary families; no pooled original-study or joint study-wide FWER claim'
    write(output/'report.json',report);write(output/'complete.json',{'report_sha256':sha(output/'report.json'),'input_lock_sha256':sha(output/'input-lock.json')})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistical-lock',type=Path,required=True);parser.add_argument('--statistical-lock-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--config',type=Path);parser.add_argument('--cache-manifest',type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    report=run_report(**vars(parser.parse_args()));print(json.dumps({'status':report['status'],'scientific_inference_performed':report['scientific_inference_performed']}))

if __name__=='__main__':main()
