#!/usr/bin/env python3
"""Plan 24 Replay execution caches; read public metadata, never run a probe."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import runpy
import tempfile

PROFILE={'builder_python':'3.11.16','candidate_python':'3.10.12','bubblewrap':'0.6.1','timeout_seconds':6.0}
EXECUTION_SOURCES=('scripts/build_eesd_replay_mechanism_cache.py','src/pbpf/eesd/replay_execution_cache.py',
 'src/pbpf/eesd/replay_runtime.py','src/pbpf/apbpf/rbr_execution.py','src/pbpf/apbpf/codearc_execution.py',
 'src/pbpf/eesd/replay_materialization.py','src/pbpf/eesd/replay_admission.py')
DOMAINS=('apps_replay','codecontests_replay')
FAMILIES=('qwen25_7b','deepseek_6p7b','seed_coder_8b','starcoder2_15b')
SEEDS=(1701,1702,1703)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def require(ok,message):
    if not ok:raise ValueError(message)


def build_cache_plan(root,generation_manifest,generation_manifest_sha256,execution_lock,execution_lock_sha256,output_root,*,operational_probe_report=None,generation_plan_builder=None):
    root,gen_path,lock_path,output_root=(Path(p).resolve() for p in (root,generation_manifest,execution_lock,output_root))
    require(sha(gen_path)==generation_manifest_sha256,'generation manifest checksum mismatch')
    require(sha(lock_path)==execution_lock_sha256,'execution lock checksum mismatch')
    generation=json.loads(gen_path.read_text());lock=json.loads(lock_path.read_text())
    require(generation.get('schema')=='eesd-replay-generation-matrix-v1' and generation.get('status')=='planned-generation-only-not-launched','wrong generation manifest')
    require(lock.get('schema')=='eesd-replay-execution-lock-v1' and lock.get('status')=='locked-not-probed','wrong execution lock')
    require(lock.get('profile')==PROFILE and lock.get('runtime',{}).get('profile')==PROFILE,'execution profile mismatch')
    require(type(lock.get('max_workers')) is int and lock['max_workers']==12,'matrix requires locked twelve-worker capacity')
    require(set(lock.get('sources',{}))==set(EXECUTION_SOURCES),'execution source inventory mismatch')
    sources=dict(generation['sources'])
    for name,value in lock['sources'].items():
        require(name not in sources or sources[name]==value,'inconsistent shared source hash')
        sources[name]=value
    own='scripts/plan_eesd_replay_caches.py';sources[own]=sha(root/own)
    for name,value in sources.items():
        path=(root/name).resolve();require(path.is_relative_to(root),'source outside repository')
        require(sha(path)==value,'generation/execution source changed')
    executables=lock['runtime'].get('executables',{})
    require(set(executables)=={'builder','candidate','bubblewrap'},'incomplete executable identity')
    for identity in executables.values():
        require(Path(identity['path']).is_absolute() and sha(identity['path'])==identity['sha256'],'runtime binary checksum changed')
    require(isinstance(lock.get('probe_code_sha256'),str) and len(lock['probe_code_sha256'])==64,'missing trusted probe identity')
    actual={(c['domain'],c['family'],c['seed']) for c in generation['cells']}
    require(len(generation['cells'])==24 and actual==set(itertools.product(DOMAINS,FAMILIES,SEEDS)),'incomplete Cartesian generation population')
    if generation_plan_builder is None:
        # Fixed local planner was source-verified above, not an input-provided program.
        generation_plan_builder=runpy.run_path(str(root/'scripts/plan_eesd_replay_generation.py'))['build_plan']
    reproduced=generation_plan_builder(root,Path(generation['bundle']),generation['admission_sha256'],Path(generation['output_root']))
    require(reproduced==generation,'generation manifest differs from current admitted public population/source/argv')
    cells=[]
    for cell in generation['cells']:
        domain,family,seed=cell['domain'],cell['family'],cell['seed']
        output=output_root/'replay-mechanism-cache'/domain/family/f'seed{seed}'
        dependencies={split:{'bank':cell['splits'][split]['output'],'components':count,
            'population_sha256':cell['splits'][split]['population_sha256'],
            'public_bindings':cell['splits'][split]['bindings'],
            'required_files':['run.json','complete.json'],
            'status':'sealed-bank-required-not-inspected-by-planner'} for split,count in [('development',200),('primary',500)]}
        argv=[str(root/'.venv/bin/python'),str(root/'scripts/build_eesd_replay_mechanism_cache.py'),
              '--bundle',generation['bundle'],'--admission-sha256',generation['admission_sha256'],
              '--generation-manifest',str(gen_path),'--generation-manifest-sha256',generation_manifest_sha256,
              '--domain',domain,'--family',family,'--seed',str(seed),
              '--development-bank',dependencies['development']['bank'],'--primary-bank',dependencies['primary']['bank'],
              '--execution-lock',str(lock_path),'--execution-lock-sha256',execution_lock_sha256,
              '--workers','12','--output',str(output)]
        cells.append({'domain':domain,'family':family,'seed':seed,'candidate_jobs':700,'test_executions':7000,
                      'dependencies':dependencies,'output':str(output),'argv':argv})
    reference=None
    if operational_probe_report is not None:
        path=Path(operational_probe_report).resolve();report=json.loads(path.read_text())
        require(report.get('status') in ('infrastructure_blocked','probe_passed'),'unknown operational report')
        observed_lock=report.get('execution_lock_sha256',report.get('readiness',{}).get('execution_lock_sha256'))
        require(observed_lock==execution_lock_sha256,'operational probe refers to another lock')
        reference={'path':str(path),'sha256':sha(path),'observed_status':report['status'],
                   'scope':'historical operational reference only; never authorizes execution or establishes current readiness'}
    require(sha(gen_path)==generation_manifest_sha256 and sha(lock_path)==execution_lock_sha256,'input changed during planning')
    for name,value in sources.items():require(sha(root/name)==value,'source changed during planning')
    return {'schema':'eesd-replay-cache-matrix-v1','status':'planned-cache-execution-not-started',
        'runtime_readiness':'not-established-by-planner','generation_manifest':{'path':str(gen_path),'sha256':generation_manifest_sha256},
        'execution_lock':{'path':str(lock_path),'sha256':execution_lock_sha256},'sources':sources,
        'bundle':generation['bundle'],'admission_sha256':generation['admission_sha256'],
        'populations':generation['populations'],'output_root':str(output_root),'workers':12,
        'cells':cells,'cell_count':24,'planned_candidate_jobs':16800,'planned_test_executions':168000,
        'operational_reference':reference,'limitations':['No bank completion or sandbox readiness inferred by planner.',
        'No evaluator/raw data read; trusted cache builder validates private data only after both banks are sealed.']}


def write_once(path,value):
    path=Path(path);data=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(temp,path)
    finally:Path(temp).unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('generation-manifest','execution-lock','output-root','manifest-out'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('generation-manifest-sha256','execution-lock-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--operational-probe-report',type=Path)
    args=parser.parse_args()
    if args.manifest_out.exists():raise FileExistsError(args.manifest_out)
    plan=build_cache_plan(Path(__file__).resolve().parents[1],args.generation_manifest,args.generation_manifest_sha256,
        args.execution_lock,args.execution_lock_sha256,args.output_root,operational_probe_report=args.operational_probe_report)
    write_once(args.manifest_out,plan)
    print(json.dumps({'status':plan['status'],'cells':24,'manifest_sha256':sha(args.manifest_out)}))

if __name__=='__main__':main()
