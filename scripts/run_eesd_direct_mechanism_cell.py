#!/usr/bin/env python3
"""One original-domain cell under an explicitly separate no-sandbox profile."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import FunctionType
from unittest.mock import patch
import yaml
from pbpf.eesd.replay_admission import rename_new_directory

ROOT=Path(__file__).resolve().parents[1]
_BUILDER=runpy.run_path(str(ROOT/'scripts/build_eesd_mechanism_cache.py'))

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def dump(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
def runtime_api():
    from pbpf.eesd import direct_runtime
    return direct_runtime

def matrix_api():return runpy.run_path(str(ROOT/'scripts/run_eesd_matrix.py'))


def work(job):
    """Spawn-picklable adapter; original projection and all ten calls unchanged."""
    from pbpf.apbpf import direct_execution
    original=_BUILDER['work'];namespace=dict(original.__globals__)
    namespace.update(execute_stdin=direct_execution.execute_stdin,execute_call=direct_execution.execute_call)
    return FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)(job)


def spawn_pool(**kwargs):
    return ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'),**kwargs)


def build_direct_cache(args,cell,banks,cache):
    original=_BUILDER['main'];namespace=dict(original.__globals__)
    namespace.update(work=work,ProcessPoolExecutor=spawn_pool)
    run=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    argv=['build_eesd_mechanism_cache.py','--domain',args.domain,'--evaluator-root',cell['evaluator_root'],
          '--output',str(cache),'--workers',str(args.workers),'--timeout','6']
    for path,_,_ in banks:argv.extend(['--bank',str(path.resolve())])
    with patch.object(sys,'argv',argv):run()


def run_evidence(args,cell,cache,directory):
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_eesd_evidence_matrix.py'),
        '--config',str(args.config.resolve()),'--cache',str(cache),'--dataset',cell['dataset'],
        '--model',cell['model'],'--seed',str(args.seed),'--visible','4','--validation-split','development',
        '--assessment-split','primary','--output',str(directory)],cwd=ROOT,check=True)


def run(args):
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    require(args.domain in ('rbr','codearc') and args.seed in (1701,1702,1703),'invalid cell identity')
    require(sha(args.manifest)==args.manifest_sha256 and sha(args.config)==args.config_sha256,'manifest/config checksum mismatch')
    runtime=runtime_api();lock=runtime.validate_execution_lock(args.execution_lock,args.execution_lock_sha256)
    require(lock['profile']=='direct-no-sandbox' and lock['timeout_seconds']==6.,'wrong direct execution profile')
    require(type(args.workers) is int and 1<=args.workers<=lock['max_workers'],'worker count outside execution lock')
    for field in ('manifest','config'):
        path=getattr(args,field)
        require(Path(lock[field]['path']).resolve()==path.resolve() and lock[field]['sha256']==getattr(args,field+'_sha256'),'execution lock input mismatch')
    manifest=yaml.safe_load(args.manifest.read_text());config=yaml.safe_load(args.config.read_text())
    require(manifest.get('schema')=='eesd-cache-manifest-v1' and config.get('schema')=='eesd-iclr2027-v1' and args.seed in config['seeds'],'protocol/seed mismatch')
    cells=[c for c in manifest['mechanism_cells'] if c['domain']==args.domain and c['family']==args.family]
    require(len(cells)==1,'unique manifest cell required');cell=cells[0]
    require(cell['model']==args.family and cell.get('visible',4)==4 and cell.get('development_components',200)==200,'cell scientific parameters differ')
    verifier=matrix_api();banks=[];sources=set()
    for split,count in [('development',200),('primary',500)]:
        path=args.results_root/'mechanism-banks'/cell['dataset']/cell['model']/f'seed{args.seed}'/split
        bound=verifier['verify_mechanism_bank'](path,cell=cell,seed=args.seed,split=split,root=ROOT)
        info=bound[1];ids=info['source_component_ids']
        require(info['components']==count and len(info['task_ids'])==count and len(set(ids))==count and not sources.intersection(ids),'bank population must be 200/500 disjoint sources')
        sources.update(ids);banks.append(bound)
    watched={str(args.manifest.resolve()):args.manifest_sha256,str(args.config.resolve()):args.config_sha256,
             str(args.execution_lock.resolve()):args.execution_lock_sha256}
    for field in ('public_root','evaluator_root'):
        for name in ('manifest.json','tasks.jsonl'):
            path=Path(cell[field]).resolve()/name;watched[str(path)]=sha(path)
    for path,_,_ in banks:
        for file in path.iterdir():watched[str(file.resolve())]=sha(file)
    def unchanged():
        for path,value in watched.items():require(sha(path)==value,'input/bank changed during direct execution')
        require(runtime.validate_execution_lock(args.execution_lock,args.execution_lock_sha256)==lock,'runtime/source lock changed')
    # Both complete-bank validations precede the trusted probes and dataset work.
    ready=runtime.probe_readiness(args.execution_lock,args.execution_lock_sha256)
    require(ready.get('status')=='ready' and ready.get('profile')=='direct-no-sandbox'
        and ready.get('execution_lock_sha256')==args.execution_lock_sha256 and ready.get('sources')==lock['sources'],'invalid direct readiness receipt')
    unchanged()
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.direct-partial-',dir=output.parent))
    try:
        cache=staging/'cache.json';build_direct_cache(args,cell,banks,cache)
        payload=read(cache)
        require(payload.get('schema')=='eesd-public-query-mechanism-cache-v1' and payload.get('dataset')==args.domain
            and payload.get('counts')=={'development':200,'primary':500} and payload.get('source_counts')=={'development':200,'primary':500},'direct cache coverage mismatch')
        records=payload['records'];require(len(records)==700 and {r['source_component_id'] for r in records}==sources,'direct cache source coverage mismatch')
        require(all(len(r['tests'])==10 and len(r['outcomes'])==10 and all(x in ('PASS','WRONG_OUTPUT','COMPILE_ERROR','RUNTIME_EXCEPTION','TIMEOUT') for x in r['outcomes']) for r in records),'direct cache outcome inventory mismatch')
        unchanged()
        report_dir=staging/'report';run_evidence(args,cell,cache,report_dir)
        verifier['verify_mechanism_report'](report_dir,cache=cache,config=args.config,cell=cell,seed=args.seed,root=ROOT)
        unchanged()
        binding={'schema':'eesd-direct-mechanism-binding-v1','execution_profile':'direct-no-sandbox',
            'claim_scope':'new execution profile; not a sandbox-backed cache or original execution seal',
            'domain':args.domain,'dataset':cell['dataset'],'family':args.family,'seed':args.seed,'workers':args.workers,
            'execution_lock':{'path':str(args.execution_lock.resolve()),'sha256':args.execution_lock_sha256},
            'readiness':ready,'sources':lock['sources'],'inputs':watched,
            'banks':[{'path':str(path.resolve()),'split':info['split'],'components':info['components'],'complete_sha256':digest} for path,info,digest in banks],
            'cache_sha256':sha(cache),'report_complete_sha256':sha(report_dir/'complete.json')}
        dump(staging/'direct-binding.json',binding)
        files={str(path.relative_to(staging)):sha(path) for path in staging.rglob('*') if path.is_file()}
        dump(staging/'complete.json',{'schema':'eesd-direct-mechanism-complete-v1','execution_profile':'direct-no-sandbox','files':files})
        rename_new_directory(staging,output)
    except BaseException:
        # Preserve the unpublished attempt for diagnosis; never publish partial success.
        raise
    return {'status':'direct-mechanism-cell-complete','execution_profile':'direct-no-sandbox','output':str(output),'complete_sha256':sha(output/'complete.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','config','results-root','execution-lock','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('manifest-sha256','config-sha256','execution-lock-sha256','family'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--domain',choices=['rbr','codearc'],required=True);parser.add_argument('--seed',type=int,choices=[1701,1702,1703],required=True)
    parser.add_argument('--workers',type=int,default=4)
    print(json.dumps(run(parser.parse_args())))

if __name__=='__main__':main()
