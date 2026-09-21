#!/usr/bin/env python3
"""Verify Replay provenance, run frozen evidence mathematics, publish atomically."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from pbpf.eesd import replay_artifacts as artifacts,replay_execution_cache as execution,replay_runtime as runtime
from pbpf.eesd.replay_admission import rename_new_directory

ROOT=Path(__file__).resolve().parents[1]


def validate_statistical_lock(*args,**kwargs):
    from pbpf.eesd.replay_statistics import validate_statistical_lock as validate
    return validate(*args,**kwargs)


def execution_helpers():
    return runpy.run_path(str(ROOT/'scripts/build_eesd_replay_mechanism_cache.py'))


def write_once(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')


def run(args):
    lock=validate_statistical_lock(args.statistical_lock,args.statistical_lock_sha256)
    config=Path(lock['config']['path']);matrix=json.loads(Path(lock['cache_manifest']['path']).read_text())
    if matrix.get('schema')!='eesd-replay-cache-matrix-v1' or matrix.get('status')!='planned-cache-execution-not-started':
        raise ValueError('wrong Replay cache matrix')
    cells=[c for c in matrix['cells'] if (c['domain'],c['family'],c['seed'])==(args.domain,args.family,args.seed)]
    if len(cells)!=1:raise ValueError('missing/duplicate evidence cell')
    cell=cells[0];cache_dir=Path(cell['output']);execution_ref=matrix['execution_lock']
    execution_lock=runtime.validate_execution_lock(execution_ref['path'],execution_ref['sha256'])
    helpers=execution_helpers()
    inputs=execution.load_replay_execution_inputs(matrix['bundle'],matrix['admission_sha256'],args.domain,args.family,args.seed,
        cell['dependencies']['development']['bank'],cell['dependencies']['primary']['bank'],
        generation_manifest=matrix['generation_manifest']['path'],generation_manifest_sha256=matrix['generation_manifest']['sha256'],
        extra_source_paths=helpers['common_sources']())
    if inputs['bindings']['execution_sources']!=execution_lock['sources']:
        raise ValueError('cache execution sources differ from locked runtime')
    ready=execution.read(cache_dir/'binding.json')['readiness']
    helpers['validate_historical_readiness'](ready,execution_lock,execution_ref['sha256'])
    artifacts.verify_replay_evidence_inputs(cache_dir=cache_dir,expected_inputs=inputs,expected_readiness=ready)
    output=Path(matrix['output_root'])/'replay-mechanism'/args.domain/args.family/f'seed{args.seed}'
    kwargs=dict(cache=cache_dir/'cache.json',config=config,domain=args.domain,model=args.family,seed=args.seed,
        expected_inputs=inputs,expected_readiness=ready,expected_bindings={'statistical_lock_sha256':args.statistical_lock_sha256,
        'execution_lock_sha256':execution_ref['sha256']},root=ROOT)
    if output.exists():
        required={'report.json','predictions.npz','complete.json','replay-binding.json'}
        if not output.is_dir() or {p.name for p in output.iterdir()}!=required:
            raise ValueError('existing evidence target incomplete or contains unexpected artifacts; preserve it')
        artifacts.load_replay_artifacts(report_dir=output,**kwargs)
        print(json.dumps({'status':'verified-complete-report-resume','output':str(output)}));return 0
    if args.preflight_only:
        print(json.dumps({'status':'verified-input-files-not-inferred','output':str(output)}));return 0
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.pending-',dir=output.parent));report_dir=staging/'report'
    try:
        command=[sys.executable,str(ROOT/'scripts/run_eesd_evidence_matrix.py'),'--config',str(config),
            '--cache',str(cache_dir/'cache.json'),'--dataset',args.domain,'--model',args.family,'--seed',str(args.seed),
            '--validation-split','development','--assessment-split','primary','--visible','4','--output',str(report_dir)]
        with (staging/'runner.log').open('x') as log:subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
        binding=artifacts.build_report_binding(**{key:kwargs[key] for key in ('cache','config','expected_inputs','expected_readiness','expected_bindings','root')})
        write_once(report_dir/'replay-binding.json',binding)
        artifacts.load_replay_artifacts(report_dir=report_dir,**kwargs)
        validate_statistical_lock(args.statistical_lock,args.statistical_lock_sha256)
        runtime.validate_execution_lock(execution_ref['path'],execution_ref['sha256'])
        rename_new_directory(report_dir,output)
    except BaseException as error:
        write_once(staging/'failure.json',{'status':'failed-not-published','error_type':type(error).__name__,'error':str(error)})
        raise
    # Retain the subprocess log beside the result without polluting its sealed inventory.
    write_once(staging/'published.json',{'status':'published','output':str(output)})
    print(json.dumps({'status':'replay-evidence-complete-not-family-inference','output':str(output)}));return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--statistical-lock',type=Path,required=True);p.add_argument('--statistical-lock-sha256',required=True)
    p.add_argument('--domain',choices=execution.generation.DOMAINS,required=True)
    p.add_argument('--family',choices=execution.generation.MODELS,required=True)
    p.add_argument('--seed',type=int,choices=[1701,1702,1703],required=True)
    p.add_argument('--preflight-only',action='store_true')
    raise SystemExit(run(p.parse_args()))

if __name__=='__main__':main()
