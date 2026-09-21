#!/usr/bin/env python3
"""Build separate Replay execution caches only after fresh sandbox readiness."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import hashlib
from datetime import datetime
import multiprocessing
from pathlib import Path
from pbpf.eesd import replay_execution_cache as cache, replay_runtime as runtime


def common_sources():
    return sorted(set(cache.EXECUTION_SOURCES)|{
        str(Path(runtime.__file__).resolve().relative_to(runtime.ROOT)),
        str(Path(__file__).resolve().relative_to(runtime.ROOT))})


def write_once(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def bounded_process_map(jobs,workers):
    """At most workers submitted tasks; preserve population order in results."""
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
    pending={};indexed=iter(enumerate(jobs));results=[None]*len(jobs)
    def submit_next():
        item=next(indexed,None)
        if item is not None:
            index,job=item;pending[pool.submit(cache.measure_job,job)]=index
    try:
        for _ in range(workers):submit_next()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            # Check every completed future before admitting further work.
            values=[(future,pending[future],future.result()) for future in done]
            for future,index,value in values:
                pending.pop(future);results[index]=value
            for _ in values:submit_next()
        return results
    finally:
        pool.shutdown(wait=True,cancel_futures=True)


def validate_historical_readiness(receipt,lock,lock_sha):
    if (receipt.get('execution_lock_sha256')!=lock_sha or receipt.get('profile')!=lock['profile']
        or receipt.get('sources')!=lock['sources']):
        raise ValueError('historical readiness differs from current execution lock')
    workers=receipt.get('worker_count')
    if type(workers) is not int or not 1<=workers<=lock['max_workers']:
        raise ValueError('historical readiness worker count outside execution lock')
    expected_probe={'code_sha256':lock['probe_code_sha256'],
        'stdout_sha256':hashlib.sha256((runtime.SENTINEL+'\n').encode()).hexdigest()}
    if receipt.get('probe')!=expected_probe:
        raise ValueError('historical trusted probe evidence differs')
    stamp=receipt.get('checked_at')
    try:
        checked=datetime.fromisoformat(stamp)
    except (TypeError,ValueError) as error:
        raise ValueError('historical readiness timestamp invalid') from error
    if checked.tzinfo is None or checked.utcoffset() is None:
        raise ValueError('historical readiness timestamp requires timezone')


def run(args):
    if args.freeze_lock:
        lock=runtime.build_execution_lock(extra_source_paths=common_sources(),max_workers=args.workers)
        write_once(args.execution_lock,lock)
        print(json.dumps(dict(status='execution-profile-locked-not-probed',path=str(args.execution_lock),sha256=runtime.sha(args.execution_lock))))
        return 0
    lock=runtime.validate_execution_lock(args.execution_lock,args.execution_lock_sha256)
    if type(args.workers) is not int or not 1<=args.workers<=lock['max_workers']:
        raise ValueError('worker count exceeds execution lock')
    if args.probe_only:
        try:
            receipt=runtime.probe_readiness(args.execution_lock,args.execution_lock_sha256)
            report={'status':'probe_passed','readiness':receipt};code=0
        except Exception as error:
            report={'status':'infrastructure_blocked','execution_lock_sha256':args.execution_lock_sha256,
                    'error_type':type(error).__name__,'error':str(error)};code=3
        if args.probe_report:write_once(args.probe_report,report)
        print(json.dumps(report));return code
    try:
        inputs=cache.load_replay_execution_inputs(args.bundle,args.admission_sha256,args.domain,args.family,args.seed,
            args.development_bank,args.primary_bank,generation_manifest=args.generation_manifest,
            generation_manifest_sha256=args.generation_manifest_sha256,extra_source_paths=common_sources())
    except FileNotFoundError as error:
        missing=[str(Path(bank)/name) for bank in (args.development_bank,args.primary_bank)
                 for name in ('run.json','complete.json') if not (Path(bank)/name).is_file()]
        if not missing or error.filename is None or Path(error.filename).resolve() not in {Path(p).resolve() for p in missing}:raise
        print(json.dumps(dict(status='waiting-for-sealed-generation-banks',missing=missing)));return 2
    if inputs['bindings']['execution_sources']!=lock['sources']:
        raise ValueError('execution lock source inventory differs from input bindings')
    if args.output.exists():
        historical=cache.read(args.output/'binding.json')['readiness']
        validate_historical_readiness(historical,lock,args.execution_lock_sha256)
        result=cache.verify_replay_cache(args.output,expected_inputs=inputs,expected_readiness=historical)
        print(json.dumps(dict(status='verified-complete-cache',verification=result)));return 0
    if args.preflight_only:
        print(json.dumps(dict(status='input-files-verified-not-executed',sources=len(inputs['jobs']),
            execution_lock_sha256=args.execution_lock_sha256)));return 0
    def readiness_check():
        ready=runtime.probe_readiness(args.execution_lock,args.execution_lock_sha256)
        ready['worker_count']=args.workers
        return ready
    try:
        measured=cache.measure_jobs(inputs,readiness_check=readiness_check,
            job_mapper=lambda jobs:bounded_process_map(jobs,args.workers))
        runtime.validate_execution_lock(args.execution_lock,args.execution_lock_sha256)
        cache.publish_cache(args.output,inputs,measured)
        result=cache.verify_replay_cache(args.output,expected_inputs=inputs,expected_readiness=measured['readiness'])
    except Exception as error:
        print(json.dumps(dict(status='infrastructure_or_artifact_failure-no-scientific-result',error_type=type(error).__name__,error=str(error))))
        return 3
    print(json.dumps(dict(status='execution-cache-complete-not-inference',verification=result)));return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    for flag in ('freeze-lock','probe-only','preflight-only'):modes.add_argument('--'+flag,action='store_true')
    parser.add_argument('--execution-lock',type=Path,required=True)
    parser.add_argument('--execution-lock-sha256')
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--probe-report',type=Path)
    for flag in ('bundle','generation-manifest','development-bank','primary-bank','output'):
        parser.add_argument('--'+flag,type=Path)
    for flag in ('admission-sha256','generation-manifest-sha256'):parser.add_argument('--'+flag)
    parser.add_argument('--domain',choices=cache.generation.DOMAINS)
    parser.add_argument('--family',choices=cache.generation.MODELS)
    parser.add_argument('--seed',type=int,choices=[1701,1702,1703])
    args=parser.parse_args()
    if not args.freeze_lock and not args.execution_lock_sha256:parser.error('--execution-lock-sha256 required')
    if not args.freeze_lock and not args.probe_only:
        required=('bundle','generation_manifest','development_bank','primary_bank','output','admission_sha256','generation_manifest_sha256','domain','family','seed')
        if any(getattr(args,key) is None for key in required):parser.error('cache modes require all data/bank/cell arguments')
    raise SystemExit(run(args))

if __name__=='__main__':main()
