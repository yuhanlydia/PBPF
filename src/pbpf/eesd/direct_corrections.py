"""Create-once correction adapters inheriting an explicitly direct execution lock.

Only a private copy of the original main's globals substitutes the two execution
functions. Original helper functions do not execute candidates. Scientific prompt,
generation, selection and scoring settings remain those of the original scripts.
New sources are bound by a separate request/completion receipt, not represented
as having been part of the older execution lock. This profile is not a sandbox.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import FunctionType
from pbpf.apbpf import direct_execution as direct
from . import direct_runtime as runtime
from .replay_admission import rename_new_directory

ROOT=Path(__file__).resolve().parents[3]
STAGES={'prepare':('prepare_eesd_recursive_corrections.py','prepare_eesd_direct_corrections.py',('public-corrections.jsonl','audit.json')),
        'generate':('generate_eesd_corrections.py','generate_eesd_direct_corrections.py',('corrections.jsonl','report.json'))}
sha=runtime.sha


def write_once(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,ensure_ascii=False,indent=2);stream.write('\n')


def direct_main(original):
    namespace=dict(original.__globals__)
    namespace.update(execute_stdin=direct.execute_stdin,execute_call=direct.execute_call)
    return FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)


def source_inventory(stage):
    original,wrapper,_=STAGES[stage]
    names=[f'scripts/{original}',f'scripts/{wrapper}','src/pbpf/eesd/direct_corrections.py',
           'src/pbpf/apbpf/direct_execution.py','src/pbpf/eesd/direct_runtime.py',
           'src/pbpf/apbpf/rbr_execution.py','src/pbpf/apbpf/codearc_execution.py',
           'src/pbpf/apbpf/codearc_bank.py','src/pbpf/eesd/replay_admission.py']
    if stage=='generate':names+=['src/pbpf/eesd/correction_prompt.py','src/pbpf/eesd/evidence.py',
        'src/pbpf/apbpf/rbr_prompt.py','src/pbpf/apbpf/code_extraction.py','src/pbpf/real_gate.py']
    return {name:sha(ROOT/name) for name in names}


def input_inventory(stage,args,lock_sha):
    paths=[]
    if stage=='prepare':
        if args.public_root is None or not args.bank:raise ValueError('prepare requires public-root and banks')
        paths=[args.public_root/'manifest.json',args.public_root/'tasks.jsonl']
        for bank in args.bank:
            if not (bank/'complete.json').is_file():raise FileNotFoundError(bank/'complete.json')
            paths.extend(p for p in sorted(bank.iterdir()) if p.is_file())
    else:
        if args.public_bank is None or args.model_config is None:raise ValueError('generate requires public-bank and model-config')
        receipt_path=args.public_bank.parent/'direct-profile.json'
        receipt=json.loads(receipt_path.read_text())
        if (receipt.get('schema')!='eesd-direct-corrections-profile-v1' or receipt.get('stage')!='prepare'
            or receipt.get('status')!='complete' or receipt.get('profile')!=runtime.PROFILE
            or receipt.get('execution_lock_sha256')!=lock_sha):raise ValueError('public bank direct profile missing/mismatched')
        for name,value in receipt['outputs'].items():
            if Path(name).name!=name or sha(args.public_bank.parent/name)!=value:raise ValueError('public direct bank output checksum mismatch')
        if receipt['outputs'].get(args.public_bank.name)!=sha(args.public_bank):raise ValueError('public bank not bound by direct receipt')
        for name,value in receipt['sources'].items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or sha(path)!=value:raise ValueError('upstream direct preparation source changed')
        paths=[args.public_bank,args.model_config,receipt_path]
        if args.adapter:
            if not args.adapter.is_dir():raise ValueError('adapter directory required')
            adapter_files=[p for p in sorted(args.adapter.rglob('*')) if p.is_file()]
            if not adapter_files:raise ValueError('empty adapter directory')
            paths.extend(adapter_files)
    return {str(p.resolve()):sha(p) for p in paths}


def run(stage,argv=None):
    if stage not in STAGES:raise ValueError('unknown correction stage')
    argv=list(sys.argv[1:] if argv is None else argv)
    lock_parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    lock_parser.add_argument('--execution-lock',type=Path,required=True)
    lock_parser.add_argument('--execution-lock-sha256',required=True)
    lock_args,forward=lock_parser.parse_known_args(argv)
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=6.)
    p.add_argument('--public-root',type=Path);p.add_argument('--bank',type=Path,action='append')
    p.add_argument('--public-bank',type=Path);p.add_argument('--model-config',type=Path);p.add_argument('--adapter',type=Path)
    args,_=p.parse_known_args(forward)
    output=args.output.resolve()
    if output.exists():raise FileExistsError('direct corrections are create-once; preserve existing output')
    lock=runtime.validate_execution_lock(lock_args.execution_lock,lock_args.execution_lock_sha256)
    if lock['profile']!=runtime.PROFILE or args.timeout!=lock['timeout_seconds']:raise ValueError('direct timeout/profile differs from inherited lock')
    inputs=input_inventory(stage,args,lock_args.execution_lock_sha256);sources=source_inventory(stage)
    request=dict(schema='eesd-direct-corrections-profile-v1',stage=stage,profile=runtime.PROFILE,status='requested',
        execution_lock_sha256=lock_args.execution_lock_sha256,execution_lock=str(lock_args.execution_lock.resolve()),
        lock_scope='inherits execution profile; new wrapper/original correction sources separately bound here',
        sources=sources,inputs=inputs,original_argv=forward,output=str(output))
    readiness=runtime.probe_readiness(lock_args.execution_lock,lock_args.execution_lock_sha256)
    output.parent.mkdir(parents=True,exist_ok=True)
    attempt=Path(tempfile.mkdtemp(prefix=output.name+'.direct-attempt-',dir=output.parent));target=attempt/'artifacts'
    write_once(attempt/'request.json',request)
    # Preserve every original option and replace only the output destination.
    forwarded=[];index=0
    while index<len(forward):
        item=forward[index]
        if item=='--output':forwarded+=['--output',str(target)];index+=2
        elif item.startswith('--output='):forwarded.append('--output='+str(target));index+=1
        else:forwarded.append(item);index+=1
    original,_,outputs=STAGES[stage];old_argv=sys.argv
    try:
        module=runpy.run_path(str(ROOT/'scripts'/original))
        sys.argv=[str(ROOT/'scripts'/original),*forwarded]
        direct_main(module['main'])()
        runtime.validate_execution_lock(lock_args.execution_lock,lock_args.execution_lock_sha256)
        if source_inventory(stage)!=sources:raise ValueError('correction source changed during execution')
        if input_inventory(stage,args,lock_args.execution_lock_sha256)!=inputs:raise ValueError('correction input changed during execution')
        if {p.name for p in target.iterdir()}!=set(outputs):raise ValueError('unexpected/incomplete original correction outputs')
        receipt={**request,'status':'complete','readiness':readiness,'outputs':{name:sha(target/name) for name in outputs}}
        write_once(target/'direct-profile.json',receipt)
        rename_new_directory(target,output)
    except BaseException as error:
        write_once(attempt/'failure.json',{'error_type':type(error).__name__,'error':str(error),'status':'failed-not-published'})
        raise
    finally:sys.argv=old_argv
    write_once(attempt/'published.json',{'output':str(output),'receipt_sha256':sha(output/'direct-profile.json')})
    print(json.dumps({'status':'direct-corrections-complete','stage':stage,'output':str(output),'receipt_sha256':sha(output/'direct-profile.json')}),flush=True)
    return 0
