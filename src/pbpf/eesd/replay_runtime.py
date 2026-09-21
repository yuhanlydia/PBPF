"""Locked runtime identity and a trusted namespace probe for Replay execution.

Checking a lock does not execute dataset programs. A readiness receipt is only
returned after the immutable stdin executor actually runs the fixed probe.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pbpf.apbpf import rbr_execution, codearc_execution
from pbpf.apbpf.rbr_execution import execute_stdin

ROOT=Path(__file__).resolve().parents[3]
PROFILE={'builder_python':'3.11.16','candidate_python':'3.10.12','bubblewrap':'0.6.1','timeout_seconds':6.0}
SENTINEL='eesd-replay-sandbox-ready'
PROBE_CODE=f'print({SENTINEL!r})\n'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def source_inventory(extra_source_paths=()):
    paths=[Path(__file__),Path(rbr_execution.__file__),Path(codearc_execution.__file__),*map(Path,extra_source_paths)]
    result={}
    for path in paths:
        path=(path if path.is_absolute() else ROOT/path).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('execution source outside repository')
        result[path.relative_to(ROOT).as_posix()]=sha(path)
    return dict(sorted(result.items()))


def runtime_identity():
    bwrap=shutil.which('bwrap',path='/usr/bin:/bin')
    if bwrap is None:raise RuntimeError('bubblewrap executable unavailable')
    candidate=Path('/usr/bin/python3')
    version=subprocess.check_output([str(candidate),'-I','-c','import platform; print(platform.python_version())'],text=True,timeout=10).strip()
    bwrap_version=subprocess.check_output([bwrap,'--version'],text=True,timeout=10).strip()
    if not bwrap_version.startswith('bubblewrap '):raise RuntimeError('unrecognized bubblewrap version')
    executables={}
    for key,path in [('builder',Path(sys.executable)),('candidate',candidate),('bubblewrap',Path(bwrap))]:
        resolved=path.resolve()
        executables[key]={'path':str(resolved),'sha256':sha(resolved)}
    return {'profile':{'builder_python':platform.python_version(),'candidate_python':version,
                       'bubblewrap':bwrap_version.split(' ',1)[1],'timeout_seconds':6.0},
            'executables':executables,'system':platform.system(),'machine':platform.machine()}


def build_execution_lock(*,extra_source_paths=(),max_workers=12):
    if type(max_workers) is not int or not 1<=max_workers<=12:raise ValueError('max_workers must be 1..12')
    runtime=runtime_identity()
    if runtime['profile']!=PROFILE:raise ValueError('runtime profile differs from declared versions')
    return {'schema':'eesd-replay-execution-lock-v1','status':'locked-not-probed',
            'profile':dict(PROFILE),'runtime':runtime,'sources':source_inventory(extra_source_paths),
            'max_workers':max_workers,'probe_code_sha256':hashlib.sha256(PROBE_CODE.encode()).hexdigest()}


def validate_execution_lock(path,expected_sha256):
    path=Path(path)
    if sha(path)!=expected_sha256:raise ValueError('execution lock checksum mismatch')
    lock=json.loads(path.read_text())
    if lock.get('schema')!='eesd-replay-execution-lock-v1' or lock.get('status')!='locked-not-probed':
        raise ValueError('execution lock schema/status mismatch')
    if lock.get('profile')!=PROFILE:raise ValueError('execution lock profile mismatch')
    if type(lock.get('max_workers')) is not int or not 1<=lock['max_workers']<=12:
        raise ValueError('invalid execution worker limit')
    sources=lock.get('sources')
    if not isinstance(sources,dict) or not set(source_inventory())<=set(sources):
        raise ValueError('execution lock missing required sources')
    if source_inventory(sources)!=sources:raise ValueError('execution source changed')
    if lock.get('probe_code_sha256')!=hashlib.sha256(PROBE_CODE.encode()).hexdigest():
        raise ValueError('trusted probe code changed')
    current=runtime_identity()
    if current['profile']!=PROFILE:raise ValueError('runtime profile changed')
    if current!=lock.get('runtime'):raise ValueError('runtime identity changed')
    return lock


def probe_readiness(path,expected_sha256):
    lock=validate_execution_lock(path,expected_sha256)
    result=execute_stdin(PROBE_CODE,{'input':'','expected':SENTINEL+'\n'},timeout=6.0)
    if (not isinstance(result,dict) or result.get('outcome')!='PASS' or result.get('returncode')!=0
        or result.get('timed_out') is not False or result.get('stdout')!=SENTINEL+'\n'
        or result.get('stderr')!='' or result.get('stdout_truncated') is not False):
        raise RuntimeError('trusted sandbox readiness probe failed')
    if validate_execution_lock(path,expected_sha256)!=lock:
        raise ValueError('execution lock changed during probe')
    return {'schema':'eesd-replay-execution-readiness-v1','status':'ready',
            'execution_lock_sha256':expected_sha256,'profile':lock['profile'],'sources':lock['sources'],
            'checked_at':datetime.now(timezone.utc).isoformat(),
            'probe':{'code_sha256':lock['probe_code_sha256'],
                     'stdout_sha256':hashlib.sha256(result['stdout'].encode()).hexdigest()}}
