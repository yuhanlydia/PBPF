"""Explicit user-authorized direct execution profile; not a sandbox."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
PROFILE='direct-no-sandbox'
SOURCES=(
    'src/pbpf/eesd/direct_runtime.py','src/pbpf/apbpf/direct_execution.py',
    'scripts/run_eesd_direct_mechanism_cell.py','scripts/build_eesd_mechanism_cache.py',
    'scripts/run_eesd_matrix.py','scripts/run_eesd_evidence_matrix.py',
    'src/pbpf/eesd/evidence.py','src/pbpf/apbpf/codearc_execution.py',
    'src/pbpf/apbpf/rbr_execution.py','src/pbpf/apbpf/codearc_bank.py',
    'src/pbpf/eesd/replay_admission.py',
)
PROBES={
    'stdin': {'code': 'import os\nprint(os.getuid())\n', 'test': {'input':'','expected':'65534\n'}},
    'call': {'code':'def add(a,b):\n    return a+b\n',
             'test':{'input':'print(add(2,3))','expected':'5','expected_error':False}},
}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def pointer(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':sha(path)}


def runtime_identity():
    if os.geteuid()!=0:raise RuntimeError('direct runner must be able to drop candidate privileges')
    if Path('/root').stat().st_mode & 0o077:raise RuntimeError('/root must remain inaccessible to candidate UID')
    return {'builder':{'path':sys.executable,'sha256':sha(sys.executable),'version':platform.python_version()},
            'candidate':{'path':'/usr/bin/python3','sha256':sha('/usr/bin/python3'),
                'version':subprocess.check_output(['/usr/bin/python3','--version'],text=True).strip()},
            'platform':platform.platform(),'candidate_uid':65534,'candidate_gid':65534,
            'supplementary_groups':[], 'isolation':'none',
            'limits':{'address_space_bytes':1024**3,'output_file_bytes':8*1024**2,
                      'open_files':64,'processes_per_uid':64,'core_bytes':0}}


def build_execution_lock(manifest,config,max_workers=4):
    if type(max_workers) is not int or not 1<=max_workers<=4:raise ValueError('direct profile allows 1–4 CPU workers')
    return {'schema':'eesd-direct-execution-lock-v1','profile':PROFILE,
            'manifest':pointer(manifest),'config':pointer(config),'runtime':runtime_identity(),
            'timeout_seconds':6.0,'max_workers':max_workers,'probe':PROBES,
            'sources':{name:sha(ROOT/name) for name in SOURCES},
            'amendment':pointer(ROOT/'docs/EESD_EXECUTION_REVIEW_20260921.md')}


def validate_execution_lock(path,expected_sha256):
    if sha(path)!=expected_sha256:raise ValueError('direct execution lock checksum mismatch')
    lock=json.loads(Path(path).read_text())
    expected=build_execution_lock(lock['manifest']['path'],lock['config']['path'],lock['max_workers'])
    if lock!=expected:raise ValueError('direct execution profile, inputs or sources changed')
    return lock


def trusted_probes():
    from pbpf.apbpf.direct_execution import execute_stdin,execute_call
    return {name:fn(PROBES[name]['code'],PROBES[name]['test'],timeout=6.)
            for name,fn in [('stdin',execute_stdin),('call',execute_call)]}


def probe_readiness(path,expected_sha256):
    lock=validate_execution_lock(path,expected_sha256)
    results=trusted_probes()
    if set(results)!={'stdin','call'} or any(
        r.get('outcome')!='PASS' or r.get('returncode')!=0 or r.get('timed_out') is not False
        for r in results.values()):
        raise RuntimeError('direct execution trusted probes failed: '+json.dumps(results))
    validate_execution_lock(path,expected_sha256)
    return {'schema':'eesd-direct-execution-readiness-v1','profile':PROFILE,'status':'ready',
            'execution_lock_sha256':expected_sha256,'sources':lock['sources'],
            'checked_at':datetime.now(timezone.utc).isoformat(),'probes':results}
