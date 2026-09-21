import hashlib
import json
import pytest
from pbpf.eesd import replay_runtime as runtime


@pytest.fixture
def locked(tmp_path,monkeypatch):
    identity={'profile':dict(runtime.PROFILE),'executables':{'fixture':{'path':'fixture','sha256':'a'*64}}}
    monkeypatch.setattr(runtime,'runtime_identity',lambda:identity)
    lock=runtime.build_execution_lock()
    path=tmp_path/'lock.json';path.write_text(json.dumps(lock))
    return path,hashlib.sha256(path.read_bytes()).hexdigest(),identity


def test_validation_never_calls_executor(locked,monkeypatch):
    path,digest,_=locked
    monkeypatch.setattr(runtime,'execute_stdin',lambda *a,**k:pytest.fail('must not execute during file validation'))
    assert runtime.validate_execution_lock(path,digest)['profile']==runtime.PROFILE


def test_wrong_lock_checksum_rejected_before_probe(locked,monkeypatch):
    monkeypatch.setattr(runtime,'execute_stdin',lambda *a,**k:pytest.fail('must not execute'))
    with pytest.raises(ValueError,match='checksum'):runtime.probe_readiness(locked[0],'b'*64)


def test_namespace_failure_cannot_produce_ready_receipt(locked,monkeypatch):
    def failed(*a,**k):raise RuntimeError('bwrap: namespace unavailable')
    monkeypatch.setattr(runtime,'execute_stdin',failed)
    with pytest.raises(RuntimeError,match='namespace'):runtime.probe_readiness(*locked[:2])


def test_probe_is_fixed_trusted_program_and_six_second_deadline(locked,monkeypatch):
    calls=[]
    def execute(code,test,**kwargs):
        calls.append((code,test,kwargs))
        return dict(outcome='PASS',stdout=runtime.SENTINEL+'\n',stderr='',returncode=0,timed_out=False,stdout_truncated=False)
    monkeypatch.setattr(runtime,'execute_stdin',execute)
    ready=runtime.probe_readiness(*locked[:2])
    assert len(calls)==1 and calls[0][0]==runtime.PROBE_CODE
    assert calls[0][1]=={'input':'','expected':runtime.SENTINEL+'\n'}
    assert calls[0][2]=={'timeout':6.0}
    assert ready['status']=='ready' and ready['execution_lock_sha256']==locked[1]


def test_runtime_drift_after_probe_invalidates_readiness(locked,monkeypatch):
    def execute(*a,**k):
        locked[2]['executables']['fixture']['sha256']='b'*64
        return dict(outcome='PASS',stdout=runtime.SENTINEL+'\n',stderr='',returncode=0,timed_out=False,stdout_truncated=False)
    monkeypatch.setattr(runtime,'execute_stdin',execute)
    with pytest.raises(ValueError,match='runtime'):runtime.probe_readiness(*locked[:2])


def test_mismatched_profile_rejected_without_execution(locked,monkeypatch):
    locked[2]['profile']['candidate_python']='3.12.0'
    monkeypatch.setattr(runtime,'execute_stdin',lambda *a,**k:pytest.fail('must not execute'))
    with pytest.raises(ValueError,match='profile'):runtime.probe_readiness(*locked[:2])
