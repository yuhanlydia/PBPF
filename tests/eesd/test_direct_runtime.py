import json
from pathlib import Path
import pytest


def test_direct_runtime_module_exists():
    from pbpf.eesd import direct_runtime
    assert direct_runtime.PROFILE == 'direct-no-sandbox'


@pytest.fixture
def setup_lock(tmp_path,monkeypatch):
    from pbpf.eesd import direct_runtime as r
    monkeypatch.setattr(r,'ROOT',tmp_path)
    monkeypatch.setattr(r,'SOURCES',('executor.py',))
    monkeypatch.setattr(r,'runtime_identity',lambda:{'verified_runtime':'test-double'})
    (tmp_path/'executor.py').write_text('source')
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/EESD_EXECUTION_REVIEW_20260921.md').write_text('amendment')
    for name in ('manifest','config'):(tmp_path/name).write_text(name)
    lock=r.build_execution_lock(tmp_path/'manifest',tmp_path/'config')
    p=tmp_path/'lock.json';p.write_text(json.dumps(lock))
    return r,p,r.sha(p)


def test_lock_roundtrip(setup_lock):
    r,p,d=setup_lock
    assert r.validate_execution_lock(p,d)['max_workers']==4


@pytest.mark.parametrize('name',['executor.py','manifest','config'])
def test_changed_execution_inputs_rejected(setup_lock,name):
    r,p,d=setup_lock;(p.parent/name).write_text('changed')
    with pytest.raises(ValueError):r.validate_execution_lock(p,d)


def test_probe_failure_never_ready(setup_lock,monkeypatch):
    r,p,d=setup_lock
    monkeypatch.setattr(r,'trusted_probes',lambda:{'stdin':{'outcome':'WRONG_OUTPUT'},'call':{'outcome':'PASS'}})
    with pytest.raises(RuntimeError):r.probe_readiness(p,d)
