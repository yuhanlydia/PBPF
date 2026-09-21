import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


def module():
    path=Path(__file__).resolve().parents[2]/'scripts/build_eesd_replay_mechanism_cache.py'
    spec=importlib.util.spec_from_file_location('replay_cache_cli',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


@pytest.fixture
def setup(tmp_path,monkeypatch):
    m=module();args=SimpleNamespace(freeze_lock=False,probe_only=False,preflight_only=True,
      execution_lock=tmp_path/'lock.json',execution_lock_sha256='a'*64,workers=2,
      bundle=tmp_path/'bundle',admission_sha256='b'*64,domain='apps_replay',family='qwen25_7b',seed=1701,
      development_bank=tmp_path/'dev',primary_bank=tmp_path/'primary',generation_manifest=tmp_path/'manifest',
      generation_manifest_sha256='c'*64,output=tmp_path/'cache',probe_report=None)
    inputs={'jobs':[{}]*700,'bindings':{'execution_sources':{'file':'d'*64}}}
    monkeypatch.setattr(m.runtime,'validate_execution_lock',lambda *a:dict(max_workers=12,sources={'file':'d'*64}))
    monkeypatch.setattr(m.cache,'load_replay_execution_inputs',lambda *a,**k:inputs)
    return m,args,inputs


def test_file_preflight_does_not_probe_or_execute(setup,monkeypatch):
    m,args,_=setup
    monkeypatch.setattr(m.runtime,'probe_readiness',lambda *a:pytest.fail('file-only preflight must not probe'))
    monkeypatch.setattr(m.cache,'measure_jobs',lambda *a,**k:pytest.fail('must not measure'))
    assert m.run(args)==0
    assert not args.output.exists()


def test_missing_banks_are_pending_not_model_failures(setup,monkeypatch):
    m,args,_=setup
    def absent(*a,**k):raise FileNotFoundError(2,'bank run missing',str(args.development_bank/'run.json'))
    monkeypatch.setattr(m.cache,'load_replay_execution_inputs',absent)
    assert m.run(args)==2
    assert not args.output.exists()


def test_worker_count_cannot_exceed_lock(setup):
    m,args,_=setup;args.workers=13
    with pytest.raises(ValueError,match='worker'):m.run(args)


def test_namespace_failure_before_any_measurement_returns_infrastructure_status(setup,monkeypatch):
    m,args,_=setup;args.preflight_only=False
    def fail(*a):raise RuntimeError('namespace blocked')
    monkeypatch.setattr(m.runtime,'probe_readiness',fail)
    def measurement(inputs,*,readiness_check,**kwargs):
        readiness_check()
        pytest.fail('must stop before mapper')
    monkeypatch.setattr(m.cache,'measure_jobs',measurement)
    assert m.run(args)==3
    assert not args.output.exists()


def test_runtime_source_map_must_equal_measured_input_sources(setup,monkeypatch):
    m,args,_=setup
    monkeypatch.setattr(m.runtime,'validate_execution_lock',lambda *a:dict(max_workers=12,sources={'different':'f'*64}))
    with pytest.raises(ValueError,match='source'):m.run(args)


def test_pool_preserves_order_and_cancels_remaining_on_failure(monkeypatch):
    from concurrent.futures import Future
    m=module();instances=[]
    class Pool:
        def __init__(self,**kwargs):self.kwargs=kwargs;self.calls=[];instances.append(self)
        def submit(self,fn,job):
            self.calls.append(job);f=Future()
            if job=='fail':f.set_exception(RuntimeError('worker died'))
            else:f.set_result({'id':job})
            return f
        def shutdown(self,**kwargs):self.closed=kwargs
    monkeypatch.setattr(m,'ProcessPoolExecutor',Pool)
    assert m.bounded_process_map(list(range(7)),2)==[{'id':i} for i in range(7)]
    assert instances[-1].kwargs['mp_context'].get_start_method()=='spawn'
    assert instances[-1].closed=={'wait':True,'cancel_futures':True}
    with pytest.raises(RuntimeError,match='worker died'):m.bounded_process_map(['fail',1,2,3,4],2)
    assert instances[-1].calls==['fail',1]
    assert instances[-1].closed=={'wait':True,'cancel_futures':True}


def test_common_source_set_covers_runtime_and_cache_dependencies():
    m=module();sources=m.common_sources()
    assert set(m.cache.EXECUTION_SOURCES)<=set(sources)
    assert 'src/pbpf/eesd/replay_runtime.py' in sources
    assert 'scripts/build_eesd_replay_mechanism_cache.py' in sources
    assert m.runtime.source_inventory(sources).keys()==dict.fromkeys(sources).keys()


def test_missing_sealed_candidate_cannot_hide_behind_other_pending_bank(setup,monkeypatch):
    m,args,_=setup
    args.development_bank.mkdir()
    for name in ('run.json','complete.json'):(args.development_bank/name).write_text('{}')
    def corrupt(*a,**k):raise FileNotFoundError(2,'sealed record missing',str(args.development_bank/'candidate.json'))
    monkeypatch.setattr(m.cache,'load_replay_execution_inputs',corrupt)
    with pytest.raises(FileNotFoundError,match='candidate.json'):m.run(args)


def test_historical_readiness_rejects_missing_probe_and_excess_workers():
    m=module()
    lock=dict(max_workers=12,profile=m.runtime.PROFILE,sources={'source':'a'*64},probe_code_sha256='b'*64)
    receipt=dict(execution_lock_sha256='c'*64,profile=lock['profile'],sources=lock['sources'],worker_count=999)
    with pytest.raises(ValueError):m.validate_historical_readiness(receipt,lock,'c'*64)


@pytest.mark.parametrize('bad_field',[None,'worker_count','probe','checked_at'])
def test_historical_receipt_full_contract(bad_field):
    import hashlib
    m=module()
    lock=dict(max_workers=12,profile=m.runtime.PROFILE,sources={'source':'a'*64},probe_code_sha256='b'*64)
    receipt=dict(execution_lock_sha256='c'*64,profile=lock['profile'],sources=lock['sources'],worker_count=2,
        probe={'code_sha256':'b'*64,'stdout_sha256':hashlib.sha256((m.runtime.SENTINEL+'\n').encode()).hexdigest()},
        checked_at='2026-09-20T12:00:00+00:00')
    if bad_field is None:m.validate_historical_readiness(receipt,lock,'c'*64)
    else:
        if bad_field=='worker_count':receipt[bad_field]=999
        else:receipt.pop(bad_field)
        with pytest.raises(ValueError):m.validate_historical_readiness(receipt,lock,'c'*64)
