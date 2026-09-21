import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def load():
    path=Path(__file__).resolve().parents[2]/'scripts/run_eesd_replay_evidence.py'
    spec=importlib.util.spec_from_file_location('replay_evidence_cli',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.fixture
def pipeline(tmp_path,monkeypatch):
    m=load();calls=[];cache_dir=tmp_path/'cache';cache_dir.mkdir();(cache_dir/'binding.json').write_text(json.dumps({'readiness':{'execution_lock_sha256':'e'*64}}))
    cell={'domain':'apps_replay','family':'qwen25_7b','seed':1701,'output':str(cache_dir),'dependencies':{s:{'bank':str(tmp_path/s)} for s in ('development','primary')}}
    matrix={'schema':'eesd-replay-cache-matrix-v1','status':'planned-cache-execution-not-started','output_root':str(tmp_path/'outputs'),'bundle':str(tmp_path/'bundle'),'admission_sha256':'a'*64,'generation_manifest':{'path':'generation','sha256':'g'*64},'execution_lock':{'path':'execution','sha256':'e'*64},'cells':[cell]}
    matrix_path=tmp_path/'matrix.json';matrix_path.write_text(json.dumps(matrix));config=tmp_path/'config.yaml';config.write_text('test')
    lock={'config':{'path':str(config)},'cache_manifest':{'path':str(matrix_path)}}
    monkeypatch.setattr(m,'validate_statistical_lock',lambda *a,**kw:lock)
    monkeypatch.setattr(m.runtime,'validate_execution_lock',lambda *a:{'sources':{'source':'hash'}})
    monkeypatch.setattr(m,'execution_helpers',lambda:{'common_sources':lambda:['source'],'validate_historical_readiness':lambda *a:calls.append('historical')})
    monkeypatch.setattr(m.execution,'load_replay_execution_inputs',lambda *a,**kw:{'bindings':{'execution_sources':{'source':'hash'}}})
    monkeypatch.setattr(m.artifacts,'verify_replay_evidence_inputs',lambda **kw:calls.append('full-cache'))
    monkeypatch.setattr(m.artifacts,'build_report_binding',lambda **kw:{'schema':'binding'})
    monkeypatch.setattr(m.artifacts,'load_replay_artifacts',lambda **kw:(calls.append('verify-report'),{}))
    def runner(argv,**kwargs):
        calls.append('runner');out=Path(argv[argv.index('--output')+1]);out.mkdir()
        for f in ('report.json','complete.json','predictions.npz'):(out/f).write_text('{}')
    monkeypatch.setattr(m.subprocess,'run',runner)
    args=SimpleNamespace(statistical_lock=tmp_path/'statlock',statistical_lock_sha256='f'*64,domain='apps_replay',family='qwen25_7b',seed=1701,preflight_only=False)
    return m,args,calls,matrix


def test_preflight_does_not_infer_or_publish(pipeline):
    m,args,calls,matrix=pipeline;args.preflight_only=True
    assert m.run(args)==0
    assert calls==['historical','full-cache']
    assert not Path(matrix['output_root']).exists()


def test_atomic_report_with_binding_and_complete_resume(pipeline):
    m,args,calls,matrix=pipeline;assert m.run(args)==0
    out=Path(matrix['output_root'])/'replay-mechanism/apps_replay/qwen25_7b/seed1701'
    assert json.loads((out/'replay-binding.json').read_text())=={'schema':'binding'}
    assert calls==['historical','full-cache','runner','verify-report']
    snapshot={p.name:p.read_bytes() for p in out.iterdir()}
    assert m.run(args)==0
    assert calls.count('runner')==1 and calls.count('verify-report')==2
    assert snapshot=={p.name:p.read_bytes() for p in out.iterdir()}


def test_existing_incomplete_report_never_overwritten(pipeline):
    m,args,calls,matrix=pipeline;out=Path(matrix['output_root'])/'replay-mechanism/apps_replay/qwen25_7b/seed1701';out.mkdir(parents=True);(out/'report.json').write_text('keep')
    with pytest.raises(ValueError,match='incomplete'):m.run(args)
    assert (out/'report.json').read_text()=='keep' and 'runner' not in calls


def test_runner_failure_no_canonical_publication(pipeline,monkeypatch):
    m,args,calls,matrix=pipeline
    def fail(*a,**kw):raise RuntimeError('synthetic runner failure')
    monkeypatch.setattr(m.subprocess,'run',fail)
    with pytest.raises(RuntimeError):m.run(args)
    out=Path(matrix['output_root'])/'replay-mechanism/apps_replay/qwen25_7b/seed1701'
    assert not out.exists() and list(out.parent.glob('seed1701.pending-*'))


def test_bad_cache_prevents_runner(pipeline,monkeypatch):
    m,args,calls,matrix=pipeline
    def fail(**kw):raise ValueError('bad cache')
    monkeypatch.setattr(m.artifacts,'verify_replay_evidence_inputs',fail)
    with pytest.raises(ValueError,match='bad cache'):m.run(args)
    assert 'runner' not in calls


def test_bad_statistical_lock_prevents_cache_access(pipeline,monkeypatch):
    m,args,calls,matrix=pipeline
    def fail(*a,**kw):raise ValueError('statistical lock mismatch')
    monkeypatch.setattr(m,'validate_statistical_lock',fail)
    with pytest.raises(ValueError,match='statistical lock'):m.run(args)
    assert calls==[]


def test_report_validation_failure_retains_diagnostics_without_publish(pipeline,monkeypatch):
    m,args,calls,matrix=pipeline
    def fail(**kw):raise ValueError('NPZ mismatch')
    monkeypatch.setattr(m.artifacts,'load_replay_artifacts',fail)
    with pytest.raises(ValueError,match='NPZ'):m.run(args)
    out=Path(matrix['output_root'])/'replay-mechanism/apps_replay/qwen25_7b/seed1701'
    assert not out.exists()
    staging=next(out.parent.glob('seed1701.pending-*'))
    assert (staging/'runner.log').exists() and (staging/'failure.json').exists()
    assert (staging/'report/replay-binding.json').exists()


def test_historical_readiness_failure_prevents_inference(pipeline,monkeypatch):
    m,args,calls,matrix=pipeline
    def fail(*a):raise ValueError('historical readiness mismatch')
    monkeypatch.setattr(m,'execution_helpers',lambda:{'common_sources':lambda:['source'],'validate_historical_readiness':fail})
    with pytest.raises(ValueError,match='historical readiness'):m.run(args)
    assert calls==[]
