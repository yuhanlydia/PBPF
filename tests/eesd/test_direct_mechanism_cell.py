"""Direct profile orchestration with fakes; never execute dataset candidates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('direct_cell',ROOT/'scripts/run_eesd_direct_mechanism_cell.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def setup(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.yaml';config=tmp_path/'config.yaml';lockpath=tmp_path/'lock.json'
    cell=dict(dataset='codearc',domain='codearc',model='qwen25_7b',family='qwen25_7b',public_root=str(tmp_path/'public'),evaluator_root=str(tmp_path/'evaluator'),development_components=200,visible=4)
    for view in ('public','evaluator'):
        p=tmp_path/view;p.mkdir();(p/'manifest.json').write_text('{}');(p/'tasks.jsonl').write_text('{}\n')
    manifest.write_text(yaml.safe_dump({'schema':'eesd-cache-manifest-v1','mechanism_cells':[cell]}));config.write_text(yaml.safe_dump({'schema':'eesd-iclr2027-v1','seeds':[1701,1702,1703]}));lockpath.write_text('{}')
    args=SimpleNamespace(manifest=manifest,manifest_sha256=m.sha(manifest),config=config,config_sha256=m.sha(config),domain='codearc',family='qwen25_7b',seed=1701,results_root=tmp_path/'old',execution_lock=lockpath,execution_lock_sha256=m.sha(lockpath),output=tmp_path/'direct',workers=4)
    lock=dict(profile='direct-no-sandbox',max_workers=4,timeout_seconds=6.,sources={},manifest={'path':str(manifest),'sha256':m.sha(manifest)},config={'path':str(config),'sha256':m.sha(config)})
    events=[]
    def probe(*a):events.append('probe');return dict(status='ready',profile='direct-no-sandbox',execution_lock_sha256=args.execution_lock_sha256,sources={})
    runtime=SimpleNamespace(validate_execution_lock=lambda *a:lock,probe_readiness=probe)
    monkeypatch.setattr(m,'runtime_api',lambda:runtime)
    def verify(bank,*,cell,seed,split,root):
        events.append('bank:'+split);count=200 if split=='development' else 500
        bank.mkdir(parents=True,exist_ok=True);(bank/'run.json').write_text('{}');(bank/'complete.json').write_text('{}')
        return bank,dict(split=split,components=count,source_component_ids=[f'{split}-{i}' for i in range(count)],task_ids=[f'{split}-{i}' for i in range(count)]),'0'*64
    def report_verify(*a,**kw):events.append('report_verify')
    monkeypatch.setattr(m,'matrix_api',lambda:dict(verify_mechanism_bank=verify,verify_mechanism_report=report_verify))
    def build(args,cell,banks,cache):
        events.append('cache');rows=[dict(split=split,source_component_id=f'{split}-{i}',task_id=f'{split}-{i}/candidate',problem_id=f'{split}-{i}',tests=[{'id':str(j),'input':str(j)} for j in range(10)],outcomes=['PASS']*10,candidate_code_sha256='0'*64) for split,n in [('development',200),('primary',500)] for i in range(n)]
        cache.write_text(json.dumps({'schema':'eesd-public-query-mechanism-cache-v1','dataset':'codearc','records':rows,'counts':{'development':200,'primary':500},'source_counts':{'development':200,'primary':500}}))
    monkeypatch.setattr(m,'build_direct_cache',build)
    def evidence(args,cell,cache,directory):
        events.append('evidence');directory.mkdir();(directory/'report.json').write_text('{}');(directory/'predictions.npz').write_bytes(b'npz');(directory/'complete.json').write_text('{}')
    monkeypatch.setattr(m,'run_evidence',evidence)
    return args,events,runtime


def test_atomic_direct_profile_completion_and_no_old_binding(setup):
    args,events,_=setup;m.run(args)
    assert events==['bank:development','bank:primary','probe','cache','evidence','report_verify']
    binding=json.loads((args.output/'direct-binding.json').read_text())
    assert binding['execution_profile']=='direct-no-sandbox'
    assert not (args.output/'cache.binding.json').exists()
    assert (args.output/'complete.json').exists()
    with pytest.raises(FileExistsError):m.run(args)


def test_probe_failure_no_cache_or_final_directory(setup):
    args,events,runtime=setup
    def fail(*a):raise RuntimeError('direct probe failed')
    runtime.probe_readiness=fail
    with pytest.raises(RuntimeError):m.run(args)
    assert events==['bank:development','bank:primary'] and not args.output.exists()


def test_bank_failure_precedes_probe(setup,monkeypatch):
    args,events,_=setup
    def fail(*a,**kw):raise ValueError('bad seal')
    monkeypatch.setattr(m,'matrix_api',lambda:dict(verify_mechanism_bank=fail))
    with pytest.raises(ValueError):m.run(args)
    assert not events and not args.output.exists()


def test_failed_evidence_never_publishes_cell(setup,monkeypatch):
    args,events,_=setup
    def fail(*a):raise RuntimeError('evidence failed')
    monkeypatch.setattr(m,'run_evidence',fail)
    with pytest.raises(RuntimeError):m.run(args)
    assert not args.output.exists()


def test_direct_work_reuses_original_projection_with_narrow_execution(monkeypatch):
    from pbpf.apbpf import direct_execution
    calls=[]
    def execute(code,test,*,timeout):calls.append((code,test,timeout));return {'outcome':'PASS'}
    monkeypatch.setattr(direct_execution,'execute_call',execute)
    job=('codearc',dict(task_id='task',source_component_id='source',split='primary',candidates=[{'candidate_id':'candidate','code':'invalid !'}]),dict(tests=[{'id':str(i),'input':f'call{i}','expected':str(i),'expected_error':False} for i in range(10)]),6.)
    row=m.work(job)
    assert len(calls)==10 and row['outcomes']==['PASS']*10
    assert set(row['tests'][0])=={'id','input'} and 'code' not in row


def test_pool_uses_spawn(monkeypatch):
    called={}
    def pool(**kwargs):called.update(kwargs);return 'pool'
    monkeypatch.setattr(m,'ProcessPoolExecutor',pool)
    assert m.spawn_pool(max_workers=4)=='pool'
    assert called['mp_context'].get_start_method()=='spawn'
