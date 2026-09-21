"""A cache plan is public-only metadata, not execution readiness."""
import importlib.util
import json
from pathlib import Path
import pytest
from test_replay_generation_plan import setup, m as generation_planner

PATH=Path(__file__).resolve().parents[2]/'scripts/plan_eesd_replay_caches.py'
spec=importlib.util.spec_from_file_location('cache_planner',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def inputs(setup,tmp_path):
    root,bundle,loader,_=setup
    plan=generation_planner.build_plan(root,bundle,m.sha(bundle/'admission.json'),tmp_path/'runs',loader=loader)
    gen=tmp_path/'generation.json';gen.write_text(json.dumps(plan))
    sources={}
    for name in m.EXECUTION_SOURCES:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# execution source '+name);sources[name]=m.sha(p)
    p=root/'scripts/plan_eesd_replay_caches.py';p.write_text('# planner')
    binary=tmp_path/'binary';binary.write_bytes(b'not executable: no invocation allowed')
    lock={'schema':'eesd-replay-execution-lock-v1','status':'locked-not-probed','profile':m.PROFILE,'sources':sources,'max_workers':12,'probe_code_sha256':'a'*64,'runtime':{'profile':m.PROFILE,'executables':{name:{'path':str(binary),'sha256':m.sha(binary)} for name in ('builder','candidate','bubblewrap')},'system':'Linux','machine':'x86_64'}}
    lp=tmp_path/'execution.json';lp.write_text(json.dumps(lock))
    def rebuild(root,bundle,admission_sha256,output_root):
        return generation_planner.build_plan(root,bundle,admission_sha256,output_root,loader=loader)
    return dict(root=root,generation_manifest=gen,generation_manifest_sha256=m.sha(gen),execution_lock=lp,execution_lock_sha256=m.sha(lp),output_root=tmp_path/'runs',generation_plan_builder=rebuild)


def test_exact24_commands_all_required_flags_without_execution(inputs,tmp_path,monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('must not execute'))
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:pytest.fail('must not run version/probe'))
    result=m.build_cache_plan(**inputs)
    assert len(result['cells'])==24 and result['planned_test_executions']==168000
    assert result['status']=='planned-cache-execution-not-started'
    assert len({(r['domain'],r['family'],r['seed']) for r in result['cells']})==24
    required={'--bundle','--admission-sha256','--generation-manifest','--generation-manifest-sha256','--domain','--family','--seed','--development-bank','--primary-bank','--execution-lock','--execution-lock-sha256','--workers','--output'}
    for cell in result['cells']:
        assert set(cell['dependencies'])=={'development','primary'}
        assert {s:x['components'] for s,x in cell['dependencies'].items()}=={'development':200,'primary':500}
        argv=cell['argv'];assert required<={x for x in argv if x.startswith('--')}
        assert argv[argv.index('--workers')+1]=='12'
        assert '--probe-only' not in argv and '--freeze-lock' not in argv
        assert Path(cell['output'])==inputs['output_root']/'replay-mechanism-cache'/cell['domain']/cell['family']/f"seed{cell['seed']}"
    assert not inputs['output_root'].exists()

@pytest.mark.parametrize('kind',['missing_cell','population','argv'])
def test_mutated_generation_plan_rejected_even_if_rehashed(inputs,kind):
    p=inputs['generation_manifest'];d=json.loads(p.read_text())
    if kind=='missing_cell':d['cells'].pop()
    elif kind=='population':d['populations']['apps_replay']['primary']['sha256']='0'*64
    else:d['cells'][0]['splits']['primary']['argv'][-1]='other-output'
    p.write_text(json.dumps(d));inputs['generation_manifest_sha256']=m.sha(p)
    with pytest.raises(ValueError):m.build_cache_plan(**inputs)

@pytest.mark.parametrize('kind',['source','binary','workers'])
def test_execution_lock_contract_drift_rejected(inputs,kind):
    p=inputs['execution_lock'];lock=json.loads(p.read_text())
    if kind=='source':(inputs['root']/next(iter(lock['sources']))).write_text('changed')
    elif kind=='binary':Path(lock['runtime']['executables']['candidate']['path']).write_bytes(b'changed')
    else:lock['max_workers']=4;p.write_text(json.dumps(lock));inputs['execution_lock_sha256']=m.sha(p)
    with pytest.raises(ValueError):m.build_cache_plan(**inputs)


def test_blocked_probe_is_bound_reference_not_readiness(inputs,tmp_path):
    p=tmp_path/'probe.json';p.write_text(json.dumps({'status':'infrastructure_blocked','execution_lock_sha256':inputs['execution_lock_sha256']}))
    result=m.build_cache_plan(**inputs,operational_probe_report=p)
    assert result['operational_reference']['observed_status']=='infrastructure_blocked'
    assert result['runtime_readiness']=='not-established-by-planner'
    assert result['operational_reference']['sha256']==m.sha(p)
    p.write_text(json.dumps({'status':'probe_passed','execution_lock_sha256':'0'*64}))
    with pytest.raises(ValueError):m.build_cache_plan(**inputs,operational_probe_report=p)


def test_matrix_write_once(tmp_path):
    p=tmp_path/'manifest.json';m.write_once(p,{'status':'planned'})
    with pytest.raises(FileExistsError):m.write_once(p,{'status':'wrong'})
    assert json.loads(p.read_text())=={'status':'planned'}
