"""Direct-profile adapters: synthetic provenance and unchanged math only."""
import pytest
from pbpf.eesd import direct_artifacts as d


def test_isolation_changes_only_private_namespace():
    def fn():return marker
    clone=d.isolated(fn,marker=17)
    assert clone()==17 and 'marker' not in fn.__globals__


def test_contrast_wrapper_uses_direct_loader_without_global_mutation(monkeypatch):
    old=d.contrasts.reconstruct_contrasts.__globals__['load_mechanism_artifacts']
    def sentinel(**kwargs):raise RuntimeError('direct loader called')
    monkeypatch.setattr(d,'load_direct_artifacts',sentinel)
    with pytest.raises(RuntimeError,match='direct loader'):d.reconstruct_contrasts()
    assert d.contrasts.reconstruct_contrasts.__globals__['load_mechanism_artifacts'] is old

import json
from pathlib import Path
import runpy
import yaml
import numpy as np
from test_mechanism_artifacts import artifacts
from test_mechanism_contrasts import complete_artifacts
from test_mechanism_runner import ROOT


@pytest.fixture
def direct_math(complete_artifacts,monkeypatch):
    args=complete_artifacts
    payload=json.loads(args['cache'].read_text());report=json.loads((args['report_dir']/'report.json').read_text())
    cfg=yaml.safe_load(args['config'].read_text());runner=runpy.run_path(str(ROOT/'scripts/run_eesd_evidence_matrix.py'))
    binding={'direct_complete_sha256':'d'*64,'cache_sha256':d.file_sha(args['cache']),'config_sha256':d.file_sha(args['config'])}
    complete=json.loads((args['report_dir']/'complete.json').read_text())
    monkeypatch.setattr(d,'context',lambda kwargs,public_only:(payload,report,cfg,runner,binding,complete))
    return args,payload


def test_six_and_49_direct_math_identity(direct_math):
    args,_=direct_math;data,receipt=d.load_direct_artifacts(**args)
    assert len(data['probabilities'])==6 and receipt['schema']=='eesd-direct-artifacts-v1'
    out,receipt=d.reconstruct_contrasts(**args)
    assert len(out)==49 and all(r['status']=='available' for r in out.values())
    assert out['A8/n4']['alias_of']=='core/tuned' and len(out['A7/permuted']['comparators'])==3
    assert out['A10/binary']['proposed'].shape[1]==2 and receipt['execution_profile']=='direct-no-sandbox'


def test_direct_public_support_guard_and_label_mutation(direct_math):
    args,payload=direct_math
    class Guard(dict):
        def __getitem__(self,key):
            if key=='outcomes':raise AssertionError('assessment label read')
            return super().__getitem__(key)
    payload['records']=[Guard(r) if r['split']=='primary' else r for r in payload['records']]
    before,receipt=d.derive_public_support(**args)
    assert receipt['schema']=='eesd-direct-public-support-v1'
    for r in payload['records']:
        if r['split']=='primary':r['outcomes']=['COMPILE_ERROR']*10
    after,_=d.derive_public_support(**args)
    for key in before:np.testing.assert_array_equal(before[key]['public_mask'],after[key]['public_mask'])


def reporter():
    return runpy.run_path(str(ROOT/'scripts/report_eesd_direct_inference.py'))


def test_missing_24_cells_retains_all_slots_no_inference(tmp_path,monkeypatch):
    module=reporter();fn=module['run_report'];ns=fn.__globals__
    cells=[dict(dataset=dataset,domain=domain,model=model,family=model,public_root=str(tmp_path/'public'),evaluator_root=str(tmp_path/'eval')) for dataset,domain in [('runbugrun','rbr'),('codearc','codearc')] for model in module['BASE']['MODELS']]
    public=tmp_path/'public';public.mkdir();(public/'manifest.json').write_text('{}');(public/'tasks.jsonl').write_text('')
    config=tmp_path/'config';config.write_text('');manifest=tmp_path/'manifest';manifest.write_text('');lockpath=tmp_path/'lock';lockpath.write_text('{}')
    lock={'execution_lock':{'path':'execution','sha256':'e'*64},'original_statistical_lock':{'path':'old','sha256':'o'*64},'amendment':{'sha256':'a'*64}}
    monkeypatch.setitem(ns,'validate_statistical_lock',lambda *a:lock)
    monkeypatch.setattr(module['runtime'],'validate_execution_lock',lambda *a:{'manifest':{'path':str(manifest)},'config':{'path':str(config)}})
    monkeypatch.setitem(module['BASE'],'validate_inputs',lambda *a:cells)
    monkeypatch.setitem(module['BASE'],'public_inventory',lambda c:{'development':[],'primary':[]})
    monkeypatch.setattr(d,'derive_public_support',lambda **kw:pytest.fail('support must not run with missing cells'))
    out=tmp_path/'out';result=fn(statistical_lock=lockpath,statistical_lock_sha256='f'*64,results_root=tmp_path/'empty',output=out)
    assert result['status']=='incomplete' and result['scientific_inference_performed'] is False
    assert len(result['coverage'])==8 and sum(len(x) for x in result['coverage'].values())==24
    assert result['execution_profile']=='direct-no-sandbox'
    assert {k:v['family_size'] for k,v in result['families'].items()}=={'primary':8,'secondary':1560}

@pytest.fixture
def direct_cell(tmp_path,monkeypatch):
    import hashlib
    public=tmp_path/'public';evaluator=tmp_path/'evaluator';directory=tmp_path/'direct'
    public.mkdir();evaluator.mkdir();(directory/'report').mkdir(parents=True)
    cell=dict(domain='rbr',dataset='runbugrun',family='qwen25_7b',model='qwen25_7b',public_root=str(public),evaluator_root=str(evaluator))
    config=tmp_path/'config';config.write_text('schema: eesd-iclr2027-v1\n')
    manifest=tmp_path/'manifest';manifest.write_text(yaml.safe_dump({'mechanism_cells':[cell]}))
    lockpath=tmp_path/'lock';lockpath.write_text('{}');locksha=d.file_sha(lockpath)
    lock={'config':{'path':str(config),'sha256':d.file_sha(config)},'manifest':{'path':str(manifest),'sha256':d.file_sha(manifest)},'sources':{'synthetic':'a'*64},'max_workers':4,'amendment':{'path':'amendment','sha256':'b'*64}}
    monkeypatch.setattr(d.runtime,'validate_execution_lock',lambda *a:lock)
    def dump(path,value):path.write_text(json.dumps(value))
    tasks=[];records=[];bindings=[];watched={str(config):d.file_sha(config),str(manifest):d.file_sha(manifest),str(lockpath):locksha}
    for split,count in [('development',200),('primary',500)]:
        bank=tmp_path/split;bank.mkdir();ids=[];sources=[];files={}
        for i in range(count):
            tid=f'{split}-{i}';source=f's-{tid}';ids.append(tid);sources.append(source)
            tests=[{'id':str(j),'input':str(j),'output':'private'} for j in range(10)]
            tasks.append(dict(task_id=tid,source_component_id=source,split=split,tests=tests))
            candidate={'candidate_id':tid+'/0','code':'print(1)'}
            dump(bank/(tid+'.json'),dict(task_id=tid,source_component_id=source,split=split,candidates=[candidate]));files[tid+'.json']=d.file_sha(bank/(tid+'.json'))
            records.append(dict(task_id=tid+'/0',problem_id=tid,source_component_id=source,split=split,candidate_code_sha256=hashlib.sha256(b'print(1)').hexdigest(),tests=[{k:t[k] for k in ('id','input')} for t in tests],outcomes=['PASS']*10))
        run=dict(model='model',revision='r',seed=1701,candidates=1,split=split,components=count,task_ids=ids,source_component_ids=sources)
        dump(bank/'run.json',run);dump(bank/'complete.json',dict(run_sha256=d.file_sha(bank/'run.json'),files=files))
        bindings.append(dict(path=str(bank),split=split,components=count,complete_sha256=d.file_sha(bank/'complete.json')))
        for p in bank.iterdir():watched[str(p)]=d.file_sha(p)
    (evaluator/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks));dump(evaluator/'manifest.json',{'evaluator_tasks_sha256':d.file_sha(evaluator/'tasks.jsonl')})
    (public/'tasks.jsonl').write_text('public');dump(public/'manifest.json',{})
    for parent in (public,evaluator):
        for p in parent.iterdir():watched[str(p)]=d.file_sha(p)
    cache=dict(schema='eesd-public-query-mechanism-cache-v1',dataset='rbr',generator_identity=['model','r',None,1701],bank_bindings=bindings,evaluator_manifest_sha256=d.file_sha(evaluator/'manifest.json'),counts={'development':200,'primary':500},source_counts={'development':200,'primary':500},tests_per_candidate=10,records=records)
    dump(directory/'cache.json',cache)
    for name in ('report.json','complete.json','predictions.npz'):(directory/'report'/name).write_text('{}')
    ready=dict(schema='eesd-direct-execution-readiness-v1',status='ready',profile='direct-no-sandbox',execution_lock_sha256=locksha,sources=lock['sources'],checked_at='2026-09-21T00:00:00+00:00',probes={name:dict(outcome='PASS',returncode=0,timed_out=False) for name in ('stdin','call')})
    binding=dict(schema='eesd-direct-mechanism-binding-v1',execution_profile='direct-no-sandbox',domain='rbr',dataset='runbugrun',family='qwen25_7b',seed=1701,workers=4,execution_lock={'path':str(lockpath),'sha256':locksha},sources=lock['sources'],readiness=ready,inputs=watched,banks=bindings,cache_sha256=d.file_sha(directory/'cache.json'),report_complete_sha256=d.file_sha(directory/'report/complete.json'))
    dump(directory/'direct-binding.json',binding);seal_direct(directory)
    real=d.runpy.run_path
    def script(path):
        if str(path).endswith('run_eesd_matrix.py'):
            return {'verify_mechanism_bank':lambda p,**kw:(p,json.loads((p/'run.json').read_text()),d.file_sha(p/'complete.json')),'verify_mechanism_report':lambda *a,**kw:None}
        return real(path)
    monkeypatch.setattr(d.runpy,'run_path',script)
    return dict(directory=directory,cell=cell,seed=1701,config=config,execution_lock=lockpath,lock_sha=locksha)


def seal_direct(directory):
    paths=[p for p in directory.rglob('*') if p.is_file() and p!=directory/'complete.json']
    (directory/'complete.json').write_text(json.dumps({'schema':'eesd-direct-mechanism-complete-v1','execution_profile':'direct-no-sandbox','files':{str(p.relative_to(directory)):d.file_sha(p) for p in paths}}))


def test_direct_provenance_700_and_public_only_guard(direct_cell,monkeypatch):
    original=d.read
    class Guard(dict):
        def __getitem__(self,key):
            if key=='outcomes':raise AssertionError('assessment outcomes read')
            return super().__getitem__(key)
    def read(path):
        value=original(path)
        if Path(path).name=='cache.json':value['records']=[Guard(r) if r['split']=='primary' else r for r in value['records']]
        return value
    monkeypatch.setattr(d,'read',read)
    data,receipt=d.verify_direct_cell(**direct_cell,public_only=True)
    assert len(data['records'])==700 and receipt['execution_profile']=='direct-no-sandbox'
    with pytest.raises(AssertionError,match='outcomes'):d.verify_direct_cell(**direct_cell,public_only=False)


@pytest.mark.parametrize('change',['test_order','seed','watched','readiness','outcome'])
def test_direct_provenance_resealed_tamper(direct_cell,change):
    root=direct_cell['directory'];cp=root/'cache.json';bp=root/'direct-binding.json';cache=json.loads(cp.read_text());binding=json.loads(bp.read_text())
    if change=='test_order':cache['records'][0]['tests'][4]['input']='wrong'
    elif change=='seed':binding['seed']=1702
    elif change=='watched':binding['inputs'].pop(next(iter(binding['inputs'])))
    elif change=='readiness':binding['readiness']['probes']['stdin']['outcome']='WRONG_OUTPUT'
    else:cache['records'][0]['outcomes']=['UNKNOWN']*10
    cp.write_text(json.dumps(cache));binding['cache_sha256']=d.file_sha(cp);bp.write_text(json.dumps(binding));seal_direct(root)
    with pytest.raises(ValueError):d.verify_direct_cell(**direct_cell,public_only=False)


def test_all_public_support_sealed_before_first_direct_reconstruction(tmp_path,monkeypatch):
    module=reporter();fn=module['run_report'];ns=fn.__globals__;base=module['BASE']
    cells=[dict(dataset=dataset,domain=domain,model=m,family=m,public_root=str(tmp_path/'public'),evaluator_root=str(tmp_path/'eval')) for dataset,domain in [('runbugrun','rbr'),('codearc','codearc')] for m in base['MODELS']]
    public=tmp_path/'public';public.mkdir();(public/'manifest.json').write_text('{}');(public/'tasks.jsonl').write_text('')
    manifest=tmp_path/'manifest';manifest.write_text('');config=tmp_path/'config';config.write_text('');lockpath=tmp_path/'lock';lockpath.write_text('{}')
    lock={'execution_lock':{'path':'execution','sha256':'e'*64},'original_statistical_lock':{'path':'old','sha256':'o'*64},'amendment':{}}
    monkeypatch.setitem(ns,'validate_statistical_lock',lambda *a:lock)
    monkeypatch.setattr(module['runtime'],'validate_execution_lock',lambda *a:{'manifest':{'path':str(manifest)},'config':{'path':str(config)}})
    monkeypatch.setitem(base,'validate_inputs',lambda *a:cells)
    identities=[{'task_id':f't{i}','source_component_id':f's{i}'} for i in range(500)]
    monkeypatch.setitem(base,'public_inventory',lambda c:{'development':[],'primary':identities})
    monkeypatch.setattr(d,'verify_direct_cell',lambda *a,**kw:None)
    root=tmp_path/'results';out=tmp_path/'out'
    for cell in cells:
        for seed in (1701,1702,1703):
            p=root/'mechanism'/cell['dataset']/cell['model']/f'seed{seed}';p.mkdir(parents=True);(p/'complete.json').write_text('{}')
    events=[]
    def support(**kwargs):
        events.append('public')
        return {k:dict(sources=[f's{i}' for i in range(500)],seeds=[kwargs['seed']]*500,query_ids=[f'q{i}' for i in range(500)],public_mask=np.ones(500,dtype=bool)) for k in d.contrasts.BIN_IDS},{}
    def reconstruct(**kwargs):
        assert events.count('public')==24 and (out/'a9-public-support.complete.json').is_file()
        assert len(json.loads((out/'a9-public-support.json').read_text()))==8
        events.append('full')
        return {k:{'status':'unsupported','reason':'synthetic unavailable slot'} for k in d.contrasts.contrast_ids()},{}
    monkeypatch.setattr(d,'derive_public_support',support);monkeypatch.setattr(d,'reconstruct_contrasts',reconstruct)
    result=fn(statistical_lock=lockpath,statistical_lock_sha256='f'*64,results_root=root,output=out)
    assert events==['public']*24+['full']*24
    assert all(not family['complete'] for family in result['families'].values())
