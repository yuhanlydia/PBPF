"""Trusted small correction fixtures; no model/GPU or dataset candidates."""
import json
import sys
from pathlib import Path
import pytest
from pbpf.eesd import direct_corrections as dc


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return dc.sha(path)

@pytest.fixture(params=["rbr","codearc"])
def prepared_inputs(tmp_path,monkeypatch,request):
    domain=request.param
    public=tmp_path/'public';public.mkdir();rows=[];banks=[]
    for split in ('train','development'):
        rows.append(dict(task_id=split,source_component_id='s-'+split,split=split,task_text='print 17',visible_tests=[dict(id=str(i),input='',expected='17',**({'expected_error':False} if domain=='codearc' else {})) for i in range(4)]))
    tasks=public/'tasks.jsonl';tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    dump(public/'manifest.json',{'schema':'apbpf-rbr-generated-materialization-v1' if domain=='rbr' else 'apbpf-codearc-replay-materialization-v1','public_tasks_sha256':dc.sha(tasks)})
    for split in ('train','development'):
        bank=tmp_path/split;bank.mkdir();row=dict(task_id=split,source_component_id='s-'+split,split=split,candidates=[{'candidate_id':split+'/0','code':'print(3)'}]);dump(bank/(split+'.json'),row)
        run=dict(schema=f'apbpf-{domain}-generation-v1',model='test',revision='a'*40,seed=1701,split=split,candidates=1,task_ids=[split],source_component_ids=['s-'+split],public_tasks_sha256=dc.sha(tasks))
        dump(bank/'run.json',run);dump(bank/'complete.json',dict(run_sha256=dc.sha(bank/'run.json'),files={split+'.json':dc.sha(bank/(split+'.json'))}));banks.append(bank)
    lock=tmp_path/'lock.json';dump(lock,{'profile':'direct-no-sandbox'})
    monkeypatch.setattr(dc.runtime,'validate_execution_lock',lambda *a:{'profile':'direct-no-sandbox','timeout_seconds':6.})
    monkeypatch.setattr(dc.runtime,'probe_readiness',lambda *a:{'profile':'direct-no-sandbox','status':'ready','execution_lock_sha256':dc.sha(lock)})
    args=['--execution-lock',str(lock),'--execution-lock-sha256',dc.sha(lock),'--domain',domain,'--public-root',str(public),'--bank',str(banks[0]),'--bank',str(banks[1]),'--output',str(tmp_path/'prepared')]
    return args,tmp_path/'prepared'


def test_actual_prepare_with_trusted_fixture_and_receipt(prepared_inputs):
    args,out=prepared_inputs;dc.run('prepare',args)
    rows=[json.loads(l) for l in (out/'public-corrections.jsonl').read_text().splitlines()]
    assert len(rows)==2 and all(r['outcomes']==['WRONG_OUTPUT']*4 for r in rows)
    receipt=json.loads((out/'direct-profile.json').read_text())
    assert receipt['profile']=='direct-no-sandbox' and receipt['status']=='complete'
    assert receipt['outputs']['public-corrections.jsonl']==dc.sha(out/'public-corrections.jsonl')
    with pytest.raises(FileExistsError):dc.run('prepare',args)


def test_changed_input_during_execution_no_publication(prepared_inputs,monkeypatch):
    args,out=prepared_inputs;name='execute_stdin' if args[args.index('--domain')+1]=='rbr' else 'execute_call';original=getattr(dc.direct,name)
    def change(code,test,**kwargs):
        result=original(code,test,**kwargs)
        public=Path(args[args.index('--public-root')+1]);(public/'manifest.json').write_text('changed')
        return result
    monkeypatch.setattr(dc.direct,name,change)
    with pytest.raises(ValueError,match='input'):dc.run('prepare',args)
    assert not out.exists()
    assert list(out.parent.glob(out.name+'.direct-attempt-*'))


def test_isolated_main_does_not_modify_original_globals():
    def original():return execute_stdin
    old=original.__globals__.get('execute_stdin')
    clone=dc.direct_main(original)
    assert clone() is dc.direct.execute_stdin
    assert original.__globals__.get('execute_stdin') is old


def test_original_generation_with_fake_model_and_trusted_direct_program(prepared_inputs,monkeypatch,tmp_path):
    import types
    import torch
    args,prepared=prepared_inputs;dc.run('prepare',args)
    config=tmp_path/'model.yaml';config.write_text('model_id: test/model\nrevision: '+('a'*40)+'\n')
    generated=tmp_path/'generated'
    class Tokenizer:
        eos_token_id=9
        def __call__(self,*a,**kw):return {'input_ids':torch.tensor([[1,2]])}
        def decode(self,*a,**kw):return 'print(17)'
    class Model:
        device='cpu'
        def eval(self):pass
        def generate(self,input_ids,**kwargs):
            assert kwargs['do_sample'] is False and kwargs['max_new_tokens']==512
            return torch.tensor([[1,2,9]])
    transformers=types.ModuleType('transformers')
    transformers.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**kw:Tokenizer())
    transformers.AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a,**kw:Model())
    transformers.BitsAndBytesConfig=lambda **kw:kw
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    real_run_path=dc.runpy.run_path
    def load(path):
        module=real_run_path(path)
        if str(path).endswith('generate_eesd_corrections.py'):
            fn=module['main'];namespace=dict(fn.__globals__)
            namespace['prepare_correction_prompt']=lambda *a:dict(prompt='test',messages=[{'role':'user','content':'test'}],rendered='test',input_ids=[1,2],prompt_metadata={},prompt_token_ids_sha256='0'*64)
            module['main']=types.FunctionType(fn.__code__,namespace)
        return module
    monkeypatch.setattr(dc.runpy,'run_path',load)
    forwarded=args[:4]+['--public-bank',str(prepared/'public-corrections.jsonl'),'--model-config',str(config),'--domain',args[args.index('--domain')+1],'--output',str(generated)]
    dc.run('generate',forwarded)
    rows=[json.loads(l) for l in (generated/'corrections.jsonl').read_text().splitlines()]
    assert len(rows)==2 and all(r['after_outcomes']==['PASS']*4 for r in rows)
    receipt=json.loads((generated/'direct-profile.json').read_text())
    assert receipt['stage']=='generate' and receipt['status']=='complete'
    assert str((prepared/'direct-profile.json').resolve()) in receipt['inputs']


def test_unbound_public_bank_rejected_before_readiness(prepared_inputs,monkeypatch,tmp_path):
    args,prepared=prepared_inputs;bank=tmp_path/'unbound.jsonl';bank.write_text('{}\n');cfg=tmp_path/'cfg.yaml';cfg.write_text('x')
    monkeypatch.setattr(dc.runtime,'probe_readiness',lambda *a:pytest.fail('must reject before probe'))
    with pytest.raises(FileNotFoundError):
        dc.run('generate',args[:4]+['--public-bank',str(bank),'--model-config',str(cfg),'--domain','rbr','--output',str(tmp_path/'bad')])
