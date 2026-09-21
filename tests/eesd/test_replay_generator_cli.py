import importlib.util
from pathlib import Path
import pytest


def generator():
    path=Path(__file__).resolve().parents[2]/'scripts/generate_eesd_replay_bank.py'
    spec=importlib.util.spec_from_file_location('replay_generator',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Tokenizer:
    eos_token_id=9;pad_token_id=9;unk_token_id=-1
    def convert_tokens_to_ids(self,text):assert text=='###';return 8
    def decode(self,ids,skip_special_tokens):return '' if list(ids)==[9] else 'not valid python !!!'


@pytest.mark.parametrize('family,expected_stop',[('qwen25_7b',None),('seed_coder_8b',None),('starcoder2_15b',[9,8])])
def test_one_decode_has_fixed_parameters_and_no_token_types(family,expected_stop):
    import torch
    module=generator()
    class Model:
        device='cpu'
        def generate(self,**kwargs):
            self.kwargs=kwargs
            return torch.tensor([[1,2,3,9]])
    model=Model()
    candidate=module.decode_one(model,Tokenizer(),{'ids':[1,2,3]},family,1701,torch)
    assert model.kwargs['input_ids'].tolist()==[[1,2,3]]
    assert model.kwargs['attention_mask'].tolist()==[[1,1,1]]
    assert 'token_type_ids' not in model.kwargs
    assert {k:model.kwargs[k] for k in ('do_sample','num_return_sequences','max_new_tokens','temperature','top_p')}==dict(do_sample=True,num_return_sequences=1,max_new_tokens=1024,temperature=.8,top_p=.95)
    assert model.kwargs.get('eos_token_id')==expected_stop
    assert candidate['generated_tokens']==1 and candidate['hit_token_cap'] is False
    assert isinstance(candidate['code'],str)  # Invalid/empty samples are kept.


def test_task_seed_is_stable_and_task_specific():
    import hashlib
    module=generator()
    assert module.task_seed(1701,'task/a')==int.from_bytes(hashlib.sha256(b'1701:task/a').digest()[:4],'big')
    assert module.task_seed(1701,'task/a')!=module.task_seed(1701,'task/b')

@pytest.fixture
def pipeline(tmp_path,monkeypatch):
    import sys,types
    module=generator()
    row=dict(task_id='t',source_component_id='s',domain='apps_replay',split='development')
    args=types.SimpleNamespace(bundle=tmp_path/'bundle',admission_sha256='hash',domain='apps_replay',split='development',family='qwen25_7b',seed=1701,output=tmp_path/'bank',preflight_only=False)
    calls=[]
    library=types.ModuleType('pbpf.eesd.replay_generation')
    library.load_public_split=lambda *a:dict(rows=[row],audits={'t':{}},model={})
    library.prepare_prompt=lambda *a:dict(prompt='prompt',ids=[1],metadata={})
    library.build_expected=lambda *a:{'identity':'expected'}
    library.verify_record=lambda *a:calls.append('verify_record')
    library.verify_replay_bank=lambda *a,**k:calls.append('verify_complete')
    library.record_filename=lambda task:task+'.json'
    library.sha=lambda path:'sha'
    monkeypatch.setitem(sys.modules,'pbpf.eesd.replay_generation',library)
    monkeypatch.setattr(module,'load_tokenizer',lambda model:Tokenizer())
    def forbidden(*a):raise AssertionError('GPU model must not load')
    monkeypatch.setattr(module,'load_model',forbidden)
    return module,library,args,calls


def test_prompt_failure_prevents_model_and_bank_creation(pipeline):
    module,library,args,_=pipeline
    def bad(*a):raise ValueError('token mismatch')
    library.prepare_prompt=bad
    with pytest.raises(ValueError,match='token mismatch'):module.execute(args)
    assert not args.output.exists()


def test_completed_resume_verifies_without_gpu(pipeline):
    import json
    module,library,args,calls=pipeline
    args.output.mkdir();(args.output/'run.json').write_text(json.dumps({'identity':'expected'}));(args.output/'complete.json').write_text('{}')
    module.execute(args)
    assert calls==['verify_complete']


def test_orphan_partial_stops_without_regenerating(pipeline):
    module,library,args,calls=pipeline
    args.output.mkdir();partial=args.output/'t.json.partial';partial.write_text('preserve')
    with pytest.raises(ValueError,match='incomplete publication'):module.execute(args)
    assert partial.read_text()=='preserve'


def test_preflight_only_creates_no_generation_output(pipeline):
    module,library,args,calls=pipeline
    args.preflight_only=True
    module.execute(args)
    assert not args.output.exists() and calls==[]
