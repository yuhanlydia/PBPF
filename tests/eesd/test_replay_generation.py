import hashlib
import json
from pathlib import Path
import pytest
from pbpf.eesd import replay_generation as g
from pbpf.eesd.replay_prompt import TEMPLATE, messages_for


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}

@pytest.fixture
def admitted(tmp_path,monkeypatch):
    root=tmp_path/'bundle'; root.mkdir()
    proofs=tmp_path/'proofs';proofs.mkdir();monkeypatch.setattr(g,'MODEL_PROOF_ROOT',proofs)
    snapshot=tmp_path/'snapshot';snapshot.mkdir();(snapshot/'model.safetensors').write_bytes(b'weight')
    for f,(m,r) in g.MODELS.items():
        dump(proofs/f'{f}-verified.json',{'model_key':f,'model_id':m,'revision':r,'status':'downloaded-and-sha256-verified','snapshot':str(snapshot),'files':[{'file':'model.safetensors','bytes':6,'sha256':g.sha(snapshot/'model.safetensors'),'publisher_lfs_sha256':g.sha(snapshot/'model.safetensors')}]})
    models=[{'key':f,'model_id':m,'revision':r,'tokenizer_files_sha256':{'tokenizer.json':'a'*64},'chat_template':'chat','chat_template_sha256':hashlib.sha256(b'chat').hexdigest()} for f,(m,r) in g.MODELS.items()]
    template=tmp_path/'probe/prompt-template.txt';template.parent.mkdir();template.write_text(TEMPLATE)
    receipt=tmp_path/'probe/receipt.json'
    dump(receipt,{'schema':'eesd-replay-token-probe-v1','status':'cpu-probe-not-admitted-not-generated','policy':'eesd-replay-synthesis-full-public-v1','input_cap':4096,'proposed_max_new_tokens':1024,'no_truncation':True,'template_sha256':g.sha(template),'models':models,'transformers_version':'4.57.6'})
    outputs={};audits=[]
    for domain in g.DOMAINS:
        rows=[]
        for i in range(700):
            row={'task_id':f'{domain}/{i}','source_component_id':f'{domain}-source{i}','domain':domain,'split':'development' if i<200 else 'primary','statement':'test','visible_tests':[{'input':'a\n','output':'b\n'}]*4}
            rows.append(row)
            audits.append({'domain':domain,'task_id':row['task_id'],'source_id':row['source_component_id'],'all_four_fit':True,'message_sha256':hashlib.sha256(json.dumps(messages_for(row['statement'],row['visible_tests']),ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),'models':{f:{'fits_4096':True,'input_tokens':3,'rendered_prompt_sha256':'b'*64,'token_ids_sha256':'c'*64} for f in g.MODELS}})
        tasks=root/domain/'public/tasks.jsonl';tasks.parent.mkdir(parents=True);tasks.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        seal={'sha256':g.sha(tasks),'bytes':tasks.stat().st_size};outputs[f'{domain}/public/tasks.jsonl']=seal
        outputs[f'{domain}/public/manifest.json']=dump(root/domain/'public/manifest.json',{'schema':'eesd-replay-public-v1','domain':domain,'task_kind':'stdin_synthesis','counts':{'development':200,'primary':500},'tasks':seal})
        # Presence must not cause the generator to open evaluator content.
        (root/domain/'evaluator').mkdir();(root/domain/'evaluator/tasks.jsonl').write_text('INVALID PRIVATE DATA')
    audit=root/'selected-token-audit.jsonl';audit.write_text(''.join(json.dumps(r)+'\n' for r in audits));outputs[audit.name]={'sha256':g.sha(audit),'bytes':audit.stat().st_size}
    dump(root/'admission.json',{'schema':'eesd-replay-admission-v1','status':'data-admitted-not-generated-not-scored','outputs':outputs,'evidence':{'token_receipt':{'path':str(receipt),'sha256':g.sha(receipt)},'prompt_template_sha256':g.sha(template),'models':models,'spec_sha256':'d'*64}})
    return root


def load(root):
    return g.load_public_split(root,g.sha(root/'admission.json'),'apps_replay','development','qwen25_7b')


def test_public_only_loader_and_identity(admitted,monkeypatch):
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert 'evaluator' not in path.parts
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    p=load(admitted)
    assert len(p['rows'])==200
    assert len(p['audits'])==200
    run=g.build_expected(p,'qwen25_7b',1701,Path(__file__))
    assert run['components']==200 and run['candidates']==1
    assert run['domain']=='apps_replay' and run['seed']==1701
    assert run['quantization']['bnb_4bit_quant_type']=='fp4'


def test_tampered_public_rejected(admitted):
    p=admitted/'apps_replay/public/tasks.jsonl';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='checksum'):load(admitted)


def test_wrong_admission_rejected(admitted):
    with pytest.raises(ValueError,match='admission'):g.load_public_split(admitted,'0'*64,'apps_replay','development','qwen25_7b')


def test_template_tamper_rejected(admitted):
    p=admitted.parent/'probe/prompt-template.txt';p.write_text(TEMPLATE+' ')
    with pytest.raises(ValueError,match='template'):load(admitted)


def test_record_and_bank_external_identity(admitted,tmp_path):
    p=load(admitted);run=g.build_expected(p,'qwen25_7b',1701,Path(__file__))
    bank=tmp_path/'bank';bank.mkdir();dump(bank/'run.json',run);files={}
    for row in p['rows']:
        audit=p['audits'][row['task_id']];m=audit['models'][run['family']]
        rec={k:row[k] for k in ('task_id','source_component_id','domain','split')}
        rec.update(seed=g.task_seed(run['seed'],row['task_id']),prompt_sha256=m['rendered_prompt_sha256'],prompt_metadata={**{k:m[k] for k in ('input_tokens','rendered_prompt_sha256','token_ids_sha256')},'message_sha256':audit['message_sha256'],'prompt_policy':'eesd-replay-synthesis-full-public-v1','input_clipped':False,'visible_observations':4},candidates=[{'candidate_id':f"{row['task_id']}/{run['family']}/0",'code':'','raw_completion':'','generated_tokens':1,'hit_token_cap':False}])
        path=bank/g.record_filename(row['task_id']);dump(path,rec);path.with_suffix('.sha256').write_text(g.sha(path)+'\n');files[path.name]=g.sha(path)
    dump(bank/'complete.json',{'schema':'eesd-replay-generation-complete-v1','run_sha256':g.sha(bank/'run.json'),'files':files})
    assert len(g.verify_replay_bank(bank,run,preflight=p)['rows'])==200
    wrong={**run,'seed':1702}
    with pytest.raises(ValueError,match='identity'):g.verify_replay_bank(bank,wrong,preflight=p)
    partial=bank/'interrupted.json.partial';partial.write_text('partial')
    with pytest.raises(ValueError,match='partial'):g.verify_replay_bank(bank,run,preflight=p)
    partial.unlink()
    path=bank/g.record_filename(p['rows'][0]['task_id']);original=path.read_bytes()
    bad=json.loads(original);bad['seed']+=1;dump(path,bad);path.with_suffix('.sha256').write_text(g.sha(path))
    with pytest.raises(ValueError,match='seed'):g.verify_record(path,run,p['rows'][0],p['audits'][p['rows'][0]['task_id']])
    path.write_bytes(original);path.with_suffix('.sha256').unlink()
    with pytest.raises(ValueError,match='checksum'):g.verify_record(path,run,p['rows'][0],p['audits'][p['rows'][0]['task_id']])


def test_task_seed_and_filename():
    assert g.task_seed(1701,'a')==int.from_bytes(hashlib.sha256(b'1701:a').digest()[:4],'big')
    with pytest.raises(ValueError):g.record_filename('../x')


def reseal(root,relative):
    path=root/relative;admission=json.loads((root/'admission.json').read_text())
    seal={'sha256':g.sha(path),'bytes':path.stat().st_size};admission['outputs'][relative]=seal
    if relative.endswith('/public/tasks.jsonl'):
        mp=path.parent/'manifest.json';m=json.loads(mp.read_text());m['tasks']=seal
        admission['outputs'][str(mp.relative_to(root))]=dump(mp,m)
    dump(root/'admission.json',admission)


@pytest.mark.parametrize('mutation',['duplicate_source','wrong_split','private_field','bad_bounds','wrong_domain','collision'])
def test_resealed_invalid_public_fails(admitted,mutation):
    path=admitted/'apps_replay/public/tasks.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
    if mutation=='duplicate_source':rows[1]['source_component_id']=rows[0]['source_component_id']
    elif mutation=='wrong_split':rows[0]['split']='train'
    elif mutation=='private_field':rows[0]['tests']=[]
    elif mutation=='bad_bounds':rows[0]['visible_tests'][0]['input']=' '*8193
    elif mutation=='wrong_domain':rows[0]['domain']='codecontests_replay'
    else:rows[1]['task_id']=rows[0]['task_id'].replace('/','-')
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));reseal(admitted,str(path.relative_to(admitted)))
    with pytest.raises(ValueError):load(admitted)


@pytest.mark.parametrize('mutation',['missing','duplicate','wrong_source','false_fit','missing_model'])
def test_resealed_invalid_audit_fails(admitted,mutation):
    path=admitted/'selected-token-audit.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(rows[0])
    elif mutation=='wrong_source':rows[0]['source_id']='other'
    elif mutation=='false_fit':rows[0]['all_four_fit']=False
    else:rows[0]['models'].pop('qwen25_7b')
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows));reseal(admitted,path.name)
    with pytest.raises(ValueError):load(admitted)


def test_prepare_rechecks_exact_ids(monkeypatch):
    monkeypatch.setattr(g.replay_prompt,'render_and_verify',lambda *a:('prompt',{'token_ids_sha256':hashlib.sha256(b'[1,2]').hexdigest()}))
    class Tokenizer:
        def __call__(self,prompt,**kwargs):
            assert kwargs==dict(add_special_tokens=False,return_token_type_ids=False,truncation=False)
            return {'input_ids':[1,3]}
    with pytest.raises(ValueError,match='ids changed'):g.prepare_prompt(Tokenizer(),{}, {},'qwen25_7b')


@pytest.mark.parametrize('change',['revision','publisher','empty','size'])
def test_weight_proof_rejected(admitted,change):
    path=g.MODEL_PROOF_ROOT/'qwen25_7b-verified.json';proof=json.loads(path.read_text())
    if change=='revision':proof['revision']='bad'
    elif change=='publisher':proof['files'][0]['publisher_lfs_sha256']='0'*64
    elif change=='empty':proof['files']=[]
    else:proof['files'][0]['bytes']=7
    dump(path,proof)
    with pytest.raises(ValueError,match='weight proof'):load(admitted)


def test_real_library_cli_generation_and_resume_integration(admitted,tmp_path,monkeypatch,capsys):
    """Exercise the frozen CLI against real public/run/record/bank validators."""
    import importlib.util
    import types
    import torch
    script=Path(__file__).resolve().parents[2]/'scripts/generate_eesd_replay_bank.py'
    spec=importlib.util.spec_from_file_location('replay_cli_integration',script)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    audit_path=admitted/'selected-token-audit.jsonl'
    audits=[json.loads(x) for x in audit_path.read_text().splitlines()]
    for audit in audits:
        for values in audit['models'].values():
            values['rendered_prompt_sha256']=hashlib.sha256(b'prompt').hexdigest()
            values['token_ids_sha256']=hashlib.sha256(b'[1,2,3]').hexdigest()
    audit_path.write_text(''.join(json.dumps(a)+'\n' for a in audits));reseal(admitted,audit_path.name)
    class Tokenizer:
        eos_token_id=9;pad_token_id=9
        def apply_chat_template(self,messages,**kwargs):return 'prompt'
        def __call__(self,prompt,**kwargs):
            assert kwargs==dict(add_special_tokens=False,return_token_type_ids=False,truncation=False)
            return {'input_ids':[1,2,3]}
        def decode(self,ids,skip_special_tokens):return 'print(1)'
    calls={'loads':0,'generations':0}
    class Model:
        device='cpu'
        def generate(self,**kwargs):
            calls['generations']+=1
            return torch.tensor([[1,2,3,9]])
    def load_model(model):
        calls['loads']+=1
        return Model(),torch
    monkeypatch.setattr(cli,'load_tokenizer',lambda model:Tokenizer())
    monkeypatch.setattr(cli,'load_model',load_model)
    args=types.SimpleNamespace(bundle=admitted,admission_sha256=g.sha(admitted/'admission.json'),domain='apps_replay',split='development',family='qwen25_7b',seed=1701,output=tmp_path/'integration-bank',preflight_only=False)
    cli.execute(args)
    assert calls=={'loads':1,'generations':200}
    preflight=load(admitted);run=g.build_expected(preflight,args.family,args.seed,script)
    verified=g.verify_replay_bank(args.output,run,preflight=preflight)
    assert len(verified['rows'])==200
    seals={p.name:p.read_bytes() for p in args.output.iterdir()}
    cli.execute(args)
    assert calls=={'loads':1,'generations':200}
    assert seals=={p.name:p.read_bytes() for p in args.output.iterdir()}
    # A genuinely not-yet-generated task in an incomplete bank is generated once.
    missing=args.output/g.record_filename(preflight['rows'][-1]['task_id'])
    missing.unlink();missing.with_suffix('.sha256').unlink();(args.output/'complete.json').unlink()
    preserved={p.name:p.read_bytes() for p in args.output.iterdir()}
    cli.execute(args)
    assert calls=={'loads':2,'generations':201}
    assert all((args.output/name).read_bytes()==data for name,data in preserved.items())
    assert len(g.verify_replay_bank(args.output,run,preflight=preflight)['rows'])==200
    # A published record without its checksum is not treated as a missing task.
    (args.output/'complete.json').unlink();missing.with_suffix('.sha256').unlink()
    record_bytes=missing.read_bytes()
    with pytest.raises(ValueError,match='checksum'):cli.execute(args)
    assert calls=={'loads':2,'generations':201} and missing.read_bytes()==record_bytes
    assert not (args.output/'complete.json').exists()
    capsys.readouterr()
