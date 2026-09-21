"""Public-only Replay admission and externally bound generation inventories."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
from . import replay_prompt

DOMAINS=('apps_replay','codecontests_replay')
MODELS={
 'qwen25_7b':('Qwen/Qwen2.5-Coder-7B-Instruct','c03e6d358207e414f1eca0bb1891e29f1db0e242'),
 'deepseek_6p7b':('deepseek-ai/deepseek-coder-6.7b-instruct','e5d64addd26a6a1db0f9b863abf6ee3141936807'),
 'seed_coder_8b':('ByteDance-Seed/Seed-Coder-8B-Instruct','c62d428b84d52a7f1bb38d2aa72a79d6d5f5e614'),
 'starcoder2_15b':('bigcode/starcoder2-15b-instruct-v0.1','ffb8dd9776ba9a66d655ecd962e882f3013e9f7c'),
}
COUNTS={'development':200,'primary':500}
MODEL_PROOF_ROOT=Path(__file__).resolve().parents[3]/'runs/eesd-setup'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def _json(path):return json.loads(Path(path).read_bytes())
def _rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]
def _digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def _verify(path,seal):
    if set(seal)!={'sha256','bytes'} or Path(path).stat().st_size!=seal['bytes'] or sha(path)!=seal['sha256']:
        raise ValueError(f'checksum mismatch: {path}')

def record_filename(task_id):
    if not isinstance(task_id,str) or not task_id or any(part in ('','.','..') for part in task_id.split('/')) or '\\' in task_id:
        raise ValueError('unsafe task filename')
    return task_id.replace('/','-')+'.json'

def task_seed(seed,task_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{task_id}'.encode()).digest()[:4],'big')

def load_public_split(bundle,admission_sha256,domain,split,family):
    """Validate public artifacts only; evaluator paths are never followed."""
    bundle=Path(bundle).resolve()
    if domain not in DOMAINS or split not in COUNTS or family not in MODELS:
        raise ValueError('unsupported Replay domain/split/family')
    admission_path=bundle/'admission.json'
    if not _digest(admission_sha256) or sha(admission_path)!=admission_sha256:
        raise ValueError('admission checksum mismatch')
    admission=_json(admission_path)
    if admission.get('schema')!='eesd-replay-admission-v1' or admission.get('status')!='data-admitted-not-generated-not-scored':
        raise ValueError('admission schema/status mismatch')
    evidence=admission['evidence'];outputs=admission['outputs']
    token_path=Path(evidence['token_receipt']['path'])
    if sha(token_path)!=evidence['token_receipt']['sha256']:raise ValueError('token receipt checksum mismatch')
    token=_json(token_path)
    if (token.get('schema')!='eesd-replay-token-probe-v1' or token.get('status')!='cpu-probe-not-admitted-not-generated'
        or token.get('policy')!=replay_prompt.POLICY or token.get('input_cap')!=4096
        or token.get('proposed_max_new_tokens')!=1024 or token.get('no_truncation') is not True):
        raise ValueError('token receipt protocol mismatch')
    template_path=token_path.parent/'prompt-template.txt'
    template_sha=hashlib.sha256(replay_prompt.TEMPLATE.encode()).hexdigest()
    if not (sha(template_path)==evidence['prompt_template_sha256']==token['template_sha256']==template_sha):
        raise ValueError('sealed template mismatch')
    models=token['models']
    if models!=evidence['models'] or len(models)!=4 or {m['key'] for m in models}!=set(MODELS):
        raise ValueError('admitted model inventory mismatch')
    for model in models:
        if (model['model_id'],model['revision'])!=MODELS[model['key']]:raise ValueError('model revision mismatch')
        if (not model['tokenizer_files_sha256'] or any(not _digest(v) for v in model['tokenizer_files_sha256'].values())
            or hashlib.sha256(model['chat_template'].encode()).hexdigest()!=model['chat_template_sha256']):
            raise ValueError('tokenizer identity invalid')
    proof_path=MODEL_PROOF_ROOT/f'{family}-verified.json'
    proof=_json(proof_path)
    if (proof.get('model_key')!=family or (proof.get('model_id'),proof.get('revision'))!=MODELS[family]
        or proof.get('status')!='downloaded-and-sha256-verified'):
        raise ValueError('model weight proof identity mismatch')
    weights=[f for f in proof['files'] if f['file'].endswith('.safetensors')]
    if not weights:raise ValueError('model weight proof has no weights')
    names=set()
    for file in weights:
        name=file['file'];path=Path(proof['snapshot'])/name
        if (name in names or Path(name).is_absolute() or '..' in Path(name).parts
            or not _digest(file['sha256']) or file['sha256']!=file['publisher_lfs_sha256']
            or type(file['bytes']) is not int or file['bytes']<=0
            or not path.is_file() or path.stat().st_size!=file['bytes']):
            raise ValueError('model weight proof checksum/size invalid')
        names.add(name)
    audit_path=bundle/'selected-token-audit.jsonl';_verify(audit_path,outputs[audit_path.name])
    audits={};sources=set();all_rows={};public_bindings={}
    for d in DOMAINS:
        mp=bundle/d/'public/manifest.json';tp=bundle/d/'public/tasks.jsonl'
        for p in (mp,tp):_verify(p,outputs[p.relative_to(bundle).as_posix()])
        manifest=_json(mp)
        if (manifest.get('schema')!='eesd-replay-public-v1' or manifest.get('domain')!=d
            or manifest.get('task_kind')!='stdin_synthesis' or manifest.get('counts')!=COUNTS
            or manifest['tasks']!=outputs[tp.relative_to(bundle).as_posix()]):raise ValueError('public manifest mismatch')
        rows=_rows(tp);counts={s:0 for s in COUNTS};names=set()
        for row in rows:
            if set(row)!={'task_id','source_component_id','split','domain','statement','visible_tests'}:
                raise ValueError('public fields outside whitelist')
            if row['domain']!=d or row['split'] not in COUNTS:raise ValueError('public domain/split mismatch')
            if not isinstance(row['source_component_id'],str) or not row['source_component_id'] or row['source_component_id'] in sources:
                raise ValueError('duplicate/invalid source across public population')
            name=record_filename(row['task_id'])
            if row['task_id'] in all_rows or name in names:raise ValueError('duplicate task or filename collision')
            names.add(name);sources.add(row['source_component_id']);counts[row['split']]+=1
            if not isinstance(row['statement'],str) or not row['statement'].strip() or len(row['statement'])>6000:
                raise ValueError('invalid complete statement')
            tests=row['visible_tests']
            if not isinstance(tests,list) or len(tests)!=4:raise ValueError('exactly four public tests required')
            for test in tests:
                if (not isinstance(test,dict) or set(test)!={'input','output'} or any(not isinstance(test[k],str) for k in test)
                    or len(test['input'])>8192 or len(test['output'])>4096):raise ValueError('invalid public test fields/bounds')
            all_rows[row['task_id']]=row
        if counts!=COUNTS:raise ValueError('public population counts mismatch')
        public_bindings[d]={'public_tasks_sha256':sha(tp),'public_manifest_sha256':sha(mp)}
    for audit in _rows(audit_path):
        task=audit['task_id']
        if task in audits or task not in all_rows:raise ValueError('duplicate/unexpected audit task')
        row=all_rows[task]
        if audit['domain']!=row['domain'] or audit['source_id']!=row['source_component_id'] or audit.get('all_four_fit') is not True:
            raise ValueError('audit population identity mismatch')
        if set(audit['models'])!=set(MODELS):raise ValueError('audit model inventory mismatch')
        for values in audit['models'].values():
            if (values.get('fits_4096') is not True or type(values['input_tokens']) is not int or not 1<=values['input_tokens']<=4096
                or any(not _digest(values[k]) for k in ('rendered_prompt_sha256','token_ids_sha256'))):raise ValueError('audit token fields invalid')
        message=json.dumps(replay_prompt.messages_for(row['statement'],row['visible_tests']),ensure_ascii=False,separators=(',',':')).encode()
        if hashlib.sha256(message).hexdigest()!=audit['message_sha256']:raise ValueError('audit message changed')
        audits[task]=audit
    if set(audits)!=set(all_rows):raise ValueError('missing audit tasks')
    selected=[r for r in all_rows.values() if r['domain']==domain and r['split']==split]
    return {'rows':selected,'audits':{r['task_id']:audits[r['task_id']] for r in selected},
        'model':next(m for m in models if m['key']==family),'domain':domain,'split':split,'family':family,
        'bindings':{'admission_sha256':admission_sha256,'token_receipt_sha256':sha(token_path),
                    'selected_token_audit_sha256':sha(audit_path),'template_sha256':template_sha,
                    'spec_sha256':evidence['spec_sha256'],'model_weight_proof_sha256':sha(proof_path),**public_bindings[domain]}}

def prepare_prompt(tokenizer,row,audit,family):
    prompt,metadata=replay_prompt.render_and_verify(tokenizer,row,audit,family)
    ids=tokenizer(prompt,add_special_tokens=False,return_token_type_ids=False,truncation=False)['input_ids']
    if hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()!=metadata['token_ids_sha256']:
        raise ValueError('token ids changed on generation preparation')
    return {'prompt':prompt,'ids':ids,'metadata':metadata}

def build_expected(preflight,family,seed,generator_path):
    if family!=preflight['family'] or type(seed) is not int or seed not in (1701,1702,1703):raise ValueError('run family/seed mismatch')
    from pbpf.apbpf import rbr_prompt
    rows=preflight['rows'];model=preflight['model']
    return {'schema':'eesd-replay-generation-v1','task_kind':'stdin_synthesis','domain':preflight['domain'],
        'family':family,'model':model['model_id'],'revision':model['revision'],'split':preflight['split'],'seed':seed,
        'task_ids':[r['task_id'] for r in rows],'source_component_ids':[r['source_component_id'] for r in rows],
        'components':len(rows),'candidates':1,'adapter':None,'max_input_tokens':4096,'max_new_tokens':1024,
        'temperature':.8,'top_p':.95,'decode_policy':'temperature0.8-top_p0.95','prompt_policy':replay_prompt.POLICY,
        'task_seed_policy':'first4-big-endian-SHA256-UTF8(seed:task_id)',
        'quantization':{'load_in_4bit':True,'bnb_4bit_compute_dtype':'bfloat16','bnb_4bit_quant_type':'fp4','bnb_4bit_use_double_quant':False},
        'torch_dtype':'bfloat16','device_map':{'':0},'local_files_only':True,
        'dependencies':{p:importlib.metadata.version(p) for p in ('torch','transformers','bitsandbytes')},
        'tokenizer':model,'bindings':preflight['bindings'],
        'source_sha256':{'generator':sha(generator_path),'replay_generation':sha(__file__),
                         'replay_prompt':sha(replay_prompt.__file__),'completion_helper':sha(rbr_prompt.__file__)}}

def verify_record(path,run,row,audit):
    path=Path(path);checksum=path.with_suffix('.sha256')
    if not path.is_file() or not checksum.is_file() or checksum.read_text().strip()!=sha(path):raise ValueError('candidate checksum missing/mismatch')
    record=_json(path)
    if path.name!=record_filename(row['task_id']) or any(record.get(k)!=row[k] for k in ('task_id','source_component_id','domain','split')):
        raise ValueError('candidate external identity mismatch')
    if type(record.get('seed')) is not int or record['seed']!=task_seed(run['seed'],row['task_id']):raise ValueError('candidate seed identity mismatch')
    m=audit['models'][run['family']]
    expected={k:m[k] for k in ('input_tokens','rendered_prompt_sha256','token_ids_sha256')}
    expected.update(message_sha256=audit['message_sha256'],prompt_policy=replay_prompt.POLICY,input_clipped=False,visible_observations=4)
    if record.get('prompt_metadata')!=expected or record.get('prompt_sha256')!=m['rendered_prompt_sha256']:raise ValueError('candidate prompt identity mismatch')
    candidates=record.get('candidates')
    if not isinstance(candidates,list) or len(candidates)!=1:raise ValueError('candidate count mismatch')
    c=candidates[0]
    if (c.get('candidate_id')!=f"{row['task_id']}/{run['family']}/0" or not isinstance(c.get('code'),str)
        or not isinstance(c.get('raw_completion'),str) or type(c.get('generated_tokens')) is not int
        or not 0<=c['generated_tokens']<=1024 or type(c.get('hit_token_cap')) is not bool):raise ValueError('candidate fields invalid')
    return record

def verify_replay_bank(bank,expected,*,preflight):
    bank=Path(bank)
    if _json(bank/'run.json')!=expected:raise ValueError('bank run external identity mismatch')
    if (expected['task_ids']!=[r['task_id'] for r in preflight['rows']]
        or expected['source_component_ids']!=[r['source_component_id'] for r in preflight['rows']]
        or expected['bindings']!=preflight['bindings'] or expected['family']!=preflight['family']
        or expected['domain']!=preflight['domain'] or expected['split']!=preflight['split']):raise ValueError('bank preflight identity mismatch')
    complete=_json(bank/'complete.json')
    if complete.get('schema')!='eesd-replay-generation-complete-v1' or complete.get('run_sha256')!=sha(bank/'run.json'):
        raise ValueError('bank completion run checksum mismatch')
    names={record_filename(r['task_id']) for r in preflight['rows']}
    if set(complete['files'])!=names:raise ValueError('bank completion inventory mismatch')
    allowed={'run.json','complete.json'}|names|{str(Path(n).with_suffix('.sha256')) for n in names}
    if {p.name for p in bank.iterdir()}!=allowed:raise ValueError('unexpected/partial bank artifact')
    rows=[]
    for row in preflight['rows']:
        path=bank/record_filename(row['task_id'])
        if sha(path)!=complete['files'][path.name]:raise ValueError('bank record checksum mismatch')
        rows.append(verify_record(path,expected,row,preflight['audits'][row['task_id']]))
    return {'run':expected,'rows':rows,'complete_sha256':sha(bank/'complete.json')}
