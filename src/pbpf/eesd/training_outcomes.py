"""CPU-only eligibility and immutable scientific training-outcome protocol."""
import hashlib
import json
import math
from pathlib import Path
from pbpf.eesd.distillation import TRAIN_RULES,validate_scored_record

NON_ESTIMABLE='non_estimable_zero_positive_training_weight'
PROTOCOL_ID='eesd-zero-positive-training-weight-v1'
ADOPTION='docs/EESD_ZERO_WEIGHT_ADOPTION_20260920.md'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def protocol_binding(root):
    return dict(training_outcome_protocol=PROTOCOL_ID,training_outcome_adoption_sha256=sha(Path(root)/ADOPTION),
        training_outcomes_source_sha256=sha(Path(__file__)))


def tree_sha(directory):
    root=Path(directory);files=sorted(p for p in root.rglob('*') if p.is_file())
    if not files:raise ValueError('parent adapter tree must be nonempty')
    digest=hashlib.sha256()
    for p in files:
        name=p.relative_to(root).as_posix().encode();digest.update(len(name).to_bytes(4,'big'));digest.update(name)
        digest.update(bytes.fromhex(sha(p)))
    return digest.hexdigest()


def read_training_rows(path,rule):
    if rule not in TRAIN_RULES or rule=='no_update':raise ValueError('valid update rule required')
    rows=[];ids=set();sources={}
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if not isinstance(row,dict):raise ValueError('scored record must be an object')
        weights=row.get('training_weights')
        if not isinstance(weights,dict) or any(type(w) not in (int,float) for w in weights.values()):
            raise ValueError('training weights must be numeric, not booleans or strings')
        validate_scored_record(row)
        split=row['split']
        if split not in {'train','development'}:raise ValueError('training split must be train/development')
        for key in ('trajectory_id','source_component_id','prompt','correction'):
            if not isinstance(row[key],str) or not row[key].strip():raise ValueError(f'nonempty {key} required')
        if row['trajectory_id'] in ids:raise ValueError('duplicate trajectory identity')
        ids.add(row['trajectory_id'])
        source=row['source_component_id']
        if source in sources and sources[source]!=split:raise ValueError('training/development source overlap')
        sources[source]=split
        before,after,relevance=(row[k] for k in ('before_outcomes','after_outcomes','relevance'))
        if (not all(isinstance(v,list) for v in (before,after,relevance)) or not before
                or len(before)!=len(after) or len(before)!=len(relevance)
                or any(type(x) is not int or not 0<=x<5 for x in before+after)
                or any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in relevance)
                or sum(relevance)<=0):raise ValueError('invalid scored trajectory vectors')
        if 'messages' in row:
            messages=row['messages'];metadata=row.get('prompt_metadata',{});token_sha=row.get('prompt_token_ids_sha256')
            if (not isinstance(messages,list) or not messages or any(not isinstance(m,dict)
                    or set(m)!={'role','content'} or m['role'] not in {'system','user'}
                    or not isinstance(m['content'],str) for m in messages)
                    or messages[-1]!={'role':'user','content':row['prompt']}
                    or not isinstance(token_sha,str) or len(token_sha)!=64
                    or any(c not in '0123456789abcdef' for c in token_sha)
                    or not isinstance(metadata,dict) or type(metadata.get('input_tokens')) is not int
                    or metadata['input_tokens']<=0):raise ValueError('invalid saved generation messages/token metadata')
        rows.append({**row,'_weight':float(row['training_weights'][rule])})
    eligibility=classify(rows)
    return rows,eligibility


def classify(rows):
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='development']
    if not train or not dev:raise ValueError('nonempty training and development rows required')
    positive=sum(r['_weight']>0 for r in train)
    return dict(status='eligible' if positive else NON_ESTIMABLE,train_rows=len(train),development_rows=len(dev),
        all_rows=len(rows),positive_training_rows=positive,positive_development_rows=sum(r['_weight']>0 for r in dev),
        training_sources=len({r['source_component_id'] for r in train}),
        development_sources=len({r['source_component_id'] for r in dev}))


def validate_parent_adapter(directory, model_config):
    """Validate a real LoRA checkpoint on CPU before classifying an empty update."""
    import yaml
    from safetensors import safe_open
    directory=Path(directory)
    try:
        cfg=json.loads((directory/'adapter_config.json').read_text())
        model=yaml.safe_load(Path(model_config).read_text())
        if (cfg.get('peft_type')!='LORA' or cfg.get('task_type')!='CAUSAL_LM'
                or cfg.get('base_model_name_or_path')!=model['model_id']
                or cfg.get('revision') not in (None,model['revision'])
                or type(cfg.get('r')) is not int or cfg['r']<1):
            raise ValueError('adapter config/model identity mismatch')
        with safe_open(directory/'adapter_model.safetensors',framework='np') as tensors:
            keys=list(tensors.keys())
            if not keys or not any('lora_' in key for key in keys):
                raise ValueError('adapter contains no LoRA tensors')
            for key in keys:
                if not tensors.get_slice(key).get_shape():raise ValueError('adapter scalar tensor')
    except Exception as exc:
        raise ValueError(f'invalid parent adapter: {exc}') from exc


def zero_report(*,root,input_path,model_config,trainer,rule,seed,budget,previous_adapter,anchor_beta,eligibility):
    if eligibility['status']!=NON_ESTIMABLE:raise ValueError('zero outcome requires verified empty selection')
    if type(budget) is not int or budget<1:raise ValueError('zero outcome requires explicit positive response-token budget')
    previous=str(Path(previous_adapter).resolve()) if previous_adapter else None
    if previous:validate_parent_adapter(previous,model_config)
    return dict(schema='eesd-zero-training-outcome-v1',status=NON_ESTIMABLE,rule=rule,seed=seed,
        input_sha256=sha(input_path),model_config_sha256=sha(model_config),trainer_source_sha256=sha(trainer),
        previous_adapter=previous,previous_adapter_tree_sha256=tree_sha(previous) if previous else None,
        anchor_beta=anchor_beta,response_token_budget=budget,response_tokens=0,optimizer_steps=0,micro_steps=0,
        eligibility=eligibility,adapter=None,main_estimate=None,confidence_interval=None,p_value=None,
        reason='valid training population contains no positive-weight trajectories; equal actual budget is non-estimable',
        descriptive_fallback_adopted=False,**protocol_binding(root))
