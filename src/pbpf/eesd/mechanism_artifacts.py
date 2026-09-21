"""Reconstruct six saved mechanism arms from fully bound offline artifacts.

This module does not execute candidates, run a model, compute new confidence
intervals, or reconstruct the remaining ablations. An evaluator root is required
because the historical NPZ has no query IDs: labels alone cannot establish order.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np
import yaml
from pbpf.apbpf.codearc_bank import file_sha, load_bank

FROZEN_RUNNER_SHA256='09d016fb8b9caf6c8b7829a56550ce99f4d1fbefb83a1091a60396a6026aff85'
FROZEN_EVIDENCE_SHA256='f825e0a70a41d2c9e3a8b96aafc89ea54d70c6b36337b90697c906e93758d7be'
FROZEN_MODEL_INVENTORY_SHA256='2e9f92a4ccdc51bce751026d5797b264e2b2e2e795086a9549ef17ce777acc4f'
ARMS=('ordinary','fixed','effective','global_mass','fixed_at_effective_params','effective_at_fixed_params')
CACHE_SOURCES=('scripts/build_eesd_mechanism_cache.py','src/pbpf/apbpf/codearc_execution.py','src/pbpf/apbpf/rbr_execution.py')
RTOL=1e-12
ATOL=1e-14


def read(path):
    return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition: raise ValueError(message)


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def close(actual,expected,name):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and np.all(np.isfinite(a)) and
            np.allclose(a,b,rtol=RTOL,atol=ATOL),f'{name} reconstruction mismatch')


def load_mechanism_artifacts(*,report_dir,cache,config,evaluator_root,domain,model,seed,root=None):
    """Return (schema-neutral arrays/identities, verification receipt).

    ``domain`` is rbr/codearc; ``model`` is the frozen generator family key.
    ``query_ids`` retain absolute test indices, not offsets within future tests.
    Numeric comparison tolerances are rtol=1e-12 and atol=1e-14. No writable state.
    """
    require(evaluator_root is not None,'sealed evaluator root required to verify ordered query inputs')
    root=Path(root) if root else Path(__file__).resolve().parents[3]
    report_dir,cache,config,evaluator_root=map(Path,(report_dir,cache,config,evaluator_root))
    require(domain in {'rbr','codearc'},'unsupported mechanism domain')
    runner_path=root/'scripts/run_eesd_evidence_matrix.py'
    evidence_path=root/'src/pbpf/eesd/evidence.py'
    generator_path=root/'scripts/generate_apbpf_rbr_bank.py'
    for path,expected in ((runner_path,FROZEN_RUNNER_SHA256),(evidence_path,FROZEN_EVIDENCE_SHA256),
                          (generator_path,FROZEN_MODEL_INVENTORY_SHA256)):
        require(file_sha(path)==expected,f'frozen source mismatch: {path.name}')
    # Read a literal mapping; never execute generation code.
    inventory=None
    for node in ast.parse(generator_path.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODELS' for t in node.targets):
            inventory=ast.literal_eval(node.value)
    require(inventory is not None and model in inventory,'unknown frozen model family')
    model_id,revision=inventory[model]
    identity=[model_id,revision,None,seed]
    complete=read(report_dir/'complete.json')
    for name,suffix in [('report','.json'),('predictions','.npz')]:
        require(file_sha(report_dir/(name+suffix))==complete.get(name+'_sha256'),f'{name} checksum mismatch')
    report=read(report_dir/'report.json')
    expected=dict(schema='eesd-evidence-matrix-v1',dataset={'rbr':'runbugrun','codearc':'codearc'}[domain],
        model=model,seed=seed,visible=4,validation_split='development',assessment_split='primary',
        cache_sha256=file_sha(cache),config_sha256=file_sha(config),
        runner_source_sha256=FROZEN_RUNNER_SHA256,evidence_source_sha256=FROZEN_EVIDENCE_SHA256)
    require(all(report.get(k)==v for k,v in expected.items()),'report protocol/source/identity mismatch')
    cfg=yaml.safe_load(config.read_text())
    require(cfg.get('schema')=='eesd-iclr2027-v1' and seed in cfg.get('seeds',[]),'config protocol/seed mismatch')
    payload=read(cache);seal=read(cache.with_suffix('.binding.json'))
    manifest=read(evaluator_root/'manifest.json')
    require(manifest.get('schema')=={'rbr':'apbpf-rbr-generated-materialization-v1',
        'codearc':'apbpf-codearc-replay-materialization-v1'}[domain],'evaluator schema mismatch')
    tasks_sha=file_sha(evaluator_root/'tasks.jsonl');manifest_sha=file_sha(evaluator_root/'manifest.json')
    require(tasks_sha==manifest.get('evaluator_tasks_sha256'),'evaluator task checksum mismatch')
    require(payload.get('schema')=='eesd-public-query-mechanism-cache-v1' and payload.get('dataset')==domain
            and payload.get('generator_identity')==identity and payload.get('tests_per_candidate')==10,
            'cache protocol/generator identity mismatch')
    require(payload.get('evaluator_manifest_sha256')==manifest_sha,'cache evaluator binding mismatch')
    bindings=payload.get('bank_bindings',[])
    require(len(bindings)==2 and {b['split'] for b in bindings}=={'development','primary'},'two distinct bank splits required')
    expected_seal=dict(cache_sha256=file_sha(cache),generator_identity=identity,
        bank_bindings=[{**b,'path':str(Path(b['path']).resolve())} for b in bindings],
        evaluator_tasks_sha256=tasks_sha,evaluator_manifest_sha256=manifest_sha,
        sources={name:file_sha(root/name) for name in CACHE_SOURCES})
    normalized_seal={**seal,'bank_bindings':[{**b,'path':str(Path(b['path']).resolve())} for b in seal.get('bank_bindings',[])]}
    require(normalized_seal==expected_seal,'cache checksum/source binding mismatch')
    tasks_list=[json.loads(line) for line in (evaluator_root/'tasks.jsonl').read_text().splitlines() if line.strip()]
    tasks={t['task_id']:t for t in tasks_list}
    require(len(tasks)==len(tasks_list),'duplicate evaluator task identity')
    bound_records=[];sources=set();candidates=set()
    for binding in bindings:
        run,bank,bank_sha=load_bank(binding['path'])
        require(bank_sha==binding['complete_sha256'] and run['components']==binding['components']
            and len(bank)==binding['components'] and run.get('schema')==f'apbpf-{domain}-generation-v1'
            and run['split']==binding['split'] and run['candidates']==1
            and [run['model'],run['revision'],run.get('adapter_sha256'),run['seed']]==identity
            and run.get('public_tasks_sha256')==manifest.get('public_tasks_sha256'),'bank binding mismatch')
        for row in bank:
            source=row['source_component_id'];candidate=row['candidates'][0];task=tasks.get(row['task_id'])
            require(source not in sources and candidate['candidate_id'] not in candidates,'source split overlap or duplicate candidate')
            sources.add(source);candidates.add(candidate['candidate_id'])
            require(task is not None and task['source_component_id']==source and task['split']==row['split'],
                    'bank/evaluator task source mismatch')
            tests=task['tests']
            require(len(tests)==10 and [str(t['id']) for t in tests]==list(map(str,range(10))),
                    'evaluator ordered test inventory mismatch')
            bound_records.append(dict(task_id=candidate['candidate_id'],problem_id=row['task_id'],
                source_component_id=source,split=row['split'],
                candidate_code_sha256=hashlib.sha256(candidate['code'].encode()).hexdigest(),
                tests=[{'id':str(t['id']),'input':t['input']} for t in tests]))
    rows=payload['records']
    require(len(rows)==len(bound_records),'record count differs from sealed candidate inventory')
    for row,bound in zip(rows,bound_records,strict=True):
        require({k:row.get(k) for k in bound}==bound,'ordered record/candidate/query binding mismatch')
        require(set(row)==set(bound)|{'outcomes'} and len(row['outcomes'])==10,'cache record schema mismatch')
    counts={s:sum(r['split']==s for r in rows) for s in ('development','primary')}
    require(payload.get('counts')==counts and payload.get('source_counts')==counts,'cache source/split counts mismatch')
    # Only the already-hash-checked pure evidence runner is loaded.
    runner=runpy.run_path(str(runner_path))
    from pbpf.eesd import evidence as imported_evidence
    require(file_sha(Path(imported_evidence.__file__))==FROZEN_EVIDENCE_SHA256,'imported evidence source mismatch')
    visible=report['visible'];dev=[r for r in rows if r['split']=='development'];assessment=[r for r in rows if r['split']=='primary']
    runner['_validate_rows'](dev,visible);runner['_validate_rows'](assessment,visible)
    memo={}
    def examples(split,h,strength,binary):
        key=(split,h,strength,binary)
        if key not in memo:memo[key]=runner['build_examples'](dev if split=='validation' else assessment,h,strength,binary=binary)
        return memo[key]
    e=cfg['evidence'];selected={};probabilities={};mass=None
    for arm,rule in [('ordinary','ordinary'),('fixed','fixed'),('effective','effective'),('global_mass','global')]:
        selection,_=runner['select_arm'](examples,'validation',visible,e['alphas'],
            [0.] if arm=='ordinary' else e['strengths'],rule,
            masses=e['global_mass_grid'] if arm=='global_mass' else None,ece_bins=e['ece_bins'])
        actual=report['selected'][arm]
        require(set(actual)==set(selection),f'{arm} selection schema mismatch')
        for key,value in selection.items():
            if value is None:require(actual[key] is None,f'{arm} mass selection mismatch')
            else:close(actual[key],value,f'{arm} development selection {key}')
        selected[arm]=selection
        probabilities[arm],masses=runner['predictions'](examples('assessment',visible,selection['strength'],False),
            selection['alpha'],rule,global_mass=selection['mass'])
        if arm=='effective':mass=masses
    for arm,selection,rule in [('fixed_at_effective_params','effective','fixed'),('effective_at_fixed_params','fixed','effective')]:
        s=selected[selection]
        probabilities[arm],_=runner['predictions'](examples('assessment',visible,s['strength'],False),s['alpha'],rule)
    labels,clusters=runner['labels_clusters'](examples('assessment',visible,selected['effective']['strength'],False))
    with np.load(report_dir/'predictions.npz',allow_pickle=False) as stored:
        require(set(stored.files)==set(ARMS)|{'labels','clusters','effective_mass'},'saved NPZ arm schema mismatch')
        require(np.array_equal(stored['labels'],labels) and np.array_equal(stored['clusters'],clusters),'saved labels/source ordering mismatch')
        for arm in ARMS:close(stored[arm],probabilities[arm],arm)
        close(stored['effective_mass'],mass,'effective mass')
    query_ids=[dict(domain=domain,source_component_id=r['source_component_id'],problem_id=r['problem_id'],
        candidate_id=r['task_id'],test_id=t['id'],test_index=i) for r in assessment
        for i,t in enumerate(r['tests']) if i>=visible]
    require(report.get('counts')==dict(validation_records=len(dev),assessment_records=len(assessment),
        assessment_examples=len(query_ids),assessment_sources=len(set(clusters))),'report count mismatch')
    receipt=dict(schema='eesd-mechanism-artifact-verification-v1',scope='six-saved-arms-only',
        complete_sha256=file_sha(report_dir/'complete.json'),report_sha256=complete['report_sha256'],
        predictions_sha256=complete['predictions_sha256'],cache_sha256=file_sha(cache),config_sha256=file_sha(config),
        cache_binding_sha256=file_sha(cache.with_suffix('.binding.json')),evaluator_tasks_sha256=tasks_sha,
        evaluator_manifest_sha256=manifest_sha,runner_source_sha256=FROZEN_RUNNER_SHA256,
        evidence_source_sha256=FROZEN_EVIDENCE_SHA256,loader_source_sha256=file_sha(Path(__file__)),
        targets_per_candidate=10-visible,bank_bindings=bindings,cache_sources=seal['sources'],
        ordered_query_ids_sha256=digest_json(query_ids),ordered_input_bindings_sha256=digest_json(bound_records),
        rtol=RTOL,atol=ATOL,domain=domain,model=model,seed=seed,arms=list(ARMS))
    data=dict(labels=labels,probabilities=probabilities,sources=clusters,seeds=np.full(len(labels),seed,dtype=np.int64),
        query_ids=query_ids,domain=domain,model=model,seed=seed)
    return data,receipt
