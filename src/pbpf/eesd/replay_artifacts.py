"""Replay provenance adapter around immutable evidence/49-contrast mathematics.

The original 49-function source SHA is FROZEN_CONTRASTS_SHA256 and is bound in
all sidecars/receipts. FunctionType uses a private globals copy replacing only
load_mechanism_artifacts; it never mutates the original module or its functions.
Trusted callers must validate the statistical/execution locks and load actual
sealed Replay execution inputs. Public support verifies bytes and ordered public
projections without accessing assessment outcome fields; it is not full label
verification. No candidate execution or readiness probe occurs in this module.
"""
from pathlib import Path
import hashlib
import json
import runpy
from types import FunctionType
import numpy as np
import yaml
from . import mechanism_contrasts as contrasts, replay_execution_cache as execution
from .mechanism_artifacts import (ARMS,FROZEN_RUNNER_SHA256,FROZEN_EVIDENCE_SHA256,
                                  close,require,file_sha,read,digest_json)

ROOT=Path(__file__).resolve().parents[3]
FROZEN_CONTRASTS_SHA256='678ad3cdd12d0c6c9e77b0cd74779207d3998f49ab8ec7e342a344d596e8ce6f'


def source_bindings(root=None):
    root=Path(root or ROOT)
    sources={'runner':file_sha(root/'scripts/run_eesd_evidence_matrix.py'),
             'evidence':file_sha(root/'src/pbpf/eesd/evidence.py'),
             'contrasts':file_sha(contrasts.__file__),'replay_artifacts':file_sha(__file__)}
    require(sources['runner']==FROZEN_RUNNER_SHA256 and sources['evidence']==FROZEN_EVIDENCE_SHA256
            and sources['contrasts']==FROZEN_CONTRASTS_SHA256,'frozen evidence/contrast source changed')
    from . import evidence
    require(file_sha(evidence.__file__)==FROZEN_EVIDENCE_SHA256,'imported evidence source changed')
    return sources


def _public_cache(cache,expected_inputs,expected_readiness):
    """Validate sealed bytes and external inputs, never assessment outcome values."""
    cache=Path(cache);directory=cache.parent
    require(cache.name=='cache.json','Replay cache filename must be cache.json')
    execution.check_inputs(expected_inputs);execution.validate_readiness(expected_readiness,expected_inputs)
    require({p.name for p in directory.iterdir()}=={'cache.json','binding.json','complete.json'},'cache inventory mismatch')
    complete=read(directory/'complete.json')
    require(complete.get('schema')=='eesd-replay-cache-complete-v1' and set(complete['files'])=={'cache.json','binding.json'},'cache complete schema mismatch')
    for name,seal in complete['files'].items():execution.verify_file(directory/name,seal)
    binding=read(directory/'binding.json')
    require(binding==dict(schema='eesd-replay-cache-binding-v1',inputs=expected_inputs['bindings'],readiness=expected_readiness),'cache external bindings mismatch')
    payload=read(cache);records=payload['records'];identity=expected_inputs['identity']
    require(payload.get('schema')=='eesd-replay-public-query-cache-v1' and payload.get('task_kind')=='stdin_synthesis'
        and all(payload.get(k)==v for k,v in identity.items()) and payload.get('tests_per_candidate')==10
        and payload.get('observation_indices')==[0,1,2,3] and payload.get('target_indices')==[4,5,6,7,8,9]
        and payload.get('counts')==execution.generation.COUNTS and payload.get('source_counts')==execution.generation.COUNTS
        and payload.get('input_bindings_sha256')==execution.digest(expected_inputs['bindings']),'cache protocol/identity mismatch')
    require(len(records)==len(expected_inputs['jobs']),'cache record count mismatch')
    seen=set()
    for row,job in zip(records,expected_inputs['jobs'],strict=True):
        projected=execution.projected_record(job)
        require(set(row)==set(projected)|{'outcomes'} and all(row[k]==v for k,v in projected.items()),'cache ordered public identity mismatch')
        source=row['source_component_id'];require(source not in seen,'overlapping/duplicate source');seen.add(source)
        require(len(row['tests'])==10 and [t['id'] for t in row['tests']]==list(map(str,range(10))),'ordered target count mismatch')
    return payload


def verify_public_cache_projection(cache,expected_inputs,expected_readiness):
    """Public-stage validation for existing caches even before reports exist."""
    return _public_cache(cache,expected_inputs,expected_readiness)


def verify_replay_evidence_inputs(*,cache_dir,expected_inputs,expected_readiness):
    return execution.verify_replay_cache(cache_dir,expected_inputs=expected_inputs,expected_readiness=expected_readiness)


def build_report_binding(*,cache,config,expected_inputs,expected_readiness,expected_bindings,root=None):
    cache=Path(cache)
    require(set(expected_bindings)=={'statistical_lock_sha256','execution_lock_sha256'}
        and all(execution.is_sha(v) for v in expected_bindings.values()),'explicit statistical/execution lock binding required')
    require(expected_bindings['execution_lock_sha256']==expected_readiness['execution_lock_sha256'],'execution lock/readiness mismatch')
    return dict(schema='eesd-replay-evidence-binding-v1',cache_sha256=file_sha(cache),
        cache_complete_sha256=file_sha(cache.parent/'complete.json'),cache_binding_sha256=file_sha(cache.parent/'binding.json'),
        config_sha256=file_sha(config),input_bindings_sha256=execution.digest(expected_inputs['bindings']),
        readiness_sha256=execution.digest(expected_readiness),**expected_bindings,sources=source_bindings(root))


def _context(*,report_dir,cache,config,domain,model,seed,expected_inputs,expected_readiness,expected_bindings,root=None):
    root=Path(root or ROOT);report_dir=Path(report_dir);cache=Path(cache);config=Path(config)
    require(domain in execution.generation.DOMAINS and model in execution.generation.MODELS
        and type(seed) is int and seed in (1701,1702,1703),'unsupported Replay report identity')
    require(all(expected_inputs['identity'].get(k)==v for k,v in [('domain',domain),('family',model),('seed',seed)]),'external cell identity mismatch')
    binding=build_report_binding(cache=cache,config=config,expected_inputs=expected_inputs,
        expected_readiness=expected_readiness,expected_bindings=expected_bindings,root=root)
    require(read(report_dir/'replay-binding.json')==binding,'Replay report binding mismatch')
    payload=_public_cache(cache,expected_inputs,expected_readiness)
    complete=read(report_dir/'complete.json')
    require(complete.get('report_sha256')==file_sha(report_dir/'report.json')
        and complete.get('predictions_sha256')==file_sha(report_dir/'predictions.npz'),'report completion checksum mismatch')
    report=read(report_dir/'report.json');cfg=yaml.safe_load(config.read_text())
    require(cfg.get('schema')=='eesd-iclr2027-v1' and seed in cfg.get('seeds',[]),'config schema/seed mismatch')
    expected=dict(schema='eesd-evidence-matrix-v1',dataset=domain,model=model,seed=seed,visible=4,
        validation_split='development',assessment_split='primary',cache_sha256=file_sha(cache),config_sha256=file_sha(config),
        runner_source_sha256=FROZEN_RUNNER_SHA256,evidence_source_sha256=FROZEN_EVIDENCE_SHA256)
    require(all(report.get(k)==v for k,v in expected.items()),'report identity mismatch')
    runner=runpy.run_path(str(root/'scripts/run_eesd_evidence_matrix.py'))
    return payload,report,cfg,runner,binding,complete


def _query_ids(records,n,domain):
    return [dict(domain=domain,source_component_id=r['source_component_id'],problem_id=r['problem_id'],
        candidate_id=r['task_id'],test_id=t['id'],test_index=i) for r in records for i,t in enumerate(r['tests']) if i>=n]


def load_replay_artifacts(**kwargs):
    payload,report,cfg,runner,binding,complete=_context(**kwargs)
    verification=verify_replay_evidence_inputs(cache_dir=Path(kwargs['cache']).parent,
        expected_inputs=kwargs['expected_inputs'],expected_readiness=kwargs['expected_readiness'])
    rows=payload['records'];dev=[r for r in rows if r['split']=='development'];assessment=[r for r in rows if r['split']=='primary']
    runner['_validate_rows'](dev,4);runner['_validate_rows'](assessment,4)
    memo={}
    def examples(split,h,strength,binary):
        key=(split,h,strength,binary)
        if key not in memo:memo[key]=runner['build_examples'](dev if split=='validation' else assessment,h,strength,binary=binary)
        return memo[key]
    e=cfg['evidence'];selected={};probabilities={};mass=None
    for arm,rule in [('ordinary','ordinary'),('fixed','fixed'),('effective','effective'),('global_mass','global')]:
        selection,_=runner['select_arm'](examples,'validation',4,e['alphas'],[0.] if arm=='ordinary' else e['strengths'],rule,
            masses=e['global_mass_grid'] if arm=='global_mass' else None,ece_bins=e['ece_bins'])
        actual=report['selected'][arm];require(set(actual)==set(selection),f'{arm} selection schema mismatch')
        for key,value in selection.items():
            if value is None:require(actual[key] is None,f'{arm} selection mismatch')
            else:close(actual[key],value,f'{arm} selection {key}')
        selected[arm]=selection
        probabilities[arm],masses=runner['predictions'](examples('assessment',4,selection['strength'],False),selection['alpha'],rule,global_mass=selection['mass'])
        if arm=='effective':mass=masses
    for arm,selection,rule in [('fixed_at_effective_params','effective','fixed'),('effective_at_fixed_params','fixed','effective')]:
        s=selected[selection];probabilities[arm],_=runner['predictions'](examples('assessment',4,s['strength'],False),s['alpha'],rule)
    labels,clusters=runner['labels_clusters'](examples('assessment',4,selected['effective']['strength'],False))
    with np.load(Path(kwargs['report_dir'])/'predictions.npz',allow_pickle=False) as stored:
        require(set(stored.files)==set(ARMS)|{'labels','clusters','effective_mass'},'NPZ arm inventory mismatch')
        require(np.array_equal(stored['labels'],labels) and np.array_equal(stored['clusters'],clusters),'NPZ labels/source order mismatch')
        for arm in ARMS:close(stored[arm],probabilities[arm],arm)
        close(stored['effective_mass'],mass,'effective mass')
    ids=_query_ids(assessment,4,kwargs['domain'])
    require(report.get('counts')==dict(validation_records=len(dev),assessment_records=len(assessment),assessment_examples=len(ids),assessment_sources=len(set(clusters))),'report counts mismatch')
    receipt=dict(schema='eesd-replay-artifacts-v1',scope='six-saved-arms-only',cache_sha256=binding['cache_sha256'],config_sha256=binding['config_sha256'],
        report_sha256=complete['report_sha256'],predictions_sha256=complete['predictions_sha256'],binding=binding,
        cache_verification=verification,ordered_query_ids_sha256=digest_json(ids))
    return dict(labels=labels,probabilities=probabilities,sources=clusters,seeds=np.full(len(labels),kwargs['seed'],dtype=np.int64),
        query_ids=ids,domain=kwargs['domain'],model=kwargs['model'],seed=kwargs['seed']),receipt


def derive_public_support(**kwargs):
    payload,report,cfg,runner,binding,_=_context(**kwargs);e=cfg['evidence']
    require(tuple(e['alphas'])==contrasts.ALPHAS and tuple(e['strengths'])==contrasts.STRENGTHS
        and e['concentration_bins_for_n4']==[1,2,3,4],'public support grids mismatch')
    dev=[];assessment=[]
    for row in payload['records']:
        public={k:row[k] for k in ('task_id','problem_id','source_component_id','split','tests')}
        if row['split']=='development':public['outcomes']=row['outcomes'];dev.append(public)
        else:public['outcomes']=['PASS']*10;assessment.append(public)
    memo={}
    def examples(split,n,strength,binary):
        require(split=='validation','public support tuning must be development only')
        key=(n,strength,binary)
        if key not in memo:memo[key]=runner['build_examples'](dev,n,strength,binary=binary)
        return memo[key]
    selection,_=runner['select_arm'](examples,'validation',4,e['alphas'],e['strengths'],'effective',ece_bins=e['ece_bins'])
    saved=report['selected']['effective'];require(set(saved)==set(selection),'public selection schema mismatch')
    for key,value in selection.items():
        if value is None:require(saved[key] is None,'public development selection mismatch')
        else:close(saved[key],value,f'public development selection {key}')
    ex=runner['build_examples'](assessment,4,selection['strength'],binary=False)
    _,mass=runner['predictions'](ex,selection['alpha'],'effective')
    ids=_query_ids(assessment,4,kwargs['domain']);encoded=[json.dumps(q,sort_keys=True,separators=(',',':')) for q in ids]
    result={}
    for name,(lo,hi) in zip(contrasts.BIN_IDS,contrasts.BINS,strict=True):
        result[name]=dict(sources=[q['source_component_id'] for q in ids],seeds=[kwargs['seed']]*len(ids),query_ids=encoded,
            public_mask=(mass>=lo)&((mass<=hi) if hi==4 else (mass<hi)),public_mass=mass.copy(),bin_edges=[lo,hi],upper_inclusive=hi==4)
    receipt=dict(schema='eesd-replay-public-support-v1',assessment_outcomes_accessed=False,binding=binding,
        development_selected_effective=selection,ordered_query_ids_sha256=digest_json(ids),
        public_mask_sha256={k:digest_json(v['public_mask'].tolist()) for k,v in result.items()},
        public_mass_sha256=digest_json(mass.tolist()),trust_boundary='sealed bytes and caller-validated external inputs; full label verification follows public support sealing')
    return result,receipt


def reconstruct_contrasts(**kwargs):
    source_bindings(kwargs.get('root'))
    original=contrasts.reconstruct_contrasts
    namespace=dict(original.__globals__);namespace['load_mechanism_artifacts']=load_replay_artifacts
    isolated=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    result,receipt=isolated(**kwargs)
    receipt={**receipt,'schema':'eesd-replay-contrasts-v1','original_contrasts_sha256':FROZEN_CONTRASTS_SHA256,
             'adapter_source_sha256':file_sha(__file__)}
    return result,receipt
