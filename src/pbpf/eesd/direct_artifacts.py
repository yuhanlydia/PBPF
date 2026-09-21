"""Direct-profile provenance; reuse immutable six-arm and 49-contrast functions.

Private FunctionType globals replace only provenance callbacks. No old cache
binding is synthesized and no original module globals/source are changed. Public
support checks sealed bytes/ordered inputs without accessing assessment labels.
"""
import hashlib
from pathlib import Path
import runpy
from types import FunctionType
from datetime import datetime
import yaml
from pbpf.apbpf.codearc_bank import load_bank
from . import direct_runtime as runtime,replay_artifacts as reuse,mechanism_contrasts as contrasts
from .mechanism_artifacts import read,require,file_sha
ROOT=Path(__file__).resolve().parents[3]


def isolated(fn,**overrides):
    namespace=dict(fn.__globals__);namespace.update(overrides)
    return FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)


def source_inventory():
    names=set(runtime.SOURCES)|{'src/pbpf/eesd/direct_artifacts.py','src/pbpf/eesd/replay_artifacts.py',
        'src/pbpf/eesd/mechanism_artifacts.py','src/pbpf/eesd/mechanism_contrasts.py',
        'src/pbpf/eesd/mechanism_inference.py','scripts/report_eesd_mechanism_inference.py',
        'scripts/report_eesd_direct_inference.py','src/pbpf/eesd/replay_execution_cache.py',
        'src/pbpf/eesd/replay_generation.py','src/pbpf/eesd/replay_prompt.py','src/pbpf/eesd/replay_materialization.py'}
    return {n:file_sha(ROOT/n) for n in sorted(names)}


def verify_direct_cell(directory,cell,seed,config,execution_lock,lock_sha,*,public_only=True):
    directory=Path(directory);config=Path(config)
    lock=runtime.validate_execution_lock(execution_lock,lock_sha)
    require(lock['config']['sha256']==file_sha(config) and Path(lock['config']['path']).resolve()==config.resolve(),'direct config differs from lock')
    manifest=yaml.safe_load(Path(lock['manifest']['path']).read_text())
    require(sum(c==cell for c in manifest['mechanism_cells'])==1,'external cell differs from locked manifest')
    complete=read(directory/'complete.json');binding=read(directory/'direct-binding.json')
    names={'cache.json','direct-binding.json','report/complete.json','report/report.json','report/predictions.npz'}
    require(complete.get('schema')=='eesd-direct-mechanism-complete-v1' and complete.get('execution_profile')=='direct-no-sandbox'
        and set(complete['files'])==names and {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}==names|{'complete.json'},'direct inventory mismatch')
    for name,value in complete['files'].items():require(file_sha(directory/name)==value,'direct seal checksum mismatch')
    require(binding.get('schema')=='eesd-direct-mechanism-binding-v1' and binding.get('execution_profile')=='direct-no-sandbox'
        and all(binding.get(k)==v for k,v in [('domain',cell['domain']),('dataset',cell['dataset']),('family',cell['family']),('seed',seed)])
        and binding.get('execution_lock')=={'path':str(Path(execution_lock).resolve()),'sha256':lock_sha}
        and binding.get('sources')==lock['sources'],'direct external profile/identity mismatch')
    require(type(binding['workers']) is int and 1<=binding['workers']<=lock['max_workers'],'direct worker limit mismatch')
    ready=binding['readiness'];require(ready.get('schema')=='eesd-direct-execution-readiness-v1' and ready.get('status')=='ready'
        and ready.get('profile')=='direct-no-sandbox' and ready.get('execution_lock_sha256')==lock_sha
        and ready.get('sources')==lock['sources'] and set(ready.get('probes',{}))=={'stdin','call'},'direct readiness mismatch')
    require(datetime.fromisoformat(ready['checked_at']).utcoffset() is not None,'readiness timestamp needs timezone')
    require(all(p.get('outcome')=='PASS' and p.get('returncode')==0 and p.get('timed_out') is False for p in ready['probes'].values()),'direct readiness probe failed')
    watched={lock['manifest']['path']:lock['manifest']['sha256'],str(config.resolve()):file_sha(config),str(Path(execution_lock).resolve()):lock_sha}
    for field in ('public_root','evaluator_root'):
        for name in ('manifest.json','tasks.jsonl'):
            path=(Path(cell[field])/name).resolve();watched[str(path)]=file_sha(path)
    evaluator=Path(cell['evaluator_root']);em=read(evaluator/'manifest.json')
    require(em['evaluator_tasks_sha256']==file_sha(evaluator/'tasks.jsonl'),'evaluator checksum mismatch')
    import json
    tasks_list=[json.loads(line) for line in (evaluator/'tasks.jsonl').read_text().splitlines() if line.strip()]
    tasks={r['task_id']:r for r in tasks_list};require(len(tasks)==len(tasks_list),'duplicate evaluator task')
    verifier=runpy.run_path(str(ROOT/'scripts/run_eesd_matrix.py'));projected=[];seen=set();identity=None
    require(len(binding['banks'])==2 and [b['split'] for b in binding['banks']]==['development','primary'],'direct bank order mismatch')
    for b,count in zip(binding['banks'],(200,500),strict=True):
        path=Path(b['path']);_,info,digest=verifier['verify_mechanism_bank'](path,cell=cell,seed=seed,split=b['split'],root=ROOT)
        require(b['components']==count and info['components']==count and b['complete_sha256']==digest,'direct bank coverage mismatch')
        run,rows,_=load_bank(path);current=[run['model'],run['revision'],run.get('adapter_sha256'),run['seed']]
        require(identity is None or identity==current,'direct bank policy mismatch');identity=current
        for p in path.iterdir():watched[str(p.resolve())]=file_sha(p)
        for row in rows:
            source=row['source_component_id'];require(source not in seen,'source overlap');seen.add(source)
            t=tasks[row['task_id']];require(t['source_component_id']==source and t['split']==row['split'],'evaluator identity mismatch')
            tests=t['tests'];require(len(tests)==10 and [str(t['id']) for t in tests]==list(map(str,range(10))),'ordered ten tests required')
            candidate=row['candidates'][0]
            projected.append(dict(task_id=candidate['candidate_id'],problem_id=row['task_id'],source_component_id=source,split=row['split'],
                candidate_code_sha256=hashlib.sha256(candidate['code'].encode()).hexdigest(),tests=[{'id':str(t['id']),'input':t['input']} for t in tests]))
    require(len(projected)==700 and {s:sum(r['split']==s for r in projected) for s in ('development','primary')}=={'development':200,'primary':500},'direct projected population mismatch')
    require(binding['inputs']==watched,'direct watched input inventory mismatch')
    for name,value in watched.items():require(file_sha(name)==value,'direct input changed')
    cache=read(directory/'cache.json');require(binding['cache_sha256']==file_sha(directory/'cache.json') and binding['report_complete_sha256']==file_sha(directory/'report/complete.json'),'direct artifact binding mismatch')
    require(cache.get('schema')=='eesd-public-query-mechanism-cache-v1' and cache.get('dataset')==cell['domain']
        and cache.get('generator_identity')==identity and cache.get('evaluator_manifest_sha256')==file_sha(evaluator/'manifest.json')
        and cache.get('bank_bindings')==binding['banks'] and cache.get('counts')=={'development':200,'primary':500}
        and cache.get('source_counts')==cache['counts'] and cache.get('tests_per_candidate')==10,'direct cache metadata mismatch')
    require(len(cache['records'])==len(projected),'direct cache count mismatch')
    for row,expected in zip(cache['records'],projected,strict=True):
        require(set(row)==set(expected)|{'outcomes'} and all(row[k]==v for k,v in expected.items()),'direct ordered query identity mismatch')
        if not public_only:require(len(row['outcomes'])==10 and all(v in ('PASS','WRONG_OUTPUT','COMPILE_ERROR','RUNTIME_EXCEPTION','TIMEOUT') for v in row['outcomes']),'direct outcome schema mismatch')
    verifier['verify_mechanism_report'](directory/'report',cache=directory/'cache.json',config=config,cell=cell,seed=seed,root=ROOT)
    return cache,dict(execution_profile='direct-no-sandbox',direct_complete_sha256=file_sha(directory/'complete.json'),
        direct_binding_sha256=file_sha(directory/'direct-binding.json'),cache_sha256=file_sha(directory/'cache.json'),config_sha256=file_sha(config),
        execution_lock_sha256=lock_sha,execution_amendment=lock['amendment'],sources=source_inventory())


def context(kwargs,public_only):
    directory=Path(kwargs['cache']).parent
    require(Path(kwargs['report_dir']).resolve()==(directory/'report').resolve(),'direct report/cache layout mismatch')
    cell=kwargs['cell'];require(kwargs['domain']==cell['domain'] and kwargs['model']==cell['family'],'direct loader cell mismatch')
    cache,binding=verify_direct_cell(directory,cell,kwargs['seed'],kwargs['config'],kwargs['execution_lock'],kwargs['lock_sha'],public_only=public_only)
    reuse.source_bindings(ROOT)
    report=read(directory/'report/report.json');complete=read(directory/'report/complete.json')
    cfg=yaml.safe_load(Path(kwargs['config']).read_text());runner=runpy.run_path(str(ROOT/'scripts/run_eesd_evidence_matrix.py'))
    return cache,report,cfg,runner,binding,complete


def load_direct_artifacts(**kwargs):
    ctx=context(kwargs,False)
    fn=isolated(reuse.load_replay_artifacts,_context=lambda **ignored:ctx,
        verify_replay_evidence_inputs=lambda **ignored:{'direct_complete_sha256':ctx[4]['direct_complete_sha256']})
    data,receipt=fn(**{**kwargs,'expected_inputs':{},'expected_readiness':{}})
    receipt['schema']='eesd-direct-artifacts-v1';return data,receipt


def derive_public_support(**kwargs):
    ctx=context(kwargs,True)
    result,receipt=isolated(reuse.derive_public_support,_context=lambda **ignored:ctx)(**kwargs)
    receipt['schema']='eesd-direct-public-support-v1';return result,receipt


def reconstruct_contrasts(**kwargs):
    reuse.source_bindings(ROOT)
    result,receipt=isolated(contrasts.reconstruct_contrasts,load_mechanism_artifacts=load_direct_artifacts)(**kwargs)
    receipt.update(schema='eesd-direct-contrasts-v1',execution_profile='direct-no-sandbox',adapter_sources=source_inventory())
    return result,receipt
