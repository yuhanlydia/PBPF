"""Trusted Replay cache adapter. No readiness or execution occurs on import.

Readiness is supplied by a trusted caller; validating its structure is not a
claim that this library independently demonstrated working namespace isolation.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from . import replay_generation as generation
from .replay_admission import rename_new_directory
from .replay_materialization import _normalized_io

ROOT=Path(__file__).resolve().parents[3]
OUTCOMES=('PASS','WRONG_OUTPUT','COMPILE_ERROR','RUNTIME_EXCEPTION','TIMEOUT')
PROFILE={'builder_python':'3.11.16','candidate_python':'3.10.12','bubblewrap':'0.6.1','timeout_seconds':6.0}
GENERATION_SOURCES=('scripts/generate_eesd_replay_bank.py','scripts/plan_eesd_replay_generation.py',
 'src/pbpf/eesd/replay_generation.py','src/pbpf/eesd/replay_prompt.py','src/pbpf/apbpf/rbr_prompt.py')
EXECUTION_SOURCES=('src/pbpf/eesd/replay_execution_cache.py','src/pbpf/apbpf/rbr_execution.py',
 'src/pbpf/apbpf/codearc_execution.py','src/pbpf/eesd/replay_materialization.py','src/pbpf/eesd/replay_admission.py')

def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=False,separators=(',',':')).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):return generation.sha(path)
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def require(value,message):
    if not value:raise ValueError(message)
def is_sha(value):return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)
def verify_file(path,seal):
    require(sha(path)==seal['sha256'] and Path(path).stat().st_size==seal['bytes'],'input file checksum/size mismatch')


def load_replay_execution_inputs(bundle,admission_sha256,domain,family,seed,development_bank,primary_bank,*,generation_manifest,generation_manifest_sha256,extra_source_paths=()):
    """Verify both complete banks before opening any evaluator data.

    This trusted path can read evaluator outputs. It must never be called by the
    candidate generator. Returned jobs contain private expected output strings.
    """
    bundle=Path(bundle).resolve();manifest_path=Path(generation_manifest).resolve()
    require(domain in generation.DOMAINS and family in generation.MODELS and type(seed) is int and seed in (1701,1702,1703),'invalid cache cell identity')
    require(sha(bundle/'admission.json')==admission_sha256,'admission checksum mismatch')
    require(sha(manifest_path)==generation_manifest_sha256,'generation manifest checksum mismatch')
    manifest=read(manifest_path)
    require(manifest['schema']=='eesd-replay-generation-matrix-v1' and manifest['status']=='planned-generation-only-not-launched','wrong generation manifest')
    require(Path(manifest['bundle']).resolve()==bundle and manifest['admission_sha256']==admission_sha256,'manifest admission mismatch')
    require(set(manifest['sources'])==set(GENERATION_SOURCES),'generation source inventory mismatch')
    for name,value in manifest['sources'].items():require(sha(ROOT/name)==value,'generation source changed')
    cells=[cell for cell in manifest['cells'] if (cell['domain'],cell['family'],cell['seed'])==(domain,family,seed)]
    require(len(cells)==1 and set(cells[0]['splits'])==set(generation.COUNTS),'missing/duplicate generation cell')
    paths={'development':Path(development_bank).resolve(),'primary':Path(primary_bank).resolve()}
    preflights={};banks={};bank_bindings=[]
    watched={str(bundle/'admission.json'):admission_sha256,str(manifest_path):generation_manifest_sha256}
    execution_paths=set(EXECUTION_SOURCES)
    require(isinstance(extra_source_paths,(list,tuple,set,frozenset)),'extra source paths must be a collection')
    for name in extra_source_paths:
        path=(ROOT/Path(name)).resolve()
        require(path.is_relative_to(ROOT),'extra execution source outside repository')
        execution_paths.add(str(path.relative_to(ROOT)))
    for name in (*GENERATION_SOURCES,*sorted(execution_paths)):watched[str(ROOT/name)]=sha(ROOT/name)
    for split,count in generation.COUNTS.items():
        planned=cells[0]['splits'][split];path=paths[split]
        require(Path(planned['output']).resolve()==path and planned['components']==count,'bank path/count differs from planned split')
        p=generation.load_public_split(bundle,admission_sha256,domain,split,family)
        expected=generation.build_expected(p,family,seed,ROOT/'scripts/generate_eesd_replay_bank.py')
        bank=generation.verify_replay_bank(path,expected,preflight=p)
        require(len(p['rows'])==count and len(bank['rows'])==count,'incomplete generation population')
        preflights[split]=p;banks[split]=bank
        bank_bindings.append(dict(path=str(path),split=split,components=count,run_sha256=sha(path/'run.json'),complete_sha256=bank['complete_sha256']))
        for file in path.iterdir():watched[str(file)]=sha(file)
    # Deliberately below BOTH sealed-bank validations.
    admission=read(bundle/'admission.json');private_root=bundle/domain/'evaluator'
    for view in ('public','evaluator'):
        for file in ('manifest.json','tasks.jsonl'):
            path=bundle/domain/view/file;seal=admission['outputs'][f'{domain}/{view}/{file}']
            verify_file(path,seal);watched[str(path)]=seal['sha256']
    private_manifest=read(private_root/'manifest.json')
    require(private_manifest==dict(schema='eesd-replay-evaluator-v1',domain=domain,task_kind='stdin_synthesis',counts=generation.COUNTS,tasks=admission['outputs'][f'{domain}/evaluator/tasks.jsonl']),'evaluator manifest identity mismatch')
    private=rows(private_root/'tasks.jsonl');indexed={r['task_id']:r for r in private}
    require(len(indexed)==len(private)==700,'evaluator population duplicates/count mismatch')
    jobs=[];sources=set();task_ids=set()
    for split in generation.COUNTS:
        for public,record in zip(preflights[split]['rows'],banks[split]['rows'],strict=True):
            task=indexed.get(public['task_id']);require(task is not None,'missing evaluator task')
            require(all(task.get(key)==public[key] for key in ('task_id','source_component_id','domain','split','statement')),'public/evaluator identity or statement mismatch')
            source=public['source_component_id'];task_id=public['task_id']
            require(source not in sources and task_id not in task_ids,'reused source/task');sources.add(source);task_ids.add(task_id)
            tests=task.get('tests');known=task.get('known_exposed_input_sha256')
            require(isinstance(tests,list) and len(tests)==10,'ten ordered tests required')
            require(isinstance(known,list) and len(set(known))==len(known) and all(is_sha(x) for x in known),'invalid exposed input inventory')
            seen=set();ordered=[]
            pools={'apps_input_output'} if domain=='apps_replay' else {'public_tests','private_tests','generated_tests'}
            for i,test in enumerate(tests):
                inp,out=test.get('input'),test.get('output')
                require(test.get('id')==str(i) and isinstance(inp,str) and isinstance(out,str) and len(inp)<=8192 and len(out)<=4096,'invalid ordered raw IO')
                normalized=_normalized_io(inp);require(normalized not in seen,'duplicate normalized test input');seen.add(normalized)
                require(test.get('pair_sha256')==hashlib.sha256((normalized+'\0'+_normalized_io(out)).encode()).hexdigest(),'pair identity mismatch')
                original=test.get('original_pools')
                require(isinstance(original,list) and bool(original) and all(isinstance(x,str) and x in pools for x in original) and len(set(original))==len(original),'invalid test provenance')
                exposed=hashlib.sha256(normalized.encode()).hexdigest() in known
                require(type(test.get('known_exposed_input')) is bool and test['known_exposed_input']==exposed,'exposure flag mismatch')
                if i<4:require(public['visible_tests'][i]=={'input':inp,'output':out},'public observation differs from raw evaluator IO')
                else:require(not exposed,'target input is already exposed')
                ordered.append({'id':str(i),'input':inp,'expected':out})
            candidate=record['candidates'][0]
            jobs.append(dict(task_id=task_id,source_component_id=source,split=split,candidate_id=candidate['candidate_id'],code=candidate['code'],tests=ordered))
    require(task_ids==set(indexed),'extra evaluator tasks')
    for name in ('selected-token-audit.jsonl',):watched[str(bundle/name)]=sha(bundle/name)
    execution_sources={name:watched[str(ROOT/name)] for name in sorted(execution_paths)}
    bindings={'admission_sha256':admission_sha256,'generation_manifest_sha256':generation_manifest_sha256,
        'banks':bank_bindings,'public_bindings':{split:preflights[split]['bindings'] for split in generation.COUNTS},
        'execution_sources':execution_sources,'watched_files':watched,'ordered_jobs_sha256':digest(jobs)}
    inputs={'jobs':jobs,'bindings':bindings,'identity':{'domain':domain,'family':family,'seed':seed,
        'model':preflights['development']['model']['model_id'],'revision':preflights['development']['model']['revision'],'adapter':None}}
    check_inputs(inputs)
    return inputs


def check_inputs(inputs):
    require(digest(inputs['jobs'])==inputs['bindings']['ordered_jobs_sha256'],'private job inputs changed')
    for file,expected in inputs['bindings']['watched_files'].items():require(sha(file)==expected,'source/input changed during execution')
    require(len(inputs['jobs'])==700 and {s:sum(j['split']==s for j in inputs['jobs']) for s in generation.COUNTS}==generation.COUNTS,'incomplete execution population')


def validate_readiness(receipt,inputs):
    require(isinstance(receipt,dict) and receipt.get('schema')=='eesd-replay-execution-readiness-v1' and receipt.get('status')=='ready','execution readiness missing/not ready')
    require(is_sha(receipt.get('execution_lock_sha256')) and receipt.get('profile')==PROFILE,'execution readiness lock/profile mismatch')
    sources=receipt.get('sources')
    require(isinstance(sources,dict) and bool(sources) and sources==inputs['bindings']['execution_sources'] and all(is_sha(v) for v in sources.values()),'execution readiness source mismatch')


def projected_record(job):
    return dict(task_id=job['candidate_id'],problem_id=job['task_id'],source_component_id=job['source_component_id'],split=job['split'],candidate_code_sha256=hashlib.sha256(job['code'].encode()).hexdigest(),tests=[{'id':t['id'],'input':t['input']} for t in job['tests']])


def measure_job(job,executor=None):
    """Ten ordered executions; call only after the parent validates readiness.

    Top-level helper is picklable by a spawn ProcessPoolExecutor. A failed test
    never short-circuits the remaining tests; infrastructure exceptions propagate.
    """
    if executor is None:
        from pbpf.apbpf.rbr_execution import execute_stdin
        executor=execute_stdin
    outcomes=[]
    for test in job['tests']:
        result=executor(job['code'],{'input':test['input'],'expected':test['expected']},timeout=6.0)
        require(isinstance(result,dict) and result.get('outcome') in OUTCOMES,'malformed executor result')
        require(isinstance(result.get('stderr'),str) and isinstance(result.get('stdout'),str),'missing executor diagnostics')
        if result['stderr'].startswith('bwrap:'):raise RuntimeError('sandbox infrastructure failure')
        require(type(result.get('timed_out')) is bool and type(result.get('stdout_truncated')) is bool
            and 'returncode' in result and (result['returncode'] is None or type(result['returncode']) is int),'malformed executor structure')
        outcome=result['outcome'];rc=result['returncode'];timed=result['timed_out']
        require((outcome=='COMPILE_ERROR' and rc is None and not timed)
            or (outcome=='TIMEOUT' and timed and type(rc) is int)
            or (outcome in ('PASS','WRONG_OUTPUT') and rc==0 and not timed)
            or (outcome=='RUNTIME_EXCEPTION' and type(rc) is int and rc!=0 and not timed),'inconsistent executor structure/category')
        outcomes.append(result['outcome'])
    return {**projected_record(job),'outcomes':outcomes}


def measure_jobs(inputs,*,readiness_check,executor=None,job_mapper=None):
    """Validate readiness before invoking a lazy serial/process-pool mapper.

    The caller's mapper accepts jobs and returns records in the same order. Pool
    creation belongs inside that callback, after readiness; never use threads
    around the immutable executor's preexec_fn. No publication occurs here.
    """
    check_inputs(inputs)
    require(job_mapper is None or executor is None,'supply executor or job_mapper, not both')
    receipt=readiness_check();validate_readiness(receipt,inputs)
    if job_mapper is None:
        records=[measure_job(job,executor=executor) for job in inputs['jobs']]
    else:
        records=list(job_mapper(inputs['jobs']))
    measured={'records':records,'readiness':receipt,'bindings_sha256':digest(inputs['bindings'])}
    validate_measured(inputs,measured)
    return measured


def validate_measured(inputs,measured):
    check_inputs(inputs);validate_readiness(measured['readiness'],inputs)
    require(measured['bindings_sha256']==digest(inputs['bindings']),'measurement input binding mismatch')
    records=measured['records'];require(len(records)==len(inputs['jobs']),'partial execution population')
    for job,row in zip(inputs['jobs'],records,strict=True):
        expected=projected_record(job)
        require(set(row)==set(expected)|{'outcomes'} and all(row[k]==v for k,v in expected.items()),'cache ordered record identity/redaction mismatch')
        require(isinstance(row['outcomes'],list) and len(row['outcomes'])==10 and all(x in OUTCOMES for x in row['outcomes']),'invalid execution labels')


def payloads(inputs,measured):
    validate_measured(inputs,measured)
    binding={'schema':'eesd-replay-cache-binding-v1','inputs':inputs['bindings'],'readiness':measured['readiness']}
    cache={'schema':'eesd-replay-public-query-cache-v1','task_kind':'stdin_synthesis',**inputs['identity'],
        'tests_per_candidate':10,'observation_indices':[0,1,2,3],'target_indices':[4,5,6,7,8,9],
        'counts':dict(generation.COUNTS),'source_counts':dict(generation.COUNTS),'records':measured['records'],
        'input_bindings_sha256':digest(inputs['bindings']),
        'claim_scope':'fixed Replay execution outcome calibration; not official benchmark accuracy',
        'visibility':'trusted evaluation cache: first four outcomes are observations; later outcomes are labels, never predictor features'}
    return cache,binding


def publish_cache(output,inputs,measured):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    cache,binding=payloads(inputs,measured)
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.partial-',dir=output.parent))
    try:
        seals={}
        for name,value in [('cache.json',cache),('binding.json',binding)]:
            data=canonical(value)+b'\n';(staging/name).write_bytes(data)
            seals[name]={'sha256':sha(staging/name),'bytes':len(data)}
        (staging/'complete.json').write_bytes(canonical({'schema':'eesd-replay-cache-complete-v1','files':seals})+b'\n')
        check_inputs(inputs)
        rename_new_directory(staging,output)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True);raise


def verify_replay_cache(cache_dir,*,expected_inputs,expected_readiness):
    path=Path(cache_dir);check_inputs(expected_inputs);validate_readiness(expected_readiness,expected_inputs)
    require({p.name for p in path.iterdir()}=={'cache.json','binding.json','complete.json'},'cache file inventory mismatch')
    complete=read(path/'complete.json')
    require(complete.get('schema')=='eesd-replay-cache-complete-v1' and set(complete.get('files',{}))=={'cache.json','binding.json'},'cache completion mismatch')
    for name,seal in complete['files'].items():verify_file(path/name,seal)
    cache,binding=read(path/'cache.json'),read(path/'binding.json')
    measured={'records':cache['records'],'readiness':expected_readiness,'bindings_sha256':digest(expected_inputs['bindings'])}
    expected_cache,expected_binding=payloads(expected_inputs,measured)
    require(cache==expected_cache and binding==expected_binding,'cache external identity/readiness mismatch')
    return {'schema':'eesd-replay-cache-verification-v1','counts':cache['counts'],'cache_sha256':sha(path/'cache.json'),'binding_sha256':sha(path/'binding.json'),'complete_sha256':sha(path/'complete.json')}
