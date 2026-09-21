"""Synthetic public/bank/evaluator fixtures; no real execution or namespace probe."""
import hashlib
import json
from pathlib import Path
import pytest
from pbpf.eesd import replay_execution_cache as c
from pbpf.eesd import replay_generation as g
from pbpf.eesd.replay_prompt import messages_for
from test_replay_generation import admitted, dump


def readiness(inputs):
    return dict(schema='eesd-replay-execution-readiness-v1',status='ready',execution_lock_sha256='f'*64,
                profile=c.PROFILE,sources=inputs['bindings']['execution_sources'])


def measured_result(outcome='PASS',stdout='',stderr=''):
    return dict(outcome=outcome,stdout=stdout,stderr=stderr,returncode=None if outcome=='COMPILE_ERROR' else -9 if outcome=='TIMEOUT' else 1 if outcome=='RUNTIME_EXCEPTION' else 0,timed_out=outcome=='TIMEOUT',stdout_truncated=False)

@pytest.fixture
def sealed(admitted,tmp_path,request):
    admission=json.loads((admitted/'admission.json').read_text())
    audit_path=admitted/'selected-token-audit.jsonl'
    audits=[json.loads(x) for x in audit_path.read_text().splitlines()]
    by_id={a['task_id']:a for a in audits}
    for domain in g.DOMAINS:
        public=admitted/domain/'public/tasks.jsonl'
        rows=[json.loads(x) for x in public.read_text().splitlines()]
        private=[]
        for row in rows:
            tests=[]
            for i in range(10):
                inp=f'{i}\r\n';out=f'{i+1}\r\n'
                tests.append(dict(id=str(i),input=inp,output=out,pair_sha256=hashlib.sha256(f'{i}\0{i+1}'.encode()).hexdigest(),original_pools=['apps_input_output'] if domain=='apps_replay' else ['generated_tests'],known_exposed_input=i<4))
            row['visible_tests']=[{k:t[k] for k in ('input','output')} for t in tests[:4]]
            by_id[row['task_id']]['message_sha256']=hashlib.sha256(c.canonical(messages_for(row['statement'],row['visible_tests']))).hexdigest()
            private.append({k:row[k] for k in ('task_id','source_component_id','domain','split','statement')}|dict(tests=tests,known_exposed_input_sha256=[hashlib.sha256(str(i).encode()).hexdigest() for i in range(4)]))
        if domain=='apps_replay':
            defect=getattr(request,'param',None)
            if defect=='visible_raw': private[0]['tests'][0]['input']='0\n'
            if defect=='exposed_target':
                private[0]['known_exposed_input_sha256'].append(hashlib.sha256(b'4').hexdigest())
                private[0]['tests'][4]['known_exposed_input']=True
            if defect=='duplicate_id': private[0]['tests'][9]['id']='8'
        for view,values in [('public',rows),('evaluator',private)]:
            path=admitted/domain/view/'tasks.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in values));seal={'sha256':g.sha(path),'bytes':path.stat().st_size}
            admission['outputs'][f'{domain}/{view}/tasks.jsonl']=seal
            admission['outputs'][f'{domain}/{view}/manifest.json']=dump(path.with_name('manifest.json'),dict(schema=f'eesd-replay-{view}-v1',domain=domain,task_kind='stdin_synthesis',counts={'development':200,'primary':500},tasks=seal))
    audit_path.write_text(''.join(json.dumps(r)+'\n' for r in audits));admission['outputs'][audit_path.name]={'sha256':g.sha(audit_path),'bytes':audit_path.stat().st_size};dump(admitted/'admission.json',admission)
    banks={};root=Path(__file__).resolve().parents[2]
    for split in ('development','primary'):
        p=g.load_public_split(admitted,g.sha(admitted/'admission.json'),'apps_replay',split,'qwen25_7b')
        run=g.build_expected(p,'qwen25_7b',1701,root/'scripts/generate_eesd_replay_bank.py')
        bank=tmp_path/split;bank.mkdir();dump(bank/'run.json',run);files={}
        for row in p['rows']:
            audit=p['audits'][row['task_id']];meta=audit['models']['qwen25_7b']
            rec={k:row[k] for k in ('task_id','source_component_id','domain','split')}
            rec.update(seed=g.task_seed(1701,row['task_id']),prompt_sha256=meta['rendered_prompt_sha256'],prompt_metadata={**{k:meta[k] for k in ('input_tokens','rendered_prompt_sha256','token_ids_sha256')},'message_sha256':audit['message_sha256'],'prompt_policy':'eesd-replay-synthesis-full-public-v1','input_clipped':False,'visible_observations':4},candidates=[dict(candidate_id=f"{row['task_id']}/qwen25_7b/0",code='invalid !',raw_completion='invalid !',generated_tokens=3,hit_token_cap=False)])
            path=bank/g.record_filename(row['task_id']);dump(path,rec);path.with_suffix('.sha256').write_text(g.sha(path)+'\n');files[path.name]=g.sha(path)
        dump(bank/'complete.json',dict(schema='eesd-replay-generation-complete-v1',run_sha256=g.sha(bank/'run.json'),files=files));banks[split]=bank
    manifest=tmp_path/'generation.json';dump(manifest,dict(schema='eesd-replay-generation-matrix-v1',status='planned-generation-only-not-launched',bundle=str(admitted),admission_sha256=g.sha(admitted/'admission.json'),sources={name:g.sha(root/name) for name in c.GENERATION_SOURCES},cells=[dict(domain='apps_replay',family='qwen25_7b',seed=1701,splits={split:dict(output=str(bank),components=200 if split=='development' else 500) for split,bank in banks.items()})]))
    return dict(bundle=admitted,admission_sha256=g.sha(admitted/'admission.json'),domain='apps_replay',family='qwen25_7b',seed=1701,development_bank=banks['development'],primary_bank=banks['primary'],generation_manifest=manifest,generation_manifest_sha256=g.sha(manifest))


def test_banks_verified_before_private_open(sealed,monkeypatch):
    (sealed['primary_bank']/'complete.json').unlink()
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert 'evaluator' not in path.parts,'private read before both banks sealed'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    with pytest.raises(FileNotFoundError):c.load_replay_execution_inputs(**sealed)


def test_exact_ten_calls_raw_mapping_and_redaction(sealed):
    inputs=c.load_replay_execution_inputs(**sealed);calls=[]
    def execute(code,test,*,timeout):
        assert code=='invalid !' and timeout==6.0
        assert set(test)=={'input','expected'} and test['input'].endswith('\r\n') and test['expected'].endswith('\r\n')
        calls.append(test)
        return measured_result(c.OUTCOMES[(len(calls)-1)%5],stdout='private actual',stderr='private error')
    measured=c.measure_jobs(inputs,readiness_check=lambda:readiness(inputs),executor=execute)
    assert len(calls)==7000 and len(measured['records'])==700
    assert measured['records'][0]['outcomes']==list(c.OUTCOMES)*2
    assert 'private actual' not in json.dumps(measured) and 'private error' not in json.dumps(measured)
    assert set(measured['records'][0]['tests'][0])=={'id','input'}


def test_namespace_failure_precedes_even_compile_invalid(sealed):
    inputs=c.load_replay_execution_inputs(**sealed);called=[]
    def not_ready():raise RuntimeError('namespace unavailable')
    with pytest.raises(RuntimeError,match='namespace'):
        c.measure_jobs(inputs,readiness_check=not_ready,executor=lambda *a,**k:called.append(1))
    assert called==[]

@pytest.mark.parametrize('bad',[{'outcome':'RUNTIME_EXCEPTION','stderr':'bwrap: namespace failed'}, {'outcome':'UNKNOWN'},None])
def test_infrastructure_or_malformed_result_aborts(sealed,bad):
    inputs=c.load_replay_execution_inputs(**sealed)
    with pytest.raises((ValueError,RuntimeError)):
        c.measure_jobs(inputs,readiness_check=lambda:readiness(inputs),executor=lambda *a,**k:bad)


def test_publish_verify_tamper_and_createonce(sealed,tmp_path):
    inputs=c.load_replay_execution_inputs(**sealed)
    measured=c.measure_jobs(inputs,readiness_check=lambda:readiness(inputs),executor=lambda *a,**k:measured_result())
    output=tmp_path/'cache';c.publish_cache(output,inputs,measured)
    result=c.verify_replay_cache(output,expected_inputs=inputs,expected_readiness=measured['readiness'])
    assert result['counts']=={'development':200,'primary':500}
    with pytest.raises(FileExistsError):c.publish_cache(output,inputs,measured)
    p=output/'cache.json';p.write_text(p.read_text().replace('PASS','TIMEOUT',1))
    with pytest.raises(ValueError):c.verify_replay_cache(output,expected_inputs=inputs,expected_readiness=measured['readiness'])


def test_partial_population_and_source_drift_never_publish(sealed,tmp_path):
    inputs=c.load_replay_execution_inputs(**sealed)
    measured=c.measure_jobs(inputs,readiness_check=lambda:readiness(inputs),executor=lambda *a,**k:measured_result())
    measured['records'].pop();out=tmp_path/'cache'
    with pytest.raises(ValueError):c.publish_cache(out,inputs,measured)
    assert not out.exists()
    (sealed['primary_bank']/'run.json').write_text('{}')
    with pytest.raises(ValueError):c.measure_jobs(inputs,readiness_check=lambda:readiness(inputs),executor=lambda *a,**k:None)


@pytest.mark.parametrize('sealed',['visible_raw','exposed_target','duplicate_id'],indirect=True)
def test_sealed_but_invalid_evaluator_contract(sealed):
    with pytest.raises(ValueError):c.load_replay_execution_inputs(**sealed)

@pytest.mark.parametrize('field',['profile','sources','status'])
def test_readiness_contract_must_match_before_executor(sealed,field):
    inputs=c.load_replay_execution_inputs(**sealed);receipt=readiness(inputs);receipt[field]={} if field!='status' else 'blocked'
    called=[]
    with pytest.raises(ValueError):c.measure_jobs(inputs,readiness_check=lambda:receipt,executor=lambda *a,**k:called.append(1))
    assert called==[]

def test_worker_exception_propagates_without_publication(sealed,tmp_path):
    inputs=c.load_replay_execution_inputs(**sealed)
    def dead(*a,**k):raise RuntimeError('worker died')
    with pytest.raises(RuntimeError,match='worker died'):
        c.measure_jobs(inputs,readiness_check=lambda:readiness(inputs),executor=dead)
    assert not (tmp_path/'cache').exists()

def test_incomplete_executor_structure_is_not_a_model_label(sealed):
    inputs=c.load_replay_execution_inputs(**sealed)
    with pytest.raises(ValueError,match='structure'):
        c.measure_jobs(inputs,readiness_check=lambda:readiness(inputs),executor=lambda *a,**k:dict(outcome='PASS',stderr='',stdout=''))

def test_extra_runtime_source_is_bound_and_watched(sealed):
    extra='src/pbpf/eesd/replay_prompt.py'
    inputs=c.load_replay_execution_inputs(**sealed,extra_source_paths=[extra])
    assert inputs['bindings']['execution_sources'][extra]==g.sha(c.ROOT/extra)
    assert inputs['bindings']['watched_files'][str(c.ROOT/extra)]==g.sha(c.ROOT/extra)
    assert set(c.EXECUTION_SOURCES)<=set(inputs['bindings']['execution_sources'])
    with pytest.raises(ValueError):c.load_replay_execution_inputs(**sealed,extra_source_paths=['../outside.py'])

def test_mapper_only_called_after_readiness_and_accepts_ordered_results(sealed):
    inputs=c.load_replay_execution_inputs(**sealed);events=[]
    def ready():events.append('ready');return readiness(inputs)
    def mapper(jobs):
        events.append('mapper')
        return [c.measure_job(job,executor=lambda *a,**k:measured_result()) for job in jobs]
    measured=c.measure_jobs(inputs,readiness_check=ready,job_mapper=mapper)
    assert events==['ready','mapper'] and len(measured['records'])==700
    events.clear()
    def failed():raise RuntimeError('namespace unavailable')
    with pytest.raises(RuntimeError):c.measure_jobs(inputs,readiness_check=failed,job_mapper=mapper)
    assert events==[]

@pytest.mark.parametrize('defect',['reversed','partial'])
def test_mapper_order_and_complete_population_required(sealed,defect):
    inputs=c.load_replay_execution_inputs(**sealed)
    def mapper(jobs):
        records=[c.measure_job(job,executor=lambda *a,**k:measured_result()) for job in jobs]
        return records[::-1] if defect=='reversed' else records[:-1]
    with pytest.raises(ValueError):c.measure_jobs(inputs,readiness_check=lambda:readiness(inputs),job_mapper=mapper)
