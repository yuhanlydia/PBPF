"""Synthetic sealed artifacts, without candidate execution or GPU imports."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from test_mechanism_runner import ROOT,load_script
from pbpf.apbpf.codearc_bank import file_sha


def dump(path,value):
    path.write_text(json.dumps(value,sort_keys=True))


@pytest.fixture(params=["rbr", "codearc"])
def artifacts(tmp_path,request):
    domain=request.param
    runner=load_script('run_eesd_evidence_matrix')
    cfg={'schema':'eesd-iclr2027-v1','seeds':[1701], 'evidence':{'alphas':[.1,1.0],
          'strengths':[0.,4.],'global_mass_grid':[1.,4.],'ece_bins':10}}
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(cfg))
    evaluator=tmp_path/'evaluator';evaluator.mkdir()
    records=[];tasks=[];bindings=[]
    identity=['Qwen/Qwen2.5-Coder-7B-Instruct','c03e6d358207e414f1eca0bb1891e29f1db0e242',None,1701]
    for split in ('development','primary'):
        bank=tmp_path/split;bank.mkdir();ids=[];sources=[];files={}
        for i in range(2):
            problem=f'{split}-{i}';source=f'source-{problem}';candidate=problem+'/c0';code=f'print({i})'
            tests=[{'id':str(j),'input':f'{i} '+('word '*(j+1)),'output':'private'} for j in range(10)]
            tasks.append(dict(task_id=problem,source_component_id=source,split=split,tests=tests))
            record=dict(task_id=candidate,problem_id=problem,source_component_id=source,split=split,
                candidate_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                tests=[{k:t[k] for k in ('id','input')} for t in tests],
                outcomes=['PASS','WRONG_OUTPUT','PASS','TIMEOUT','PASS','PASS','COMPILE_ERROR','PASS','WRONG_OUTPUT','PASS'])
            records.append(record);ids.append(problem);sources.append(source)
            row=dict(task_id=problem,source_component_id=source,split=split,
                candidates=[dict(candidate_id=candidate,code=code)])
            dump(bank/f'{problem}.json',row);files[f'{problem}.json']=file_sha(bank/f'{problem}.json')
        run=dict(schema=f'apbpf-{domain}-generation-v1',model=identity[0],revision=identity[1],seed=1701,
                 adapter_sha256=None,candidates=1,components=2,split=split,task_ids=ids,source_component_ids=sources,
                 public_tasks_sha256='a'*64)
        dump(bank/'run.json',run);dump(bank/'complete.json',dict(run_sha256=file_sha(bank/'run.json'),files=files))
        bindings.append(dict(split=split,path=str(bank),complete_sha256=file_sha(bank/'complete.json'),components=2))
    (evaluator/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    dump(evaluator/'manifest.json',dict(schema='apbpf-rbr-generated-materialization-v1' if domain=='rbr' else 'apbpf-codearc-replay-materialization-v1',
        evaluator_tasks_sha256=file_sha(evaluator/'tasks.jsonl'),public_tasks_sha256='a'*64))
    cache=tmp_path/'cache.json'
    payload=dict(schema='eesd-public-query-mechanism-cache-v1',dataset=domain,generator_identity=identity,
        tests_per_candidate=10,records=records,counts={'development':2,'primary':2},
        source_counts={'development':2,'primary':2},bank_bindings=bindings,
        evaluator_manifest_sha256=file_sha(evaluator/'manifest.json'))
    dump(cache,payload)
    source_names=('scripts/build_eesd_mechanism_cache.py','src/pbpf/apbpf/codearc_execution.py','src/pbpf/apbpf/rbr_execution.py')
    dump(cache.with_suffix('.binding.json'),dict(cache_sha256=file_sha(cache),generator_identity=identity,
        bank_bindings=bindings,evaluator_tasks_sha256=file_sha(evaluator/'tasks.jsonl'),
        evaluator_manifest_sha256=file_sha(evaluator/'manifest.json'),sources={n:file_sha(ROOT/n) for n in source_names}))
    def examples(split,h,s,b):
        return runner.build_examples([r for r in records if r['split']==('development' if split=='validation' else 'primary')],h,s,binary=b)
    selected={};probs={}
    for arm,rule in [('ordinary','ordinary'),('fixed','fixed'),('effective','effective'),('global_mass','global')]:
        selected[arm],_=runner.select_arm(examples,'validation',4,[.1,1.],
            [0.] if arm=='ordinary' else [0.,4.],rule,masses=[1.,4.] if arm=='global_mass' else None)
        s=selected[arm];result=runner.evaluate(examples('assessment',4,s['strength'],False),s,rule,ece_bins=10)
        probs[arm]=result['probabilities']
        if arm=='effective':probs['effective_mass']=result['masses']
    for arm,selection,rule in [('fixed_at_effective_params','effective','fixed'),('effective_at_fixed_params','fixed','effective')]:
        s=selected[selection];probs[arm]=runner.predictions(examples('assessment',4,s['strength'],False),s['alpha'],rule)[0]
    labels,clusters=runner.labels_clusters(examples('assessment',4,0.,False))
    out=tmp_path/'report';out.mkdir()
    report=dict(schema='eesd-evidence-matrix-v1',dataset='runbugrun' if domain=='rbr' else 'codearc',model='qwen25_7b',seed=1701,
        visible=4,validation_split='development',assessment_split='primary',selected=selected,
        counts=dict(validation_records=2,assessment_records=2,assessment_examples=12,assessment_sources=2),
        cache_sha256=file_sha(cache),config_sha256=file_sha(config),
        runner_source_sha256=file_sha(ROOT/'scripts/run_eesd_evidence_matrix.py'),
        evidence_source_sha256=file_sha(ROOT/'src/pbpf/eesd/evidence.py'))
    dump(out/'report.json',report);np.savez(out/'predictions.npz',labels=labels,clusters=clusters,**probs)
    reseal(out)
    return dict(report_dir=out,cache=cache,config=config,evaluator_root=evaluator,
                domain=domain,model='qwen25_7b',seed=1701,root=ROOT)


def reseal(out):
    dump(out/'complete.json',dict(report_sha256=file_sha(out/'report.json'),predictions_sha256=file_sha(out/'predictions.npz')))


def test_reconstruct_six_arms_with_absolute_query_identity(artifacts):
    from pbpf.eesd.mechanism_artifacts import load_mechanism_artifacts
    data,receipt=load_mechanism_artifacts(**artifacts)
    assert len(data['labels'])==12 and len(data['probabilities'])==6
    assert data['query_ids'][0]['test_index']==4
    assert data['query_ids'][6]['problem_id']=='primary-1'
    assert set(data['seeds'])=={1701}
    assert receipt['scope']=='six-saved-arms-only'


@pytest.mark.parametrize('change',['npz_probability','npz_labels','runner_hash','model','seed','cache_query','candidate','source_overlap','source_hash','missing_evaluator','test_order','record_order','selection','npz_clusters'])
def test_reject_corrupted_or_resealed_wrong_identity(artifacts,change):
    from pbpf.eesd.mechanism_artifacts import load_mechanism_artifacts
    out=artifacts['report_dir'];cache=artifacts['cache']
    if change.startswith('npz_'):
        with np.load(out/'predictions.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        if change=='npz_probability':arrays['effective'][0]=np.roll(arrays['effective'][0],1)
        elif change=='npz_clusters':arrays['clusters']=arrays['clusters'][::-1]
        else:arrays['labels'][0]=(arrays['labels'][0]+1)%5
        np.savez(out/'predictions.npz',**arrays);reseal(out)
    elif change in {'runner_hash','model','seed'}:
        rp=out/'report.json';r=json.loads(rp.read_text());r[{'runner_hash':'runner_source_sha256'}.get(change,change)]='bad';dump(rp,r);reseal(out)
    elif change=='selection':
        rp=out/'report.json';r=json.loads(rp.read_text());r['selected']['effective']['alpha']=123.;dump(rp,r);reseal(out)
    elif change=='missing_evaluator':artifacts['evaluator_root']=None
    elif change=='source_hash':
        p=cache.with_suffix('.binding.json');r=json.loads(p.read_text());r['sources']['scripts/build_eesd_mechanism_cache.py']='bad';dump(p,r)
    else:
        r=json.loads(cache.read_text())
        if change=='cache_query':r['records'][2]['tests'][4]['input']='wrong query with same label'
        elif change=='candidate':r['records'][2]['task_id']='another-candidate'
        elif change=='test_order':r['records'][2]['tests'][4:6]=r['records'][2]['tests'][4:6][::-1]
        elif change=='record_order':r['records'][2:4]=r['records'][2:4][::-1]
        else:r['records'][2]['source_component_id']=r['records'][0]['source_component_id']
        dump(cache,r)
        seal=cache.with_suffix('.binding.json');s=json.loads(seal.read_text());s['cache_sha256']=file_sha(cache);dump(seal,s)
        rp=out/'report.json';r=json.loads(rp.read_text());r['cache_sha256']=file_sha(cache);dump(rp,r);reseal(out)
    with pytest.raises((ValueError,TypeError)):
        load_mechanism_artifacts(**artifacts)
