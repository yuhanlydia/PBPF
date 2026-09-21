import hashlib
import json
from pathlib import Path

import pytest
import yaml

from pbpf.eesd.distillation import score_trajectory
from pbpf.eesd.correction_credibility import load_public_bundle, evaluate_bundle, summarize_credibility


def dump(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in value) if lines else json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, shared=False, candidate_task_ids=False):
    gen, scored, public, evaluator = [tmp_path/n for n in ['generated','scored','public','evaluator']]
    cfg=tmp_path/'config.yaml'
    cfg.write_text(yaml.safe_dump({'schema':'eesd-iclr2027-v1','distillation':{'transition_utility':[1,-1,0,0]}}))
    configsha=hashlib.sha256(cfg.read_bytes()).hexdigest()
    generated=[];scoredrows=[];publicrows=[];tasks=[]
    for i in range(4):
        tests=[{'id':str(j),'input':str(j),'expected':'ok','hidden':j>=4} for j in range(10)]
        bank_task=f't{i}/model/0' if candidate_task_ids else f't{i}'
        source='shared' if shared and i < 3 else f's{i}'
        original=f'original{i}';correction=f'corrected{i}'
        before=['WRONG_OUTPUT']*4;after=['PASS']*(i+1)+['WRONG_OUTPUT']*(3-i)
        pub={'schema':'eesd-public-correction-row-v1','task_id':bank_task, 'problem_id':f't{i}', 'source_component_id':source,
             'split':'development','candidate':original,'candidate_code_sha256':hashlib.sha256(original.encode()).hexdigest(),
             'tests':[{k:t[k] for k in ('id','input','expected')} for t in tests[:4]],'outcomes':before}
        publicrows.append(pub)
        row={'schema':'eesd-correction-trajectory-v1','trajectory_id':f'id{i}','task_id':bank_task,'problem_id':f't{i}',
             'source_component_id':source,'split':'development','original':original,'correction':correction,
             'correction_code_sha256':hashlib.sha256(correction.encode()).hexdigest(),
             'before_outcomes':before,'after_outcomes':after,'relevance':[1]*4}
        generated.append(row)
        diagnostics=score_trajectory([1]*4,[0]*(i+1)+[1]*(3-i),[1]*4,alpha=1,
                                     utility=[1,-1,0,0],uncertainty_penalty=0)
        weights=diagnostics.pop('weights')
        scoredrows.append({**row,'before_outcomes':[1]*4,'after_outcomes':[0]*(i+1)+[1]*(3-i),
                           'training_weights':weights,'eesd_diagnostics':diagnostics})
        tasks.append({'task_id':f't{i}','source_component_id':source,'split':'development','tests':tests})
    publicsha=dump(public/'public-corrections.jsonl',publicrows,True)
    dump(public/'audit.json',{'schema':'eesd-public-correction-bank-v1','rows':4,'public_bank_sha256':publicsha})
    gensha=dump(gen/'corrections.jsonl',generated,True)
    dump(gen/'report.json',{'schema':'eesd-correction-generation-v1','domain':'rbr','trajectories':4,
         'public_bank_sha256':publicsha,'output_sha256':gensha,'private_fields_available_to_generator':False})
    scoredsha=dump(scored/'scored-corrections.jsonl',scoredrows,True)
    dump(scored/'summary.json',{'schema':'eesd-scored-corrections-v1','input_sha256':gensha,
         'scored_sha256':scoredsha,'config_sha256':configsha,'trajectories':4,'alpha':1,'uncertainty_penalty':0})
    tasksha=dump(evaluator/'tasks.jsonl',tasks,True)
    dump(evaluator/'manifest.json',{'schema':'apbpf-rbr-generated-materialization-v1','evaluator_tasks_sha256':tasksha})
    return gen,scored,public,cfg,evaluator


def fake_execute(code,test,timeout):
    assert test['id'] in ['4','5','6','7','8','9']
    # s0 FIX, s1 REGRESSION, s2 PRESERVED, s3 UNRESOLVED on independent hidden tests.
    passed=code in ['corrected0','original1','original2','corrected2']
    return {'outcome':'PASS' if passed else 'WRONG_OUTPUT'}


def load(paths):return load_public_bundle(*paths[:4],domain='rbr')


def test_hidden_only_quantiles_hand_computable_and_ties_not_split(tmp_path):
    paths=fixture(tmp_path);bundle=load(paths)
    evaluation=evaluate_bundle(bundle,paths[4],execute=fake_execute)
    result=summarize_credibility(bundle,evaluation,bins=2)
    groups=result['by_split']['development']['groupings']
    assert result['label_scope']=='hidden-only'
    assert result['cohort']=='visible-failing-originals'
    utility=groups['public_conservative_utility']['bins']
    assert [b['sources'] for b in utility]==[2,2]
    assert utility[0]['fixes']==1 and utility[0]['regressions']==1
    assert utility[0]['final_correctness']==0.5
    assert utility[0]['fix_rate_on_hidden_wrong']==1
    assert utility[0]['regression_rate_on_hidden_correct']==1
    mass=groups['public_effective_evidence_mass']['bins']
    assert sorted(b['sources'] for b in mass)==[0,4]
    assert result['by_split']['development']['sources']==4


@pytest.mark.parametrize('change',['missing','extra','source','hash','duplicate','binding','tests','test_digest'])
def test_rejects_misaligned_independent_evaluations(tmp_path,change):
    paths=fixture(tmp_path);bundle=load(paths);evaluation=evaluate_bundle(bundle,paths[4],execute=fake_execute)
    if change=='missing':evaluation['records'].pop()
    if change=='extra':evaluation['records'].append({**evaluation['records'][0],'trajectory_id':'unexpected'})
    if change=='source':evaluation['records'][0]['source_component_id']='other'
    if change=='hash':evaluation['records'][0]['correction_sha256']='0'*64
    if change=='duplicate':evaluation['records'].append(evaluation['records'][0])
    if change=='binding':evaluation['input_binding']['scored_sha256']='0'*64
    if change=='tests':evaluation['records'][0]['after_hidden'].pop()
    if change=='test_digest':evaluation['records'][0]['hidden_test_set_sha256']='x'*64
    with pytest.raises(ValueError):summarize_credibility(bundle,evaluation)


def test_tampered_public_scores_fail_before_private_execution(tmp_path):
    paths=fixture(tmp_path)
    p=paths[1]/'scored-corrections.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()]
    rows[0]['eesd_diagnostics']['effective_conservative_utility']['conservative']=99
    digest=dump(p,rows,True)
    summary=json.loads((paths[1]/'summary.json').read_text());summary['scored_sha256']=digest
    dump(paths[1]/'summary.json',summary)
    with pytest.raises(ValueError,match='diagnostic'):load(paths)


def test_duplicate_source_rejected_even_with_distinct_trajectory(tmp_path):
    paths=fixture(tmp_path);bundle=load(paths)
    bundle['records'][1]['source_component_id']=bundle['records'][0]['source_component_id']
    with pytest.raises(ValueError,match='source'):evaluate_bundle(bundle,paths[4],execute=fake_execute)


def test_sandbox_failure_propagates_without_fabricating_outcomes(tmp_path):
    paths=fixture(tmp_path);bundle=load(paths)
    def unavailable(*a,**k):raise RuntimeError('sandbox unavailable')
    with pytest.raises(RuntimeError,match='sandbox unavailable'):
        evaluate_bundle(bundle,paths[4],execute=unavailable)


def test_multiple_trajectories_per_source_are_not_dropped_or_pooled(tmp_path):
    paths=fixture(tmp_path,shared=True);bundle=load(paths)
    evaluation=evaluate_bundle(bundle,paths[4],execute=fake_execute)
    result=summarize_credibility(bundle,evaluation,bins=1)['by_split']['development']
    assert result['sources']==2 and result['trajectories']==4
    # Shared source has 2/3 final correct, second source 0: equal-source mean is 1/3.
    assert result['final_correctness']==pytest.approx(1/3)
    assert result['fix_rate']==pytest.approx(1/6)


@pytest.mark.parametrize('kind',['generation_hash','missing_scored','public_hash','evaluator_hash','public_tests'])
def test_rejects_incomplete_seals_and_evaluator_mismatch(tmp_path,kind):
    paths=fixture(tmp_path)
    if kind=='generation_hash':
        with (paths[0]/'corrections.jsonl').open('a') as f:f.write('\n')
    elif kind=='missing_scored':
        rows=[json.loads(l) for l in (paths[1]/'scored-corrections.jsonl').read_text().splitlines()][:-1]
        sha=dump(paths[1]/'scored-corrections.jsonl',rows,True)
        summary=json.loads((paths[1]/'summary.json').read_text());summary['scored_sha256']=sha
        dump(paths[1]/'summary.json',summary)
    elif kind=='public_hash':
        with (paths[2]/'public-corrections.jsonl').open('a') as f:f.write('\n')
    elif kind=='evaluator_hash':
        with (paths[4]/'tasks.jsonl').open('a') as f:f.write('\n')
    else:
        rows=[json.loads(l) for l in (paths[4]/'tasks.jsonl').read_text().splitlines()]
        rows[0]['tests'][0]['input']='changed'
        sha=dump(paths[4]/'tasks.jsonl',rows,True)
        m=json.loads((paths[4]/'manifest.json').read_text());m['evaluator_tasks_sha256']=sha
        dump(paths[4]/'manifest.json',m)
    executed=[]
    def must_not_execute(*a,**kw):executed.append(1);raise AssertionError('executed invalid bank')
    with pytest.raises(ValueError):
        bundle=load(paths)
        evaluate_bundle(bundle,paths[4],execute=must_not_execute)
    assert executed==[]


def test_postprocessing_cli_reads_sealed_evaluator_artifacts_without_execution(tmp_path):
    import subprocess
    import sys
    paths=fixture(tmp_path);bundle=load(paths)
    report=evaluate_bundle(bundle,paths[4],execute=fake_execute)
    previous=tmp_path/'prior-evaluation'
    locksha=dump(previous/'input-lock.json',{'schema':'eesd-correction-input-lock-v1',
        'input_binding':bundle['input_binding']})
    report['input_lock_sha256']=locksha
    evalsha=dump(previous/'evaluation.json',report)
    dump(previous/'complete.json',{'schema':'eesd-correction-credibility-complete-v1',
        'files':{'input-lock.json':locksha,'evaluation.json':evalsha}})
    script=Path(__file__).resolve().parents[2]/'scripts/report_eesd_correction_credibility.py'
    command=[sys.executable,str(script),'--generation-dir',str(paths[0]),'--scored-dir',str(paths[1]),
        '--public-dir',str(paths[2]),'--config',str(paths[3]),'--domain','rbr',
        '--evaluation-dir',str(previous),'--output',str(tmp_path/'out'),'--bins','2']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    summary=json.loads((tmp_path/'out/credibility.json').read_text())
    assert summary['by_split']['development']['fixes']==1
    assert (tmp_path/'out/complete.json').exists()
    with (previous/'evaluation.json').open('a') as f:f.write(' ')
    command[command.index('--output')+1]=str(tmp_path/'tampered')
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode!=0
    assert not (tmp_path/'tampered/complete.json').exists()


def test_initial_bank_candidate_task_ids_bind_to_explicit_problem_id(tmp_path):
    paths=fixture(tmp_path,candidate_task_ids=True);bundle=load(paths)
    evaluation=evaluate_bundle(bundle,paths[4],execute=fake_execute)
    assert evaluation['records'][0]['task_id']=='t0/model/0'
    assert evaluation['records'][0]['evaluator_task_id']=='t0'
    assert summarize_credibility(bundle,evaluation)['by_split']['development']['sources']==4


def test_four_transition_cells_and_source_weighted_net_gain(tmp_path):
    paths=fixture(tmp_path,shared=True);bundle=load(paths)
    result=summarize_credibility(bundle,evaluate_bundle(bundle,paths[4],execute=fake_execute),bins=2)
    stats=result['by_split']['development']
    assert [stats[k] for k in ('fixes','regressions','preserved','unresolved')]==[1,1,1,1]
    assert stats['preserved_rate']==pytest.approx(1/6)
    assert stats['unresolved_rate']==pytest.approx(1/2)
    assert sum(stats[k] for k in ('fix_rate','regression_rate','preserved_rate','unresolved_rate'))==pytest.approx(1)
    assert stats['net_gain']==pytest.approx(stats['fix_rate']-stats['regression_rate'])
    assert stats['net_gain_count']==0
    assert stats['retained_correct_rate']==pytest.approx(1/2)
    for grouping in stats['groupings'].values():
        for cell in grouping['bins']:
            if cell['sources']:
                assert sum(cell[k] for k in ('fix_rate','regression_rate','preserved_rate','unresolved_rate'))==pytest.approx(1)
            else:
                assert cell['net_gain'] is None
