import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml
from test_mechanism_runner import ROOT,load_script

ZERO='non_estimable_zero_positive_training_weight'


def setup_pipeline(tmp_path,monkeypatch,policy,empty):
    m=load_script('run_eesd_recursive')
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump({'schema':'eesd-iclr2027-v1','seeds':[1701],
        'distillation':{'transition_utility':[1,-1,0,0]}}))
    model=tmp_path/'model.yaml';model.write_text('model_id: mock\nrevision: '+ 'a'*40+'\n')
    for name in ('public','evaluator'):
        (tmp_path/name).mkdir();(tmp_path/name/'manifest.json').write_text('{}')
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['recursive','--config',str(config),'--domain','rbr','--public-root',str(tmp_path/'public'),
        '--evaluator-root',str(tmp_path/'evaluator'),'--model-config',str(model),'--family','qwen25_7b',
        '--output',str(out),'--seed','1701','--rounds','3','--response-token-budget','53','--experience-policy',policy])
    calls={'collect':[],'train':[],'eval':[],'compare':[]};cached={}
    monkeypatch.setattr(m,'generate_bank',lambda **kw:None)
    def collect(**kw):
        calls['collect'].append(kw)
        kw['output'].mkdir(parents=True,exist_ok=True)
        from pbpf.eesd.distillation import TRAIN_RULES
        r=1+int(next(part.removeprefix('round') for part in reversed(kw['output'].parts) if part.startswith('round')))
        rows=[{'trajectory_id':split,'source_component_id':split,'split':split,'prompt':'repair','correction':'pass',
               'before_outcomes':[1],'after_outcomes':[0],'relevance':[1.],
               'training_weights':{rule:0. if (r,rule)==empty and split=='train' else 1. for rule in TRAIN_RULES}}
              for split in ('train','development')]
        path=kw['output']/'scored.jsonl';path.write_text('\n'.join(map(json.dumps,rows))+'\n')
        return path
    def train(**kw):
        path=kw['output'];r=int(path.parent.parent.name.removeprefix('round'));key=(r,kw['rule'])
        task=m._TRAINING['make_training_task'](root=ROOT,scored=kw['scored'],model_config=kw['model_config'],
            rule=kw['rule'],previous_adapter=kw['previous_adapter'],output=path,seed=kw['seed'],
            budget=kw['response_token_budget'],max_steps=kw['max_steps'],anchor_beta=kw['anchor_beta'])
        if task['status']!='pending':return m._TRAINING['check_training_binding'](task)
        calls['train'].append(kw)
        path.mkdir(parents=True)
        is_zero=task['eligibility']['status']==ZERO
        if not is_zero:
            adapter=path/'adapter';adapter.mkdir();(adapter/'weights').write_bytes(str(key).encode())
        report=path/('training-outcome.json' if is_zero else 'training-report.json')
        report.write_text(json.dumps(task['expected_report']))
        outcome=m._TRAINING['check_training_binding'](task,seal=True)
        cached[key]=outcome
        return outcome
    def evaluate(**kw):
        calls['eval'].append(kw);kw['output'].mkdir(parents=True,exist_ok=True)
        adapter=kw['output'].parent/'training/adapter'
        adapter_sha=m._TRAINING['adapter_tree_digest'](adapter) if adapter.exists() else None
        (kw['output']/'report.json').write_text(json.dumps({'fresh_all_tests_pass_at_1':.5,'adapter_sha256':adapter_sha}))
    def compare(root,baseline,method,output):
        calls['compare'].append(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text('{}')
    for name,fn in [('collect_experience',collect),('train_next',train),('evaluate_primary',evaluate),('compare_reports',compare)]:
        monkeypatch.setattr(m,name,fn)
    return m,out,calls,cached


@pytest.mark.parametrize('policy,empty,collects,trains,evals',[
    ('shared_eesd_teacher',(1,'eesd_full'),1,2,2),
    ('shared_eesd_teacher',(1,'equal_weight'),3,6,6),
    ('arm-specific',(1,'eesd_full'),3,4,4),
])
def test_zero_preserves_rows_and_blocks_only_dependent_updates(tmp_path,monkeypatch,policy,empty,collects,trains,evals):
    m,out,calls,_=setup_pipeline(tmp_path,monkeypatch,policy,empty)
    m.main()
    report=json.loads((out/'recursive-report.json').read_text())
    assert len(calls['collect'])==collects and len(calls['train'])==trains and len(calls['eval'])==evals
    assert len(report['summary'])==3
    assert report['summary'][0]['arms'][empty[1]]['status']==ZERO
    assert report['summary'][0]['arms'][empty[1]]['pass_at_1'] is None
    assert report['summary'][0]['eesd_vs_equal'] is None
    assert report['coverage']['complete'] is False
    for r in (2,3):
        for rule in m.RULES:
            row=report['summary'][r-1]['arms'][rule]
            blocked=(empty[1]=='eesd_full' and (policy=='shared_eesd_teacher' or rule=='eesd_full'))
            assert row['status']==('blocked_dependency' if blocked else 'trained')
            if blocked:
                assert row['pass_at_1'] is None and row['dependency']['round']==1
                assert row['dependency']['rule']=='eesd_full'
    calls['train'].clear();m.main();assert not calls['train']


def test_tampered_terminal_receipt_rejected_on_recursive_resume(tmp_path,monkeypatch):
    m,out,_,cached=setup_pipeline(tmp_path,monkeypatch,'shared_eesd_teacher',(1,'eesd_full'))
    m.main()
    Path(cached[(1,'eesd_full')]['receipt']).write_text('tampered')
    with pytest.raises((ValueError,json.JSONDecodeError),match='receipt|binding|Expecting'):
        m.main()


def test_offline_incomplete_summary_retains_zero_and_blocked_rows(tmp_path,monkeypatch):
    m,out,_,_=setup_pipeline(tmp_path,monkeypatch,'shared_eesd_teacher',(1,'eesd_full'))
    m.main()
    summary=load_script('summarize_eesd_recursive_trajectories')
    target=tmp_path/'incomplete.json'
    monkeypatch.setattr(sys,'argv',['summary','--study-root',str(out),'--rule','equal_weight','--output',str(target)])
    summary.main()
    result=json.loads(target.read_text())
    assert result['status']=='non_estimable_incomplete_recursive_study'
    assert result['coverage']['complete'] is False
    assert [r['status'] for r in result['by_round'][1:]]==['trained','blocked_dependency','blocked_dependency']
    assert result['by_round'][2]['pass_at_1'] is None
    assert 'trajectories' not in result
    receipt=out/'round2/outcomes.json';receipt.write_text('{}')
    monkeypatch.setattr(sys,'argv',['summary','--study-root',str(out),'--rule','equal_weight','--output',str(tmp_path/'bad.json')])
    with pytest.raises(ValueError,match='outcome'):
        summary.main()


@pytest.mark.parametrize('tamper',['missing_outcome','zero_tokens','receipt_schema','trained_as_zero'])
def test_incomplete_summary_requires_real_terminal_contract(tmp_path,monkeypatch,tamper):
    m,out,_,_=setup_pipeline(tmp_path,monkeypatch,'shared_eesd_teacher',(1,'eesd_full'))
    m.main()
    training=out/'round1/eesd_full/training'
    if tamper=='missing_outcome':(training/'training-outcome.json').unlink()
    elif tamper=='zero_tokens':
        p=training/'training-outcome.json';v=json.loads(p.read_text());v['response_tokens']=53;p.write_text(json.dumps(v))
    elif tamper=='receipt_schema':
        (training/'training-binding.json').write_text('{"round":1,"rule":"eesd_full"}')
    else:
        (training/'training-binding.json').write_bytes((out/'round1/equal_weight/training/training-binding.json').read_bytes())
    # Rehash outer containers: raw checksum verification alone must not suffice.
    receipt=training/'training-binding.json';receipt_sha=hashlib.sha256(receipt.read_bytes()).hexdigest()
    round_path=out/'round1/outcomes.json';round_value=json.loads(round_path.read_text())
    round_value['arms']['eesd_full']['receipt_sha256']=receipt_sha;round_path.write_text(json.dumps(round_value))
    report_path=out/'recursive-report.json';report=json.loads(report_path.read_text())
    report['summary'][0]['arms']=round_value['arms']
    report['summary'][0]['outcomes_sha256']=hashlib.sha256(round_path.read_bytes()).hexdigest();report_path.write_text(json.dumps(report))
    summary=load_script('summarize_eesd_recursive_trajectories')
    target=tmp_path/'bad.json';monkeypatch.setattr(sys,'argv',['summary','--study-root',str(out),'--rule','equal_weight','--output',str(target)])
    with pytest.raises((ValueError,FileNotFoundError)):
        summary.main()
    assert not target.exists()


@pytest.mark.parametrize('tamper',['shared_path','teacher_sha','blocked_plans','blocked_dependencies'])
def test_incomplete_shared_contract_rejects_resealed_changes(tmp_path,monkeypatch,tamper):
    m,out,_,_=setup_pipeline(tmp_path,monkeypatch,'shared_eesd_teacher',(1,'eesd_full'))
    m.main()
    r=2 if tamper.startswith('blocked') else 1
    update_path=out/f'round{r}/update-inputs.json';update=json.loads(update_path.read_text())
    if tamper=='shared_path':
        plan=update['arms']['equal_weight'];copy=tmp_path/'same-bytes.jsonl'
        copy.write_bytes(Path(plan['scored']).read_bytes());plan['scored']=str(copy)
    elif tamper=='teacher_sha':update['teacher_adapter_sha256']='a'*64
    elif tamper=='blocked_plans':update['arms']['equal_weight']={'scored':'not-a-real-update'}
    else:update['blocked_dependencies']={}
    update_path.write_text(json.dumps(update));update_sha=hashlib.sha256(update_path.read_bytes()).hexdigest()
    outcome_path=out/f'round{r}/outcomes.json';outcome=json.loads(outcome_path.read_text())
    outcome['update_inputs_sha256']=update_sha;outcome_path.write_text(json.dumps(outcome))
    report_path=out/'recursive-report.json';report=json.loads(report_path.read_text());item=report['summary'][r-1]
    item['update_inputs_sha256']=update_sha;item['outcomes_sha256']=hashlib.sha256(outcome_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))
    summary=load_script('summarize_eesd_recursive_trajectories');target=tmp_path/'bad.json'
    monkeypatch.setattr(sys,'argv',['summary','--study-root',str(out),'--rule','equal_weight','--output',str(target)])
    with pytest.raises(ValueError,match='shared|blocked'):
        summary.main()
    assert not target.exists()
