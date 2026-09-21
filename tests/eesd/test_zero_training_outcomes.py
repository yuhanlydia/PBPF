import builtins
import json
import sys
from pathlib import Path
import pytest
from test_mechanism_runner import ROOT,load_script
from pbpf.eesd.distillation import TRAIN_RULES


def scored_rows(train_weight=0.,dev_weight=1.):
    return [dict(trajectory_id=f'id-{split}',source_component_id=f'src-{split}',split=split,
        prompt='Solve this task',correction='print(1)',before_outcomes=[0]*4,after_outcomes=[0]*4,relevance=[1.]*4,
        training_weights={r:(0. if r=='no_update' else weight) for r in TRAIN_RULES})
        for split,weight in [('train',train_weight),('development',dev_weight)]]


@pytest.fixture
def task_args(tmp_path):
    data=tmp_path/'scored.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in scored_rows()))
    model=tmp_path/'model.yaml';model.write_text('model_id: test/model\nrevision: '+ 'a'*40+'\n')
    return dict(root=ROOT,scored=data,model_config=model,rule='eesd_full',previous_adapter=None,
        output=tmp_path/'out',seed=1701,budget=53,max_steps=2,anchor_beta=.03)


def test_zero_preflight_and_cpu_trainer_receipt_resume(task_args,monkeypatch):
    wrapper=load_script('run_eesd_downstream');trainer=load_script('run_eesd_weighted_sft')
    task=wrapper.make_training_task(**task_args)
    assert task['eligibility']['status']=='non_estimable_zero_positive_training_weight'
    original=builtins.__import__
    def no_model_import(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','transformers','peft'}:raise AssertionError('zero arm loaded model stack')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_model_import)
    monkeypatch.setattr(sys,'argv',task['command'][1:]);trainer.main()
    outcome=wrapper.check_training_binding(task,seal=True)
    assert outcome['status']=='non_estimable_zero_positive_training_weight' and outcome['adapter'] is None
    report=json.loads((task_args['output']/'training-outcome.json').read_text())
    assert report['response_tokens']==0 and report['response_token_budget']==53
    assert report['optimizer_steps']==0 and report['eligibility']['positive_development_rows']==1
    assert not (task_args['output']/'adapter').exists()
    assert not (task_args['output']/'training-report.json').exists()
    assert wrapper.make_training_task(**task_args)['status']=='verified_non_estimable'
    report['response_tokens']=53;(task_args['output']/'training-outcome.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):wrapper.check_training_binding(task)


@pytest.mark.parametrize('change',['missing_train','missing_dev','bad_weight','duplicate','bad_response','bad_messages','bad_vectors'])
def test_invalid_input_is_not_scientific_empty_selection(task_args,change):
    rows=scored_rows()
    if change=='missing_train':rows=rows[1:]
    elif change=='missing_dev':rows=rows[:1]
    elif change=='bad_weight':rows[0]['training_weights']['eesd_full']=float('nan')
    elif change=='duplicate':rows[1]['trajectory_id']=rows[0]['trajectory_id']
    elif change=='bad_response':rows[0]['correction']=''
    elif change=='bad_messages':rows[0]['messages']=[{'role':'assistant','content':'wrong'}]
    else:rows[0]['before_outcomes']=[]
    task_args['scored'].write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):load_script('run_eesd_downstream').make_training_task(**task_args)
    assert not task_args['output'].exists()


def test_downstream_continues_other_arms_and_keeps_eight_rows(tmp_path,monkeypatch):
    import yaml,subprocess
    from test_downstream_cli import fake_training_job
    runner=load_script('run_eesd_downstream');trainer=load_script('run_eesd_weighted_sft')
    output=tmp_path/'out';model=tmp_path/'model.yaml';model.write_text('model_id: test/model\nrevision: '+'a'*40+'\n')
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump({'schema':'eesd-iclr2027-v1','seeds':[1701],
        'distillation':{'transition_utility':[1,-1,0,0]}}))
    manifest=tmp_path/'manifest.yaml';manifest.write_text(yaml.safe_dump({'schema':'eesd-cache-manifest-v1',
        'correction_cells':[dict(dataset='d',model='m',round=1,model_config=str(model))]}))
    scored=output/'corrections/d/m/round1/scored-corrections.jsonl';scored.parent.mkdir(parents=True)
    rows=scored_rows(train_weight=1.)
    rows[0]['training_weights']['final_correctness']=0.
    scored.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    calls=[]
    def launch(command,**kwargs):
        calls.append(command[command.index('--rule')+1])
        if calls[-1]=='final_correctness':
            with monkeypatch.context() as context:
                context.setattr(sys,'argv',command[1:]);trainer.main()
            return subprocess.CompletedProcess(command,0)
        return fake_training_job(command,**kwargs)
    monkeypatch.setattr(runner.subprocess,'run',launch)
    monkeypatch.setattr(sys,'argv',['runner','--config',str(config),'--manifest',str(manifest),'--output',str(output),
        '--stage','train','--response-token-budget','53'])
    runner.main()
    record=json.loads(next(output.glob('downstream-launches/*/launch.json')).read_text())
    assert len(calls)==7 and calls[-1]=='eesd_full'
    assert len(record['training_coverage'])==8
    assert record['status']=='completed_with_non_estimable'
    zero=[r for r in record['training_coverage'] if r['rule']=='final_correctness'][0]
    assert zero['status']=='non_estimable_zero_positive_training_weight'
    assert not (output/'training/d/m/round1/final_correctness/seed1701/adapter').exists()


@pytest.mark.parametrize('weight',[True,'0'])
def test_coerced_weights_are_invalid(task_args,weight):
    rows=scored_rows();rows[0]['training_weights']['eesd_full']=weight
    task_args['scored'].write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):load_script('run_eesd_downstream').make_training_task(**task_args)


def test_nonempty_invalid_parent_is_not_scientific_zero(task_args,tmp_path):
    previous=tmp_path/'fake_adapter';previous.mkdir();(previous/'log.txt').write_text('not a checkpoint')
    task_args['previous_adapter']=previous
    with pytest.raises(ValueError,match='adapter'):
        load_script('run_eesd_downstream').make_training_task(**task_args)


def test_fresh_and_transfer_keep_zero_main_rows_without_fallback(task_args,monkeypatch,tmp_path):
    runner=load_script('run_eesd_downstream');trainer=load_script('run_eesd_weighted_sft')
    rootout=tmp_path/'study';scored=rootout/'corrections/d/m/round1/scored-corrections.jsonl'
    scored.parent.mkdir(parents=True);scored.write_bytes(task_args['scored'].read_bytes())
    task_args.update(scored=scored,output=rootout/'training/d/m/round1/eesd_full/seed1701')
    task=runner.make_training_task(**task_args)
    monkeypatch.setattr(sys,'argv',task['command'][1:]);trainer.main();runner.check_training_binding(task,seal=True)
    correction=dict(dataset='d',model='m',round=1,model_config=str(task_args['model_config']))
    cell=dict(dataset='humaneval',model='m',source_dataset='d',source_round=1,model_config=str(task_args['model_config']),
        domain='rbr',family='qwen25_7b',public_root=str(tmp_path),evaluator_root=str(tmp_path))
    manifest=dict(correction_cells=[correction],fresh_cells=[cell],transfer_cells=[cell])
    planned=runner.plan_fresh(root=ROOT,manifest=manifest,output=rootout,seeds=[1701],rules=['no_update','eesd_full'])
    assert len(planned)==2 and planned[1]['status']=='non_estimable_zero_positive_training_weight'
    assert planned[1].get('command') is None
    import pbpf.eesd.evalplus_public as public
    monkeypatch.setattr(public,'load_public_dataset',lambda *a: ([],{'tasks':2}))
    transfer=runner.plan_transfer(root=ROOT,manifest=manifest,output=rootout,seeds=[1701],rules=['no_update','eesd_full'],
        public_root=tmp_path,public_manifest_sha256='a'*64)
    assert len(transfer)==2 and transfer[1].get('command') is None
    assert transfer[1]['status']=='non_estimable_zero_positive_training_weight'
    # Execute only mocked fresh routing: preserve all eight slots while calling
    # the frozen runner only for policies that exist, with baseline initialization.
    import yaml,subprocess
    config=tmp_path/'fresh-config.yaml';config.write_text(yaml.safe_dump({'schema':'eesd-iclr2027-v1',
        'seeds':[1701],'distillation':{'transition_utility':[1,-1,0,0]}}))
    manifest_path=tmp_path/'fresh-manifest.yaml'
    manifest_path.write_text(yaml.safe_dump({'schema':'eesd-cache-manifest-v1',**manifest}))
    commands=[]
    def launch(command,**kwargs):
        commands.append(command)
        actual=yaml.safe_load(Path(command[command.index('--manifest')+1]).read_text())
        assert actual['fresh_cells']==[cell]
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(runner.subprocess,'run',launch)
    monkeypatch.setattr(sys,'argv',['runner','--config',str(config),'--manifest',str(manifest_path),
        '--output',str(rootout),'--stage','fresh'])
    runner.main()
    assert len(commands)==7
    assert all(c[c.index('--rules')+1]=='no_update' and 'eesd_full' not in c for c in commands)
    launch_record=json.loads(next(rootout.glob('downstream-launches/*/launch.json')).read_text())
    assert len(launch_record['fresh_tasks'])==8
    assert launch_record['status']=='completed_with_non_estimable'

    # Receipt must not be transplanted into a cell whose external parent differs.
    manifest['correction_cells'][0]['previous_adapter']=str(tmp_path/'different_parent')
    with pytest.raises((ValueError,FileNotFoundError)):
        runner.plan_fresh(root=ROOT,manifest=manifest,output=rootout,seeds=[1701],rules=['eesd_full'])


def test_valid_parent_checkpoint_is_bound_without_creating_adapter(task_args,tmp_path):
    import numpy as np
    from safetensors.numpy import save_file
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','task_type':'CAUSAL_LM',
        'base_model_name_or_path':'test/model','revision':'a'*40,'r':2}))
    save_file({'layer.lora_A.weight':np.zeros((2,3),dtype=np.float32)},str(parent/'adapter_model.safetensors'))
    task_args['previous_adapter']=parent
    runner=load_script('run_eesd_downstream');task=runner.make_training_task(**task_args)
    assert task['expected_report']['previous_adapter']==str(parent.resolve())
    assert len(task['expected_report']['previous_adapter_tree_sha256'])==64
    cfg=json.loads((parent/'adapter_config.json').read_text());cfg['base_model_name_or_path']='other/model'
    (parent/'adapter_config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='adapter'):
        runner.make_training_task(**task_args)
