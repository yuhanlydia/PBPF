import importlib.util
from pathlib import Path
import pytest


def script():
    p=Path(__file__).resolve().parents[2]/'scripts/plan_eesd_training_budget.py'
    spec=importlib.util.spec_from_file_location('budget_plan',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_reference_budget_matches_legacy_equal_weight_exposure():
    budget=script().reference_budget([2,3,5],seed=1701,steps=3,accumulation=4)
    assert budget==40 # Four complete passes through three responses.


def test_partial_pass_is_seeded_and_reproducible():
    m=script()
    budgets=[m.reference_budget([2,3,9],seed=s,steps=1,accumulation=2) for s in range(12)]
    assert all(b in (5,11,12) for b in budgets)
    assert len(set(budgets))>1
    assert budgets==[m.reference_budget([2,3,9],seed=s,steps=1,accumulation=2) for s in range(12)]


def test_budget_refuses_empty_or_zero_response_population():
    m=script()
    for lengths in ([],[0],[2,-1]):
        with pytest.raises(ValueError):m.reference_budget(lengths,seed=1701,steps=3,accumulation=4)


def invoke_plan(tmp_path, monkeypatch, *, duplicate=False, wrong_counts=False, max_length=4608):
    import hashlib
    import json
    import sys
    import types
    m=script()
    from pbpf.eesd.distillation import TRAIN_RULES
    rows=[{'trajectory_id':str(i),'source_component_id':str(i),'split':split,
           'training_weights':{rule:1.0 for rule in TRAIN_RULES},
           'before_outcomes':[1], 'after_outcomes':[0], 'relevance':[1.0], 'model_id':'mock','model_revision':'a'*40,
           'prompt':'p','correction':'x'*n}
          for i,(split,n) in enumerate([('train',2),('train',4),('development',100)])]
    if duplicate: rows[1]['trajectory_id']=rows[0]['trajectory_id']
    bank=tmp_path/'scored.jsonl';bank.write_text('\n'.join(map(json.dumps,rows)))
    summary=tmp_path/'summary.json';summary.write_text(json.dumps({
        'schema':'eesd-scored-corrections-v1','scored_sha256':hashlib.sha256(bank.read_bytes()).hexdigest(),
        'trajectories':4 if wrong_counts is True else 3,
        'sources':4 if wrong_counts=='sources' else 3,
        'split_counts':{'train':1,'development':2} if wrong_counts=='splits' else {'train':2,'development':1}}))
    cfg=tmp_path/'model.yaml';cfg.write_text('model_id: mock\nrevision: '+ 'a'*40+'\n')
    class Tokenizer:
        eos_token='!'
        def apply_chat_template(self,*args,**kwargs):return 'P'
        def __call__(self,text,**kwargs):return {'input_ids':list(text.encode())}
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())))
    output=tmp_path/'budget.json'
    monkeypatch.setattr(sys,'argv',['plan','--input',str(bank),'--scoring-summary',str(summary),
        '--model-config',str(cfg),'--output',str(output),'--seeds','1701','--max-length',str(max_length)])
    m.main()
    return json.loads(output.read_text())


def test_cli_default_200_steps_16_accumulation_counts_only_training_response_and_eos(tmp_path,monkeypatch):
    report=invoke_plan(tmp_path,monkeypatch)
    # 3200 draws: 1600 of each response, lengths 3 and 5 including EOS.
    assert report['budgets']=={'1701':12800}
    assert report['reference_steps']==200 and report['gradient_accumulation']==16


@pytest.mark.parametrize('option,match', [('duplicate','duplicate|unique'),('wrong_counts','population')])
def test_cli_rejects_inconsistent_sealed_population(tmp_path,monkeypatch,option,match):
    with pytest.raises(ValueError,match=match):invoke_plan(tmp_path,monkeypatch,**{option:True})
    assert not (tmp_path/'budget.json').exists()


def test_cli_rejects_sequence_limit_trainer_cannot_use(tmp_path,monkeypatch):
    with pytest.raises(SystemExit):invoke_plan(tmp_path,monkeypatch,max_length=63)


@pytest.mark.parametrize('field',['sources','splits'])
def test_cli_verifies_each_summary_population_field(tmp_path,monkeypatch,field):
    with pytest.raises(ValueError,match='population'):
        invoke_plan(tmp_path,monkeypatch,wrong_counts=field)
