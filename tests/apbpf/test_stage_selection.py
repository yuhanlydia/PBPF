import copy
import json

import numpy as np
import pytest
import torch

from pbpf.apbpf.config import ROOT, resolve_config
from pbpf.apbpf.stage_cache import build_stage_cache
from pbpf.apbpf.stage_prediction import DOMAINS
from pbpf.apbpf.stage_selection import fit_selection, aggregate_selection, NEURAL_BASELINES, warm_start
from pbpf.apbpf.selection import SuccessHead
from pbpf.apbpf.stages import _validate_gate_decision
from test_stage_cache import population
from test_stage_training import module


def cache():
    payload=build_stage_cache(*population(),domain='rbr')
    primary=[r for r in payload['records'] if r['split']=='test'];payload['records']=[r for r in payload['records'] if r['split']!='test']
    for i in range(5):
        for row in primary:
            copied=copy.deepcopy(row);copied.update(task_id=f'{row["task_id"]}-source{i}',problem_id=f'p{i}',source_component_id=f'p{i}')
            payload['records'].append(copied)
    return payload


def test_selection_weights_and_scores_ignore_primary_future_content_and_labels(tmp_path):
    payload=cache();trainer=module('selection_training_fixture','run_apbpf_train_worker.py');helper=module('selection_helper_fixture','run_rbr_prediction_gate.py')
    protocol={'difficulty_dim':8,'diagnosis_dim':24,'particles':8,'lambda_assoc':1.,'lambda_inv':.1,'association_margin':.03}
    args=dict(seed=1701,steps=1,batch_size=8,learning_rate=.0003,feature_dim=16,hidden_dim=16,protocol=protocol,helper=helper)
    for kind in ('belief','baselines'):trainer.train_models(payload,tmp_path/kind,kind=kind,**args)
    checkpoints={arm:tmp_path/'baselines'/f'{arm}.pt' for arm in NEURAL_BASELINES}
    baseline=torch.load(checkpoints['pair_aware'],weights_only=True)
    head=SuccessHead(16,32,16,'pair_aware');warm_start(head,baseline)
    assert torch.equal(head.network[0].weight,baseline['model']['context.0.weight'])
    first=fit_selection(payload,tmp_path/'belief/belief.pt',checkpoints,tmp_path/'first',seed=1701,cache_sha256='a'*64,steps=2)
    changed=copy.deepcopy(payload)
    for row in changed['records']:
        if row['split']=='test':
            for test in row['tests'][4:]:test.update(input='CHANGED_FUTURE_INPUT',expected='CHANGED_FUTURE_ANSWER',outcome='PASS')
            row['outcomes'][4:]=['PASS']*6
    second=fit_selection(changed,tmp_path/'belief/belief.pt',checkpoints,tmp_path/'second',seed=1701,cache_sha256='b'*64,steps=2)
    # Checkpoint files bind their different cache plans, so compare actual tensors.
    for path in (tmp_path/'first').glob('*.pt'):
        a=torch.load(path,weights_only=True)['state'];b=torch.load(tmp_path/'second'/path.name,weights_only=True)['state']
        assert all(torch.equal(a[k],b[k]) for k in a)
    with np.load(tmp_path/'first/candidate-scores.npz') as a,np.load(tmp_path/'second/candidate-scores.npz') as b:
        assert not np.array_equal(a['labels'],b['labels'])
        assert all(np.array_equal(a[k],b[k]) for k in a.files if k!='labels')
    assert first['selections']==second['selections']
    assert len(first['fitting'])==8
    assert {r['comparator'] for r in first['crossfit']} <= set(first['selected_successes'])-{'particle'}
    for fold in first['crossfit']:
        assert set(fold['fitting_sources']).isdisjoint(fold['assessment_sources'])


def test_selection_aggregation_uses_all_seeds_with_source_level_resampling():
    reports={}
    for seed,success in zip((1701,1702,1703),([1,0],[0,1],[0,0])):
        reports[seed]={'seed':seed,'sources':['a','b'],'population':{'primary_candidates':16},
                       'selected_successes':{'particle':success},'cross_fitted_successes':[0,0],
                       'advantage':np.mean(success),'ci95':[0,1],'selected_pass1':{}}
    result=aggregate_selection(reports)
    assert result['absolute_selected_pass1_advantage']==pytest.approx(1/3)
    assert result['ci95']==pytest.approx([1/3,1/3])
    reports[1703]['sources'].reverse()
    with pytest.raises(ValueError,match='source order'):aggregate_selection(reports)


@pytest.mark.parametrize('bad',[False,True])
def test_selection_gate_requires_each_domain_and_agrees_with_runner(bad):
    worker=module('selection_gate_under_test','run_apbpf_selection_gate_worker.py')
    config=resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml','local_exploratory').config
    row={'seeds':[1701,1702,1703],'bootstrap_draws':10000,'primary_sources':500,'primary_candidates':4000,
         'absolute_selected_pass1_advantage':.04,'ci95':[.01,.07]}
    evidence={'schema':'apbpf-stage-selection-v1','comparator':'strongest_cross_fitted_deterministic','domains':{d:copy.deepcopy(row) for d in DOMAINS.values()}}
    if bad:evidence['domains']['codearc_replay']['ci95'][0]=-.001
    gate=worker.decision(evidence,config);assert gate['passed'] is (not bad)
    _validate_gate_decision({'gate':gate},{'backend':'real','stage':'selection_gate'},config)
