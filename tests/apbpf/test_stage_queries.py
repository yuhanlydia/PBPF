import copy

import numpy as np
import pytest

from pbpf.apbpf.config import ROOT, resolve_config
from pbpf.apbpf.stage_cache import build_stage_cache
from pbpf.apbpf.stage_prediction import DOMAINS
from pbpf.apbpf.stage_queries import replay_queries, summarize_queries, POLICIES
from pbpf.apbpf.stages import _validate_gate_decision
from test_stage_cache import population
from test_stage_training import module


@pytest.fixture
def trained(tmp_path):
    trainer = module('query_training_fixture','run_apbpf_train_worker.py')
    helper = module('query_training_helper','run_rbr_prediction_gate.py')
    payload = build_stage_cache(*population(),domain='rbr')
    protocol = {'difficulty_dim':8,'diagnosis_dim':24,'particles':8,'lambda_assoc':1.,'lambda_inv':.1,'association_margin':.03}
    output = tmp_path/'training'
    trainer.train_models(payload,output,kind='belief',seed=1701,steps=1,batch_size=8,learning_rate=.0003,
        feature_dim=16,hidden_dim=16,protocol=protocol,helper=helper)
    return [r for r in payload['records'] if r['split']=='test'],output/'belief.pt'


def test_query_oracle_includes_fixed_and_random_and_has_no_subset_headroom_at_four(trained):
    rows,checkpoint = trained
    records = list(replay_queries(rows,checkpoint,seed=1701,mode='oracle',batch_size=3))
    assert len(records)==8
    for row in records:
        for budget in (1,2,4):
            policies = row['budgets'][str(budget)]
            for reference in ('fixed','random_canonical'):
                assert np.mean(policies['oracle']['future_nll']) <= np.mean(policies[reference]['future_nll'])+1e-7
            if budget==4:
                assert policies['oracle']['future_nll']==policies['fixed']['future_nll']==policies['random_canonical']['future_nll']
    summary = summarize_queries({s:[{**r,'seed':s} for r in records] for s in (1701,1702,1703)},mode='oracle',draws=100)
    assert summary['source_components']==1
    assert summary['summary']['4']['oracle']['against_fixed']['mean_nll_gap']==0


def test_active_acquisition_cannot_depend_on_hidden_labels(trained):
    rows,checkpoint = trained
    first = list(replay_queries(rows,checkpoint,seed=1701,mode='active',batch_size=4))
    changed = copy.deepcopy(rows)
    for row in changed:
        for test in row['tests'][4:]:test['outcome']='PASS'
        row['outcomes'][4:]=['PASS']*6
    second = list(replay_queries(changed,checkpoint,seed=1701,mode='active',batch_size=4))
    assert first[0]['budgets']['4']['fixed']['future_nll'] != second[0]['budgets']['4']['fixed']['future_nll']
    for a,b in zip(first,second,strict=True):
        for budget in (1,2,3,4):
            assert set(a['budgets'][str(budget)])==set(POLICIES)
            for policy in POLICIES:
                x,y=a['budgets'][str(budget)][policy],b['budgets'][str(budget)][policy]
                assert x['selected']==y['selected']
                assert len(set(x['selected']))==budget
                if policy=='diagnostic_mi' and budget<4:
                    assert x['remaining_information']==y['remaining_information']
    damaged=copy.deepcopy(first);damaged[0]['budgets']['4']['fixed']['selected']=[0,0,1,2]
    with pytest.raises(ValueError,match='exact public budget'):
        summarize_queries({s:[{**r,'seed':s} for r in damaged] for s in (1701,1702,1703)},mode='active',draws=100)


@pytest.mark.parametrize('stage',['oracle_headroom_gate','active_testing_gate'])
@pytest.mark.parametrize('bad',[False,True])
def test_query_gate_matches_runner_and_requires_both_domains(stage,bad):
    worker=module('query_gate_under_test','run_apbpf_query_gate_worker.py')
    config=resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml','local_exploratory').config
    part={'against_fixed':{'mean_nll_gap':.04},'against_random':{'mean_nll_gap':.04}}
    row={'seeds':[1701,1702,1703],'source_components':500,'candidates_per_seed':4000,
         'all_policies_execute_exact_budget':True,'summary':{'2':{'oracle':copy.deepcopy(part)},'4':{'diagnostic_mi':copy.deepcopy(part)}},
         'stopping':{'matched_hidden_quality':False,'test_reduction':.5}}
    evidence={'schema':'apbpf-stage-query-replay-v1','stage':stage.removesuffix('_gate'),'public_pool':4,'oracle_gate_budget':2,
              'domains':{d:copy.deepcopy(row) for d in DOMAINS.values()}}
    if bad:
        target=evidence['domains']['codearc_replay']['summary']['4' if stage=='active_testing_gate' else '2']
        target['diagnostic_mi' if stage=='active_testing_gate' else 'oracle']['against_random']['mean_nll_gap']=-.01
    gate=worker.decision(evidence,config,stage)
    assert gate['passed'] is (not bad)
    _validate_gate_decision({'gate':gate},{'backend':'real','stage':stage},config)
