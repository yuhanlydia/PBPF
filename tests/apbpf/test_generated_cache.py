import copy

import pytest

from pbpf.apbpf.generated_cache import build_generated_cache
from pbpf.real_gate import public_test_text


def inputs(domain='rbr'):
    name = domain+'/one'
    public = {name: {'task_id':name, 'source_component_id':'source1', 'split':'train',
                    'task_text':'Describe the task', 'visible_tests':[
                        {'id':str(i),'input':str(i),'expected':f'public{i}'} for i in range(4)]}}
    private = {name: {'task_id':name, 'source_component_id':'source1', 'split':'train',
                     'tests':[{'id':str(i),'input':str(i),'expected':f'public{i}' if i<4 else f'PRIVATE{i}',
                               'expected_error':False,'hidden':i>=4} for i in range(10)]}}
    bank = [{'task_id':name,'source_component_id':'source1','split':'train',
             'candidates':[{'candidate_id':f'{name}/{j}','code':'print(0)'} for j in range(8)]}]
    execution = [{'task_id':name,'candidate_id':candidate['candidate_id'], 'tests':[
                    {'test_id':str(i),'stdout':'0','stderr':'','returncode':0,'timed_out':False,
                     'outcome':'WRONG_OUTPUT'} for i in range(10)]} for candidate in bank[0]['candidates']]
    return bank, execution, public, private


@pytest.mark.parametrize('domain', ['rbr','codearc'])
def test_future_answers_never_enter_features_and_all_failed_candidates_remain(domain):
    values = inputs(domain)
    cache = build_generated_cache(*values, domain=domain)
    assert len(cache['records']) == 8
    assert all(r['outcomes'] == ['WRONG_OUTPUT']*10 for r in cache['records'])
    feature = lambda payload: [[public_test_text(t, expected_is_public=domain=='rbr') for t in r['tests']]
                               for r in payload['records']]
    encoded = feature(cache)
    assert 'PRIVATE' not in str(encoded)
    if domain == 'rbr': assert 'public0' in encoded[0][0]
    for row in cache['records']:
        assert all(t['expected']=='' and t['expected_redacted'] for t in row['tests'][4:])
    altered = copy.deepcopy(values)
    for task in altered[-1].values():
        for test in task['tests'][4:]: test['expected'] = 'DIFFERENT_HIDDEN_ANSWER'
    assert feature(build_generated_cache(*altered, domain=domain)) == encoded


def test_primary_or_incomplete_execution_inventory_is_rejected():
    values = inputs()
    primary = copy.deepcopy(values); primary[0][0]['split']='primary'
    with pytest.raises(ValueError,match='primary'):
        build_generated_cache(*primary,domain='rbr')
    incomplete=copy.deepcopy(values);incomplete[1].pop()
    with pytest.raises(ValueError,match='inventory'):
        build_generated_cache(*incomplete,domain='rbr')


def test_public_and_private_visible_examples_must_agree():
    bank, execution, public, private = inputs()
    private['rbr/one']['tests'][0]['expected']='DIFFERENT_PUBLIC_ANSWER'
    with pytest.raises(ValueError,match='public examples'):
        build_generated_cache(bank,execution,public,private,domain='rbr')
