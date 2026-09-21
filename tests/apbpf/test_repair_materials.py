import copy
import json

import pytest

from pbpf.apbpf.repair_materials import actor_rows, build_repair_materials
from pbpf.apbpf.stage_cache import build_stage_cache
from test_stage_cache import population


def fixture(domain):
    banks, executions, public, private = population()
    for task in private.values():
        task['reference_code'] = 'PRIVATE_PRIMARY_REFERENCE' if task['split'] == 'primary' else 'print("TRAIN_TARGET")'
        if domain == 'codearc':
            for test in task['tests']:
                test['expected_error'] = True
    if domain == 'codearc':
        for task in public.values():
            for test in task['visible_tests']:
                test['expected_error'] = True
    cache = build_stage_cache(banks, executions, public, private, domain=domain)
    return cache, public, private


@pytest.mark.parametrize('domain', ['rbr', 'codearc'])
def test_repair_actor_has_no_primary_reference_or_future_execution(domain):
    cache, public, private = fixture(domain)
    targets, evaluator, context = build_repair_materials(cache, public, private, domain=domain)
    rows = actor_rows(cache, targets, context)
    assert 'PRIVATE_PRIMARY_REFERENCE' not in json.dumps((targets, evaluator, context, rows))
    assert 'answer4' not in json.dumps(rows)
    assert 'answer4' in json.dumps(evaluator)
    assert all(len(r['tests']) == len(r['outcomes']) == 4 for r in rows)
    assert all(('reference_code' in r) is (r['split'] != 'primary') for r in rows)
    assert {r['split'] for r in targets['records']} == {'train', 'development'}
    if domain == 'codearc':
        assert all(t['expected_error'] is True for r in rows for t in r['tests'])
    changed = copy.deepcopy(private)
    for task in changed.values():
        if task['split'] == 'primary':
            task['reference_code'] = 'DIFFERENT_SECRET_REFERENCE'
        for test in task['tests'][4:]:
            test['expected'] = 'CHANGED_FUTURE_ANSWER'
    new_targets, new_evaluator, new_context = build_repair_materials(cache, public, changed, domain=domain)
    assert targets == new_targets and context == new_context
    assert actor_rows(cache, new_targets, new_context) == rows
    assert evaluator != new_evaluator


@pytest.mark.parametrize('damage', ['source', 'test_order', 'public', 'reference', 'candidate'])
def test_repair_materials_reject_misbound_data(damage):
    cache, public, private = fixture('rbr')
    if damage == 'source': private['task0']['source_component_id'] = 'wrong'
    elif damage == 'test_order': private['task0']['tests'].reverse()
    elif damage == 'public': public['task0']['visible_tests'][0]['input'] = 'wrong'
    elif damage == 'reference': private['task0']['reference_code'] = ''
    else: cache['records'].pop()
    with pytest.raises(ValueError):
        build_repair_materials(cache, public, private, domain='rbr')


def test_actor_rejects_primary_target_injection():
    cache, public, private = fixture('rbr')
    targets, _, context = build_repair_materials(cache, public, private, domain='rbr')
    targets['records'].append({'task_id': 'task3', 'split': 'primary', 'reference_code': 'secret'})
    with pytest.raises(ValueError, match='primary repair reference'):
        actor_rows(cache, targets, context)
