import copy

import pytest

from pbpf.apbpf.development import development_cache
from pbpf.apbpf.stage_cache import build_stage_cache, join_primary_executions
from pbpf.real_gate import public_test_text


def population():
    banks, executions, public, private = [], [], {}, {}
    for index, split in enumerate(['train', 'train', 'development', 'primary']):
        name = f'task{index}'
        common = {'task_id': name, 'source_component_id': name, 'split': split}
        tests = [{'id': str(i), 'input': str(i), 'expected': f'answer{i}',
                  'hidden': i >= 4} for i in range(10)]
        public[name] = {**common, 'task_text': f'Problem {name}', 'visible_tests': copy.deepcopy(tests[:4])}
        private[name] = {**common, 'tests': tests}
        candidates = [{'candidate_id': f'{name}/{j}', 'code': '' if j == 0 else 'print(0)'} for j in range(8)]
        banks.append({**common, 'candidates': candidates})
        for candidate in candidates:
            executions.append({'task_id': name, 'candidate_id': candidate['candidate_id'], 'tests': [
                {'test_id': str(i), 'stdout': '0', 'stderr': '', 'returncode': 0, 'timed_out': False,
                 'outcome': 'PASS' if i < 4 else 'WRONG_OUTPUT'} for i in range(10)]})
    return banks, executions, public, private


@pytest.mark.parametrize('domain', ['rbr', 'codearc'])
def test_full_stage_preserves_primary_only_for_assessment_and_redacts_answers(domain):
    values = population()
    payload = build_stage_cache(*values, domain=domain)
    assert payload['counts'] == {'train': 16, 'development': 8, 'test': 8}
    assert all(r['original_split'] == 'primary' for r in payload['records'] if r['split'] == 'test')
    assert sum(r['empty_candidate'] for r in payload['records']) == 4
    assert all(t['expected'] == '' for r in payload['records'] for t in r['tests'][4:])
    changed = copy.deepcopy(values)
    for task in changed[-1].values():
        for test in task['tests'][4:]: test['expected'] = 'SECRET_DIFFERENT'
    assert build_stage_cache(*changed, domain=domain) == payload
    dev = development_cache(payload)
    assert not any(r['problem_id'] == 'task3' for r in dev['records'])
    assert {r['problem_id'] for r in dev['records'] if r['split'] == 'test'} == {'task2'}
    assert 'SECRET' not in str([[public_test_text(t, expected_is_public=domain == 'rbr')
                                for t in r['tests']] for r in payload['records']])


def test_join_primary_phases_retains_every_execution_and_rejects_inventory_drift():
    rows = population()[1][-8:]
    visible = [{**r, 'tests': r['tests'][:4]} for r in rows]
    hidden = [{**r, 'tests': r['tests'][4:]} for r in rows]
    assert join_primary_executions(visible, list(reversed(hidden))) == rows
    with pytest.raises(ValueError, match='inventories differ'):
        join_primary_executions(visible, hidden[:-1])
    with pytest.raises(ValueError, match='phase test inventory'):
        join_primary_executions(hidden, visible)
    with pytest.raises(ValueError, match='duplicate'):
        join_primary_executions(visible + visible[:1], hidden)


@pytest.mark.parametrize('damage', ['source', 'candidate', 'phase', 'test_order', 'public'])
def test_full_cache_rejects_population_or_evidence_mismatch(damage):
    banks, executions, public, private = population()
    if damage == 'source': banks[-1]['source_component_id'] = banks[0]['source_component_id']
    elif damage == 'candidate': banks[-1]['candidates'].pop()
    elif damage == 'phase': executions[-1]['tests'] = executions[-1]['tests'][4:]
    elif damage == 'test_order': executions[-1]['tests'].reverse()
    elif damage == 'public': public['task0']['visible_tests'][0]['expected'] = 'changed'
    with pytest.raises(ValueError):
        build_stage_cache(banks, executions, public, private, domain='rbr')
