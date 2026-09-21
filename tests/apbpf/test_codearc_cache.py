import pytest

from pbpf.apbpf.codearc_cache import build_cache
from pbpf.apbpf.development import development_cache
from pbpf.real_gate import public_test_text


def fixture():
    banks, executions, public, private = [], [], {}, {}
    for group in range(3):
        key, source = f'task{group}', f'source{group}'
        split = 'development' if group == 2 else 'train'
        banks.append({'task_id': key, 'source_component_id': source, 'split': split,
                      'candidates': [{'candidate_id': f'{key}/0', 'code': ''}]})
        public[key] = {'task_text': 'Infer solution from examples', 'split': split, 'source_component_id': source}
        private[key] = {**public[key], 'reference_code': 'GOLD-CANARY',
                        'tests': [{'id': str(i), 'input': f'print(solution({i}))',
                                   'expected': 'EXPECTED-CANARY', 'expected_error': False} for i in range(10)]}
        executions.append({'task_id': key, 'candidate_id': f'{key}/0',
            'tests': [{'test_id': str(i), 'stdout': 'ACTUAL-CANARY', 'stderr': '', 'returncode': 0,
                       'timed_out': False, 'outcome': 'WRONG_OUTPUT'} for i in range(10)]})
    return banks, executions, public, private


def test_conversion_retains_failed_empty_candidates_without_gold_or_feature_leaks():
    payload = build_cache(*fixture())
    assert len(payload['records']) == 3
    assert all(row['empty_candidate'] for row in payload['records'])
    assert 'GOLD-CANARY' not in repr(payload)
    for row in payload['records']:
        for case in row['tests']:
            features = public_test_text(case)
            assert 'EXPECTED-CANARY' not in features
            assert 'ACTUAL-CANARY' not in features
    derived = development_cache(payload)
    assert derived['dataset'] == 'codearc_replay'
    assert derived['evaluation_role'] == 'development_assessment_only'
    assert set(derived['counts']) == {'train', 'development', 'test'}


def test_conversion_rejects_primary_and_incomplete_execution_populations():
    data = fixture()
    data[0][0]['split'] = 'primary'
    with pytest.raises(ValueError, match='primary'):
        build_cache(*data)
    data = fixture()
    data[1].pop()
    with pytest.raises(ValueError, match='inventory'):
        build_cache(*data)
    data = fixture()
    data[1][0]['tests'].reverse()
    with pytest.raises(ValueError, match='reordered'):
        build_cache(*data)
