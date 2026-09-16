import json
from pbpf.apbpf.rbr_materialize import build_records


def test_source_split_and_public_firewall_do_not_depend_on_fixed_program():
    bugs = [{'id': str(i), 'problem_id': f'p{i}', 'buggy_code': 'print(input())',
             'fixed_code': 'SECRET_FIXED'} for i in range(6)]
    tests = [{'id': f'{i}-{j}', 'problem_id': f'p{i}', 'input': str(j), 'output': f'answer{i}-{j}'}
             for i in range(6) for j in range(12)]
    descriptions = {f'p{i}': 'description' for i in range(6)}
    public, private, inventory = build_records(bugs, tests, descriptions, seed=1701,
        development_components=2, primary_components=2)
    assert inventory['component_counts'] == {'primary': 2, 'development': 2, 'train': 2}
    assert 'SECRET_FIXED' not in json.dumps(public)
    for visible, hidden in zip(public, private):
        assert len(visible['visible_tests']) == 4 and len(hidden['tests']) == 10
        assert all(t['hidden'] is (i >= 4) for i, t in enumerate(hidden['tests']))
        for test in hidden['tests'][4:]:
            assert json.dumps(test['expected']) not in json.dumps(visible)
    for bug in bugs:
        bug['fixed_code'] = 'this is invalid syntax !!!'
    again, _, same = build_records(bugs, tests, descriptions, seed=1701,
        development_components=2, primary_components=2)
    assert again == public and same == inventory
