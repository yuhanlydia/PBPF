import copy
import hashlib

import pytest

from pbpf.eesd.replay_materialization import choose_representatives, select_population, select_training_reserve


MODELS = ('qwen25_7b', 'deepseek_6p7b', 'seed_coder_8b', 'starcoder2_15b')


def member(domain, task_id, source_id):
    return {'domain': domain, 'task_id': task_id, 'source_id': source_id,
            'statement': 'Read a number and print it.',
            'visible_tests': [{'input': str(i), 'output': str(i)} for i in range(4)]}


def token_rows(rows):
    return [{**{k: r[k] for k in ('domain', 'task_id', 'source_id')},
             'models': {key: {'input_tokens': 100} for key in MODELS}}
            for r in rows]


def test_cc_first_uses_fitting_member_only():
    rows = [member('apps_replay', 'apps:1', 'shared'),
            member('codecontests_replay', 'cc:1', 'shared')]
    tokens = token_rows(rows)
    assert [r['task_id'] for r in choose_representatives(rows, tokens, set())] == ['cc:1']
    tokens[1]['models']['starcoder2_15b']['input_tokens'] = 4097
    tokens[1]['all_four_fit'] = True  # Never trust a stale derived fit flag.
    assert [r['task_id'] for r in choose_representatives(rows, tokens, set())] == ['apps:1']


def test_quarantine_survives_fitting_members():
    rows = [member('apps_replay', 'apps:1', 'bridge')]
    assert choose_representatives(rows, token_rows(rows), {'bridge'}) == []


def test_quarantine_rejects_a_bare_string_instead_of_splitting_it():
    rows = [member('apps_replay', 'apps:1', 'bridge')]
    with pytest.raises(ValueError, match='collection'):
        choose_representatives(rows, token_rows(rows), 'bridge')


def test_representative_uses_hash_not_shortest_prompt():
    rows = [member('apps_replay', f'apps:{i}', 'shared') for i in range(3)]
    expected = min(rows, key=lambda r: (
        hashlib.sha256(('eesd-replay-representative-v1|' + r['task_id']).encode()).hexdigest(),
        r['task_id']))
    tokens = token_rows(rows)
    for row in tokens:
        for value in row['models'].values():
            value['input_tokens'] = 4096 if row['task_id'] == expected['task_id'] else 50
    assert choose_representatives(rows[::-1], tokens[::-1], set()) == [expected]


@pytest.mark.parametrize('mutation', [
    lambda rows, tokens: tokens.pop(),
    lambda rows, tokens: tokens[0]['models'].pop('seed_coder_8b'),
    lambda rows, tokens: tokens[0].update(source_id='another'),
    lambda rows, tokens: tokens[0]['models']['qwen25_7b'].update(input_tokens=True),
    lambda rows, tokens: tokens[0]['models']['qwen25_7b'].update(input_tokens=0),
    lambda rows, tokens: rows.append(copy.deepcopy(rows[0])),
    lambda rows, tokens: tokens.append(copy.deepcopy(tokens[0])),
])
def test_invalid_or_unpaired_token_audit_is_rejected(mutation):
    rows = [member('apps_replay', 'apps:1', 'one')]
    tokens = token_rows(rows)
    mutation(rows, tokens)
    with pytest.raises(ValueError):
        choose_representatives(rows, tokens, set())


def test_source_split_is_exact_disjoint_and_order_invariant():
    rows = [member(domain, f'{domain}:{i}', f'{domain}:source:{i}')
            for domain in ('apps_replay', 'codecontests_replay') for i in range(7)]
    selected = select_population(rows, token_rows(rows), set(), development=2, primary=3)
    assert selected == select_population(rows[::-1], token_rows(rows)[::-1], set(), development=2, primary=3)
    sources = []
    for domain, values in selected.items():
        assert [split for split, row in values] == ['development'] * 2 + ['primary'] * 3
        assert all(row['domain'] == domain for split, row in values)
        sources.extend(row['source_id'] for split, row in values)
    assert len(sources) == len(set(sources)) == 10


def test_insufficient_domain_does_not_silently_reduce_population():
    rows = [member('apps_replay', f'apps:{i}', str(i)) for i in range(5)]
    with pytest.raises(ValueError, match='codecontests_replay'):
        select_population(rows, token_rows(rows), set(), development=2, primary=3)


def test_training_reserve_follows_assessment_prefix_without_source_overlap():
    rows = [member(domain, f'{domain}:{i}', f'{domain}:source:{i}')
            for domain in ('apps_replay', 'codecontests_replay') for i in range(9)]
    assessment = select_population(rows, token_rows(rows), set(), development=2, primary=3)
    train = select_training_reserve(rows, token_rows(rows), set(), development=2, primary=3, train=2)
    assert train == select_training_reserve(rows[::-1], token_rows(rows)[::-1], set(),
                                            development=2, primary=3, train=2)
    assessment_sources = {r['source_id'] for values in assessment.values() for _, r in values}
    training_sources = {r['source_id'] for values in train.values() for split, r in values if split == 'train'}
    assert len(assessment_sources) == 10
    assert len(training_sources) == 4
    assert assessment_sources.isdisjoint(training_sources)


def test_training_reserve_requires_full_population():
    rows = [member(domain, f'{domain}:{i}', f'{domain}:source:{i}')
            for domain in ('apps_replay', 'codecontests_replay') for i in range(6)]
    with pytest.raises(ValueError, match='requires 7'):
        select_training_reserve(rows, token_rows(rows), set(), development=2, primary=3, train=2)
