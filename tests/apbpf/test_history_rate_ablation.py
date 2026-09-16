import copy

import numpy as np
import pytest

from pbpf.real_gate import OUTCOMES
from test_stage_training import module


def test_fixed_rate_ignores_future_labels_and_all_semantic_content():
    worker = module('history_rate_ablation_test', 'run_apbpf_history_rate_ablation.py')
    row = {'outcomes': ['PASS', 'PASS', 'TIMEOUT', 'WRONG_OUTPUT', *['COMPILE_ERROR']*6],
           'task': 'original', 'tests': [{'expected': 'private answer'}]*10}
    expected = np.ones(5)/9
    expected[OUTCOMES.index('PASS')] = 3/9
    expected[OUTCOMES.index('TIMEOUT')] = 2/9
    expected[OUTCOMES.index('WRONG_OUTPUT')] = 2/9
    value = worker.history_predictions([row])
    np.testing.assert_allclose(value, np.tile(expected, (6, 1)))
    changed = copy.deepcopy(row)
    changed.update(task='different', tests=[{'expected': 'secret'}]*10)
    changed['outcomes'][4:] = ['PASS']*6
    np.testing.assert_array_equal(value, worker.history_predictions([changed]))
    changed['outcomes'][0] = 'TIMEOUT'
    assert not np.array_equal(value, worker.history_predictions([changed]))


def test_rate_is_invariant_to_visible_order_but_rejects_missing_observations():
    worker = module('history_rate_permutation_test', 'run_apbpf_history_rate_ablation.py')
    row = {'outcomes': ['PASS', 'WRONG_OUTPUT', 'PASS', 'TIMEOUT', *['PASS']*6]}
    permuted = {'outcomes': row['outcomes'][:4][::-1]+row['outcomes'][4:]}
    np.testing.assert_array_equal(worker.history_predictions([row]), worker.history_predictions([permuted]))
    with pytest.raises(ValueError, match='four public'):
        worker.history_predictions([{'outcomes': ['PASS']*3}])
