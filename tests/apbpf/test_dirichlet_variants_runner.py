import copy
from pathlib import Path
import runpy

import numpy as np


def test_future_outcomes_and_private_execution_fields_do_not_enter_predictions():
    api = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/run_dirichlet_variants.py'))
    row = {'outcomes': ['PASS', 'WRONG_OUTPUT', 'PASS', 'WRONG_OUTPUT', 'PASS', 'PASS'],
           'tests': [{'input': str(i), 'actual': 'SECRET', 'stderr': 'SECRET'} for i in range(6)]}
    changed = copy.deepcopy(row)
    changed['outcomes'][4:] = ['TIMEOUT', 'COMPILE_ERROR']
    for t in changed['tests']:
        t['actual'] = 'DIFFERENT'; t['stderr'] = 'DIFFERENT'
    first = api['predict_rows']([row], .1, 4)
    np.testing.assert_array_equal(first, api['predict_rows']([changed], .1, 4))
    assert first.shape == (2, 5)
    np.testing.assert_allclose(first.sum(-1), 1)
