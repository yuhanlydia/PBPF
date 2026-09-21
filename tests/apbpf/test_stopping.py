import numpy as np
import pytest

from pbpf.apbpf.stopping import lock_threshold, stopped_budgets


def test_stopping_uses_first_crossing_and_never_exceeds_four():
    information = [[.2, .1, .01], [.001, .2, .3], [.4, .3, .2]]
    assert stopped_budgets(information, .02).tolist() == [3, 1, 4]
    assert stopped_budgets(information, None).tolist() == [4, 4, 4]
    with pytest.raises(ValueError):
        stopped_budgets([[.1, np.nan, .2]], .1)


def test_calibration_rejects_cheap_bad_quality_and_has_fixed_four_fallback():
    information = [[.2, .1, .01], [.2, .1, .01]]
    result = lock_threshold(information, [[.8, .6, .4, .4], [.8, .6, .4, .4]], [.02, .15, .3])
    assert result['threshold'] == .02
    result = lock_threshold(information, [[.8, .6, .5, .4], [.8, .6, .5, .4]], [.02, .15, .3])
    assert result['threshold'] is None
