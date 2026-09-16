import numpy as np
import torch

from pbpf.apbpf.selection import SuccessHead, cross_fitted_choices, public_batch
from pbpf.real_gate import FrozenTextEncoder


def test_selector_features_ignore_all_future_content():
    row = {'task_text': 'task', 'candidate': 'code',
           'tests': [{'input': str(i), 'expected': 'x'} for i in range(10)],
           'outcomes': ['PASS']*10}
    first = public_batch([row], FrozenTextEncoder(16))
    row['tests'][4:] = [None]*6
    row['outcomes'][4:] = ['secret']*6
    second = public_batch([row], FrozenTextEncoder(16))
    for name in ('task', 'candidate', 'tests', 'outcomes'):
        assert torch.equal(getattr(first, name), getattr(second, name))
    for arm in ('particle', 'pair_aware', 'deep_sets', 'no_particle_bottleneck'):
        head = SuccessHead(16, 8, 12, arm)
        value = head(first, torch.zeros(1, 3, 8), torch.zeros(1, 3))
        assert value.shape == (1,) and 0 <= value.item() <= 1


def test_crossfit_does_not_choose_comparator_using_its_own_fold_labels():
    sources = [f's{i}' for i in range(10)]
    values = {'a': np.ones(10), 'b': np.zeros(10)}
    _, before = cross_fitted_choices(sources, values, seed=1701)
    held = set(before[0]['assessment_sources'])
    values['a'] = np.array([0 if s in held else 1 for s in sources])
    values['b'] = np.array([1 if s in held else 0 for s in sources])
    _, after = cross_fitted_choices(sources, values, seed=1701)
    assert before[0]['comparator'] == after[0]['comparator'] == 'a'
    assert all(set(c['fitting_sources']).isdisjoint(c['assessment_sources']) for c in after)
