import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('acquisition_replay',
    Path(__file__).resolve().parents[2] / 'scripts/run_apbpf_acquisition_replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class Adapter:
    def __init__(self, *, leak=False):
        self.observed = []
        self.leak = leak

    def remaining(self, state):
        values = {str(i): np.array([[.7, .3], [.3, .7]]) for i in range(4) if str(i) not in state.executed}
        if self.leak:
            values['hidden'] = np.array([[1., 0.], [0., 1.]])
        return values

    def update(self, state, test_id, outcome):
        self.observed.append((test_id, outcome))
        updated = SimpleNamespace(component_probs=state.component_probs, executed=(*state.executed, test_id))
        return updated, self.remaining(updated)


@pytest.mark.parametrize('policy', ['fixed', 'random', 'diagnostic_mi', 'predictive_entropy'])
def test_replay_observes_exact_public_budget(policy):
    adapter = Adapter()
    initial = SimpleNamespace(component_probs=np.array([.5, .5]), executed=())
    state, selected = replay.follow(adapter, initial, {str(i): i for i in range(4)},
        budget=2, policy=policy, rng=np.random.default_rng(1))
    assert len(selected) == len(set(selected)) == len(adapter.observed) == 2
    assert set(selected) <= {'0', '1', '2', '3'}
    assert state.executed == tuple(selected)
    assert initial.executed == ()


def test_replay_rejects_hidden_query_before_observing_it():
    adapter = Adapter(leak=True)
    initial = SimpleNamespace(component_probs=np.array([.5, .5]), executed=())
    with pytest.raises(ValueError, match='hidden or repeated'):
        replay.follow(adapter, initial, {str(i): i for i in range(4)}, budget=1,
                      policy='diagnostic_mi', rng=np.random.default_rng(1))
    assert adapter.observed == []
