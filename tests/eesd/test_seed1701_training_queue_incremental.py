"""A sealed cell can enter the training queue before the other cells finish."""
import json
from pathlib import Path
import runpy

import pytest


QUEUE = Path(__file__).resolve().parents[2] / (
    'runs/eesd-direct-20260921/operations/queue-seed1701-training-arms.py')


def test_incremental_plan_admission_is_idempotent_and_sealed(tmp_path):
    module = runpy.run_path(str(QUEUE))
    available = module['available_plans']
    state = module['initial_state']({})
    globals_ = available.__globals__
    cell = ('codearc', 'gemma3_4b')
    path = tmp_path / 'training-commands.json'
    globals_['plan_paths'] = lambda: {cell: path}
    globals_['save'] = lambda value: None

    assert available() == {}
    rules = [f'rule_{index}' for index in range(7)]
    tasks = []
    for rule in rules:
        tasks.append({'request': {'rule': rule, 'seed': 1701,
                      'scored': str(globals_['run'] /
                                    'corrections/codearc/gemma3_4b/round1/scored-corrections.jsonl')},
                      'output': str(globals_['run'] /
                                    f'training/codearc/gemma3_4b/round1/{rule}/seed1701'),
                      'eligibility': {'status': 'estimable'}})
    plan = {'schema': 'eesd-training-command-plan-v1',
            'baseline_rows': [{'rule': 'no_update', 'seed': 1701}],
            'training_tasks': tasks}
    path.write_text(json.dumps(plan))

    globals_['add_plans'](state, available())
    assert len(state['tasks']) == 7
    assert len(state['plan_sha256']) == 1
    globals_['add_plans'](state, available())
    assert len(state['tasks']) == 7

    plan['extra'] = 'changed after admission'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='changed after admission'):
        globals_['add_plans'](state, available())
