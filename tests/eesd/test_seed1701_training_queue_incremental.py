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


def test_oom_attempt_is_archived_for_bounded_retry(tmp_path):
    module = runpy.run_path(str(QUEUE))
    output = tmp_path / 'seed1701'
    output.mkdir()
    (output / 'tokenization-audit.json').write_text('{}')
    log = tmp_path / 'train.log'
    log.write_text('torch.OutOfMemoryError: CUDA out of memory\n')
    entry = {'task': {'output': str(output)}, 'log': str(log),
             'returncode': 1, 'retry_count': 0}

    assert module['archive_oom_attempt'](entry)
    assert not output.exists()
    assert (tmp_path / 'seed1701-oom-attempt1' /
            'tokenization-audit.json').exists()
    assert entry['failed_attempt_output'].endswith('seed1701-oom-attempt1')
    assert not module['archive_oom_attempt'](entry)


def test_fresh_oom_retry_gets_distinct_log_path():
    module = runpy.run_path(str(QUEUE))
    log_path = module['training_log_path']
    key = 'runbugrun/qwen25_7b/fixed_mass_dirichlet'
    original = log_path(key, {'retry_count': 0, 'resume': False})
    retry = log_path(key, {'retry_count': 1, 'resume': False})
    assert original.name == 'train-runbugrun-qwen25_7b-fixed_mass_dirichlet.log'
    assert retry.name == 'train-runbugrun-qwen25_7b-fixed_mass_dirichlet-retry1.log'
    assert retry != original
