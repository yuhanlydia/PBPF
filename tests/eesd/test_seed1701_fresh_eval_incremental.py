"""Fresh evaluation admits each sealed training arm before the matrix finishes."""
import json
from pathlib import Path
import runpy

import pytest


QUEUE = Path(__file__).resolve().parents[2] / (
    'runs/eesd-direct-20260921/operations/queue-seed1701-fresh-evaluations.py')


def test_incremental_training_admission_preserves_binding(tmp_path):
    module = runpy.run_path(str(QUEUE))
    admit = module['admit_completed_training']
    globals_ = admit.__globals__
    globals_['tree_sha'] = lambda path: 'adapter-seal'
    globals_['output_for'] = lambda key, slot: tmp_path / f'{key}-shard{slot}'
    globals_['eval_output'] = lambda key: tmp_path / f'{key}-eval'
    key = 'codearc/qwen25_7b/eesd_full'
    output = tmp_path / 'train'
    output.mkdir()
    binding = output / 'training-binding.json'
    binding.write_text(json.dumps({'schema': 'eesd-training-binding-v1',
                                   'eligibility': {'status': 'estimable'},
                                   'adapter_tree_sha256': 'adapter-seal'}))
    training = {'status': 'running', 'tasks': {
        key: {'status': 'pending', 'task': {'output': str(output)}}}}
    state = {'cells': {}}
    assert not admit(state, training)
    training['tasks'][key]['status'] = 'complete'
    assert admit(state, training)
    assert len(state['cells'][key]['shards']) == 5
    assert not admit(state, training)
    assert len(state['cells']) == 1
    binding.write_text(json.dumps({'schema': 'eesd-training-binding-v1',
                                   'eligibility': {'status': 'estimable'},
                                   'adapter_tree_sha256': 'changed'}))
    with pytest.raises(ValueError, match='binding changed'):
        admit(state, training)
