import copy
import importlib.util
from pathlib import Path


def test_protocol_difference_excludes_original_heldout_outcomes_and_counts_recovery():
    spec = importlib.util.spec_from_file_location('stdin_replay',
        Path(__file__).resolve().parents[2]/'scripts/run_local_rbr_stdin_replay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = {'records': [
        {'task_id': 'a', 'split': 'train', 'outcomes': ['PASS', 'WRONG_OUTPUT']},
        {'task_id': 'heldout', 'split': 'test', 'outcomes': ['PASS']}]}
    new = copy.deepcopy(old)
    new['records'][0]['outcomes'][1] = 'PASS'
    new['records'].append({'task_id': 'recovered', 'split': 'development', 'outcomes': ['PASS']})
    result = module.development_difference(old, new)
    assert result['train']['changed_test_outcomes'] == 1
    assert result['development']['recovered_candidates'] == ['recovered']
    new['records'][1]['outcomes'] = ['WRONG_OUTPUT'] * 100
    assert module.development_difference(old, new) == result
