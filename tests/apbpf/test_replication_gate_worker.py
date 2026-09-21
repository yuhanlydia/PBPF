import copy

import pytest

from pbpf.apbpf.config import ROOT, resolve_config
from pbpf.apbpf.stages import _validate_gate_decision
from test_stage_training import module


def evidence():
    return {'schema': 'apbpf-stage-replication-v1', 'cells': [
        {'domain': domain, 'family': family, 'seeds': [1701, 1702, 1703],
         'primary_sources': 500, 'primary_candidates': 4000, 'bootstrap_draws': 10000,
         'association_gap': .01, 'selection_advantage': .01}
        for domain in ('runbugrun', 'codearc_replay') for family in ('qwen', 'deepseek')]}


@pytest.mark.parametrize('gap', [.01, 0., -.01])
def test_replication_gate_matches_locked_direction_rule(gap):
    worker = module('replication_gate_test', 'run_apbpf_replication_gate_worker.py')
    config = resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory').config
    value = evidence()
    value['cells'][3]['association_gap'] = gap
    result = worker.decision(value, config)
    assert result['passed'] is (gap > 0)
    _validate_gate_decision({'gate': result}, {'stage': 'replication_gate', 'backend': 'real'}, config)


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'seed', 'population', 'nonfinite'])
def test_replication_gate_rejects_partial_or_invalid_evidence(damage):
    worker = module('replication_gate_reject_test', 'run_apbpf_replication_gate_worker.py')
    config = resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory').config
    value = evidence()
    if damage == 'missing': value['cells'].pop()
    elif damage == 'duplicate': value['cells'].append(copy.deepcopy(value['cells'][0]))
    elif damage == 'seed': value['cells'][0]['seeds'] = [1701]
    elif damage == 'population': value['cells'][0]['primary_sources'] = 16
    else: value['cells'][0]['selection_advantage'] = float('nan')
    with pytest.raises(ValueError):
        worker.decision(value, config)
