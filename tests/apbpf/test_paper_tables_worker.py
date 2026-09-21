import copy
import csv

import pytest

from pbpf.apbpf.stages import DEPENDENCIES
from test_stage_training import module


def test_tables_preserve_negative_values_and_failed_gates(tmp_path):
    worker = module('paper_tables_test', 'run_apbpf_paper_tables_worker.py')
    results = {stage: {'stage': stage, 'schema': 'apbpf-stage-result-v1',
                      'gate': {'passed': False, 'reason': 'negative effect',
                               'metrics': {'domains': {'runbugrun': {'gap': -.02, 'ci95': [-.03, -.01]}}}}}
               for stage in DEPENDENCIES['paper_tables']}
    gates, values = worker.gate_rows(results, claim_status='exploratory-predeclared')
    assert len(gates) == 7 and all(not row['passed'] for row in gates)
    assert all(row['value_json'] == '-0.02' for row in values if row['metric'].endswith('.gap'))
    assert all(row['value_json'] == '[-0.03, -0.01]' for row in values if row['metric'].endswith('.ci95'))
    path = tmp_path/'gates.csv'
    worker.write_csv(path, gates, ['stage', 'passed', 'reason'])
    with path.open() as stream:
        assert all(row['passed'] == 'False' for row in csv.DictReader(stream))
    changed = copy.deepcopy(results)
    changed.pop('replication_gate')
    with pytest.raises(ValueError, match='every declared'):
        worker.gate_rows(changed, claim_status='exploratory-predeclared')
    with pytest.raises(ValueError, match='claim identity'):
        worker.gate_rows(results, claim_status='prospective-confirmatory')
