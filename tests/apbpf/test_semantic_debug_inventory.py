import copy
import json

from pbpf.apbpf.stage_cache import build_stage_cache
from test_stage_cache import population
from test_stage_training import module


def test_semantic_debug_excludes_primary_and_execution_evidence():
    worker = module('semantic_debug_inventory', 'run_local_codearc_semantic_debug.py')
    source = build_stage_cache(*population(), domain='rbr')
    source.update(schema='apbpf-codearc-rich-cache-v1', dataset='codearc_replay')
    payload, inventory = worker.prepare_inventory(source)
    changed = copy.deepcopy(source)
    for row in changed['records']:
        if row['split'] == 'test':
            row.update(task_text='PRIVATE_PRIMARY_TASK', candidate='PRIVATE_PRIMARY_CODE')
            for case in row['tests']:
                case['input'] = 'PRIVATE_PRIMARY_TEST'
        for case in row['tests']:
            case.update(expected='PRIVATE_EXPECTED', actual='PRIVATE_ACTUAL', stderr='PRIVATE_STDERR')
    other, other_inventory = worker.prepare_inventory(changed)
    assert inventory == other_inventory
    assert not any(token in json.dumps(other_inventory) for token in
                   ('PRIVATE_PRIMARY', 'PRIVATE_EXPECTED', 'PRIVATE_ACTUAL', 'PRIVATE_STDERR'))
    assert other['evaluation_role'] == 'development_assessment_only'
    assert {r['source_component_id'] for r in other['records']} == {
        r['source_component_id'] for r in source['records'] if r['split'] != 'test'}
    assert payload['problem_counts'] == other['problem_counts']
