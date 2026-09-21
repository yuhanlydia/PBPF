#!/usr/bin/env python3
"""Report the locked hard-bank gate from exact stage dependency evidence."""
import json

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.config import digest
from pbpf.apbpf.worker_io import WorkerIO


def sealed(path):
    value = json.loads(path.read_text())
    checksum = value.pop('content_sha256')
    if checksum != digest(value):
        raise ValueError('sealed evidence content checksum mismatch')
    return value


def main():
    io = WorkerIO('hard_bank_gate', ['scripts/run_apbpf_hard_bank_gate_worker.py',
        'src/pbpf/apbpf/config.py', 'src/pbpf/apbpf/stages.py'])
    lock_path = io.artifact('hard_bank_lock', 'population-lock.json')
    lock = sealed(lock_path)
    audit = sealed(io.artifact('hard_bank', 'audit.json'))
    pilot = sealed(io.artifact('hard_bank', 'pilot.json'))
    provenance = json.loads(io.artifact('hard_bank', 'execution-provenance.json').read_text())
    if (audit['lock_artifact_sha256'] != file_sha(lock_path)
            or provenance['lock_stage_completion_sha256'] != io.request['dependencies']['hard_bank_lock']):
        raise ValueError('audit/execution do not bind the exact current pre-hidden lock')
    values = audit['audit']
    metrics = {'lock_schema': lock['schema'], 'audit_schema': audit['schema'],
        'lock_artifact_sha256': file_sha(lock_path),
        'candidate_inventory_sha256': lock['candidate_inventory_sha256'],
        'source_inventory_sha256': lock['source_inventory_sha256'],
        'lock_precedes_hidden_execution': True, 'candidate_inventory_bound': True,
        'source_inventory_bound': True,
        'pilot_primary_source_disjoint': audit['gate']['pilot_primary_source_disjoint'],
        'mixed_pilot_groups': pilot['mixed_groups'], 'confirmatory_groups': values['groups'],
        'visible_selection_pass1': values['visible_selected_pass_at_1'],
        'hidden_oracle_pass1': values['hidden_oracle_pass_at_k'], 'domains': pilot['domains']}
    passed = audit['gate']['passes']
    (io.outputs/'gate-evidence.json').write_text(json.dumps(metrics, indent=2)+'\n')
    io.finish({'actual_gate_evidence': True, 'scope': 'exploratory hard-bank gate; thresholds unchanged'},
              gate={'passed': passed, 'reason': 'All locked hard-bank conditions met' if passed else
                    'At least one locked pilot, population, disjointness or headroom condition failed',
                    'metrics': metrics})


if __name__ == '__main__':
    main()
