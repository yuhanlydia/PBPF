#!/usr/bin/env python3
"""Check direction agreement over the complete fixed domain/family matrix."""
import json
import math
import shutil

from pbpf.apbpf.stage_prediction import DOMAINS
from pbpf.apbpf.worker_io import WorkerIO


def decision(evidence, config):
    if evidence['schema'] != 'apbpf-stage-replication-v1':
        raise ValueError('stage-bound replication evidence required')
    expected = {(d, m['family']) for d in DOMAINS.values() for m in config['models'].values()}
    cells = evidence['cells']
    keys = [(row['domain'], row['family']) for row in cells]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('all domain/family cells required exactly once')
    for row in cells:
        if (row['seeds'] != config['protocol']['seeds'] or row['primary_sources'] != 500
                or row['primary_candidates'] != 4000 or row['bootstrap_draws'] != 10000):
            raise ValueError('replication requires all seeds and the full locked populations')
        for key in ('association_gap', 'selection_advantage'):
            if type(row[key]) not in (int, float) or not math.isfinite(row[key]):
                raise ValueError('finite numerical replication effects required')
    metrics = {'cells': [{k: row[k] for k in ('domain', 'family', 'association_gap', 'selection_advantage')}
                         for row in cells]}
    passed = all(row['association_gap'] > 0 and row['selection_advantage'] > 0 for row in cells)
    return {'passed': passed, 'metrics': metrics,
            'reason': 'Both effect directions are positive in every domain/family cell' if passed else
                      'At least one domain/family cell lacks positive association or selection advantage'}


def main():
    io = WorkerIO('replication_gate', ['scripts/run_apbpf_replication_gate_worker.py',
                                      'src/pbpf/apbpf/stage_prediction.py'])
    path = io.artifact('replication', 'replication.json')
    evidence = json.loads(path.read_text())
    gate = decision(evidence, io.config)
    (io.outputs/'gate-evidence.json').write_text(json.dumps(gate, indent=2)+'\n')
    shutil.copyfile(path, io.outputs/'replication-evidence.json')
    io.finish({'complete_domain_family_matrix': True,
               'scope': 'exploratory direction agreement; stricter upstream gates remain binding'}, gate=gate)


if __name__ == '__main__':
    main()
