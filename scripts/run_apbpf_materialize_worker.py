#!/usr/bin/env python3
"""Real two-domain materialization worker for the exploratory local protocol."""
import hashlib
import json
import os
from pathlib import Path

from pbpf.apbpf.config import digest as config_digest
from pbpf.apbpf.codearc_materialize import materialize as codearc_materialize
from pbpf.apbpf.rbr_materialize import materialize as rbr_materialize


def main():
    request_path = Path(os.environ['APBPF_REQUEST'])
    request = json.loads(request_path.read_text())
    if (request['stage'] != 'materialize' or request['dependencies'] or request['inputs']
            or request['fingerprint'] != config_digest(request['config'])):
        raise ValueError('requires a valid root materialize request')
    if request['confirmatory'] or request['claim_status'] != 'exploratory-predeclared':
        raise ValueError('new RBR source partition requires predeclared exploratory execution')
    config = request['config']
    expected = {'runbugrun': '374251a9d65410f37e1136049cb7ff5dcca3d0ae',
                'codearc_replay': '32f2a1e1ba3bdf8a7d5057aacbd665edaf441ce0'}
    if any(config['datasets'][name]['revision'] != revision for name, revision in expected.items()):
        raise ValueError('worker dataset revision contract mismatch')
    root = Path(config['site']['working_directory'])
    for name in ('scripts/run_apbpf_materialize_worker.py', 'src/pbpf/apbpf/rbr_materialize.py',
                 'src/pbpf/apbpf/codearc_materialize.py', 'src/pbpf/real_gate.py'):
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != config['source_hashes'][name]:
            raise ValueError('worker source changed after fingerprinting')
    outputs = Path(request['outputs_directory'])
    if outputs.resolve() != Path(os.environ['APBPF_OUTPUTS']).resolve():
        raise ValueError('output directory mismatch')
    data = Path(config['site']['paths']['dataset_root'])
    seed = config['protocol']['seeds'][0]
    rbr = rbr_materialize(data/'runbugrun-v0.0.1', data/'project_codenet/problem_descriptions.tar.gz',
                         outputs/'rbr-public', outputs/'rbr-evaluator', seed=seed)
    codearc = codearc_materialize(data/'codearc', outputs/'codearc-public', outputs/'codearc-evaluator', seed=seed)
    summary = {'scientific_measurements': False, 'actual_raw_data_materialized': True,
               'domains': {name: {'source_components': value['source_components'],
                                  'component_counts': value['component_counts']}
                           for name, value in [('runbugrun', rbr), ('codearc_replay', codearc)]},
               'rbr_fresh_belief_training_required': True,
               'claim_status': 'exploratory-predeclared',
               'firewall': 'generator must mount only the domain public directory; evaluator directories never mounted'}
    (outputs/'materialization-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    artifacts = [{'path': str(path.relative_to(outputs.parent)),
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                 for path in sorted(outputs.rglob('*')) if path.is_file()]
    result = {'schema': 'apbpf-stage-result-v1', 'stage': request['stage'],
              'fingerprint': request['fingerprint'], 'dependencies': request['dependencies'],
              'summary': summary, 'artifacts': artifacts}
    with Path(os.environ['APBPF_RESULT']).open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
