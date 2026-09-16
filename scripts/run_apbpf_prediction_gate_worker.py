#!/usr/bin/env python3
"""Apply fixed association or pair-invariance gates to actual stage evidence."""
import argparse
import json

from pbpf.apbpf.stage_prediction import DOMAINS
from pbpf.apbpf.worker_io import WorkerIO


def decision(evidence, config, stage):
    if (evidence['schema'] != 'apbpf-stage-association-v1' or evidence['population'] != 'full_locked_population'
            or set(evidence['domains']) != set(DOMAINS.values())
            or evidence['bootstrap_draws'] != config['protocol']['bootstrap_draws']):
        raise ValueError('gate requires both complete domains and the locked bootstrap protocol')
    domains = {}
    for name, report in evidence['domains'].items():
        if report['seeds'] != config['protocol']['seeds'] or report['source_components'] != 500 or report['primary_candidates'] != 4000:
            raise ValueError('association report must retain all fixed seeds and 500-by-8 sources')
        gaps = report['comparisons']
        if stage == 'association_gate':
            gap = gaps['outcome_shuffled']
            domains[name] = {'gap_nats_per_test': gap['mean_nll_gap'], 'clustered_lower_bound': gap['ci95'][0]}
        else:
            domains[name] = {'shuffle_gap': gaps['outcome_shuffled']['mean_nll_gap'],
                'joint_reversal_degradation': max(0., gaps['joint_reversed']['mean_nll_gap']),
                'presentation_permutation_degradation': max(0., gaps['presentation_permuted']['mean_nll_gap'])}
    metrics = {'population': 'full_locked_population', 'domains': domains}
    if stage == 'association_gate':
        passed = all(r['gap_nats_per_test'] >= config['gates']['association']['minimum_gap_nats_per_test']
                     and r['clustered_lower_bound'] > 0 for r in domains.values())
    else:
        fraction = config['gates']['pair_invariance']['maximum_joint_reversal_fraction_of_shuffle_gap']
        passed = all(r['shuffle_gap'] > 0 and r['joint_reversal_degradation'] <= fraction*r['shuffle_gap']
                     and r['presentation_permutation_degradation'] <= fraction*r['shuffle_gap'] for r in domains.values())
    return {'passed': passed, 'metrics': metrics, 'reason':
            'Both domains meet the unchanged full-population gate' if passed else
            'At least one domain fails the unchanged full-population gate'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['association_gate', 'pair_invariance_gate'], required=True)
    args = parser.parse_args()
    io = WorkerIO(args.stage, ['scripts/run_apbpf_prediction_gate_worker.py', 'src/pbpf/apbpf/stage_prediction.py'])
    evidence = json.loads(io.artifact('association', 'association.json').read_text())
    gate = decision(evidence, io.config, args.stage)
    (io.outputs/'gate-evidence.json').write_text(json.dumps(gate, indent=2)+'\n')
    io.finish({'actual_association_evidence': True, 'scope': 'exploratory fixed-threshold prediction gate'}, gate=gate)


if __name__ == '__main__':
    main()
