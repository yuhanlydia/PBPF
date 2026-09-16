#!/usr/bin/env python3
"""Compare actual staged belief predictions with every matched baseline."""
import json

import numpy as np

from pbpf.apbpf.stage_prediction import BASELINES, DOMAINS, aggregate_predictions, load_training
from pbpf.apbpf.worker_io import WorkerIO


def main():
    io = WorkerIO('baseline_fairness_gate', ['scripts/run_apbpf_fairness_worker.py',
        'src/pbpf/apbpf/stage_prediction.py', 'src/pbpf/real_gate.py'])
    belief, baselines = load_training(io, 'belief'), load_training(io, 'baselines')
    domains = {}
    matched = ['seed', 'steps', 'batch_size', 'learning_rate', 'feature_dim', 'hidden_dim',
               'latent_dim', 'protocol', 'expected_is_public', 'encoder', 'checkpoint_selection_split']
    for domain, dataset in DOMAINS.items():
        seed_values, populations, bindings = {}, {}, []
        for seed in io.config['protocol']['seeds']:
            a, b = belief[domain, seed], baselines[domain, seed]
            if (a['entry']['cache_sha256'] != b['entry']['cache_sha256']
                    or a['report']['primary_population'] != b['report']['primary_population']
                    or a['report']['counts'] != b['report']['counts']
                    or any(a['report']['config'][key] != b['report']['config'][key] for key in matched)
                    or not np.array_equal(a['predictions']['labels'], b['predictions']['labels'])):
                raise ValueError('belief/baseline data, optimization budget or prediction population mismatch')
            seed_values[seed] = {**b['predictions'], 'aligned': a['predictions']['aligned']}
            populations[seed] = a['report']['primary_population']
            bindings.append({'seed': seed, 'cache_sha256': a['entry']['cache_sha256'],
                'belief_report_sha256': a['training_report_sha256'], 'baseline_report_sha256': b['training_report_sha256'],
                'belief_predictions_sha256': a['predictions_sha256'], 'baseline_predictions_sha256': b['predictions_sha256']})
        if len({b['cache_sha256'] for b in bindings}) != 1:
            raise ValueError('fixed seeds must use the identical execution cache')
        domains[dataset] = aggregate_predictions(seed_values, populations, bootstrap_seed=201701,
            replicates=io.config['protocol']['bootstrap_draws'])
        domains[dataset]['bindings'] = bindings
    comparisons = {name: {
        'nll_advantage': min(d['comparisons'][arm]['mean_nll_gap'] for d in domains.values()),
        'clustered_lower_bound': min(d['comparisons'][arm]['ci95'][0] for d in domains.values())}
        for arm, name in BASELINES.items()}
    rule = io.config['gates']['baseline_fairness']
    passed = all(r['nll_advantage'] >= rule['minimum_nll_advantage'] and r['clustered_lower_bound'] > 0
                 for r in comparisons.values())
    metrics = {'baselines': comparisons, 'domain_reduction': 'minimum gap and minimum lower bound across both domains',
               'domains': domains}
    (io.outputs/'fairness.json').write_text(json.dumps(metrics, indent=2)+'\n')
    io.finish({'actual_prediction_comparison': True, 'full_primary_population': True,
               'scope': 'exploratory matched baseline fairness; all fixed seeds and domains retained'},
        gate={'passed': passed, 'reason': 'Every baseline margin and CI passes in both domains' if passed else
              'At least one baseline or domain fails the fixed margin or positive lower bound', 'metrics': metrics})


if __name__ == '__main__':
    main()
