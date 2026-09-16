#!/usr/bin/env python3
"""Summarize every fixed particle budget, with paired development-only intervals."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_apbpf_particle_diagnostic import validate_development
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_prediction import aggregate_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    status = json.loads((args.run/'status.json').read_text())
    plan = json.loads((args.run/'plan.json').read_text())
    if (status['status'] != 'complete' or status['plan_sha256'] != file_sha(args.run/'plan.json')
            or status['results_sha256'] != file_sha(args.run/'results.json')
            or file_sha(args.cache) != plan['cache_sha256']):
        raise ValueError('complete bound particle diagnostic required')
    results = json.loads((args.run/'results.json').read_text())['results']
    rows = validate_development(json.loads(args.cache.read_text()))
    population = [{'candidate_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
    expected = {f'{k}-seed{s}-p{p}' for k in ('lexical', 'semantic') for s in plan['seeds'] for p in plan['inference_particles']}
    if set(status['runs']) != expected:
        raise ValueError('no partial or selected-budget summary allowed')
    arrays = {}
    for name, run in status['runs'].items():
        path = args.run/f'{name}.npz'
        if run['status'] != 'complete' or file_sha(path) != run['predictions_sha256']:
            raise ValueError('prediction inventory checksum mismatch')
        with np.load(path, allow_pickle=False) as archive:
            arrays[name] = {k: archive[k].copy() for k in ('aligned', 'labels')}
    summary = []
    for kind in ('lexical', 'semantic'):
        for count in plan['inference_particles']:
            report = results[kind][str(count)]
            association = report['comparisons']['outcome_shuffled']
            row = {'encoder': kind, 'particles': count, 'aligned_nll': report['metrics']['aligned']['nll'],
                   'association_gap': association['mean_nll_gap'], 'association_ci95': association['ci95'],
                   'joint_reversal_gap': report['comparisons']['joint_reversed']['mean_nll_gap'],
                   'presentation_permutation_gap': report['comparisons']['presentation_permuted']['mean_nll_gap'],
                   'mean_unique_initial_particles': float(np.mean([report['particle_diversity'][str(s)][-1]['mean_unique_initial_particles'] for s in plan['seeds']]))}
            if count != 8:
                values = {}
                for seed in plan['seeds']:
                    current, original = arrays[f'{kind}-seed{seed}-p{count}'], arrays[f'{kind}-seed{seed}-p8']
                    if not np.array_equal(current['labels'], original['labels']):
                        raise ValueError('particle-budget comparison labels differ')
                    values[seed] = {**current, 'eight_particle_prediction': original['aligned']}
                paired = aggregate_predictions(values, {s: population for s in plan['seeds']},
                    bootstrap_seed=201701, replicates=10000)
                row['nll_improvement_over_eight_particles'] = paired['comparisons']['eight_particle_prediction']
            summary.append(row)
    result = {'plan_sha256': status['plan_sha256'], 'results_sha256': status['results_sha256'],
              'summary': summary, 'sources': 400, 'candidates': 3200, 'seeds': plan['seeds'],
              'scope': 'development-only frozen-model inference diagnostic; all budgets retained; pointwise exploratory CIs, no primary or trained-method claim',
              'summary_source_sha256': file_sha(__file__)}
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
