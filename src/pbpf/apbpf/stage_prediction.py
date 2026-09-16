"""Validate training evidence and aggregate fixed-seed paired prediction losses."""
from collections import Counter
import json

import numpy as np

from pbpf.real_gate import clustered_nll_gap, compare_predictions
from .codearc_bank import file_sha

DOMAINS = {'rbr': 'runbugrun', 'codearc': 'codearc_replay'}
BASELINES = {'pair_aware': 'pair_aware_deterministic', 'deep_sets': 'exchangeable_deep_sets',
             'tuned_dirichlet': 'tuned_dirichlet_rate', 'no_particle_bottleneck': 'no_particle_bottleneck'}
SEED_ESTIMAND = 'mean paired per-example loss across all three fixed seeds; whole-source resampling across seeds'


def load_training(io, kind):
    dependency = 'train_' + kind
    index = json.loads(io.artifact(dependency, 'training-index.json').read_text())
    expected = {(d, s) for d in DOMAINS for s in io.config['protocol']['seeds']}
    keys = [(r['domain'], r['seed']) for r in index['runs']]
    if index['kind'] != kind or len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('training evidence must cover both domains and all fixed seeds exactly')
    result = {}
    for item in index['runs']:
        domain, seed, directory = item['domain'], item['seed'], item['directory']
        report_path = io.artifact(dependency, directory+'/training.json')
        report = json.loads(report_path.read_text())
        config = report['config']
        if (report['schema'] != 'apbpf-stage-training-v1' or report['dataset'] != DOMAINS[domain]
                or config['kind'] != kind or config['seed'] != seed
                or config['protocol'] != io.config['protocol']
                or config['checkpoint_selection_split'] != 'development'
                or config['primary_used_for_training_or_selection'] is not False
                or config['encoder'] != 'frozen_hash_text' or report['counts'] != item['counts']):
            raise ValueError('training report identity or selection protocol mismatch')
        population = report['primary_population']
        sources = Counter(r['source_component_id'] for r in population)
        if (len(population) != 4000 or len(sources) != 500 or set(sources.values()) != {8}
                or len({r['candidate_id'] for r in population}) != len(population)
                or report['counts']['test'] != len(population)):
            raise ValueError('training prediction inventory is not the full 500-by-8 primary population')
        pred_path = io.artifact(dependency, directory+'/primary-predictions.npz')
        with np.load(pred_path, allow_pickle=False) as archive:
            values = {k: archive[k].copy() for k in archive.files}
        arms = {'aligned'} if kind == 'belief' else set(BASELINES)
        if (set(values) != arms | {'labels'} or values['labels'].shape != (len(population)*6,)
                or values['labels'].dtype.kind not in 'iu'
                or any(values[arm].shape != (len(population)*6, 5) for arm in arms)):
            raise ValueError('training predictions have wrong arms or future-test count')
        # Reuse strict finite probability and label validation.
        compare_predictions(values['labels'], {'baseline': values[next(iter(arms))],
                                               **{k: values[k] for k in arms}})
        checkpoints = {}
        for name, arm in report['arms'].items():
            if name == 'tuned_dirichlet':
                smoothing = json.loads(io.artifact(dependency, directory+'/tuned_dirichlet.json').read_text())
                if smoothing != arm:
                    raise ValueError('Dirichlet parameter differs from training evidence')
            else:
                path = io.artifact(dependency, directory+'/'+arm['checkpoint'])
                if file_sha(path) != arm['checkpoint_sha256']:
                    raise ValueError('checkpoint differs from training evidence')
                checkpoints[name] = path
        expected_models = {'belief'} if kind == 'belief' else set(BASELINES)
        if set(report['arms']) != expected_models:
            raise ValueError('training model inventory differs from required arms')
        result[domain, seed] = {'entry': item, 'report': report, 'predictions': values,
                               'checkpoints': checkpoints, 'training_report_sha256': file_sha(report_path),
                               'predictions_sha256': file_sha(pred_path)}
    return result


def aggregate_predictions(seed_values, populations, *, bootstrap_seed, replicates=10000):
    """Average seed-specific losses, not probabilities or selected best seeds.

    All repeated source observations share a cluster across seeds. This does not
    treat three repetitions as three independent source populations.
    """
    seeds = sorted(seed_values)
    if seeds != [1701, 1702, 1703] or set(populations) != set(seeds):
        raise ValueError('all three fixed seeds are required for aggregation')
    population = populations[seeds[0]]
    if any(populations[s] != population for s in seeds):
        raise ValueError('candidate/source order differs across seeds')
    arms = set(seed_values[seeds[0]]) - {'labels'}
    if 'aligned' not in arms or len(arms) < 2:
        raise ValueError('aligned and comparator predictions are required')
    labels = seed_values[seeds[0]]['labels']
    if len(labels) != len(population)*6:
        raise ValueError('six future labels per candidate required')
    per_seed = {}
    clusters = np.repeat([r['source_component_id'] for r in population], 6)
    for seed in seeds:
        values = seed_values[seed]
        if set(values) != arms | {'labels'} or not np.array_equal(values['labels'], labels):
            raise ValueError('prediction arms or labels differ across seeds')
        per_seed[str(seed)] = {arm: clustered_nll_gap(labels, values['aligned'], values[arm], clusters,
            seed=bootstrap_seed, replicates=replicates) for arm in sorted(arms - {'aligned'})}
    stacked_labels = np.tile(labels, len(seeds))
    stacked = {arm: np.concatenate([seed_values[s][arm] for s in seeds]) for arm in arms}
    metrics = compare_predictions(stacked_labels, {'baseline': stacked['aligned'], **stacked})
    gaps = {arm: clustered_nll_gap(stacked_labels, stacked['aligned'], stacked[arm], np.tile(clusters, len(seeds)),
        seed=bootstrap_seed, replicates=replicates) for arm in sorted(arms - {'aligned'})}
    return {'seed_estimand': SEED_ESTIMAND, 'seeds': seeds, 'primary_candidates': len(population),
            'source_components': len(set(clusters)), 'future_examples_per_seed': len(labels),
            'metrics': metrics, 'comparisons': gaps, 'per_seed_comparisons': per_seed}
