"""Legacy-cache development screen; real repair/allocation results remain pending."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
from pbpf.apbpf.dirichlet import predict_rate, similarity_weights
from pbpf.real_gate import FrozenTextEncoder, compare_predictions, clustered_nll_gap, validate_rbr_cache
from pbpf.registry import OUTCOMES

ALPHAS = (.01, .1, .25, .5, 1., 2., 5., 10.)
STRENGTHS = (0., 1., 4., 16.)
_encode = lru_cache(maxsize=20000)(FrozenTextEncoder(256))


def predict_rows(rows, alpha, strength):
    predictions = []
    for row in rows:
        visible = [OUTCOMES.index(x) for x in row['outcomes'][:4]]
        history = np.stack([_encode(t['input']) for t in row['tests'][:4]])
        for test in row['tests'][4:]:
            weights = similarity_weights(_encode(test['input']), history, strength)
            predictions.append(predict_rate(visible, alpha, weights=weights))
    return np.asarray(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(args.cache.read_text())
    validate_rbr_cache(payload)
    parts = runpy.run_path(str(root / 'scripts/run_apbpf_parameter_diagnostic.py'))[
        'partition_development'](payload['records'])
    args.output.mkdir(parents=True, exist_ok=False)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def write(name, data):
        with (args.output / name).open('x') as f:
            json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')
    source_paths = [Path(__file__), root / 'src/pbpf/apbpf/dirichlet.py',
                    root / 'src/pbpf/real_gate.py', root / 'scripts/run_apbpf_parameter_diagnostic.py']
    plan = {'claim_status': 'exploratory-legacy-literal-stdin-not-corrected-protocol',
        'alpha_grid': ALPHAS, 'strength_grid': STRENGTHS, 'feature': 'public test input lexical cosine,256dimensions',
        'weight_normalization': 'sum equals number of visible observations; no evidence-mass inflation',
        'expected_outputs_used': False, 'original_test_used': False,
        'candidates': {k: len(v) for k, v in parts.items()},
        'cache_sha256': sha(args.cache),
        'partition_sha256': hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest(),
        'source_sha256': {str(p.relative_to(root)): sha(p) for p in source_paths},
        'blocked_real_arms': {
            'discounted_inheritance': 'parent-child repair trajectories absent locally',
            'rollout_allocation': 'complete action banks absent and namespace isolation unavailable'},
        'primary_endpoint': 'equal-budget final selected pass rate remains unmeasured',
        'screen_endpoint': 'future execution outcome NLL on original development',
        'bootstrap': '10000source-cluster draws; descriptive, not selection-adjusted'}
    write('plan.json', plan)
    labels = {split: np.asarray([OUTCOMES.index(y) for r in rows for y in r['outcomes'][4:]])
              for split, rows in parts.items() if split != 'train'}
    scores = []
    for strength in STRENGTHS:
        for alpha in ALPHAS:
            p = predict_rows(parts['validation'], alpha, strength)
            nll = compare_predictions(labels['validation'], {'baseline': p})['baseline']['nll']
            scores.append({'alpha': alpha, 'strength': strength, 'validation_nll': nll})
    selected = {
        'dirichlet': min((s for s in scores if s['strength'] == 0), key=lambda s: s['validation_nll']),
        'weighted': min(scores, key=lambda s: (s['validation_nll'], s['strength'], s['alpha']))}
    write('selection.json', {'selected': selected, 'all_validation_scores': scores})
    predictions = {arm: predict_rows(parts['assessment'], s['alpha'], s['strength'])
                   for arm, s in selected.items()}
    # Negative control: erase correspondence between public features and visible outcomes.
    from pbpf.apbpf.counterfactual import outcome_derangement
    outcomes = np.asarray([[OUTCOMES.index(y) for y in r['outcomes']] for r in parts['assessment']])
    shuffled = outcome_derangement(outcomes, 4, 1701)
    changed = [{**r, 'outcomes': [OUTCOMES[int(y)] for y in values]}
               for r, values in zip(parts['assessment'], shuffled)]
    s = selected['weighted']
    predictions['weighted_shuffled'] = predict_rows(changed, s['alpha'], s['strength'])
    clusters = np.asarray([r.get('source_component_id', r['problem_id'])
                           for r in parts['assessment'] for _ in r['outcomes'][4:]])
    report = {'claim_status': plan['claim_status'], 'selected': selected,
        'metrics': compare_predictions(labels['assessment'], {'baseline': predictions['dirichlet'], **predictions}),
        'weighted_gain_over_dirichlet': clustered_nll_gap(labels['assessment'], predictions['weighted'],
            predictions['dirichlet'], clusters, seed=2701),
        'association_gap': clustered_nll_gap(labels['assessment'], predictions['weighted'],
            predictions['weighted_shuffled'], clusters, seed=2701),
        'real_rollout_or_inheritance_results': None}
    np.savez_compressed(args.output / 'predictions.npz', **predictions,
                        labels=labels['assessment'], clusters=clusters)
    write('report.json', report)
    write('complete.json', {'files': {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}})
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
