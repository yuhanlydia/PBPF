"""Frozen-model probability combination: exploratory development data only."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np


def blend_probabilities(rate, neural, weight):
    rate, neural = np.asarray(rate), np.asarray(neural)
    if (rate.ndim != 2 or rate.shape != neural.shape or not rate.size
            or not np.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError('invalid probability shapes or weight')
    for p in (rate, neural):
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1., atol=1e-5):
            raise ValueError('invalid probabilities')
    return (1 - weight) * rate + weight * neural


def select_weight(rate, neural, labels):
    """Fit using supplied inner-validation labels only; ties favor rate model."""
    return float(min(np.linspace(0, 1, 21), key=lambda w:
        -np.log(blend_probabilities(rate, neural, w)[np.arange(len(labels)), labels].clip(1e-12)).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from pbpf.real_gate import validate_rbr_cache, clustered_nll_gap
    repo = Path(__file__).resolve().parents[1]
    gate = runpy.run_path(str(repo / 'scripts/run_rbr_prediction_gate.py'))
    diagnostic = runpy.run_path(str(repo / 'scripts/run_apbpf_parameter_diagnostic.py'))
    payload = json.loads(args.cache.read_text())
    validate_rbr_cache(payload)
    parts = diagnostic['partition_development'](payload['records'])
    old_plan = json.loads((args.checkpoints / 'plan.json').read_text())
    partition_hash = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    if partition_hash != old_plan['partition_sha256']:
        raise ValueError('checkpoint training/validation partition mismatch')
    manifest = json.loads((args.checkpoints / 'complete.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def write(name, value):
        with (args.output / name).open('x') as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')
    checkpoints = {seed: args.checkpoints / f'{seed}-learning_rate_low.pt' for seed in (1701, 1702)}
    for path in checkpoints.values():
        if sha(path) != manifest['files'][path.name]:
            raise ValueError('checkpoint checksum mismatch')
    write('plan.json', {
        'claim_status': 'exploratory-development-only-not-jointly-trained-module',
        'formula': '(1-w)*Dirichlet + w*A-PBPF', 'weight_grid': np.linspace(0, 1, 21).tolist(),
        'selection_split': 'inner_validation', 'assessment_split': 'original_development',
        'expected_is_public': True, 'original_test_used': False,
        'partition_sha256': partition_hash, 'torch': torch.__version__, 'particles': 128,
        'prediction_policy': 'mean probability over inference seeds 51701,51702,51703 for ALL neural arms',
        'checkpoint_sha256': {str(seed): sha(path) for seed, path in checkpoints.items()},
        'parent_plan_sha256': sha(args.checkpoints / 'plan.json'),
        'source_sha256': {str(p.relative_to(repo)): sha(p) for p in [Path(__file__),
            repo / 'scripts/run_apbpf_parameter_diagnostic.py', repo / 'scripts/run_rbr_prediction_gate.py',
            repo / 'src/pbpf/belief/model.py', repo / 'src/pbpf/real_gate.py']},
    })
    batches = {k: gate['_tensorize'](parts[k], gate['FrozenTextEncoder'](256), torch.device('cpu'),
                expected_is_public=True) for k in ('validation', 'assessment')}
    labels = {k: b.outcomes[:, 4:].reshape(-1).numpy() for k, b in batches.items()}
    metric = gate['compare_predictions']
    alpha = min((.01, .1, .25, .5, 1., 2., 5., 10.), key=lambda a:
        metric(labels['validation'], {'baseline': gate['_history_rate'](batches['validation'], a)})['baseline']['nll'])
    rate = {k: gate['_history_rate'](b, alpha) for k, b in batches.items()}
    reports = []
    for seed, checkpoint in checkpoints.items():
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        model = gate['NeuralBeliefModel'](256, 32, 192, difficulty_dim=8)
        model.load_state_dict(saved['model'])
        model.eval()
        neural = {}
        for split, batch in batches.items():
            neural[split] = np.mean([gate['_predict'](model, batch, particles=128,
                seed=inference_seed, mode='aligned') for inference_seed in (51701, 51702, 51703)], axis=0)
        weight = select_weight(rate['validation'], neural['validation'], labels['validation'])
        # Freeze the coefficient before any assessment metrics are computed.
        write(f'{seed}-selection.json', {'alpha': alpha, 'weight': weight, 'split': 'inner_validation'})
        combined = blend_probabilities(rate['assessment'], neural['assessment'], weight)
        predictions = {'baseline': rate['assessment'], 'apbpf': neural['assessment'], 'combined': combined}
        clusters = np.repeat([r.get('source_component_id', r['problem_id']) for r in parts['assessment']], 6)
        report = {'seed': seed, 'weight': weight, 'alpha': alpha,
            'metrics': metric(labels['assessment'], predictions),
            'combined_gain': {name: clustered_nll_gap(labels['assessment'], combined, predictions[name],
                  clusters, seed=2701, replicates=10000) for name in ('baseline', 'apbpf')},
            'claim_status': 'exploratory-development-only'}
        np.savez_compressed(args.output / f'{seed}-predictions.npz', **predictions,
                            labels=labels['assessment'], clusters=clusters)
        write(f'{seed}-report.json', report)
        reports.append(report)
        print(json.dumps(report), flush=True)
    write('complete.json', {'reports': reports, 'files': {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
