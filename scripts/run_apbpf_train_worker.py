#!/usr/bin/env python3
"""Fit real exploratory belief or deterministic baselines from declared caches.

All seeds use frozen hash-text features. Checkpoints are selected only on the
original development split. Primary predictions are emitted after fitting; no
scientific gate or cross-fitted selector claim is made by this training stage.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.worker_io import WorkerIO
from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, validate_rbr_cache


def train_models(payload, output, *, kind, seed, steps, batch_size, learning_rate,
                 feature_dim, hidden_dim, protocol, helper):
    """Fit and save every required arm; never use primary labels for selection."""
    validate_rbr_cache(payload)
    if payload.get('evaluation_role') != 'exploratory_locked_primary_assessment':
        raise ValueError('requires source-bound full-stage cache, not development reassignment')
    if any(t['expected'] for r in payload['records'] for t in r['tests'][4:]):
        raise ValueError('future expected answers must be redacted before fitting')
    if any((r.get('original_split') == 'primary') != (r['split'] == 'test') for r in payload['records']):
        raise ValueError('primary membership differs from test partition')
    output.mkdir(parents=True, exist_ok=False)
    rows = {s: [r for r in payload['records'] if r['split'] == s] for s in ('train', 'development', 'test')}
    if any(not values for values in rows.values()):
        raise ValueError('all original source splits must be nonempty')
    encoder = FrozenTextEncoder(feature_dim)
    expected_public = payload['dataset'] == 'runbugrun'
    device = torch.device('cpu')
    # Primary tensors and outcomes are not supplied to the optimization loop.
    batches = {s: helper._tensorize(rows[s], encoder, device, expected_is_public=expected_public)
               for s in ('train', 'development')}
    latent_dim = protocol['difficulty_dim'] + protocol['diagnosis_dim']
    config = {'kind': kind, 'seed': seed, 'steps': steps, 'batch_size': batch_size,
              'learning_rate': learning_rate, 'feature_dim': feature_dim, 'hidden_dim': hidden_dim,
              'latent_dim': latent_dim, 'protocol': protocol, 'expected_is_public': expected_public,
              'apbpf': kind == 'belief', 'particles': protocol['particles'],
              'difficulty_dim': protocol['difficulty_dim'], 'diagnosis_dim': protocol['diagnosis_dim'],
              'association_weight': protocol['lambda_assoc'], 'invariance_weight': protocol['lambda_inv'],
              'association_margin': protocol['association_margin'],
              'encoder': 'frozen_hash_text', 'checkpoint_selection_split': 'development',
              'primary_used_for_training_or_selection': False}
    models, reports = {}, {}
    arms = ['belief'] if kind == 'belief' else ['pair_aware', 'deep_sets', 'no_particle_bottleneck']
    for offset, arm in enumerate(arms):
        torch.manual_seed(seed + offset)
        if arm == 'belief':
            model = NeuralBeliefModel(feature_dim, latent_dim, hidden_dim,
                                      difficulty_dim=protocol['difficulty_dim']).to(device)
        else:
            model = helper._DeterministicPredictor(feature_dim, hidden_dim, latent_dim, arm).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
        rng = np.random.default_rng(seed)
        best, best_state, best_step, history = float('inf'), None, None, []
        for step in range(1, steps + 1):
            model.train()
            indices = torch.tensor(rng.integers(0, len(rows['train']), size=batch_size), device=device)
            batch = helper._subset(batches['train'], indices)
            if arm == 'belief':
                metrics = helper._apbpf_step(model, batch, optimizer, particles=protocol['particles'],
                    association_weight=protocol['lambda_assoc'], invariance_weight=protocol['lambda_inv'],
                    association_margin=protocol['association_margin'], shuffle_seed=seed + step)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(batch).reshape(-1, 5), batch.outcomes[:, 4:].reshape(-1))
                loss.backward(); optimizer.step(); metrics = {'loss': float(loss.detach())}
            if step == 1 or step % max(1, steps // 20) == 0:
                model.eval()
                with torch.no_grad():
                    if arm == 'belief':
                        probabilities = helper._predict(model, batches['development'],
                            particles=protocol['particles'], seed=seed + 50000, mode='aligned')
                        labels = batches['development'].outcomes[:, 4:].reshape(-1).cpu().numpy()
                        nll = float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean())
                    else:
                        nll = float(torch.nn.functional.cross_entropy(model(batches['development']).reshape(-1, 5),
                            batches['development'].outcomes[:, 4:].reshape(-1)))
                history.append({'step': step, 'development_nll': nll, 'training': metrics})
                if nll < best:
                    best, best_state, best_step = nll, copy.deepcopy(model.state_dict()), step
        if best_state is None:
            raise RuntimeError('no finite development checkpoint')
        model.load_state_dict(best_state); model.eval(); models[arm] = model
        checkpoint = output/f'{arm}.pt'
        with checkpoint.open('xb') as stream:
            torch.save({'model': best_state, 'arm': arm, **config}, stream)
        reports[arm] = {'checkpoint': checkpoint.name, 'checkpoint_sha256': file_sha(checkpoint),
                        'best_step': best_step, 'development_nll': best,
                        'parameters': sum(p.numel() for p in model.parameters()), 'history': history}
    if kind == 'baselines':
        grid = [.01, .1, .25, .5, 1., 2., 5., 10.]
        labels = batches['development'].outcomes[:, 4:].reshape(-1).cpu().numpy()
        losses = {a: float(-np.log(np.clip(helper._history_rate(batches['development'], a)[np.arange(len(labels)), labels], 1e-12, 1)).mean()) for a in grid}
        alpha = min(grid, key=losses.get)
        reports['tuned_dirichlet'] = {'alpha': alpha, 'grid': grid, 'development_nll': losses[alpha]}
        (output/'tuned_dirichlet.json').write_text(json.dumps(reports['tuned_dirichlet'], indent=2)+'\n')
    primary = helper._tensorize(rows['test'], encoder, device, expected_is_public=expected_public)
    predictions = {}
    with torch.no_grad():
        for arm, model in models.items():
            if arm == 'belief':
                predictions['aligned'] = helper._predict(model, primary, particles=protocol['particles'],
                                                         seed=seed + 100000, mode='aligned')
            else:
                predictions[arm] = model(primary).softmax(-1).reshape(-1, 5).cpu().numpy()
        if kind == 'baselines':
            predictions['tuned_dirichlet'] = helper._history_rate(primary, reports['tuned_dirichlet']['alpha'])
    np.savez_compressed(output/'primary-predictions.npz', **predictions,
                        labels=primary.outcomes[:, 4:].reshape(-1).cpu().numpy())
    population = [{'candidate_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows['test']]
    report = {'schema': 'apbpf-stage-training-v1', 'config': config, 'dataset': payload['dataset'],
              'arms': reports, 'primary_population': population,
              'counts': {s: len(v) for s, v in rows.items()},
              'scope': 'exploratory fitting and primary predictions; no gate decision or selection claim'}
    (output/'training.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['belief', 'baselines'], required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--feature-dim', type=int, default=256)
    p.add_argument('--hidden-dim', type=int, default=192)
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.learning_rate, args.feature_dim, args.hidden_dim) <= 0:
        raise ValueError('positive training dimensions and budgets required')
    root = Path(__file__).resolve().parents[1]
    sources = [str(path.relative_to(root)) for path in sorted((root/'src/pbpf').rglob('*.py'))]
    io = WorkerIO('train_' + args.kind, sources + ['scripts/run_apbpf_train_worker.py', 'scripts/run_rbr_prediction_gate.py'])
    protocol = io.config['protocol']
    if protocol['lambda_future'] != 1.0:
        raise ValueError('training helper implements lambda_future=1 only')
    spec = importlib.util.spec_from_file_location('apbpf_stage_prediction_helper', root/'scripts/run_rbr_prediction_gate.py')
    helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper; spec.loader.exec_module(helper)
    index = json.loads(io.artifact('execution_cache', 'execution-index.json').read_text())
    outputs = []
    for domain in ('rbr', 'codearc'):
        entry = index['training_caches'][domain]['full']
        path = io.artifact('execution_cache', entry['path'])
        if file_sha(path) != entry['sha256']:
            raise ValueError('training cache differs from execution index')
        payload = json.loads(path.read_text())
        for seed in protocol['seeds']:
            io.check_sources()
            name = f'{domain}-seed{seed}'
            report = train_models(payload, io.outputs/name, kind=args.kind, seed=seed, steps=args.steps,
                batch_size=args.batch_size, learning_rate=args.learning_rate, feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim, protocol=protocol, helper=helper)
            outputs.append({'domain': domain, 'seed': seed, 'directory': name, 'cache_sha256': file_sha(path),
                            'counts': report['counts']})
    (io.outputs/'training-index.json').write_text(json.dumps({'kind': args.kind, 'runs': outputs}, indent=2)+'\n')
    io.finish({'actual_training': True, 'runs': outputs, 'encoder': 'frozen_hash_text',
               'primary_used_for_training_or_selection': False,
               'scope': 'exploratory trained checkpoints; scientific gates remain downstream'})


if __name__ == '__main__':
    main()
