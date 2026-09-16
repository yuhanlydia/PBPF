"""Development-only diagnostic ablations; no held-out test evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rbr_prediction_gate as legacy
from pbpf.belief.diagnostic import (HistoryISBeliefModel, InteractionHistoryISBeliefModel,
                                    DeterministicInteractionPredictor, select_diagnostic_checkpoint,
                                    train_diagnostic_step)
from pbpf.belief.features import BeliefBatch
from pbpf.belief.model import NeuralBeliefModel
from pbpf.belief.semantic_features import load_features, training_view
from pbpf.real_gate import FrozenTextEncoder, association_strata, clustered_nll_gap, validate_rbr_cache
from pbpf.train_belief import train_apbpf_step


def controlled_batches(train_size, dev_size, feature_dim, seed, device):
    """Every history has 2 PASS/2 FAIL; only paired test signs reveal the bug.

    Hidden diagnosis is a random sign independent of task/candidate nuisance
    features. All future labels follow that same sign. This is a learnability
    check, not a model of real code semantics or a disentanglement benchmark.
    """
    if min(train_size, dev_size) < 1 or feature_dim < 2:
        raise ValueError('positive population sizes and at least two features required')
    rng = np.random.default_rng(seed)
    batches = {}
    for split, size in [('train', train_size), ('development', dev_size)]:
        task = rng.normal(0, .1, (size, feature_dim)).astype('float32')
        candidate = rng.normal(0, .1, (size, feature_dim)).astype('float32')
        direction = rng.choice([-1., 1.], size=size)
        signs = np.stack([np.concatenate([rng.permutation([-1., -1., 1., 1.]),
                                         rng.permutation([-1., -1., 1., 1.])]) for _ in range(size)])
        tests = np.zeros((size, 8, feature_dim), dtype='float32')
        tests[:, :, 0] = signs * rng.uniform(.5, 1.5, (size, 8))
        tests[:, :, 1] = 1.
        outcomes = (signs * direction[:, None] > 0).astype('int64')
        batches[split] = BeliefBatch(*(torch.tensor(v, device=device) for v in (task, candidate, tests, outcomes)))
    return batches, dict(scientific_claim='controlled_learnability_only',
        description='balanced binary sign-trigger diagnosis with independent nuisance features',
        counts={k: len(v.task) for k, v in batches.items()},
        source_ids={k: [f'{k}-{i}' for i in range(len(v.task))] for k, v in batches.items()},
        histogram_oracle_nll=math.log(2), paired_oracle_nll=0.)


def real_batches(cache, feature_dim, device, feature_cache=None):
    payload = validate_rbr_cache(json.loads(cache.read_text()))
    # The test inventory is never tensorized, scored, or used for selection.
    rows = {split: [r for r in payload['records'] if r['split'] == split]
            for split in ('train', 'development')}
    if any(not values for values in rows.values()):
        raise ValueError('nonempty train and development splits required')
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    feature_metadata = None
    if feature_cache is None:
        batches = {split: legacy._tensorize(values, FrozenTextEncoder(feature_dim), device,
                                            expected_is_public=False) for split, values in rows.items()}
    else:
        ids = {split: [r['task_id'] for r in values] for split, values in rows.items()}
        features, feature_metadata = load_features(feature_cache, digest, ids, feature_dim)
        batches = {}
        for split, values in rows.items():
            f = features[split]
            outcomes = [[legacy.OUTCOMES.index(y) for y in r['outcomes']] for r in values]
            if f['tests'].shape[1] != len(outcomes[0]):
                raise ValueError('feature test count mismatch')
            batches[split] = BeliefBatch(*(torch.tensor(f[k], device=device)
                for k in ('task', 'candidate', 'tests')), torch.tensor(outcomes, device=device))
    return batches, dict(scientific_claim='real_development_only',
        cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        encoder='frozen_codebert' if feature_cache else 'frozen_lexical_hash', expected_is_public=False,
        feature_metadata=feature_metadata,
        feature_sha256=hashlib.sha256(feature_cache.read_bytes()).hexdigest() if feature_cache else None,
        counts={k: len(v) for k, v in rows.items()},
        source_ids={k: [r.get('source_component_id', r['problem_id']) for r in v] for k, v in rows.items()})


def mean_nll(labels, probabilities):
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.)).mean())


@torch.no_grad()
def evaluate(model, batch, *, particles, seed):
    model.eval()
    labels = batch.outcomes[:, 4:].reshape(-1).cpu().numpy()
    predictions = {arm: legacy._predict(model, batch, particles=particles, seed=seed, mode=arm)
                   for arm in ('aligned', 'outcome_shuffled', 'joint_reversed', 'presentation_permuted')}
    nlls = {arm: mean_nll(labels, value) for arm, value in predictions.items()}
    row = dict(aligned_nll=nlls['aligned'], association_gap=nlls['outcome_shuffled'] - nlls['aligned'],
               pair_order_effect=max(abs(nlls[arm] - nlls['aligned'])
                                     for arm in ('joint_reversed', 'presentation_permuted')),
               nll=nlls)
    return row, predictions


@torch.no_grad()
def evaluate_deterministic(model, batch, seed):
    model.eval()
    sources = {'aligned': batch,
               'outcome_shuffled': batch.outcome_counterfactual(visible_steps=4, seed=seed)[0]}
    for arm in ('joint_reversed', 'presentation_permuted'):
        indices = np.tile(np.arange(batch.tests.shape[1]), (len(batch.task), 1))
        if arm == 'joint_reversed':
            indices[:, :4] = indices[:, :4][:, ::-1]
        else:
            indices, _ = legacy.joint_permutation(indices, batch.outcomes.cpu().numpy(), 4, seed)
        order = torch.tensor(indices, device=batch.task.device)
        sources[arm] = BeliefBatch(batch.task, batch.candidate,
            batch.tests.gather(1, order[..., None].expand_as(batch.tests)), batch.outcomes.gather(1, order))
    predictions = {arm: model(source).softmax(-1).reshape(-1, 5).cpu().numpy()
                   for arm, source in sources.items()}
    labels = batch.outcomes[:, 4:].reshape(-1).cpu().numpy()
    nll = {arm: mean_nll(labels, value) for arm, value in predictions.items()}
    return dict(aligned_nll=nll['aligned'], association_gap=nll['outcome_shuffled']-nll['aligned'],
                pair_order_effect=max(abs(nll[a]-nll['aligned']) for a in
                                      ('joint_reversed', 'presentation_permuted')), nll=nll), predictions


def fit_development_baselines(batches, args, output, source_ids):
    """Same features and optimizer; report *development*, not held-out claims."""
    train, dev = batches['train'], batches['development']
    labels = dev.outcomes[:, 4:].reshape(-1).cpu().numpy()
    grid = [.01, .1, .25, .5, 1., 2., 5., 10.]
    rate_nll = [(mean_nll(labels, legacy._history_rate(dev, a)), a) for a in grid]
    value, alpha = min(rate_nll)
    results = {'tuned_dirichlet': dict(nll=value, alpha=alpha, selection_split='development')}
    baseline_dir = output / 'baselines'
    baseline_dir.mkdir()
    for offset, arm in enumerate(('pair_aware', 'deep_sets', 'no_particle_bottleneck', 'deterministic_interaction')):
        torch.manual_seed(args.seed + offset + 1000)
        model = (DeterministicInteractionPredictor(args.feature_dim, args.hidden_dim, args.latent_dim)
                 if arm == 'deterministic_interaction' else
                 legacy._DeterministicPredictor(args.feature_dim, args.hidden_dim, args.latent_dim, arm)).to(dev.task.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
        rng = np.random.default_rng(args.seed)
        best, best_step, best_state, history = math.inf, None, None, []
        for step in range(1, args.steps + 1):
            index = torch.tensor(rng.integers(0, len(train.task), size=args.batch_size), device=train.task.device)
            b = legacy._subset(train, index)
            if args.shuffle_train_tests:
                b = training_view(b, args.seed + 1000000 + step)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(b).reshape(-1, 5), b.outcomes[:, 4:].reshape(-1))
            loss.backward()
            optimizer.step()
            if step == 1 or step % max(1, args.steps // 20) == 0 or step == args.steps:
                with torch.no_grad():
                    current = float(torch.nn.functional.cross_entropy(model(dev).reshape(-1, 5), dev.outcomes[:, 4:].reshape(-1)))
                history.append(dict(step=step, nll=current))
                if current < best:
                    best, best_step = current, step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(dict(model=best_state, arm=arm, selected_step=best_step,
                        feature_dim=args.feature_dim, hidden_dim=args.hidden_dim,
                        latent_dim=args.latent_dim, seed=args.seed + offset + 1000), baseline_dir / f'{arm}.pt')
        model.load_state_dict(best_state)
        metrics, predictions = evaluate_deterministic(model, dev, args.seed + 50000)
        bootstrap = clustered_nll_gap(labels, predictions['aligned'], predictions['outcome_shuffled'],
            np.repeat(source_ids, dev.tests.shape[1] - 4), seed=args.seed + 200000,
            replicates=args.bootstrap_replicates)
        results[arm] = dict(nll=best, development=metrics, development_bootstrap=bootstrap, parameters=sum(p.numel() for p in model.parameters()),
                            selection_split='development', selected_step=best_step,
                            checkpoint=f'baselines/{arm}.pt', validation_history=history)
    return results


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / 'src/pbpf').rglob('*.py')) + [Path(__file__).resolve(), Path(legacy.__file__).resolve(), root / 'scripts/prepare_semantic_features.py']
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def run(args):
    start = time.monotonic()
    if args.arm == 'legacy' and (args.association_weight, args.evidence_weight, args.invariance_weight) != (1., 1., .1):
        raise ValueError('legacy arm freezes its coefficients; nondefault overrides are not supported')
    evaluation_particles = args.eval_particles if args.eval_particles is not None else args.particles
    if evaluation_particles < 1:
        raise ValueError('positive evaluation particle count required')
    if args.feature_cache and args.source != 'runbugrun':
        raise ValueError('feature cache requires real data')
    if min(args.steps, args.particles, args.batch_size, args.bootstrap_replicates) < 1:
        raise ValueError('steps, particles, batch size and bootstrap replicates must be positive')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    device = torch.device(args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    manifest = dict(config=config, source_sha256=source_hashes(),
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    python_version=sys.version, device=str(device),
                    cuda_version=torch.version.cuda,
                    device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                    torch_threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    scope='development_only_exploratory', test_evaluated=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    try:
        source_root = Path(__file__).resolve().parents[1]
        for name, digest in manifest['source_sha256'].items():
            raw = (source_root / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError('source changed while materializing run snapshot')
            target = output / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        batches, metadata = (controlled_batches(args.train_size, args.dev_size, args.feature_dim, args.seed, device)
                             if args.source == 'controlled' else real_batches(args.cache, args.feature_dim, device, args.feature_cache))
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cls = {'history_is': HistoryISBeliefModel, 'interaction': InteractionHistoryISBeliefModel}.get(args.arm, NeuralBeliefModel)
        model = cls(args.feature_dim, args.latent_dim, args.hidden_dim, difficulty_dim=args.difficulty_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
        rng = np.random.default_rng(args.seed)
        train, dev = batches['train'], batches['development']
        eligibility = {k: int(association_strata(b.outcomes.cpu().numpy())['eligible'].sum()) for k, b in batches.items()}
        history, validation = [], []
        checkpoint_dir = output / 'checkpoints'
        checkpoint_dir.mkdir()
        print(json.dumps(dict(phase='train', arm=args.arm, data=metadata['counts'], eligible=eligibility)), flush=True)
        for step in range(1, args.steps + 1):
            index = torch.tensor(rng.integers(0, len(train.task), size=args.batch_size), device=device)
            b = legacy._subset(train, index)
            if args.shuffle_train_tests:
                b = training_view(b, args.seed + 1000000 + step)
            if args.arm == 'legacy':
                metrics = train_apbpf_step(model, b, optimizer, particles=args.particles,
                    shuffle_seed=args.seed + step, invariance_weight=.1)
            else:
                metrics = train_diagnostic_step(model, b, optimizer, particles=args.particles,
                    shuffle_seed=args.seed + step, association_weight=args.association_weight,
                    evidence_weight=args.evidence_weight, invariance_weight=args.invariance_weight,
                    measure_gradients=(step == 1))
            if step == 1 or step % max(1, args.steps // 20) == 0 or step == args.steps:
                dev_row, _ = evaluate(model, dev, particles=evaluation_particles, seed=args.seed + 50000)
                row = dict(step=step, training=metrics, development=dev_row)
                history.append(row)
                validation.append(dev_row)
                snapshot = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                            for k, v in model.state_dict().items()}
                torch.save(dict(model=snapshot, config=config, step=step,
                                model_class=cls.__name__, data=metadata), checkpoint_dir / f'{step:06d}.pt')
                print(json.dumps(row, sort_keys=True), flush=True)
        selection = select_diagnostic_checkpoint(validation)
        if args.arm == 'legacy':
            # Reproduce legacy minimum-NLL selection; report revised screen separately.
            selection['index'] = selection['best_nll_index']
            selected = validation[selection['index']]
            selection['development_gate_passed'] = bool(selected['association_gap'] >= .03
                and selected['pair_order_effect'] <= .25 * selected['association_gap'])
            selection['policy'] = 'legacy_minimum_nll'
        else:
            selection['policy'] = 'diagnostic_constraints_with_absolute_nll_guard'
        selected_step = history[selection['index']]['step']
        checkpoint = checkpoint_dir / f'{selected_step:06d}.pt'
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved['model'])
        shutil.copyfile(checkpoint, output / 'model.pt')
        selected_metrics, predictions = evaluate(model, dev, particles=evaluation_particles, seed=args.seed + 50000)
        labels = dev.outcomes[:, 4:].reshape(-1).cpu().numpy()
        clusters = np.repeat(metadata['source_ids']['development'], dev.tests.shape[1] - 4)
        bootstrap = clustered_nll_gap(labels, predictions['aligned'], predictions['outcome_shuffled'], clusters,
                                      seed=args.seed + 200000, replicates=args.bootstrap_replicates)
        mc_repeats = [evaluate(model, dev, particles=evaluation_particles, seed=args.seed + offset)[0]
                      for offset in (70000, 80000, 90000)]
        baselines = fit_development_baselines(batches, args, output, metadata['source_ids']['development'])
        result = dict(schema='apbpf-diagnostic-debug-v1', scope='development_only_exploratory',
            test_evaluated=False, config=config, data=metadata, eligible=eligibility,
            evaluation_particles=evaluation_particles,
            training_views='uniform_pair_permutation' if args.shuffle_train_tests else 'fixed_prefix',
            parameters=sum(p.numel() for p in model.parameters()), device=str(device),
            inference='prefix_is' if isinstance(model, HistoryISBeliefModel) else 'legacy_smc',
            selection={**selection, 'step': selected_step}, development=selected_metrics,
            development_bootstrap=bootstrap, bootstrap_caveat='descriptive only; checkpoint selected on these same development data',
            monte_carlo_repeats=mc_repeats, baselines=baselines, training_history=history,
            elapsed_seconds=time.monotonic() - start)
        (output / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
        checksums = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(output.rglob('*')) if p.is_file()}
        (output / 'checksums.json').write_text(json.dumps(checksums, sort_keys=True, indent=2) + '\n')
        print(json.dumps(dict(phase='complete', selection=result['selection'], development=selected_metrics,
                              baselines=baselines, elapsed_seconds=result['elapsed_seconds'])), flush=True)
        return result
    except Exception:
        (output / 'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()), indent=2) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['controlled', 'runbugrun'], required=True)
    parser.add_argument('--arm', choices=['legacy', 'objective', 'history_is', 'interaction'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('/root/pbpf-runs/association-20260916-seed1701/dataset.json'))
    parser.add_argument('--feature-cache', type=Path)
    parser.add_argument('--eval-particles', type=int)
    parser.add_argument('--shuffle-train-tests', action='store_true')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--train-size', type=int, default=512)
    parser.add_argument('--dev-size', type=int, default=128)
    parser.add_argument('--feature-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=192)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--difficulty-dim', type=int, default=8)
    parser.add_argument('--particles', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--association-weight', type=float, default=1.)
    parser.add_argument('--evidence-weight', type=float, default=1.)
    parser.add_argument('--invariance-weight', type=float, default=.1)
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
