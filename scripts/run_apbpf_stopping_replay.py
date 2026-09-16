#!/usr/bin/env python3
"""Public-MI threshold stopping, calibrated on disjoint development sources."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pbpf.apbpf.smc_acquisition import acquire, acquisition_scores
from pbpf.apbpf.stopping import lock_threshold, stopped_budgets
from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, public_test_text, validate_rbr_cache
from pbpf.registry import OUTCOMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('stopping replays are create-once')
    payload = validate_rbr_cache(json.loads(args.cache.read_text()))
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if payload.get('evaluation_role') != 'development_assessment_only' or saved.get('evaluation_role') != 'development_assessment_only':
        raise ValueError('both data and checkpoint must be development-only')
    if not saved.get('apbpf') or saved.get('encoder_type') == 'frozen_code_model':
        raise ValueError('this replay requires a factored lexical-feature checkpoint')
    rows = [r for r in payload['records'] if r['split'] == 'test']
    sources = sorted({r['source_component_id'] for r in rows},
                     key=lambda s: hashlib.sha256(f'{args.seed}:stopping:{s}'.encode()).digest())
    if len(sources) < 4:
        raise ValueError('at least four assessment sources required')
    calibration = set(sources[:len(sources)//2])
    thresholds = [0., .001, .003, .01, .03, .1, .3, 1.]
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), root/'src/pbpf/apbpf/smc_acquisition.py',
             root/'src/pbpf/apbpf/stopping.py', root/'src/pbpf/belief/model.py', root/'src/pbpf/real_gate.py']
    plan = {'schema': 'apbpf-development-threshold-stopping-v1', 'seed': args.seed,
            'cache_sha256': sha(args.cache), 'checkpoint_sha256': sha(args.checkpoint),
            'source_sha256': {str(f.relative_to(root)): sha(f) for f in files},
            'calibration_sources': sorted(calibration), 'assessment_sources': sorted(set(sources)-calibration),
            'thresholds': thresholds, 'minimum_tests': 1, 'maximum_tests': 4,
            'policy': 'diagnostic_mi', 'stop_signal': 'maximum remaining public-test diagnostic MI',
            'quality_reference': 'same diagnostic-MI policy after all four public observations',
            'scope': 'nonconfirmatory; upstream gates failed; original held-out test sources excluded',
            'budget_semantics': 'cached public observations, not new sandbox executions',
            'assessment_policy': 'threshold locked on calibration sources before assessment metrics are computed'}
    args.output.mkdir(parents=True)
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    model = NeuralBeliefModel(saved['feature_dim'], saved['latent_dim'], saved['hidden_dim'],
                             difficulty_dim=saved['difficulty_dim'])
    model.load_state_dict(saved['model']); model.eval()
    encode = lru_cache(maxsize=20000)(FrozenTextEncoder(saved['feature_dim']))
    records = []
    with torch.no_grad(), (args.output/'trajectories.jsonl').open('x') as stream:
        for start in range(0, len(rows), args.batch_size):
            part = rows[start:start+args.batch_size]
            task = torch.tensor(np.stack([encode(r['task_text']) for r in part]))
            code = torch.tensor(np.stack([encode(r['candidate']) for r in part]))
            tests = torch.tensor(np.stack([[encode(public_test_text(t, expected_is_public=saved['expected_is_public']))
                                           for t in r['tests']] for r in part]))
            labels = torch.tensor([[OUTCOMES.index(o) for o in r['outcomes']] for r in part])
            noises, uniforms = [], []
            for row in part:
                seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{row["task_id"]}'.encode()).digest()[:4], 'big')
                g = torch.Generator().manual_seed(seed)
                noises.append(torch.randn(saved['particles'], model.latent_dim, generator=g))
                uniforms.append(torch.rand(4, generator=g))
            snapshots = acquire(model, task, code, tests[:, :4], labels[:, :4], policy='diagnostic_mi',
                                noise=torch.stack(noises), uniforms=torch.stack(uniforms),
                                random_order=torch.arange(4).expand(len(part), -1), budgets=(1, 2, 3, 4))
            information, nlls, selections = [], [], []
            for budget, (z, weights, selected) in snapshots.items():
                if budget < 4:
                    scores = acquisition_scores(model, task, code, tests[:, :4], z, weights, 'diagnostic_mi')
                    scores.scatter_(1, selected, float('-inf'))
                    information.append(scores.max(1).values.numpy())
                # Future labels are used only for evaluator scoring, never acquisition or stop signals.
                predicted = model.future_predict(task, code, tests[:, 4:], z, weights)
                nlls.append(-predicted.gather(-1, labels[:, 4:, None]).squeeze(-1).mean(1).numpy())
                selections.append(selected.tolist())
            for i, row in enumerate(part):
                record = {'candidate': row['task_id'], 'source': row['source_component_id'],
                          'split': 'calibration' if row['source_component_id'] in calibration else 'assessment',
                          'remaining_information': [float(x[i]) for x in information],
                          'future_nll': [float(x[i]) for x in nlls],
                          'selected_prefixes': [x[i] for x in selections]}
                records.append(record); stream.write(json.dumps(record)+'\n')
            stream.flush()
            print(json.dumps({'completed': len(records), 'total': len(rows)}), flush=True)
    cal = [r for r in records if r['split'] == 'calibration']
    lock = lock_threshold([r['remaining_information'] for r in cal], [r['future_nll'] for r in cal], thresholds)
    lock.update(calibration_sources=sorted(calibration), plan_sha256=sha(args.output/'plan.json'))
    (args.output/'threshold-lock.json').write_text(json.dumps(lock, indent=2)+'\n')
    assessment = [r for r in records if r['split'] == 'assessment']
    values = np.array([r['future_nll'] for r in assessment])
    source_ids = sorted({r['source'] for r in assessment})
    indices = [np.array([i for i, r in enumerate(assessment) if r['source'] == s]) for s in source_ids]
    counts = np.array([len(i) for i in indices])
    draws = np.random.default_rng(args.seed).integers(0, len(indices), size=(10000, len(indices)))
    curve = []
    for threshold in (*thresholds, None):
        budgets = stopped_budgets([r['remaining_information'] for r in assessment], threshold)
        nll = values[np.arange(len(values)), budgets-1]
        gap = values[:, -1]-nll
        sums = np.array([gap[i].sum() for i in indices])
        boot = sums[draws].sum(1)/counts[draws].sum(1)
        curve.append({'threshold': threshold, 'mean_tests': float(budgets.mean()),
                      'test_reduction': float(1-budgets.mean()/4), 'nll': float(nll.mean()),
                      'nll_advantage_over_four': float(gap.mean()),
                      'ci95': np.quantile(boot, [.025, .975]).tolist(),
                      'selected_by_calibration': threshold == lock['threshold']})
    chosen = next(r for r in curve if r['selected_by_calibration'])
    report = {**plan, 'calibration_candidates': len(cal), 'assessment_candidates': len(assessment),
              'locked_threshold': lock['threshold'], 'assessment_curve': curve,
              'locked_policy_result': chosen, 'threshold_lock_sha256': sha(args.output/'threshold-lock.json'),
              'bootstrap': '10000 whole-source resamples; candidate-weighted mean over six future tests',
              'formal_gate_claim': False,
              'limitations': ['Exploratory development split; a curve is not a confirmatory active-testing gate.',
                              'No physical execution savings measured; labels were already cached.',
                              'Four-query pool only; same-policy fixed-four quality reference.']}
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'locked_policy_result': chosen}), flush=True)


if __name__ == '__main__':
    main()
