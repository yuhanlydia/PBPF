#!/usr/bin/env python3
"""Fit public-only success heads, then evaluate a full development candidate bank."""
import argparse
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pbpf.apbpf.selection import SuccessHead, cross_fitted_choices, public_batch
from pbpf.belief.features import BeliefBatch
from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, validate_rbr_cache


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def subset(batch, indices):
    return BeliefBatch(*(getattr(batch, key)[indices] for key in ('task', 'candidate', 'tests', 'outcomes')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--steps', type=int, default=1000)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('selection outputs are create-once')
    payload = validate_rbr_cache(json.loads(args.cache.read_text()))
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if payload.get('evaluation_role') != 'development_assessment_only' or saved.get('evaluation_role') != 'development_assessment_only':
        raise ValueError('requires development-only data and checkpoint')
    if saved.get('encoder_type') == 'frozen_code_model' or not saved.get('apbpf') or args.steps < 1:
        raise ValueError('requires factored lexical checkpoint and positive steps')
    report_path = args.checkpoint.with_suffix('.json')
    association = json.loads(report_path.read_text())
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if association['cache_sha256'] != digest or association['config_sha256'] != saved['config_sha256']:
        raise ValueError('checkpoint report/cache identity mismatch')
    checksums = json.loads(args.checkpoint.with_suffix('.checksums.json').read_text())
    for path in (args.checkpoint, report_path):
        if checksums[path.name] != sha(path):
            raise ValueError('checkpoint or association report checksum mismatch')
    rows = {split: [r for r in payload['records'] if r['split'] == split]
            for split in ('train', 'development', 'test')}
    if any(not r for r in rows.values()):
        raise ValueError('all three source-disjoint splits required')
    source_sets = {k: {r['source_component_id'] for r in v} for k, v in rows.items()}
    if any(source_sets[a] & source_sets[b] for a, b in [('train', 'development'), ('train', 'test'), ('development', 'test')]):
        raise ValueError('source leakage between fitting/validation/assessment')
    groups = {}
    for index, row in enumerate(rows['test']):
        groups.setdefault(row['source_component_id'], []).append(index)
    if any(len(indices) != 8 for indices in groups.values()):
        raise ValueError('full assessment requires exactly eight candidates per source group')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root/'src/pbpf/apbpf/selection.py',
               root/'src/pbpf/belief/model.py', root/'src/pbpf/real_gate.py']
    plan = {'schema': 'apbpf-development-selection-v1', 'dataset': payload['dataset'],
            'seed': args.seed, 'steps': args.steps, 'device': 'cpu',
            'cache_sha256': sha(args.cache), 'checkpoint_sha256': sha(args.checkpoint),
            'upstream_association_sha256': sha(report_path), 'upstream_gate': association['gate'],
            'source_sha256': {str(f.relative_to(root)): sha(f) for f in sources},
            'population': {k: {'sources': len(source_sets[k]), 'candidates': len(v)} for k, v in rows.items()},
            'feature_contract': 'task/candidate, four public inputs/outcomes only; no future test content',
            'utility_target': 'all six evaluator-hidden calls pass; labels used only in fitting or scoring',
            'particle_utility': 'shared sigmoid success head per joint particle, posterior-weighted mixture',
            'deterministic_controls': ['pair_aware', 'deep_sets', 'no_particle_bottleneck', 'tuned_dirichlet'],
            'training': 'same 1000-step default, batch64, AdamW lr0.0003 wd0.01; inner-validation checkpoint selection',
            'comparator': 'five-source-fold selection among independently fitted deterministic heads and visible pass rate',
            'tie_break': 'immutable candidate inventory order',
            'scope': 'nonconfirmatory development; failed upstream gates retained; no primary data'}
    args.output.mkdir(parents=True)
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    encode = lru_cache(maxsize=20000)(FrozenTextEncoder(saved['feature_dim']))
    batches = {k: public_batch(v, encode, expected_is_public=saved['expected_is_public']) for k, v in rows.items()}
    targets = {k: torch.tensor([all(o == 'PASS' for o in r['outcomes'][4:]) for r in v], dtype=torch.float32)
               for k, v in rows.items()}
    model = NeuralBeliefModel(saved['feature_dim'], saved['latent_dim'], saved['hidden_dim'],
                             difficulty_dim=saved['difficulty_dim'])
    model.load_state_dict(saved['model']); model.eval()
    posterior = {}
    with torch.no_grad():
        for split, batch in batches.items():
            latents, weights = [], []
            for start in range(0, len(batch.task), 64):
                part = subset(batch, slice(start, start+64))
                generator = torch.Generator().manual_seed(args.seed+300000+start)
                trace = model.filter(part, particles=saved['particles'], visible_steps=4, generator=generator)
                latents.append(trace.latents[:, -1]); weights.append(trace.log_weights[:, -1])
            posterior[split] = (torch.cat(latents), torch.cat(weights))
    scores, fitting = {}, {}
    for offset, arm in enumerate(('particle', 'pair_aware', 'deep_sets', 'no_particle_bottleneck')):
        torch.manual_seed(args.seed+offset)
        head = SuccessHead(saved['feature_dim'], saved['latent_dim'], saved['hidden_dim'], arm)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.0003, weight_decay=.01)
        rng = np.random.default_rng(args.seed)
        best, best_state, best_step = float('inf'), None, None
        for step in range(1, args.steps+1):
            indices = torch.tensor(rng.integers(0, len(batches['train'].task), size=64))
            batch = subset(batches['train'], indices)
            z, weights = posterior['train']
            prediction = head(batch, z[indices], weights[indices]).clamp(1e-7, 1-1e-7)
            loss = torch.nn.functional.binary_cross_entropy(prediction, targets['train'][indices])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if step == 1 or step % max(1, args.steps//20) == 0:
                with torch.no_grad():
                    predicted = head(batches['development'], *posterior['development']).clamp(1e-7, 1-1e-7)
                    validation = float(torch.nn.functional.binary_cross_entropy(predicted, targets['development']))
                if validation < best:
                    best, best_state, best_step = validation, copy.deepcopy(head.state_dict()), step
        head.load_state_dict(best_state); head.eval()
        with torch.no_grad():
            scores[arm] = head(batches['test'], *posterior['test']).numpy()
        fitting[arm] = {'best_validation_nll': best, 'best_step': best_step,
                        'parameters': sum(p.numel() for p in head.parameters())}
        torch.save({'state': best_state, 'arm': arm, 'plan_sha256': sha(args.output/'plan.json')}, args.output/(arm+'.pt'))
        print(json.dumps({'arm': arm, **fitting[arm]}), flush=True)
    # For a six-call success target the rate estimate is raised to six.
    from pbpf.registry import OUTCOMES
    pass_index = OUTCOMES.index('PASS')
    def rate(split, alpha):
        counts = (batches[split].outcomes == pass_index).sum(1).numpy()
        return ((counts+alpha)/(4+2*alpha))**6
    grid = [.01, .1, .25, .5, 1., 2., 5., 10.]
    def nll(prob):
        y = targets['development'].numpy(); prob = prob.clip(1e-7, 1-1e-7)
        return float(-(y*np.log(prob)+(1-y)*np.log1p(-prob)).mean())
    alpha = min(grid, key=lambda a: nll(rate('development', a)))
    scores['tuned_dirichlet'] = rate('test', alpha)
    scores['visible_pass_rate'] = (batches['test'].outcomes == pass_index).float().mean(1).numpy()
    fitting['tuned_dirichlet'] = {'alpha': alpha, 'grid': grid}
    source_ids = list(groups)
    labels = targets['test'].numpy()
    successes, selections = {}, {}
    for arm, values in scores.items():
        chosen = [indices[int(values[indices].argmax())] for indices in groups.values()]
        successes[arm] = labels[chosen]
        selections[arm] = [rows['test'][i]['task_id'] for i in chosen]
    comparator, folds = cross_fitted_choices(source_ids, {k: v for k, v in successes.items() if k != 'particle'}, seed=args.seed)
    gap = successes['particle']-comparator
    draws = np.random.default_rng(args.seed).integers(0, len(groups), size=(10000, len(groups)))
    ci = np.quantile(gap[draws].mean(1), [.025, .975]).tolist()
    metrics = {arm: float(values.mean()) for arm, values in successes.items()}
    metrics.update(cross_fitted_deterministic=float(comparator.mean()),
                   random_expectation=float(np.mean([labels[i].mean() for i in groups.values()])),
                   oracle=float(np.mean([labels[i].max() for i in groups.values()])))
    result = {**plan, 'fitting': fitting, 'selected_pass1': metrics, 'advantage': float(gap.mean()), 'ci95': ci,
              'descriptive_selection_threshold_met': bool(gap.mean() >= .03 and ci[0] > 0),
              'formal_gate_claim': False, 'crossfit': folds, 'selections': selections,
              'bootstrap': '10000 source-group draws, paired selected-success differences',
              'limitations': ['Standalone learned utility extension; no confirmatory claim after failed gates.',
                              'Comparator choice is cross-fitted; utility models fit independent train/inner-validation sources.',
                              'Whole-suite target is six hidden replay calls, not an exhaustive semantic specification.']}
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'selected_pass1': metrics, 'advantage': result['advantage'], 'ci95': ci}), flush=True)


if __name__ == '__main__':
    main()
