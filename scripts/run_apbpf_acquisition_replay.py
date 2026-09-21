#!/usr/bin/env python3
"""Development-only cached replay of active testing and oracle subset headroom.

Uses the shipped diagnostic mean-field adapter, four public query candidates,
and six fixed evaluator targets. No primary/test-source records are accepted.
Cached observation counts are logical budgets, not newly executed sandbox calls.
"""
import argparse
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from pbpf.apbpf.acquisition import NeuralDiagnosticAdapter, expected_information_gain
from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, public_test_text, validate_rbr_cache
from pbpf.registry import OUTCOMES


class CachedAdapter(NeuralDiagnosticAdapter):
    """Particles stay fixed in this adapter; only mean-field weights update."""
    def prime(self, state):
        self.cached = {t: super(CachedAdapter, self)._joint_outcomes(state, t) for t in self.tests}

    def _joint_outcomes(self, state, test_id):
        return self.cached[test_id]


def entropy(values):
    values = np.asarray(values)
    return float(-np.sum(values * np.log(np.clip(values, 1e-15, 1))))


def follow(adapter, initial, public_outcomes, *, budget, policy, rng):
    state = initial
    random_order = list(rng.permutation(list(public_outcomes)))
    selected = []
    for step in range(budget):
        remaining = adapter.remaining(state)
        if policy == 'fixed':
            test_id = list(public_outcomes)[step]
        elif policy == 'random':
            test_id = random_order[step]
        elif policy == 'diagnostic_mi':
            test_id = max(remaining, key=lambda t: expected_information_gain(state.component_probs, remaining[t]))
        elif policy == 'predictive_entropy':
            test_id = max(remaining, key=lambda t: entropy(state.component_probs @ remaining[t]))
        else:
            raise ValueError('unknown public policy')
        if test_id not in public_outcomes or test_id in selected:
            raise ValueError('policy tried a hidden or repeated test')
        state, _ = adapter.update(state, test_id, public_outcomes[test_id])
        selected.append(test_id)
    if len(selected) != budget:
        raise ValueError('policy did not execute exact budget')
    return state, selected


def future_nll(state, future_joint, targets):
    probabilities = np.einsum('k,j,tkjc->tc', state.component_probs, state.difficulty_probs, future_joint)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(-1), 1):
        raise ValueError('invalid posterior prediction')
    return -np.log(np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1701)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('replay outputs are create-once')
    payload = validate_rbr_cache(json.loads(args.cache.read_text()))
    if payload.get('evaluation_role') != 'development_assessment_only':
        raise ValueError('this diagnostic requires an explicitly development-only cache')
    rows = [r for r in payload['records'] if r['split'] == 'test']
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if weights.get('encoder_type') == 'frozen_code_model':
        raise ValueError('semantic checkpoints require their frozen feature cache; hash fallback forbidden')
    if weights.get('evaluation_role') != 'development_assessment_only' or not weights.get('apbpf'):
        raise ValueError('requires a factored development-only checkpoint')
    model = NeuralBeliefModel(weights['feature_dim'], weights['latent_dim'], weights['hidden_dim'],
                             difficulty_dim=weights['difficulty_dim'])
    model.load_state_dict(weights['model']); model.eval()
    encode = lru_cache(maxsize=20000)(FrozenTextEncoder(weights['feature_dim']))
    source_root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), source_root / 'src/pbpf/apbpf/acquisition.py',
               source_root / 'src/pbpf/belief/model.py', source_root / 'src/pbpf/real_gate.py']
    config = {'schema': 'apbpf-development-acquisition-replay-v1', 'seed': args.seed,
              'cache_sha256': hashlib.sha256(args.cache.read_bytes()).hexdigest(),
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'source_sha256': {str(x.relative_to(source_root)): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
              'particles': weights['particles'], 'budgets': [1, 2, 4], 'public_pool': 4, 'future_targets': 6,
              'policies': ['fixed', 'random', 'diagnostic_mi', 'predictive_entropy'],
              'oracle': 'minimum hidden NLL over public subsets in canonical order; evaluator upper bound only',
              'inference': 'shipped fixed-particle parallel mean-field Bayes adapter, initialized from trained root prior; not amortized SMC proposal updates',
              'scope': 'nonconfirmatory development-only cached replay; upstream association/fairness gates failed',
              'budget_semantics': 'exact count of cached public observations; no new sandbox executions',
              'population': [{'candidate': r['task_id'], 'source': r['source_component_id']} for r in rows]}
    args.output.mkdir(parents=True)
    (args.output / 'plan.json').write_text(json.dumps(config, indent=2) + '\n')
    records = []
    with torch.no_grad(), (args.output / 'records.jsonl').open('x') as stream:
        for index, row in enumerate(rows):
            task, candidate = encode(row['task_text']), encode(row['candidate'])
            features = {str(i): encode(public_test_text(t, expected_is_public=weights['expected_is_public']))
                        for i, t in enumerate(row['tests'])}
            public = CachedAdapter(model, task, candidate, {str(i): features[str(i)] for i in range(4)})
            evaluator = NeuralDiagnosticAdapter(model, task, candidate, {str(i): features[str(i)] for i in range(4, 10)})
            local_seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{row["task_id"]}'.encode()).digest()[:4], 'big')
            generator = torch.Generator().manual_seed(local_seed)
            prior = model.root(torch.as_tensor(task)[None], torch.as_tensor(candidate)[None])
            particles = prior.mean[0] + prior.std[0] * torch.randn((weights['particles'], model.latent_dim), generator=generator)
            initial = public.state(particles, np.full(weights['particles'], -np.log(weights['particles'])))
            public.prime(initial)
            future_joint = np.stack([evaluator._joint_outcomes(initial, str(i)) for i in range(4, 10)])
            targets = np.array([OUTCOMES.index(x) for x in row['outcomes'][4:]])
            public_outcomes = {str(i): row['outcomes'][i] for i in range(4)}
            result = {'candidate': row['task_id'], 'source': row['source_component_id'], 'budgets': {}}
            for budget in config['budgets']:
                policies = {}
                for policy in config['policies']:
                    state, selected = follow(public, initial, public_outcomes, budget=budget, policy=policy,
                                             rng=np.random.default_rng(local_seed))
                    policies[policy] = {'selected': selected, 'future_nll': future_nll(state, future_joint, targets).tolist()}
                oracle = []
                for subset in itertools.combinations(public_outcomes, budget):
                    state = initial
                    for test_id in subset:
                        state, _ = public.update(state, test_id, public_outcomes[test_id])
                    nll = future_nll(state, future_joint, targets)
                    oracle.append((float(nll.mean()), subset, nll))
                _, subset, nll = min(oracle, key=lambda x: x[0])
                if nll.mean() > np.mean(policies['fixed']['future_nll']) + 1e-10:
                    raise AssertionError('oracle must include the fixed canonical subset')
                if budget == 4 and not np.allclose(nll, policies['fixed']['future_nll']):
                    raise AssertionError('all-four canonical oracle and fixed must coincide')
                policies['oracle'] = {'selected': list(subset), 'future_nll': nll.tolist()}
                result['budgets'][str(budget)] = policies
            records.append(result);stream.write(json.dumps(result) + '\n');stream.flush()
            if (index + 1) % 32 == 0:
                print(json.dumps({'completed': index + 1, 'total': len(rows)}), flush=True)
    sources = sorted({r['source'] for r in records})
    rng = np.random.default_rng(args.seed)
    summary = {}
    for budget in config['budgets']:
        part = {}
        for policy in [*config['policies'], 'oracle']:
            values = np.array([r['budgets'][str(budget)][policy]['future_nll'] for r in records])
            fixed = np.array([r['budgets'][str(budget)]['fixed']['future_nll'] for r in records])
            gaps = (fixed-values).mean(1)
            sums = np.array([sum(g for g,r in zip(gaps,records) if r['source']==s) for s in sources])
            counts = np.array([sum(r['source']==s for r in records) for s in sources])
            draw = rng.integers(0, len(sources), size=(10000, len(sources)))
            boot = sums[draw].sum(1)/counts[draw].sum(1)
            part[policy] = {'nll': float(values.mean()), 'nll_advantage_over_fixed': float(gaps.mean()),
                            'ci95': np.quantile(boot,[.025,.975]).tolist()}
        summary[str(budget)] = part
    report = {**config, 'population': {'candidates': len(rows), 'source_components': len(sources)},
              'summary': summary, 'bootstrap': '10000 whole-source draws, candidate/future-example weighted',
              'limitations': ['Public pool has only four tests; budget four changes ordering only.',
                 'No threshold stopping quality claim; no joint-particle MI ablation yet.',
                 'Not a sealed active-testing gate or confirmation of any downstream claim.']}
    (args.output / 'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'summary': summary, 'scope': config['scope']}),flush=True)


if __name__ == '__main__':
    main()
