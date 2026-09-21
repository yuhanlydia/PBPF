"""Full-population SMC query replays with explicit public/evaluator separation."""
from functools import lru_cache
import hashlib
import itertools

import numpy as np
import torch

from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, public_test_text
from pbpf.registry import OUTCOMES
from .smc_acquisition import acquire, acquisition_scores, filter_history

POLICIES = ('fixed', 'random', 'diagnostic_mi', 'predictive_entropy', 'joint_particle_mi')


def replay_queries(rows, checkpoint, *, seed, mode, batch_size=64):
    """Yield exact cached-query traces; future labels reach evaluator scoring only.

    Oracle compares canonical subsets, including a canonicalized random subset,
    to isolate subset choice from order-sensitive inference. Active replay keeps
    each policy's actual sequential order and reports that limitation separately.
    """
    if mode not in {'oracle', 'active'} or not rows or batch_size < 1:
        raise ValueError('nonempty rows and known query replay mode required')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['seed'] != seed or not saved['apbpf'] or saved['encoder'] != 'frozen_hash_text':
        raise ValueError('requires the exact fixed-seed factored hash-feature checkpoint')
    if any(t['expected'] for row in rows for t in row['tests'][4:]):
        raise ValueError('future expected answers cannot enter predictor features')
    model = NeuralBeliefModel(saved['feature_dim'], saved['latent_dim'], saved['hidden_dim'],
                              difficulty_dim=saved['difficulty_dim'])
    model.load_state_dict(saved['model']); model.eval()
    encode = lru_cache(maxsize=20000)(FrozenTextEncoder(saved['feature_dim']))
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start+batch_size]; n = len(part)
            task = torch.tensor(np.stack([encode(r['task_text']) for r in part]))
            code = torch.tensor(np.stack([encode(r['candidate']) for r in part]))
            features = torch.tensor(np.stack([[encode(public_test_text(t, expected_is_public=saved['expected_is_public']))
                                               for t in row['tests']] for row in part]))
            labels = torch.tensor([[OUTCOMES.index(o) for o in r['outcomes']] for r in part])
            noises, uniforms, orders = [], [], []
            for row in part:
                local_seed = int.from_bytes(hashlib.sha256(f'{seed}:{row["task_id"]}'.encode()).digest()[:4], 'big')
                generator = torch.Generator().manual_seed(local_seed)
                noises.append(torch.randn(saved['particles'], model.latent_dim, generator=generator))
                uniforms.append(torch.rand(4, generator=generator))
                orders.append(np.random.default_rng(local_seed).permutation(4))
            noise, uniform, order = torch.stack(noises), torch.stack(uniforms), torch.tensor(np.stack(orders))
            records = [{'candidate':r['task_id'], 'source':r['source_component_id'], 'seed':seed,
                        'budgets':{str(b):{} for b in ((1,2,4) if mode=='oracle' else (1,2,3,4))}}
                       for r in part]

            def score(z, weights):
                # These six feature/outcome slots are never passed to acquisition.
                logp = model.future_predict(task, code, features[:,4:], z, weights)
                return -logp.gather(-1, labels[:,4:,None]).squeeze(-1).double().numpy()

            def store(budget, policy, selected, nll):
                choices = selected.tolist()
                if any(len(set(x)) != budget or any(t not in range(4) for t in x) for x in choices):
                    raise ValueError('policy violated exact distinct-public-query budget')
                for i in range(n):
                    records[i]['budgets'][str(budget)][policy] = {'selected':choices[i], 'future_nll':nll[i].tolist()}

            if mode == 'active':
                for policy in POLICIES:
                    snapshots = acquire(model, task, code, features[:,:4], labels[:,:4], policy=policy,
                        noise=noise, uniforms=uniform, random_order=order, budgets=(1,2,3,4))
                    for budget, (z, weights, selected) in snapshots.items():
                        store(budget, policy, selected, score(z, weights))
                        if policy == 'diagnostic_mi' and budget < 4:
                            scores = acquisition_scores(model, task, code, features[:,:4], z, weights, policy)
                            scores.scatter_(1, selected, float('-inf'))
                            information = scores.max(1).values.tolist()
                            for i in range(n):
                                records[i]['budgets'][str(budget)][policy]['remaining_information'] = information[i]
            else:
                for budget in (1,2,4):
                    subsets = list(itertools.combinations(range(4), budget)); losses = []
                    for subset in subsets:
                        selected = torch.tensor(subset).expand(n,-1)
                        z, weights = filter_history(model, task, code, features[:,:4], labels[:,:4], selected, noise, uniform)
                        losses.append(score(z, weights))
                    losses = np.stack(losses)
                    store(budget, 'fixed', torch.tensor(subsets[0]).expand(n,-1), losses[0])
                    random_selected = order[:,:budget].sort(1).values
                    lookup = {x:i for i,x in enumerate(subsets)}
                    random_index = np.array([lookup[tuple(x)] for x in random_selected.tolist()])
                    store(budget, 'random_canonical', random_selected, losses[random_index, np.arange(n)])
                    best = losses.mean(-1).argmin(0)
                    store(budget, 'oracle', torch.tensor([subsets[i] for i in best]), losses[best, np.arange(n)])
                    # Candidate-level best subsets are evaluator upper bounds only.
                    for i in range(n):
                        records[i]['budgets'][str(budget)]['oracle']['subsets_considered'] = len(subsets)
            yield from records


def paired_loss_gap(left, right, sources, *, seed=201701, draws=10000):
    """Positive gap means right beats left; whole-source clusters span all seeds."""
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 6 or len(a) != len(sources) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('six finite paired future losses per candidate required')
    gap = (a-b).mean(1); keys = sorted(set(sources)); sources = np.asarray(sources)
    sums = np.array([gap[sources==s].sum() for s in keys]); counts = np.array([(sources==s).sum() for s in keys])
    rng = np.random.default_rng(seed); samples = []
    for start in range(0, draws, 256):
        indices = rng.integers(0,len(keys),size=(min(256,draws-start),len(keys)))
        samples.extend(sums[indices].sum(1)/counts[indices].sum(1))
    return {'mean_nll_gap':float(gap.mean()), 'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'clusters':len(keys), 'replicates':draws, 'unit':'source_component_across_all_fixed_seeds'}


def summarize_queries(records_by_seed, *, mode, draws=10000):
    if set(records_by_seed) != {1701,1702,1703}:
        raise ValueError('all fixed seeds required')
    if any(r.get('seed') != seed for seed, rows in records_by_seed.items() for r in rows):
        raise ValueError('record seed differs from declared replication cell')
    first = [(r['candidate'],r['source']) for r in records_by_seed[1701]]
    if not first or len(set(first)) != len(first) or any([(r['candidate'],r['source']) for r in rows] != first for rows in records_by_seed.values()):
        raise ValueError('seed populations differ or contain duplicate candidates')
    records = [r for seed in (1701,1702,1703) for r in records_by_seed[seed]]
    policies = ('fixed','random_canonical','oracle') if mode=='oracle' else POLICIES
    summary = {}
    for budget in (1,2,4):
        losses = {}
        for policy in policies:
            for row in records:
                chosen = row['budgets'][str(budget)][policy]['selected']
                if len(chosen) != budget or len(set(chosen)) != budget or not set(chosen) <= set(range(4)):
                    raise ValueError('record does not execute exact public budget')
            losses[policy] = np.array([r['budgets'][str(budget)][policy]['future_nll'] for r in records])
        summary[str(budget)] = {policy:{'nll':float(values.mean()), 'against_fixed':paired_loss_gap(losses['fixed'],values,
            [r['source'] for r in records],draws=draws)} for policy,values in losses.items()}
        if mode=='oracle':
            summary[str(budget)]['oracle']['against_random'] = paired_loss_gap(losses['random_canonical'],losses['oracle'],[r['source'] for r in records],draws=draws)
        else:
            summary[str(budget)]['diagnostic_mi']['against_random'] = paired_loss_gap(losses['random'],losses['diagnostic_mi'],[r['source'] for r in records],draws=draws)
    return {'seeds':[1701,1702,1703], 'candidates_per_seed':len(first),
            'source_components':len({s for _,s in first}), 'summary':summary,
            'all_policies_execute_exact_budget':True,
            'seed_estimand':'mean paired loss over all three fixed seeds; shared source clusters',
            'budget_semantics':'cached public-observation counts; no new execution or wall-clock savings'}
