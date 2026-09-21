#!/usr/bin/env python3
"""Measure loss-component gradients on every fitting candidate without updates."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

import pbpf.belief.losses as losses
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.semantic_cache import FrozenSemanticCache
from pbpf.belief.model import NeuralBeliefModel
from pbpf.train_belief import train_apbpf_step
from run_apbpf_particle_diagnostic import validate_development
from run_apbpf_prior_training_debug import PriorTrainingBelief


class GradientRecorder:
    """Optimizer-shaped observer: validate backward, never update a parameter."""
    def __init__(self, model):
        self.named = list(model.named_parameters())
        self.gradients = None
        self.record = None

    def zero_grad(self, set_to_none=True):
        for _, parameter in self.named:
            parameter.grad = None

    def capture(self, loss, *, association_weight, invariance_weight, future_weight):
        terms = {'fivo': loss.fivo, 'future': future_weight*loss.future,
                 'association': association_weight*loss.association,
                 'invariance': invariance_weight*loss.invariance}
        self.gradients = {name: torch.autograd.grad(term, [p for _, p in self.named],
                          retain_graph=True, allow_unused=True) for name, term in terms.items()}
        self.record = {'weighted_losses': {n: float(t.detach()) for n, t in terms.items()}, 'groups': {}}
        for group in ['all', 'root_head', 'difficulty_head', 'diagnosis_head', 'root_proposal_head', 'child_proposal_head', 'transition_head']:
            indices = [i for i, (n, _) in enumerate(self.named) if group == 'all' or n.startswith(group+'.')]
            squared = {n: sum(float(gs[i].double().square().sum()) for i in indices if gs[i] is not None)
                       for n, gs in self.gradients.items()}
            base_sq = dot = 0.
            for i in indices:
                p = self.named[i][1]
                base = sum((self.gradients[n][i] if self.gradients[n][i] is not None else torch.zeros_like(p))
                           for n in ['fivo', 'future']).double()
                assoc = self.gradients['association'][i]
                base_sq += float(base.square().sum())
                if assoc is not None:
                    dot += float((base*assoc.double()).sum())
            denominator = (base_sq*squared['association'])**.5
            self.record['groups'][group] = {'component_l2': {n: v**.5 for n, v in squared.items()},
                'base_l2': base_sq**.5,
                'association_to_base_norm_ratio': (squared['association']/base_sq)**.5 if base_sq else None,
                'base_association_cosine': dot/denominator if denominator else None,
                'unused_parameter_tensors': {n: sum(gs[i] is None for i in indices) for n, gs in self.gradients.items()}}

    def step(self):
        if self.gradients is None:
            raise ValueError('loss hook did not capture gradients')
        maximum = 0.
        for i, (_, parameter) in enumerate(self.named):
            expected = sum((gs[i] if gs[i] is not None else torch.zeros_like(parameter)) for gs in self.gradients.values())
            actual = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            if not torch.isfinite(actual).all() or not torch.allclose(expected, actual, atol=2e-6, rtol=2e-4):
                raise ValueError('component gradient sum does not reproduce actual training backward')
            maximum = max(maximum, float((expected-actual).abs().max()))
        self.record['backward_sum_max_absolute_error'] = maximum


@contextlib.contextmanager
def observe_loss(recorder):
    original = losses.association_aware_belief_loss

    def wrapped(*args, **kwargs):
        loss = original(*args, **kwargs)
        recorder.capture(loss, association_weight=kwargs.get('association_weight', 1.),
                         invariance_weight=kwargs.get('invariance_weight', 1.), future_weight=kwargs.get('future_weight', 1.))
        return loss

    losses.association_aware_belief_loss = wrapped
    try:
        yield
    finally:
        losses.association_aware_belief_loss = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['development-root', 'training-root', 'reference-training-root', 'comparison-root', 'assignment-root', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    development, training, old_training, comparison, assignment, output = (p.resolve() for p in
        [args.development_root, args.training_root, args.reference_training_root, args.comparison_root, args.assignment_root, args.output])
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'validating', 'pid': os.getpid(), 'runs': {}}

    def save():
        temp = output/'status.partial'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(output/'status.json')

    save()
    try:
        if torch.cuda.is_available():
            raise ValueError('CPU-only diagnostic required')
        statuses, plans = {}, {}
        for name, folder in [('comparison', comparison), ('assignment', assignment)]:
            statuses[name] = json.loads((folder/'status.json').read_text())
            plans[name] = json.loads((folder/'plan.json').read_text())
            st = statuses[name]
            if st['status'] != 'complete' or st['plan_sha256'] != file_sha(folder/'plan.json') or st['results_sha256'] != file_sha(folder/'results.json'):
                raise ValueError('completed source-bound upstream evidence required')
        previous = json.loads((comparison/'results.json').read_text())
        cache = development/'development-cache.json'
        payload = json.loads(cache.read_text())
        validate_development(payload)
        rows = sorted([r for r in payload['records'] if r['split'] == 'train'], key=lambda r: r['task_id'])
        if (len(rows) != 1360 or len({r['source_component_id'] for r in rows}) != 170
                or any(r['original_split'] != 'train' for r in rows)):
            raise ValueError('all and only 170 fitting sources required')
        fit_plan = json.loads((training/'plan.json').read_text())
        if (plans['comparison']['cache_sha256'] != file_sha(cache)
                or plans['assignment']['comparison_results_sha256'] != statuses['comparison']['results_sha256']
                or plans['comparison']['training_plan_sha256'] != file_sha(training/'plan.json')
                or plans['comparison']['reference_training_plan_sha256'] != file_sha(old_training/'plan.json')):
            raise ValueError('upstream identities differ')
        sources = dict(plans['assignment']['source_sha256'])
        for n, h in sources.items():
            if file_sha(root/n) != h:
                raise ValueError('upstream source changed')
        sources[str(Path(__file__).resolve().relative_to(root))] = file_sha(Path(__file__))
        encoder = FrozenSemanticCache(development/'semantic-features')
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if encoder.manifest_sha256 != fit_plan['feature_manifest_sha256'] or encoder.manifest['source_payload_sha256'] != digest:
            raise ValueError('feature identity differs')
        plan = {'schema': 'apbpf-training-gradient-audit-v1', 'source_sha256': sources,
                'cache_sha256': file_sha(cache), 'feature_manifest_sha256': encoder.manifest_sha256,
                'comparison_results_sha256': statuses['comparison']['results_sha256'],
                'assignment_results_sha256': statuses['assignment']['results_sha256'],
                'seeds': [1701, 1702, 1703], 'batch_size': 64, 'particles': 32,
                'fitting_candidate_ids': [r['task_id'] for r in rows], 'fitting_sources': 170,
                'roles': {'old': 'original learned-proposal/ESS0.5 training path', 'new': 'prior-proposal/no-resampling training path'},
                'weights': {'future': 1., 'association': 1., 'invariance': .1}, 'margin': .03,
                'randomness': 'per-seed/batch generator and shuffle seeds = seed + 200000 + batch_index; same draws across old/new paths',
                'estimands': ['per-component per-head gradient L2 norm', 'weighted association/base norm ratio', 'base/association gradient cosine'],
                'scope': 'descriptive checkpoint-local gradients on every fitting candidate in fixed sorted batches; original train_apbpf_step backward checked; no parameter updates, Adam-state reconstruction, assessment gradients, primary use, gate or causal attribution'}
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        state.update(status='running', plan_sha256=file_sha(output/'plan.json'))
        save()
        spec = importlib.util.spec_from_file_location('gradient_audit_helper', root/'scripts/run_rbr_prediction_gate.py')
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        batch = helper._tensorize(rows, encoder, torch.device('cpu'), expected_is_public=False)
        summaries = {}
        for seed in plan['seeds']:
            binding = previous['bindings'][str(seed)]
            sidecar_path = training/f'prior-no-resample-seed{seed}.variant.json'
            sidecar = json.loads(sidecar_path.read_text())
            if (file_sha(sidecar_path) != binding['variant_sha256'] or sidecar['inference_variant'] != 'prior_no_resample'
                    or sidecar['model_class'] != fit_plan['model_class'] or sidecar['seed'] != seed
                    or sidecar['artifact_receipt'] != binding['receipts']['new']):
                raise ValueError('model variant mismatch')
            for role, folder, stem, cls in [('old', old_training, f'semantic-p32-seed{seed}', NeuralBeliefModel),
                                          ('new', training, f'prior-no-resample-seed{seed}', PriorTrainingBelief)]:
                key = f'{role}-seed{seed}'
                state.update(current=key, batch_index=0)
                save()
                receipt = binding['receipts'][role]
                if any(Path(n).name != n or file_sha(folder/n) != h for n, h in receipt.items()):
                    raise ValueError('checkpoint receipt mismatch')
                model = cls(512, 32, 192, difficulty_dim=8)
                model.load_state_dict(torch.load(folder/f'{stem}.pt', map_location='cpu', weights_only=True)['model'])
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                recorder = GradientRecorder(model)
                records = []
                for index, start in enumerate(range(0, len(rows), 64)):
                    indices = torch.arange(start, min(start+64, len(rows)))
                    with observe_loss(recorder):
                        metrics = train_apbpf_step(model, helper._subset(batch, indices), recorder,
                            particles=32, visible_steps=4, future_weight=1., association_weight=1., invariance_weight=.1,
                            margin=.03, shuffle_seed=seed+200000+index, generator=torch.Generator().manual_seed(seed+200000+index))
                    records.append({'batch_index': index, 'candidate_offset': start, 'candidate_count': len(indices),
                                    'metrics': metrics, **recorder.record})
                    state['batch_index'] = index
                    save()
                if any(not torch.equal(p, before[n]) for n, p in model.named_parameters()):
                    raise ValueError('diagnostic unexpectedly changed parameters')
                if any(file_sha(root/n) != h for n, h in sources.items()) or file_sha(cache) != plan['cache_sha256']:
                    raise ValueError('frozen audit identity changed')
                path = output/f'{key}.json'
                path.write_text(json.dumps({'records': records, 'model_receipt': receipt}, indent=2)+'\n')
                summary = {'batches': len(records), 'candidates': sum(x['candidate_count'] for x in records),
                           'eligible_candidates': sum(x['metrics']['eligible_count'] for x in records), 'groups': {}}
                for group in records[0]['groups']:
                    summary['groups'][group] = {}
                    for name in ['association_to_base_norm_ratio', 'base_association_cosine']:
                        vals = [x['groups'][group][name] for x in records if x['groups'][group][name] is not None]
                        summary['groups'][group][name] = None if not vals else {'unweighted_batch_mean': float(np.mean(vals)),
                            'quantiles_0_50_100': np.quantile(vals, [0, .5, 1]).tolist(), 'negative_batches': sum(x < 0 for x in vals)}
                summaries[key] = summary
                state['runs'][key] = {'records_sha256': file_sha(path), 'parameters_unchanged': True,
                                     'max_backward_sum_error': max(x['backward_sum_max_absolute_error'] for x in records)}
                save()
        result = {'plan_sha256': state['plan_sha256'], 'cells': summaries, 'bindings': state['runs'], 'scope': plan['scope']}
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        state.update(status='complete', results_sha256=file_sha(output/'results.json'))
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
