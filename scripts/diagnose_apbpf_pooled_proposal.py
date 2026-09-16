#!/usr/bin/env python3
"""Evaluate one exchangeable moment-pooled proposal on frozen development models."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from run_apbpf_particle_diagnostic import validate_development
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.semantic_cache import FrozenSemanticCache
from pbpf.apbpf.stage_prediction import aggregate_predictions
from pbpf.belief.model import GaussianParams, NeuralBeliefModel

ARMS = ('aligned', 'outcome_shuffled', 'joint_reversed', 'presentation_permuted')


class PooledProposalBelief(NeuralBeliefModel):
    """A single Gaussian matching marginal moments of visible proposal Gaussians.

    This is an explicitly approximate proposal, not a Gaussian-mixture density.
    The base filter retains the exact prior/proposal importance correction.
    """
    _pooled_proposal = None

    def pooled_proposal(self, batch, visible_steps):
        proposals = [super(PooledProposalBelief, self).proposal(
            batch.task, batch.candidate, batch.tests[:, j], batch.outcomes[:, j]) for j in range(visible_steps)]
        means = torch.stack([q.mean for q in proposals]).double()
        variances = torch.stack([q.std.square() for q in proposals]).double()
        mean = means.mean(0)
        variance = (variances + (means-mean).square()).mean(0)
        return GaussianParams(mean.to(batch.task.dtype),
            (0.5*variance.clamp_min(1e-12).log()).to(batch.task.dtype))

    def proposal(self, task, candidate, test, outcome, *, parent_z=None, diff=None):
        if self._pooled_proposal is None or parent_z is not None or diff is not None:
            raise ValueError('pooled proposal is only available inside root filtering')
        return self._pooled_proposal

    def filter(self, batch, **kwargs):
        visible_steps = kwargs.get('visible_steps', 4)
        if kwargs.get('parents') is not None or visible_steps != 4:
            raise ValueError('diagnostic requires exactly four visible root observations')
        self._pooled_proposal = self.pooled_proposal(batch, visible_steps)
        kwargs['ess_fraction'] = 0.0
        try:
            return super().filter(batch, **kwargs)
        finally:
            self._pooled_proposal = None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-root', type=Path, required=True)
    p.add_argument('--order-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    development, order, output = (p.resolve() for p in (args.development_root, args.order_root, args.output))
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'validating', 'pid': os.getpid(), 'runs': {}}
    def save():
        temporary = output/'status.partial'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(output/'status.json')
    save()
    try:
        upstream = json.loads((order/'status.json').read_text())
        order_plan = json.loads((order/'plan.json').read_text())
        cache = development/'development-cache.json'
        payload = json.loads(cache.read_text())
        rows = validate_development(payload)
        if (upstream['status'] != 'complete' or upstream['results_sha256'] != file_sha(order/'results.json')
                or upstream['plan_sha256'] != file_sha(order/'plan.json')
                or file_sha(cache) != order_plan['cache_sha256']):
            raise ValueError('completed matching order diagnostic required')
        for name, checksum in order_plan['source_sha256'].items():
            if file_sha(root/name) != checksum:
                raise ValueError('original order diagnostic source changed')
        features = development/'semantic-features'
        encoder = FrozenSemanticCache(features)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if (encoder.manifest_sha256 != order_plan['feature_manifest_sha256']
                or encoder.manifest['source_payload_sha256'] != digest
                or encoder.manifest['expected_is_public']):
            raise ValueError('frozen feature population differs')
        sources = {str(p.relative_to(root)): file_sha(p) for p in
            [Path(__file__).resolve(), root/'scripts/run_apbpf_particle_diagnostic.py',
             root/'scripts/run_rbr_prediction_gate.py', *sorted((root/'src/pbpf').rglob('*.py'))]}
        reference_hashes = {k: v['predictions_sha256'] for k, v in upstream['runs'].items()}
        plan = {'schema': 'apbpf-pooled-proposal-diagnostic-v1', 'seeds': [1701, 1702, 1703],
                'particles': 32, 'controls': ARMS, 'source_sha256': sources,
                'cache_sha256': file_sha(cache), 'feature_manifest_sha256': encoder.manifest_sha256,
                'order_plan_sha256': upstream['plan_sha256'], 'order_results_sha256': upstream['results_sha256'],
                'model_bindings': order_plan['bindings'], 'reference_predictions': reference_hashes,
                'proposal': 'single diagonal Gaussian with equal-weight mean and marginal variance of the four per-observation proposal Gaussians; exact Gaussian prior/proposal correction; no resampling',
                'hypothesis': 'pooling all four visible proposals may retain informative proposal quality while eliminating first-observation order dependence',
                'selection': 'one fixed variant, all three seeds and all400 development sources; no refitting or primary evaluation',
                'limitations': 'four proposal evaluations increase compute; moment matching need not approximate the posterior well; all comparisons exploratory',
                'scope': 'development-only inference diagnostic; original protocol and gates remain unchanged'}
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        state.update(status='running', plan_sha256=file_sha(output/'plan.json'))
        save()
        spec = importlib.util.spec_from_file_location('pooled_proposal_helper', root/'scripts/run_rbr_prediction_gate.py')
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        batch = helper._tensorize(rows, encoder, torch.device('cpu'), expected_is_public=False)
        labels = batch.outcomes[:, 4:].reshape(-1).numpy()
        population = [{'candidate_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
        values, references = {}, {k: {} for k in ('learned_resample', 'learned_no_resample', 'prior_no_resample')}
        for seed in plan['seeds']:
            if (any(file_sha(root/n) != h for n, h in sources.items()) or file_sha(cache) != plan['cache_sha256']
                    or file_sha(features/'manifest.json') != plan['feature_manifest_sha256']):
                raise ValueError('declared source or data changed')
            receipt = plan['model_bindings'][str(seed)]['model_receipt']
            if any(Path(n).name != n or file_sha(development/n) != h for n, h in receipt.items()):
                raise ValueError('model receipts changed')
            saved = torch.load(development/f'semantic-seed{seed}.pt', map_location='cpu', weights_only=True)
            report = json.loads((development/f'semantic-seed{seed}.json').read_text())
            if any(saved.get(k) != v for k, v in report['config'].items()):
                raise ValueError('model configuration mismatch')
            model = PooledProposalBelief(512, 32, 192, difficulty_dim=8)
            model.load_state_dict(saved['model'])
            model.eval()
            state['current_seed'] = seed
            save()
            with torch.no_grad():
                predictions = {arm: helper._predict(model, batch, particles=32, seed=seed+100000, mode=arm) for arm in ARMS}
            values[seed] = {**predictions, 'labels': labels}
            path = output/f'seed{seed}-predictions.npz'
            np.savez_compressed(path, **values[seed])
            state['runs'][str(seed)] = {'predictions_sha256': file_sha(path),
                'max_pair_order_prediction_difference': max(float(np.abs(predictions[arm]-predictions['aligned']).max())
                    for arm in ('joint_reversed', 'presentation_permuted'))}
            for variant in references:
                name = f'{variant}-seed{seed}'
                path = order/f'{name}.npz'
                if file_sha(path) != reference_hashes[name]:
                    raise ValueError('reference predictions changed')
                with np.load(path, allow_pickle=False) as data:
                    if not np.array_equal(data['labels'], labels):
                        raise ValueError('reference population labels differ')
                    references[variant][seed] = data['aligned'].copy()
            save()
        populations = {s: population for s in plan['seeds']}
        comparisons = {}
        for variant, old in references.items():
            paired = {s: {'aligned': values[s]['aligned'], 'reference': old[s], 'labels': labels} for s in plan['seeds']}
            comparisons[variant] = aggregate_predictions(paired, populations, bootstrap_seed=201701,
                replicates=10000)['comparisons']['reference']
        result = {'schema': plan['schema'], 'plan_sha256': state['plan_sha256'], 'prediction_bindings': state['runs'],
            'aggregate': aggregate_predictions(values, populations, bootstrap_seed=201701, replicates=10000),
            'paired_nll_improvements': comparisons, 'scope': plan['scope']}
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        state.update(status='complete', results_sha256=file_sha(output/'results.json'))
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
