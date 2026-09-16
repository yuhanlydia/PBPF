#!/usr/bin/env python3
"""Measure outcome-assignment sensitivity on common prior particle support."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

from run_apbpf_particle_diagnostic import validate_development
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.counterfactual import outcome_derangement
from pbpf.apbpf.semantic_cache import FrozenSemanticCache
from pbpf.belief.model import NeuralBeliefModel


def describe(values):
    return {'count': len(values), 'mean': float(np.mean(values)),
            'quantiles_0_50_90_99_100': np.quantile(values, [0, .5, .9, .99, 1]).tolist()}


@torch.no_grad()
def compare_targets(model, batch, z, shuffled_outcomes):
    """Same prior samples and likelihood evaluations; only assignments change."""
    weights = [z.new_full(z.shape[:2], -math.log(z.shape[1])) for _ in range(2)]
    totals = [torch.zeros_like(weights[0]) for _ in range(2)]
    for j in range(4):
        likelihood = model.likelihood(z, batch.task, batch.candidate, batch.tests[:, j])
        for k, outcomes in enumerate((batch.outcomes, shuffled_outcomes)):
            term = likelihood.gather(-1, outcomes[:, j, None, None].expand(-1, z.shape[1], 1)).squeeze(-1)
            totals[k] = totals[k] + term
            corrected = weights[k] + term
            weights[k] = corrected - torch.logsumexp(corrected, -1, keepdim=True)
    delta = totals[0] - totals[1]
    centered = delta - delta.mean(-1, keepdim=True)
    predictions = [model.future_predict(batch.task, batch.candidate, batch.tests[:, 4:], z, w).exp()
                   .reshape(-1, 5).numpy() for w in weights]
    return predictions, {'centered_log_likelihood_ratio_rms': centered.square().mean(-1).sqrt().numpy(),
        'posterior_total_variation': (0.5*(weights[0].exp()-weights[1].exp()).abs().sum(-1)).numpy(),
        'future_probability_max_difference': np.abs(predictions[0]-predictions[1]).reshape(len(z), -1).max(-1),
        'centered_log_likelihood_ratio': centered.numpy()}


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
        path = output/'status.partial'
        path.write_text(json.dumps(state, indent=2)+'\n')
        path.replace(output/'status.json')
    save()
    try:
        prior_status = json.loads((order/'status.json').read_text())
        prior_plan = json.loads((order/'plan.json').read_text())
        cache = development/'development-cache.json'
        payload = json.loads(cache.read_text())
        rows = validate_development(payload)
        if (prior_status['status'] != 'complete' or prior_status['results_sha256'] != file_sha(order/'results.json')
                or prior_status['plan_sha256'] != file_sha(order/'plan.json')
                or prior_plan['cache_sha256'] != file_sha(cache)):
            raise ValueError('completed matching order diagnostic required')
        for name, sha in prior_plan['source_sha256'].items():
            if file_sha(root/name) != sha:
                raise ValueError('upstream implementation changed')
        encoder = FrozenSemanticCache(development/'semantic-features')
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if (encoder.manifest_sha256 != prior_plan['feature_manifest_sha256']
                or encoder.manifest['source_payload_sha256'] != digest):
            raise ValueError('feature/cache binding differs')
        sources = {str(path.relative_to(root)): file_sha(path) for path in
            [Path(__file__).resolve(), root/'scripts/run_apbpf_particle_diagnostic.py',
             root/'scripts/run_rbr_prediction_gate.py', *sorted((root/'src/pbpf').rglob('*.py'))]}
        plan = {'schema': 'apbpf-likelihood-assignment-audit-v1', 'source_sha256': sources,
                'seeds': [1701, 1702, 1703], 'particles': 32, 'cache_sha256': file_sha(cache),
                'order_plan_sha256': prior_status['plan_sha256'], 'model_bindings': prior_plan['bindings'],
                'reference_predictions': {str(s): prior_status['runs'][f'prior_no_resample-seed{s}']['predictions_sha256']
                                          for s in (1701, 1702, 1703)},
                'estimands': ['particle-centered log likelihood assignment ratio RMS',
                              'aligned/shuffled posterior total variation on identical support',
                              'maximum future probability difference per candidate'],
                'hypothesis': 'near-zero association after invariant inference may reflect weak likelihood assignment dependence on the sampled support',
                'scope': 'descriptive development-only mechanism audit on common32-prior-draw support; not exact continuous posterior, new fit, gate or primary evaluation'}
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        state.update(status='running', plan_sha256=file_sha(output/'plan.json'))
        save()
        spec = importlib.util.spec_from_file_location('likelihood_assignment_helper', root/'scripts/run_rbr_prediction_gate.py')
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        batch = helper._tensorize(rows, encoder, torch.device('cpu'), expected_is_public=False)
        labels = batch.outcomes[:, 4:].reshape(-1).numpy()
        eligible = np.any(batch.outcomes[:, :4].numpy() != batch.outcomes[:, :1].numpy(), axis=1)
        results = {}
        for seed in plan['seeds']:
            if any(file_sha(root/n) != sha for n, sha in sources.items()) or file_sha(cache) != plan['cache_sha256']:
                raise ValueError('declared input or implementation changed')
            receipt = plan['model_bindings'][str(seed)]['model_receipt']
            if any(Path(n).name != n or file_sha(development/n) != sha for n, sha in receipt.items()):
                raise ValueError('model receipt mismatch')
            state['current_seed'] = seed
            save()
            saved = torch.load(development/f'semantic-seed{seed}.pt', map_location='cpu', weights_only=True)
            model = NeuralBeliefModel(512, 32, 192, difficulty_dim=8)
            model.load_state_dict(saved['model'])
            model.eval()
            with torch.no_grad():
                rng = torch.Generator().manual_seed(seed+100000)
                noise = torch.randn(len(rows), 32, 32, generator=rng)
                prior = model.root(batch.task, batch.candidate)
                z = prior.mean[:, None] + prior.std[:, None] * noise
                shuffled = torch.tensor(outcome_derangement(batch.outcomes.numpy(), 4, seed+100000), dtype=torch.long)
                predictions, arrays = compare_targets(model, batch, z, shuffled)
            reference = order/f'prior_no_resample-seed{seed}.npz'
            if file_sha(reference) != plan['reference_predictions'][str(seed)]:
                raise ValueError('reference predictions changed')
            with np.load(reference, allow_pickle=False) as archive:
                if not np.array_equal(labels, archive['labels']):
                    raise ValueError('reference labels differ')
                errors = {arm: float(np.abs(predictions[k]-archive[arm]).max())
                          for k, arm in enumerate(('aligned', 'outcome_shuffled'))}
                if max(errors.values()) > 1e-6:
                    raise ValueError('direct target evaluation fails to reproduce completed control')
            path = output/f'seed{seed}-diagnostics.npz'
            np.savez_compressed(path, **arrays, eligible=eligible,
                                candidate_ids=np.asarray([r['task_id'] for r in rows]),
                                source_ids=np.asarray([r['source_component_id'] for r in rows]))
            results[str(seed)] = {name: {stratum: describe(values[mask]) for stratum, mask in
                [('full', np.ones(len(rows), dtype=bool)), ('mixed_visible', eligible), ('constant_visible', ~eligible)]}
                for name, values in arrays.items() if values.ndim == 1}
            state['runs'][str(seed)] = {'diagnostics_sha256': file_sha(path), 'reference_reproduction_errors': errors}
            save()
        result = {'plan_sha256': state['plan_sha256'], 'seeds': results, 'bindings': state['runs'],
                  'candidate_counts': {'full': len(rows), 'mixed_visible': int(eligible.sum()),
                                       'constant_visible': int((~eligible).sum())}, 'scope': plan['scope']}
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        state.update(status='complete', results_sha256=file_sha(output/'results.json'))
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
