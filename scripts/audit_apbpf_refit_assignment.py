#!/usr/bin/env python3
"""Audit likelihood assignment and history use before/after prior refitting."""
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

from audit_apbpf_likelihood_association import compare_targets, describe
from run_apbpf_prior_training_debug import PriorTrainingBelief
from run_apbpf_particle_diagnostic import validate_development
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.counterfactual import outcome_derangement
from pbpf.apbpf.semantic_cache import FrozenSemanticCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('development-root', 'training-root', 'reference-training-root', 'comparison-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    development, training, old_training, comparison, output = (p.resolve() for p in
        (args.development_root, args.training_root, args.reference_training_root, args.comparison_root, args.output))
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'validating', 'pid': os.getpid(), 'runs': {}}

    def save():
        temp = output/'status.partial'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(output/'status.json')

    save()
    try:
        if torch.cuda.is_available():
            raise ValueError('CPU-only audit; GPU workers must remain uninterrupted')
        upstream = json.loads((comparison/'status.json').read_text())
        upstream_plan = json.loads((comparison/'plan.json').read_text())
        result = json.loads((comparison/'results.json').read_text())
        fitting_plan = json.loads((training/'plan.json').read_text())
        cache = development/'development-cache.json'
        payload = json.loads(cache.read_text())
        rows = validate_development(payload)
        if (upstream['status'] != 'complete'
                or upstream['plan_sha256'] != file_sha(comparison/'plan.json')
                or upstream['results_sha256'] != file_sha(comparison/'results.json')
                or result['plan_sha256'] != upstream['plan_sha256']
                or upstream_plan['cache_sha256'] != file_sha(cache)
                or upstream_plan['training_plan_sha256'] != file_sha(training/'plan.json')
                or upstream_plan['reference_training_plan_sha256'] != file_sha(old_training/'plan.json')
                or fitting_plan['inference_variant'] != 'prior_no_resample'):
            raise ValueError('completed, matching three-seed comparison required')
        for name, checksum in upstream_plan['source_sha256'].items():
            if file_sha(root/name) != checksum:
                raise ValueError('comparison implementation changed')
        encoder = FrozenSemanticCache(development/'semantic-features')
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if (encoder.manifest_sha256 != fitting_plan['feature_manifest_sha256']
                or encoder.manifest['source_payload_sha256'] != digest):
            raise ValueError('feature population mismatch')
        sources = dict(upstream_plan['source_sha256'])
        for path in [Path(__file__).resolve(), root/'scripts/audit_apbpf_likelihood_association.py', root/'local/sandbox.sh']:
            sources[str(path.relative_to(root))] = file_sha(path)
        plan = {'schema': 'apbpf-refit-assignment-audit-v1', 'source_sha256': sources,
                'comparison_plan_sha256': upstream['plan_sha256'], 'comparison_results_sha256': upstream['results_sha256'],
                'training_plan_sha256': upstream_plan['training_plan_sha256'],
                'reference_training_plan_sha256': upstream_plan['reference_training_plan_sha256'],
                'cache_sha256': file_sha(cache), 'feature_manifest_sha256': encoder.manifest_sha256,
                'seeds': [1701, 1702, 1703], 'particles': 32, 'roles': ['old', 'new'],
                'hypothesis': 'refitting may improve static predictions without meaningful likelihood assignment sensitivity',
                'estimands': ['centered assignment log-likelihood ratio RMS', 'aligned/shuffled posterior total variation',
                              'aligned/shuffled maximum future probability change', 'history posterior versus prior predictive total variation',
                              'prior versus history posterior candidate-average future NLL'],
                'randomness': 'same standard-normal noise by seed; model-specific learned prior transforms; aligned/shuffled within each model share exact particles',
                'scope': 'descriptive development-only six-cell audit; all seeds retained; fixed32 sampled support, not exact continuous posterior; no new fitting, primary evaluation, gate substitution or compute-matched claim'}
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        state.update(status='running', plan_sha256=file_sha(output/'plan.json'))
        save()

        def check():
            if (any(file_sha(root/n) != h for n, h in sources.items())
                    or file_sha(cache) != plan['cache_sha256']
                    or file_sha(comparison/'plan.json') != plan['comparison_plan_sha256']
                    or file_sha(comparison/'results.json') != plan['comparison_results_sha256']
                    or file_sha(training/'plan.json') != plan['training_plan_sha256']
                    or file_sha(old_training/'plan.json') != plan['reference_training_plan_sha256']
                    or file_sha(development/'semantic-features/manifest.json') != plan['feature_manifest_sha256']):
                raise ValueError('declared audit identity changed')

        spec = importlib.util.spec_from_file_location('refit_assignment_helper', root/'scripts/run_rbr_prediction_gate.py')
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        batch = helper._tensorize(rows, encoder, torch.device('cpu'), expected_is_public=False)
        labels = batch.outcomes[:, 4:].reshape(-1).numpy()
        eligible = np.any(batch.outcomes[:, :4].numpy() != batch.outcomes[:, :1].numpy(), axis=1)
        summaries = {}
        for seed in plan['seeds']:
            bindings = result['bindings'][str(seed)]
            reference = comparison/f'seed{seed}-predictions.npz'
            if file_sha(reference) != bindings['predictions_sha256']:
                raise ValueError('reference predictions changed')
            sidecar_path = training/f'prior-no-resample-seed{seed}.variant.json'
            sidecar = json.loads(sidecar_path.read_text())
            if (file_sha(sidecar_path) != bindings['variant_sha256']
                    or sidecar['plan_sha256'] != plan['training_plan_sha256']
                    or sidecar['inference_variant'] != 'prior_no_resample'
                    or sidecar['model_class'] != fitting_plan['model_class']
                    or sidecar['seed'] != seed or sidecar['artifact_receipt'] != bindings['receipts']['new']):
                raise ValueError('mandatory model variant binding changed')
            for role, folder, stem, arms in [
                ('old', old_training, f'semantic-p32-seed{seed}', ('old_prior_inference', 'old_prior_outcome_shuffled')),
                ('new', training, f'prior-no-resample-seed{seed}', ('aligned', 'outcome_shuffled'))]:
                check()
                key = f'{role}-seed{seed}'
                state['current'] = key
                save()
                receipt = bindings['receipts'][role]
                if any(Path(n).name != n or file_sha(folder/n) != h for n, h in receipt.items()):
                    raise ValueError('model receipt mismatch')
                saved = torch.load(folder/f'{stem}.pt', map_location='cpu', weights_only=True)
                model = PriorTrainingBelief(512, 32, 192, difficulty_dim=8)
                model.load_state_dict(saved['model'])
                model.eval()
                with torch.no_grad():
                    rng = torch.Generator().manual_seed(seed+100000)
                    noise = torch.randn(len(rows), 32, 32, generator=rng)
                    prior = model.root(batch.task, batch.candidate)
                    z = prior.mean[:, None] + prior.std[:, None]*noise
                    shuffled = torch.tensor(outcome_derangement(batch.outcomes.numpy(), 4, seed+100000), dtype=torch.long)
                    predictions, arrays = compare_targets(model, batch, z, shuffled)
                    uniform = z.new_full(z.shape[:2], -math.log(z.shape[1]))
                    prior_prediction = model.future_predict(batch.task, batch.candidate, batch.tests[:, 4:], z, uniform).exp().reshape(-1, 5).numpy()
                with np.load(reference, allow_pickle=False) as archive:
                    if not np.array_equal(labels, archive['labels']):
                        raise ValueError('reference labels differ')
                    errors = {arm: float(np.max(np.abs(predictions[k]-archive[arm]))) for k, arm in enumerate(arms)}
                    if max(errors.values()) > 1e-6:
                        raise ValueError('direct likelihood replay fails to reproduce completed comparison')
                arrays['prior_future_nll'] = -np.log(prior_prediction[np.arange(len(labels)), labels].astype(np.float64).clip(1e-12, 1)).reshape(len(rows), -1).mean(-1)
                arrays['posterior_future_nll'] = -np.log(predictions[0][np.arange(len(labels)), labels].astype(np.float64).clip(1e-12, 1)).reshape(len(rows), -1).mean(-1)
                arrays['history_nll_gain'] = arrays['prior_future_nll']-arrays['posterior_future_nll']
                arrays['history_future_total_variation'] = (0.5*np.abs(predictions[0]-prior_prediction).sum(-1)).reshape(len(rows), -1).mean(-1)
                if any(not np.isfinite(a).all() for a in arrays.values()):
                    raise ValueError('nonfinite audit quantity')
                for name in ['centered_log_likelihood_ratio_rms', 'posterior_total_variation', 'future_probability_max_difference']:
                    if np.max(np.abs(arrays[name][~eligible])) > 1e-7:
                        raise ValueError('constant visible outcomes must have no assignment effect')
                path = output/f'{key}.npz'
                np.savez_compressed(path, **arrays, prior_prediction=prior_prediction, eligible=eligible,
                                    candidate_ids=np.asarray([r['task_id'] for r in rows]),
                                    source_ids=np.asarray([r['source_component_id'] for r in rows]))
                summaries[key] = {name: {stratum: describe(a[mask]) for stratum, mask in
                    [('full', np.ones(len(rows), dtype=bool)), ('mixed_visible', eligible), ('constant_visible', ~eligible)]}
                    for name, a in arrays.items() if a.ndim == 1}
                state['runs'][key] = {'diagnostics_sha256': file_sha(path), 'reference_reproduction_errors': errors,
                                      'model_receipt': receipt, 'comparison_predictions_sha256': bindings['predictions_sha256']}
                save()
        check()
        audit = {'plan_sha256': state['plan_sha256'], 'cells': summaries, 'bindings': state['runs'],
                 'candidate_counts': {'full': len(rows), 'mixed_visible': int(eligible.sum()), 'constant_visible': int((~eligible).sum())},
                 'scope': plan['scope']}
        (output/'results.json').write_text(json.dumps(audit, indent=2)+'\n')
        state.update(status='complete', results_sha256=file_sha(output/'results.json'))
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
