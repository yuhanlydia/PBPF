#!/usr/bin/env python3
"""Fixed development-only 2x2 proposal/resampling decomposition at 32 particles."""
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
from pbpf.belief.model import NeuralBeliefModel

ARMS = ('aligned', 'outcome_shuffled', 'joint_reversed', 'presentation_permuted')
VARIANTS = ('learned_resample', 'learned_no_resample', 'prior_resample', 'prior_no_resample')


class DiagnosticBelief(NeuralBeliefModel):
    """Change only inference proposal/resampling; keep trained parameters intact."""
    variant = 'learned_resample'

    def proposal(self, task, candidate, test, outcome, *, parent_z=None, diff=None):
        if self.variant.startswith('prior_'):
            if parent_z is not None or diff is not None:
                raise ValueError('diagnostic supports root candidates only')
            return self.root(task, candidate)
        return super().proposal(task, candidate, test, outcome, parent_z=parent_z, diff=diff)

    def filter(self, batch, **kwargs):
        kwargs['ess_fraction'] = 0.0 if self.variant.endswith('no_resample') else 0.5
        return super().filter(batch, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-root', type=Path, required=True)
    p.add_argument('--sensitivity-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    development, sensitivity, output = (p.resolve() for p in
        (args.development_root, args.sensitivity_root, args.output))
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'validating', 'pid': os.getpid(), 'runs': {}}

    def save():
        path = output/'status.partial'
        path.write_text(json.dumps(state, indent=2)+'\n')
        path.replace(output/'status.json')

    save()
    try:
        cache = development/'development-cache.json'
        payload = json.loads(cache.read_text())
        rows = validate_development(payload)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        previous = json.loads((sensitivity/'status.json').read_text())
        old_plan = json.loads((sensitivity/'plan.json').read_text())
        if (previous['status'] != 'complete' or previous['plan_sha256'] != file_sha(sensitivity/'plan.json')
                or previous['results_sha256'] != file_sha(sensitivity/'results.json')
                or old_plan['cache_sha256'] != file_sha(cache)):
            raise ValueError('completed matching frozen-model particle sensitivity required')
        encoder = FrozenSemanticCache(development/'semantic-features')
        if (encoder.manifest['source_payload_sha256'] != digest
                or encoder.manifest['expected_is_public']):
            raise ValueError('feature population or visibility mismatch')
        bindings = {}
        for seed in (1701, 1702, 1703):
            name = f'semantic-seed{seed}'
            receipt = json.loads((development/f'{name}.checksums.json').read_text())
            if (receipt != old_plan['model_bindings'][name]
                    or any(Path(n).name != n or file_sha(development/n) != h for n, h in receipt.items())):
                raise ValueError('original eight-particle-trained semantic checkpoints changed')
            old = sensitivity/f'{name}-p32.npz'
            if file_sha(old) != previous['runs'][f'{name}-p32']['predictions_sha256']:
                raise ValueError('reference inference predictions changed')
            bindings[str(seed)] = {'model_receipt': receipt, 'reference_predictions_sha256': file_sha(old)}
        paths = [Path(__file__).resolve(), root/'scripts/run_apbpf_particle_diagnostic.py',
                 root/'scripts/run_rbr_prediction_gate.py', *sorted((root/'src/pbpf').rglob('*.py'))]
        sources = {str(p.relative_to(root)): file_sha(p) for p in paths}
        plan = {'schema': 'apbpf-order-decomposition-v1', 'seeds': [1701, 1702, 1703],
                'variants': VARIANTS, 'controls': ARMS, 'training_particles': 8, 'inference_particles': 32,
                'encoder': 'frozen_semantic512', 'cache_sha256': file_sha(cache),
                'feature_manifest_sha256': encoder.manifest_sha256,
                'source_sha256': sources, 'bindings': bindings,
                'hypothesis': 'first-observation-dependent proposal and intermediate resampling both contribute to pair-order sensitivity',
                'design': 'cross learned first-observation proposal versus root-prior proposal with ESS0.5 resampling versus no resampling; same frozen models, sources, particle count and random draws',
                'selection_rule': 'all 12 cells and four controls retained; no selection of best seed or primary evaluation',
                'scope': 'development-only numerical mechanism diagnostic; no training, protocol replacement or positive-method claim'}
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        state.update(status='running', plan_sha256=file_sha(output/'plan.json'))
        save()
        spec = importlib.util.spec_from_file_location('order_diagnostic_helper', root/'scripts/run_rbr_prediction_gate.py')
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        batch = helper._tensorize(rows, encoder, torch.device('cpu'), expected_is_public=False)
        labels = batch.outcomes[:, 4:].reshape(-1).numpy()
        population = [{'candidate_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
        values = {v: {} for v in VARIANTS}

        def check():
            if (any(file_sha(root/n) != h for n, h in sources.items())
                    or file_sha(cache) != plan['cache_sha256']
                    or file_sha(development/'semantic-features/manifest.json') != plan['feature_manifest_sha256']):
                raise ValueError('declared inputs or source changed')

        for seed in plan['seeds']:
            check()
            binding = bindings[str(seed)]
            if any(file_sha(development/n) != h for n, h in binding['model_receipt'].items()):
                raise ValueError('model receipt changed')
            report = json.loads((development/f'semantic-seed{seed}.json').read_text())
            saved = torch.load(development/f'semantic-seed{seed}.pt', map_location='cpu', weights_only=True)
            if (any(saved.get(k) != v for k, v in report['config'].items())
                    or saved['feature_cache_manifest_sha256'] != encoder.manifest_sha256):
                raise ValueError('checkpoint configuration mismatch')
            model = DiagnosticBelief(512, 32, 192, difficulty_dim=8)
            model.load_state_dict(saved['model'])
            model.eval()
            for variant in VARIANTS:
                check()
                key = f'{variant}-seed{seed}'
                state['current'] = key
                save()
                model.variant = variant
                with torch.no_grad():
                    predictions = {arm: helper._predict(model, batch, particles=32, seed=seed+100000, mode=arm) for arm in ARMS}
                errors = {}
                if variant == 'learned_resample':
                    old = sensitivity/f'semantic-seed{seed}-p32.npz'
                    if file_sha(old) != binding['reference_predictions_sha256']:
                        raise ValueError('reference predictions changed during replay')
                    with np.load(old, allow_pickle=False) as archive:
                        if not np.array_equal(archive['labels'], labels):
                            raise ValueError('reference labels differ')
                        errors = {arm: float(np.abs(predictions[arm]-archive[arm]).max()) for arm in ARMS}
                        if max(errors.values()) > 1e-6:
                            raise ValueError('standard variant fails to reproduce original predictions')
                values[variant][seed] = {**predictions, 'labels': labels}
                path = output/f'{key}.npz'
                np.savez_compressed(path, **values[variant][seed])
                state['runs'][key] = {'status': 'complete', 'predictions_sha256': file_sha(path),
                    'standard_reproduction_max_absolute_errors': errors,
                    'pair_order_prediction_max_absolute_differences': {
                        arm: float(np.abs(predictions[arm]-predictions['aligned']).max())
                        for arm in ('joint_reversed', 'presentation_permuted')}}
                save()
        populations = {s: population for s in plan['seeds']}
        results = {}
        for variant in VARIANTS:
            aggregate = aggregate_predictions(values[variant], populations, bootstrap_seed=201701, replicates=10000)
            if variant != 'learned_resample':
                paired = {s: {'aligned': values[variant][s]['aligned'], 'standard': values['learned_resample'][s]['aligned'],
                              'labels': labels} for s in plan['seeds']}
                aggregate['nll_improvement_over_standard'] = aggregate_predictions(paired, populations,
                    bootstrap_seed=201701, replicates=10000)['comparisons']['standard']
            results[variant] = aggregate
        check()
        (output/'results.json').write_text(json.dumps({'plan_sha256': state['plan_sha256'],
            'variants': results, 'prediction_bindings': state['runs'], 'scope': plan['scope']}, indent=2)+'\n')
        state.update(status='complete', results_sha256=file_sha(output/'results.json'))
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
