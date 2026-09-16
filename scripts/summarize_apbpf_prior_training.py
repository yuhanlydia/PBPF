#!/usr/bin/env python3
"""Compare all prior-proposal refits under matched 32-particle inference."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from run_apbpf_prior_training_debug import PriorTrainingBelief
from run_apbpf_particle_diagnostic import validate_development
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.semantic_cache import FrozenSemanticCache
from pbpf.apbpf.stage_prediction import aggregate_predictions
from pbpf.belief.model import NeuralBeliefModel

ARMS = ('aligned', 'outcome_shuffled', 'joint_reversed', 'presentation_permuted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--reference-training-root', type=Path, required=True)
    parser.add_argument('--reference-replay-root', type=Path, required=True)
    parser.add_argument('--development-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    training, old_training, old_replay, development, output = (p.resolve() for p in
        (args.training_root, args.reference_training_root, args.reference_replay_root, args.development_root, args.output))
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'validating', 'pid': os.getpid()}
    def save():
        path = output/'status.partial'
        path.write_text(json.dumps(state, indent=2)+'\n')
        path.replace(output/'status.json')
    save()
    try:
        fitting_plan = json.loads((training/'plan.json').read_text())
        reference_plan = json.loads((old_training/'plan.json').read_text())
        reference_status = json.loads((old_replay/'status.json').read_text())
        if (fitting_plan['inference_variant'] != 'prior_no_resample'
                or reference_status['status'] != 'complete'
                or reference_status['results_sha256'] != file_sha(old_replay/'results.json')):
            raise ValueError('matching prior refits and completed reference replay required')
        reference_result = json.loads((old_replay/'results.json').read_text())
        cache = development/'development-cache.json'
        payload = json.loads(cache.read_text())
        rows = validate_development(payload)
        for plan in (fitting_plan, reference_plan):
            if (plan['particles'] != 32 or plan['seeds'] != [1701, 1702, 1703]
                    or plan['cache_sha256'] != file_sha(cache)):
                raise ValueError('same source population, seeds and training particle budget required')
        files = [Path(__file__).resolve(), root/'scripts/run_apbpf_prior_training_debug.py',
                 root/'scripts/diagnose_apbpf_order_sensitivity.py', root/'scripts/run_apbpf_particle_diagnostic.py',
                 root/'scripts/run_rbr_prediction_gate.py', *sorted((root/'src/pbpf').rglob('*.py'))]
        sources = {str(p.relative_to(root)): file_sha(p) for p in files}
        plan = {'schema': 'apbpf-prior-training-comparison-v1', 'seeds': [1701, 1702, 1703],
                'particles': 32, 'source_sha256': sources, 'cache_sha256': file_sha(cache),
                'training_plan_sha256': file_sha(training/'plan.json'),
                'reference_training_plan_sha256': file_sha(old_training/'plan.json'),
                'reference_replay_sha256': reference_status['results_sha256'],
                'contrasts': ['new prior-trained/prior-inferred versus old learned-trained/prior-inferred',
                              'new prior-trained/prior-inferred versus old learned-trained/standard-inferred'],
                'scope': 'all three development-only seeds; first contrast holds inference fixed; second changes both training and inference; 32 particles throughout but compute not matched; no gate or primary substitution'}
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        state.update(status='waiting_for_all_refits', plan_sha256=file_sha(output/'plan.json'))
        save()
        while True:
            upstream = json.loads((training/'status.json').read_text())
            if upstream['plan_sha256'] != plan['training_plan_sha256']:
                raise ValueError('training declaration changed')
            if upstream['status'] == 'complete':
                break
            process = Path(f"/proc/{upstream['pid']}")
            if upstream['status'] == 'needs_debug' or not process.exists():
                raise RuntimeError('prior-proposal training exited incomplete')
            command = (process/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'run_apbpf_prior_training_debug.py' not in command or str(training) not in command:
                raise RuntimeError('upstream PID is not the declared training process')
            if not args.wait:
                raise RuntimeError('all three refits must finish before comparison')
            time.sleep(20)
        def check():
            if (any(file_sha(root/n) != h for n, h in sources.items())
                    or file_sha(cache) != plan['cache_sha256']
                    or file_sha(training/'plan.json') != plan['training_plan_sha256']
                    or file_sha(old_training/'plan.json') != plan['reference_training_plan_sha256']
                    or file_sha(old_replay/'results.json') != plan['reference_replay_sha256']):
                raise ValueError('declared comparison identity changed')
        check()
        encoder = FrozenSemanticCache(development/'semantic-features')
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if (encoder.manifest_sha256 != fitting_plan['feature_manifest_sha256']
                or encoder.manifest_sha256 != reference_plan['feature_manifest_sha256']
                or encoder.manifest['source_payload_sha256'] != digest):
            raise ValueError('frozen feature population mismatch')
        spec = importlib.util.spec_from_file_location('prior_comparison_helper', root/'scripts/run_rbr_prediction_gate.py')
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        batch = helper._tensorize(rows, encoder, torch.device('cpu'), expected_is_public=False)
        labels = batch.outcomes[:, 4:].reshape(-1).numpy()
        population = [{'candidate_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
        values, paired, old_prior, bindings = {}, {}, {}, {}
        state.update(status='replaying')
        save()
        for seed in plan['seeds']:
            check()
            name = f'prior-no-resample-seed{seed}'
            sidecar = json.loads((training/f'{name}.variant.json').read_text())
            if (file_sha(training/f'{name}.variant.json') != upstream['runs'][name]['variant_sha256']
                    or sidecar['plan_sha256'] != plan['training_plan_sha256']
                    or sidecar['inference_variant'] != 'prior_no_resample'
                    or sidecar['model_class'] != fitting_plan['model_class'] or sidecar['seed'] != seed):
                raise ValueError('mandatory model-variant binding differs')
            models, reports, receipts = {}, {}, {}
            for role, folder, stem in [('new', training, name), ('old', old_training, f'semantic-p32-seed{seed}')]:
                receipt = json.loads((folder/f'{stem}.checksums.json').read_text())
                if any(Path(n).name != n or file_sha(folder/n) != h for n, h in receipt.items()):
                    raise ValueError('model receipt mismatch')
                expected = sidecar['artifact_receipt'] if role == 'new' else reference_result['bindings'][str(seed)]['new_model_receipt']
                if receipt != expected:
                    raise ValueError('model differs from bound training/replay receipt')
                report = json.loads((folder/f'{stem}.json').read_text())
                saved = torch.load(folder/f'{stem}.pt', map_location='cpu', weights_only=True)
                if (report['cache_sha256'] != digest or report['seed'] != seed
                        or any(saved.get(k) != v for k, v in report['config'].items())):
                    raise ValueError('checkpoint configuration or population mismatch')
                model = PriorTrainingBelief(512, 32, 192, difficulty_dim=8)
                model.load_state_dict(saved['model'])
                model.eval()
                models[role], reports[role], receipts[role] = model, report, receipt
            reference = old_replay/f'seed{seed}-predictions.npz'
            if file_sha(reference) != reference_result['bindings'][str(seed)]['comparison_predictions_sha256']:
                raise ValueError('reference standard predictions changed')
            with np.load(reference, allow_pickle=False) as archive:
                if not np.array_equal(archive['labels'], labels):
                    raise ValueError('reference labels differ')
                standard = archive['aligned'].copy()
            with torch.no_grad():
                normal = NeuralBeliefModel(512, 32, 192, difficulty_dim=8)
                normal.load_state_dict(models['old'].state_dict())
                normal.eval()
                reproduced = helper._predict(normal, batch, particles=32, seed=seed+100000, mode='aligned')
                if np.max(np.abs(reproduced-standard)) > 1e-6:
                    raise ValueError('reference standard predictions fail to reproduce')
                values[seed] = {arm: helper._predict(models['new'], batch, particles=32, seed=seed+100000, mode=arm) for arm in ARMS}
                old_prior[seed] = {arm: helper._predict(models['old'], batch, particles=32, seed=seed+100000, mode=arm)
                                   for arm in ('aligned', 'outcome_shuffled')}
            errors = {}
            for arm, prediction in values[seed].items():
                nll = float(-np.log(prediction[np.arange(len(labels)), labels].clip(1e-12, 1)).mean())
                errors[arm] = abs(nll-reports['new']['metrics'][arm]['nll'])
                if errors[arm] > 1e-6:
                    raise ValueError('new checkpoint does not reproduce variant-aware report')
            paired[seed] = {'aligned': values[seed]['aligned'], 'old_prior_inference': old_prior[seed]['aligned'],
                            'old_standard_inference': standard, 'labels': labels}
            values[seed]['labels'] = labels
            old_prior[seed]['labels'] = labels
            path = output/f'seed{seed}-predictions.npz'
            np.savez_compressed(path, **values[seed], old_prior_inference=old_prior[seed]['aligned'], old_standard_inference=standard,
                                old_prior_outcome_shuffled=old_prior[seed]['outcome_shuffled'])
            bindings[str(seed)] = {'receipts': receipts, 'variant_sha256': upstream['runs'][name]['variant_sha256'],
                'predictions_sha256': file_sha(path), 'new_report_nll_reproduction_errors': errors,
                'original_new_gate': reports['new']['gate']}
            state['current_seed'] = seed
            save()
        populations = {s: population for s in plan['seeds']}
        result = {'plan_sha256': state['plan_sha256'], 'bindings': bindings,
            'new_association': aggregate_predictions(values, populations, bootstrap_seed=201701, replicates=10000),
            'old_prior_association': aggregate_predictions(old_prior, populations, bootstrap_seed=201701, replicates=10000),
            'paired_comparisons': aggregate_predictions(paired, populations, bootstrap_seed=201701, replicates=10000),
            'scope': plan['scope']}
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        state.update(status='complete', results_sha256=file_sha(output/'results.json'))
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
