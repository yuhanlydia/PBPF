#!/usr/bin/env python3
"""Compare all three weight100 refits with weight1 under identical inference."""
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

ARMS = ('aligned', 'outcome_shuffled', 'joint_reversed', 'presentation_permuted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-root', 'reference-training-root', 'reference-replay-root', 'development-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    training, reference, replay, development, output = (p.resolve() for p in
        (args.training_root, args.reference_training_root, args.reference_replay_root, args.development_root, args.output))
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'validating', 'pid': os.getpid()}

    def save():
        temp = output/'status.partial'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(output/'status.json')

    save()
    try:
        fit = json.loads((training/'plan.json').read_text())
        old_fit = json.loads((reference/'plan.json').read_text())
        old_status = json.loads((replay/'status.json').read_text())
        old_result = json.loads((replay/'results.json').read_text())
        cache = development/'development-cache.json'
        payload = json.loads(cache.read_text())
        rows = validate_development(payload)
        if (old_status['status'] != 'complete' or old_status['results_sha256'] != file_sha(replay/'results.json')
                or old_status['plan_sha256'] != file_sha(replay/'plan.json')
                or json.loads((replay/'plan.json').read_text())['training_plan_sha256'] != file_sha(reference/'plan.json')
                or fit['association_weight'] != 100. or fit['reference_association_weight'] != 1.):
            raise ValueError('weight100 declaration and completed weight1 replay required')
        for declaration in (fit, old_fit):
            if (declaration['inference_variant'] != 'prior_no_resample'
                    or declaration['model_class'] != 'run_apbpf_prior_training_debug.PriorTrainingBelief'
                    or declaration['particles'] != 32 or declaration['steps'] != 1000
                    or declaration['seeds'] != [1701, 1702, 1703] or declaration['cache_sha256'] != file_sha(cache)):
                raise ValueError('matched prior-inference training declarations required')
        sources = dict(fit['source_sha256'])
        sources['scripts/summarize_apbpf_association_weight.py'] = file_sha(Path(__file__))
        plan = {'schema': 'apbpf-association-weight-comparison-v1', 'source_sha256': sources,
                'training_plan_sha256': file_sha(training/'plan.json'),
                'reference_training_plan_sha256': file_sha(reference/'plan.json'),
                'reference_replay_sha256': old_status['results_sha256'],
                'cache_sha256': file_sha(cache), 'feature_manifest_sha256': fit['feature_manifest_sha256'],
                'seeds': [1701, 1702, 1703], 'weights': [1., 100.], 'particles': 32, 'controls': ARMS,
                'contrasts': 'weight1 NLL minus weight100 NLL for each arm separately; positive means lower weight100 loss',
                'interpretation': 'inspect aligned and shuffled changes together; larger association gap can arise from worse counterfactual predictions and is not alone evidence of better aligned prediction',
                'scope': 'all three fixed seeds and all400 development sources; identical inference class and particle count; original gates retained; no primary assessment or best-seed substitution'}
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        state.update(status='waiting_for_all_refits', plan_sha256=file_sha(output/'plan.json'))
        save()

        def check():
            if (any(file_sha(root/n) != h for n, h in sources.items())
                    or file_sha(training/'plan.json') != plan['training_plan_sha256']
                    or file_sha(reference/'plan.json') != plan['reference_training_plan_sha256']
                    or file_sha(replay/'results.json') != plan['reference_replay_sha256']
                    or file_sha(cache) != plan['cache_sha256']
                    or file_sha(development/'semantic-features/manifest.json') != plan['feature_manifest_sha256']):
                raise ValueError('comparison identity changed')

        check()
        while True:
            upstream = json.loads((training/'status.json').read_text())
            if upstream['plan_sha256'] != plan['training_plan_sha256']:
                raise ValueError('training identity changed')
            if upstream['status'] == 'complete':
                break
            process = Path(f"/proc/{upstream['pid']}")
            if upstream['status'] == 'needs_debug' or not process.exists():
                raise RuntimeError('weight100 training exited incomplete')
            command = (process/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'run_apbpf_association_weight_debug.py' not in command or str(training) not in command:
                raise RuntimeError('upstream PID is not the declared trainer')
            if not args.wait:
                raise RuntimeError('all three refits required')
            time.sleep(20)
        check()
        encoder = FrozenSemanticCache(development/'semantic-features')
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if (encoder.manifest_sha256 != old_fit['feature_manifest_sha256']
                or encoder.manifest_sha256 != plan['feature_manifest_sha256']
                or encoder.manifest['source_payload_sha256'] != digest):
            raise ValueError('feature population mismatch')
        spec = importlib.util.spec_from_file_location('weight_comparison_helper', root/'scripts/run_rbr_prediction_gate.py')
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        batch = helper._tensorize(rows, encoder, torch.device('cpu'), expected_is_public=False)
        labels = batch.outcomes[:, 4:].reshape(-1).numpy()
        population = [{'candidate_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
        values = {role: {} for role in ('weight1', 'weight100')}
        bindings = {}
        state['status'] = 'replaying'
        save()
        for seed in plan['seeds']:
            check()
            state['current_seed'] = seed
            save()
            old_binding = old_result['bindings'][str(seed)]
            reference_archive = replay/f'seed{seed}-predictions.npz'
            if file_sha(reference_archive) != old_binding['predictions_sha256']:
                raise ValueError('reference archive changed')
            receipts, errors, gates = {}, {}, {}
            for role, folder, stem, coefficient, declaration in [
                ('weight1', reference, f'prior-no-resample-seed{seed}', 1., old_fit),
                ('weight100', training, f'association-weight100-seed{seed}', 100., fit)]:
                sidecar_path = folder/f'{stem}.variant.json'
                sidecar = json.loads(sidecar_path.read_text())
                expected_sidecar = (old_binding['variant_sha256'] if role == 'weight1'
                                    else upstream['runs'][stem]['variant_sha256'])
                if (file_sha(sidecar_path) != expected_sidecar or sidecar['plan_sha256'] != file_sha(folder/'plan.json')
                        or sidecar['model_class'] != declaration['model_class']
                        or sidecar['inference_variant'] != 'prior_no_resample' or sidecar['seed'] != seed
                        or (role == 'weight100' and sidecar['association_weight'] != coefficient)):
                    raise ValueError('variant binding mismatch')
                receipt = json.loads((folder/f'{stem}.checksums.json').read_text())
                if (receipt != sidecar['artifact_receipt']
                        or (role == 'weight1' and receipt != old_binding['receipts']['new'])
                        or any(Path(n).name != n or file_sha(folder/n) != h for n, h in receipt.items())):
                    raise ValueError('checkpoint/report receipt mismatch')
                report = json.loads((folder/f'{stem}.json').read_text())
                saved = torch.load(folder/f'{stem}.pt', map_location='cpu', weights_only=True)
                if (report['cache_sha256'] != digest or report['seed'] != seed
                        or report['config']['association_weight'] != coefficient
                        or any(saved.get(k) != v for k, v in report['config'].items())):
                    raise ValueError('checkpoint/configuration mismatch')
                model = PriorTrainingBelief(512, 32, 192, difficulty_dim=8)
                model.load_state_dict(saved['model'])
                model.eval()
                with torch.no_grad():
                    predictions = {arm: helper._predict(model, batch, particles=32, seed=seed+100000, mode=arm) for arm in ARMS}
                error = {}
                for arm, prediction in predictions.items():
                    nll = float(-np.log(prediction[np.arange(len(labels)), labels].clip(1e-12, 1)).mean())
                    error[arm] = abs(nll-report['metrics'][arm]['nll'])
                if max(error.values()) > 1e-6:
                    raise ValueError('report NLL does not reproduce')
                if role == 'weight1':
                    with np.load(reference_archive, allow_pickle=False) as archive:
                        if not np.array_equal(labels, archive['labels']) or any(np.max(np.abs(predictions[a]-archive[a])) > 1e-6 for a in ARMS):
                            raise ValueError('original weight1 predictions do not reproduce')
                values[role][seed] = {**predictions, 'labels': labels}
                receipts[role], errors[role], gates[role] = receipt, error, report['gate']
            path = output/f'seed{seed}-predictions.npz'
            np.savez_compressed(path, labels=labels, **{f'{role}_{arm}': values[role][seed][arm] for role in values for arm in ARMS})
            bindings[str(seed)] = {'receipts': receipts, 'report_reproduction_nll_errors': errors,
                                   'original_gates': gates, 'predictions_sha256': file_sha(path)}
        populations = {s: population for s in plan['seeds']}
        comparison = {}
        for arm in ARMS:
            paired = {s: {'aligned': values['weight100'][s][arm], 'weight1': values['weight1'][s][arm], 'labels': labels} for s in plan['seeds']}
            comparison[arm] = aggregate_predictions(paired, populations, bootstrap_seed=201701, replicates=10000)
        result = {'plan_sha256': state['plan_sha256'], 'bindings': bindings,
                  'association_by_weight': {role: aggregate_predictions(v, populations, bootstrap_seed=201701, replicates=10000) for role, v in values.items()},
                  'paired_nll_improvements_by_arm': comparison, 'scope': plan['scope'], 'interpretation': plan['interpretation']}
        check()
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        state.update(status='complete', results_sha256=file_sha(output/'results.json'))
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
