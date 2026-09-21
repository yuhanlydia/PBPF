#!/usr/bin/env python3
"""Compare all development refits at the same 32-particle inference budget."""
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

from run_apbpf_particle_diagnostic import validate_development
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.semantic_cache import FrozenSemanticCache
from pbpf.apbpf.stage_prediction import aggregate_predictions
from pbpf.belief.model import NeuralBeliefModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--development-root', type=Path, required=True)
    parser.add_argument('--sensitivity-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    training, development, sensitivity = (p.resolve() for p in
        (args.training_root, args.development_root, args.sensitivity_root))
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    fitting_plan = json.loads((training/'plan.json').read_text())
    sensitivity_status = json.loads((sensitivity/'status.json').read_text())
    cache = development/'development-cache.json'
    payload = json.loads(cache.read_text()); rows = validate_development(payload)
    if (fitting_plan['particles'] != 32 or fitting_plan['seeds'] != [1701, 1702, 1703]
            or file_sha(cache) != fitting_plan['cache_sha256']
            or sensitivity_status['status'] != 'complete'
            or file_sha(sensitivity/'results.json') != fitting_plan['sensitivity_results_sha256']):
        raise ValueError('matching completed sensitivity study and declared development refits required')
    sources = {str(p.relative_to(root)): file_sha(p) for p in
        [Path(__file__).resolve(), root/'scripts/run_apbpf_particle_diagnostic.py',
         root/'scripts/run_rbr_prediction_gate.py', *sorted((root/'src/pbpf').rglob('*.py'))]}
    old_predictions = {}
    for seed in fitting_plan['seeds']:
        name = f'semantic-seed{seed}-p32'
        path = sensitivity/f'{name}.npz'
        if file_sha(path) != sensitivity_status['runs'][name]['predictions_sha256']:
            raise ValueError('old same-inference-budget predictions changed')
        old_predictions[str(seed)] = file_sha(path)
    plan = {'schema': 'apbpf-particle-training-comparison-v1', 'training_plan_sha256': file_sha(training/'plan.json'),
        'cache_sha256': file_sha(cache), 'source_sha256': sources, 'old_predictions_sha256': old_predictions,
        'seeds': fitting_plan['seeds'], 'inference_particles': 32,
        'comparison': '32-particle-trained model versus8-particle-trained model, both evaluated with32 particles',
        'scope': 'all-seed development-only paired comparison; increased training compute, no primary or fixed-protocol claim'}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    state = {'status': 'waiting_for_all_refits', 'pid': os.getpid(), 'plan_sha256': file_sha(output/'plan.json')}
    def save():
        p = output/'status.partial'; p.write_text(json.dumps(state, indent=2)+'\n'); p.replace(output/'status.json')
    def check():
        if (file_sha(training/'plan.json') != plan['training_plan_sha256'] or file_sha(cache) != plan['cache_sha256']
                or any(file_sha(root/n) != sha for n, sha in sources.items())):
            raise ValueError('declared comparison source or data changed')
    save()
    try:
        while True:
            upstream = json.loads((training/'status.json').read_text())
            if upstream['plan_sha256'] != plan['training_plan_sha256']:
                raise ValueError('upstream fitting declaration changed')
            if upstream['status'] == 'complete': break
            process = Path(f"/proc/{upstream['pid']}")
            if upstream['status'] == 'needs_debug' or not process.exists():
                raise RuntimeError('refitting supervisor exited incomplete')
            command = (process/'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'run_local_particle_training_debug.py' not in command or str(training) not in command:
                raise RuntimeError('refitting PID does not identify the declared upstream')
            if not args.wait: raise RuntimeError('all three refits must finish before comparison')
            time.sleep(20)
        check(); state['status'] = 'running'; save()
        encoder = FrozenSemanticCache(development/'semantic-features')
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if (encoder.manifest_sha256 != fitting_plan['feature_manifest_sha256']
                or encoder.manifest['source_payload_sha256'] != digest):
            raise ValueError('frozen semantic features differ from fitting population')
        spec = importlib.util.spec_from_file_location('particle_refit_comparison_helper', root/'scripts/run_rbr_prediction_gate.py')
        helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper; spec.loader.exec_module(helper)
        batch = helper._tensorize(rows, encoder, torch.device('cpu'), expected_is_public=False)
        labels = batch.outcomes[:, 4:].reshape(-1).numpy()
        population = [{'candidate_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
        populations = {s: population for s in plan['seeds']}
        paired, controls, bindings = {}, {}, {}
        for seed in plan['seeds']:
            check(); name = f'semantic-p32-seed{seed}'
            receipt = json.loads((training/f'{name}.checksums.json').read_text())
            if any(Path(n).name != n or file_sha(training/n) != sha for n, sha in receipt.items()):
                raise ValueError('new model report/checkpoint receipt mismatch')
            report = json.loads((training/f'{name}.json').read_text())
            if (file_sha(training/f'{name}.json') != upstream['runs'][name]['results_sha256']
                    or report['cache_sha256'] != digest or report['config']['particles'] != 32
                    or report['seed'] != seed or report['config']['feature_cache_manifest_sha256'] != encoder.manifest_sha256):
                raise ValueError('completed refit does not match this comparison')
            saved = torch.load(training/f'{name}.pt', map_location='cpu', weights_only=True)
            if any(saved.get(k) != v for k, v in report['config'].items()):
                raise ValueError('checkpoint configuration differs from report')
            model = NeuralBeliefModel(512, 32, 192, difficulty_dim=8)
            model.load_state_dict(saved['model']); model.eval()
            with torch.no_grad():
                values = {arm: helper._predict(model, batch, particles=32, seed=seed+100000, mode=arm)
                          for arm in ('aligned', 'outcome_shuffled')}
            nll = float(-np.log(values['aligned'][np.arange(len(labels)), labels].clip(1e-12, 1)).mean())
            if abs(nll-report['metrics']['aligned']['nll']) > 1e-6:
                raise ValueError('refit aligned predictions do not reproduce the saved report')
            original = sensitivity/f'semantic-seed{seed}-p32.npz'
            if file_sha(original) != old_predictions[str(seed)]:
                raise ValueError('old predictions changed while waiting')
            with np.load(original, allow_pickle=False) as archive:
                if not np.array_equal(archive['labels'], labels):
                    raise ValueError('comparison future labels differ')
                old = archive['aligned'].copy()
            controls[seed] = {**values, 'labels': labels}
            paired[seed] = {'aligned': values['aligned'], 'old_training8_inference32': old, 'labels': labels}
            path = output/f'seed{seed}-predictions.npz'; np.savez_compressed(path, **controls[seed], old_training8_inference32=old)
            bindings[str(seed)] = {'new_model_receipt': receipt, 'old_predictions_sha256': file_sha(original),
                                   'comparison_predictions_sha256': file_sha(path), 'original_gate': report['gate']}
        result = {'schema': plan['schema'], 'plan_sha256': state['plan_sha256'], 'bindings': bindings,
                  'matched_inference_comparison': aggregate_predictions(paired, populations, bootstrap_seed=201701, replicates=10000),
                  'new_association': aggregate_predictions(controls, populations, bootstrap_seed=201701, replicates=10000),
                  'scope': plan['scope']}
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        state.update(status='complete', results_sha256=file_sha(output/'results.json')); save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error)); save(); raise


if __name__ == '__main__':
    main()
