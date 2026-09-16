#!/usr/bin/env python3
"""Bounded development-only refits with a prior proposal and no resampling."""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

from diagnose_apbpf_order_sensitivity import DiagnosticBelief
from run_apbpf_particle_diagnostic import validate_development
from pbpf.apbpf.codearc_bank import file_sha


class PriorTrainingBelief(DiagnosticBelief):
    variant = 'prior_no_resample'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-root', type=Path, required=True)
    p.add_argument('--likelihood-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    development, audit, output = (p.resolve() for p in (args.development_root, args.likelihood_audit, args.output))
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'validating', 'pid': os.getpid(), 'runs': {}}
    def save():
        path = output/'status.partial'
        path.write_text(json.dumps(state, indent=2)+'\n')
        path.replace(output/'status.json')
    save()
    try:
        if torch.cuda.is_available():
            raise ValueError('this follow-up is CPU-only; leave GPU work uninterrupted')
        cache, features = development/'development-cache.json', development/'semantic-features'
        payload = json.loads(cache.read_text())
        validate_development(payload)
        prior = json.loads((audit/'status.json').read_text())
        audit_plan = json.loads((audit/'plan.json').read_text())
        if (prior['status'] != 'complete' or prior['results_sha256'] != file_sha(audit/'results.json')
                or prior['plan_sha256'] != file_sha(audit/'plan.json')
                or audit_plan['cache_sha256'] != file_sha(cache)):
            raise ValueError('completed matching likelihood audit required')
        for name, checksum in audit_plan['source_sha256'].items():
            if file_sha(root/name) != checksum:
                raise ValueError('audited implementation changed')
        files = [Path(__file__).resolve(), root/'scripts/diagnose_apbpf_order_sensitivity.py',
                 root/'scripts/run_apbpf_particle_diagnostic.py', root/'scripts/run_rbr_prediction_gate.py',
                 root/'local/sandbox.sh', *sorted((root/'src/pbpf').rglob('*.py'))]
        sources = {str(p.relative_to(root)): file_sha(p) for p in files}
        plan = {'schema': 'apbpf-prior-training-diagnostic-v1', 'seeds': [1701, 1702, 1703],
                'steps': 1000, 'particles': 32, 'feature_dim': 512, 'source_sha256': sources,
                'cache_sha256': file_sha(cache), 'feature_manifest_sha256': file_sha(features/'manifest.json'),
                'likelihood_audit_sha256': prior['results_sha256'],
                'model_class': 'run_apbpf_prior_training_debug.PriorTrainingBelief',
                'inference_variant': 'prior_no_resample',
                'hypothesis': 'training without the first-observation proposal and resampling may direct the association objective into likelihood assignment dependence',
                'design': 'one fixed variant; prior proposal/no resampling for fitting, inner validation and assessment; existing A-PBPF losses, optimizer and32-particle training budget; all four strong baselines refit',
                'selection': '170 fitting/42 inner-validation/400 development-assessment sources; original500 primary sources excluded; every seed retained',
                'loading_requirement': 'checkpoint tensor shapes match the old model but filtering differs; use this plan and mandatory per-seed variant sidecar with PriorTrainingBelief, never the default model',
                'limitations': 'proposal-head parameters are present but unused; compute differs; exploratory development iteration, not a change to the original protocol',
                'scope': 'bounded development-only refitting; no primary tuning or positive-method claim'}
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        state.update(status='running', plan_sha256=file_sha(output/'plan.json'))
        save()
        spec = importlib.util.spec_from_file_location('prior_training_helper', root/'scripts/run_rbr_prediction_gate.py')
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        # Local dependency injection: reuse unchanged fitting/baseline/assessment code.
        helper.NeuralBeliefModel = PriorTrainingBelief
        for seed in plan['seeds']:
            if (any(file_sha(root/n) != h for n, h in sources.items())
                    or file_sha(cache) != plan['cache_sha256']
                    or file_sha(features/'manifest.json') != plan['feature_manifest_sha256']):
                raise ValueError('declared training identity changed')
            name = f'prior-no-resample-seed{seed}'
            state['current'] = name
            state['runs'][name] = {'status': 'running'}
            save()
            with (output/f'{name}.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                report = helper.train_and_evaluate(payload, output/f'{name}.json', feature_dim=512,
                    latent_dim=32, hidden_dim=192, particles=32, steps=1000, batch_size=64,
                    learning_rate=.0003, seed=seed, apbpf=True, difficulty_dim=8,
                    association_weight=1., invariance_weight=.1, association_margin=.03,
                    expected_is_public=False, strong_baselines=True, bootstrap_replicates=10000,
                    feature_cache=features)
            receipt = json.loads((output/f'{name}.checksums.json').read_text())
            if any(Path(n).name != n or file_sha(output/n) != h for n, h in receipt.items()):
                raise ValueError('finished model/report receipt mismatch')
            sidecar = {'plan_sha256': state['plan_sha256'], 'model_class': plan['model_class'],
                       'inference_variant': plan['inference_variant'], 'seed': seed, 'artifact_receipt': receipt}
            (output/f'{name}.variant.json').write_text(json.dumps(sidecar, indent=2)+'\n')
            state['runs'][name].update(status='complete', results_sha256=file_sha(output/f'{name}.json'),
                variant_sha256=file_sha(output/f'{name}.variant.json'),
                gate=report['gate'], association=report['cluster_bootstrap']['outcome_shuffled'])
            save()
        state['status'] = 'complete'
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
