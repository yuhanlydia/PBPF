#!/usr/bin/env python3
"""Replay all fixed generated-development models and pool source-level evidence."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_prediction import aggregate_predictions
from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, validate_rbr_cache

ARMS = ('aligned', 'outcome_shuffled', 'joint_reversed', 'presentation_permuted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream = args.development_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'validating', 'pid': os.getpid()}

    def save():
        temporary = output/'status.partial'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(output/'status.json')

    save()
    try:
        frozen = json.loads((upstream/'status.json').read_text())
        cache = upstream/'development-cache.json'
        payload = json.loads(cache.read_text())
        validate_rbr_cache(payload)
        if (frozen['status'] != 'complete' or frozen['seeds'] != [1701, 1702, 1703]
                or frozen['cache_sha256'] != file_sha(cache)
                or payload['evaluation_role'] != 'development_assessment_only'):
            raise ValueError('completed fixed three-seed development study required')
        for name, checksum in frozen['source_sha256'].items():
            if file_sha(root/name) != checksum:
                raise ValueError('original training source changed')
        rows = [r for r in payload['records'] if r['split'] == 'test']
        assessment = {r['source_component_id'] for r in rows}
        fitting = {r['source_component_id'] for r in payload['records'] if r['split'] != 'test'}
        if (len(rows) != 3200 or len(assessment) != 400 or assessment & fitting
                or assessment != set(payload['assessment_source_ids'])
                or any(t['expected'] for r in payload['records'] for t in r['tests'][4:])):
            raise ValueError('full source-disjoint redacted development population required')
        population = [{'task_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        sources = {str(p.relative_to(root)): file_sha(p) for p in
                   [Path(__file__).resolve(), root/'scripts/run_rbr_prediction_gate.py',
                    *sorted((root/'src/pbpf').rglob('*.py'))]}
        bindings, reports = {}, {}
        for seed in frozen['seeds']:
            name = f'prediction-seed{seed}'
            receipt = json.loads((upstream/f'{name}.checksums.json').read_text())
            if any(Path(n).name != n or file_sha(upstream/n) != h for n, h in receipt.items()):
                raise ValueError('model/report/population receipt mismatch')
            report = json.loads((upstream/f'{name}.json').read_text())
            pop = json.loads((upstream/f'{name}.population.json').read_text())
            if (report['seed'] != seed or report['config']['seed'] != seed
                    or report['cache_sha256'] != digest or pop['population'] != population
                    or report['locked_population'] != population
                    or report['evaluation_role'] != 'development_assessment_only'
                    or report['config'].get('feature_cache_manifest_sha256')
                    or frozen['runs'][name]['status'] != 'complete'):
                raise ValueError('only matching fixed lexical development models are supported')
            bindings[str(seed)] = receipt
            reports[seed] = report
        plan = {'schema': 'apbpf-generated-development-summary-v1', 'seeds': frozen['seeds'],
                'controls': ARMS, 'cache_sha256': file_sha(cache), 'source_sha256': sources,
                'training_status_sha256': file_sha(upstream/'status.json'), 'model_bindings': bindings,
                'scope': 'all three frozen development models; original primary absent from upstream development cache; no training, tuning or sealed-stage claim'}
        (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        state.update(status='replaying', plan_sha256=file_sha(output/'plan.json'), runs={})
        save()
        spec = importlib.util.spec_from_file_location('generated_development_replay', root/'scripts/run_rbr_prediction_gate.py')
        helper = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = helper
        spec.loader.exec_module(helper)
        values, populations = {}, {}
        for seed in frozen['seeds']:
            if (any(file_sha(root/n) != h for n, h in sources.items())
                    or file_sha(cache) != plan['cache_sha256']):
                raise ValueError('declared replay identity changed')
            receipt = bindings[str(seed)]
            if any(file_sha(upstream/n) != h for n, h in receipt.items()):
                raise ValueError('checkpoint receipt changed during replay')
            state['current_seed'] = seed
            save()
            config = reports[seed]['config']
            saved = torch.load(upstream/f'prediction-seed{seed}.pt', map_location='cpu', weights_only=True)
            if any(saved.get(k) != v for k, v in config.items()):
                raise ValueError('saved checkpoint configuration differs')
            model = NeuralBeliefModel(config['feature_dim'], config['latent_dim'], config['hidden_dim'],
                                     difficulty_dim=config['difficulty_dim'])
            model.load_state_dict(saved['model'])
            model.eval()
            batch = helper._tensorize(rows, FrozenTextEncoder(config['feature_dim']), torch.device('cpu'),
                                      expected_is_public=config['expected_is_public'])
            labels = batch.outcomes[:, 4:].reshape(-1).numpy()
            with torch.no_grad():
                predictions = {arm: helper._predict(model, batch, particles=config['particles'],
                    seed=seed+100000, mode=arm, problem_ids=[r['source_component_id'] for r in rows]) for arm in ARMS}
            reproduced = {}
            for arm, prediction in predictions.items():
                nll = float(-np.log(prediction[np.arange(len(labels)), labels].clip(1e-12, 1)).mean())
                reproduced[arm] = abs(nll-reports[seed]['metrics'][arm]['nll'])
                if reproduced[arm] > 1e-6:
                    raise ValueError(f'{seed}/{arm} NLL does not reproduce original report')
            values[seed] = {**predictions, 'labels': labels}
            populations[seed] = [{'candidate_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
            path = output/f'seed{seed}-predictions.npz'
            np.savez_compressed(path, **values[seed])
            state['runs'][str(seed)] = {'predictions_sha256': file_sha(path), 'nll_reproduction_errors': reproduced}
            save()
        report = aggregate_predictions(values, populations, bootstrap_seed=201701, replicates=10000)
        result = {'schema': plan['schema'], 'plan_sha256': state['plan_sha256'],
                  'prediction_bindings': state['runs'], 'aggregate': report,
                  'original_seed_gates': {str(s): reports[s]['gate'] for s in frozen['seeds']}, 'scope': plan['scope']}
        (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
        state.update(status='complete', results_sha256=file_sha(output/'results.json'))
        save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
