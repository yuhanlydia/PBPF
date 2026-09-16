#!/usr/bin/env python3
"""Replay association and pair-preserving controls on complete primary banks."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_prediction import DOMAINS, aggregate_predictions, load_training
from pbpf.apbpf.worker_io import WorkerIO
from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, OUTCOMES, association_strata, validate_rbr_cache

CONTROLS = ('baseline', 'aligned', 'outcome_shuffled', 'joint_reversed',
            'presentation_permuted', 'orderless', 'semantics_masked', 'wrong_candidate', 'random_latent')


def replay(payload, training, *, helper):
    validate_rbr_cache(payload)
    if payload.get('evaluation_role') != 'exploratory_locked_primary_assessment':
        raise ValueError('association requires the original locked primary assessment cache')
    if any(t['expected'] for row in payload['records'] for t in row['tests'][4:]):
        raise ValueError('future expected answers must be absent from predictor features')
    rows = [r for r in payload['records'] if r['split'] == 'test']
    population = [{'candidate_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
    if (population != training['report']['primary_population']
            or any(r.get('original_split') != 'primary' for r in rows)):
        raise ValueError('association population differs from training primary inventory')
    config = training['report']['config']
    saved = torch.load(training['checkpoints']['belief'], map_location='cpu', weights_only=True)
    if any(saved.get(key) != value for key, value in config.items()):
        raise ValueError('checkpoint configuration differs from its training report')
    model = NeuralBeliefModel(config['feature_dim'], config['latent_dim'], config['hidden_dim'],
                              difficulty_dim=config['difficulty_dim'])
    model.load_state_dict(saved['model']); model.eval()
    batch = helper._tensorize(rows, FrozenTextEncoder(config['feature_dim']), torch.device('cpu'),
                              expected_is_public=config['expected_is_public'])
    labels = batch.outcomes[:, 4:].reshape(-1).numpy()
    if not np.array_equal(labels, training['predictions']['labels']):
        raise ValueError('primary labels differ from original training-stage prediction evidence')
    values = {mode: helper._predict(model, batch, particles=config['particles'], seed=config['seed'] + 100000,
                                   mode=mode, problem_ids=[r['source_component_id'] for r in rows])
              for mode in CONTROLS}
    if not np.allclose(values['aligned'], training['predictions']['aligned'], rtol=1e-6, atol=1e-7):
        raise ValueError('restored checkpoint does not reproduce the original aligned predictions')
    values['labels'] = labels
    return values, population, association_strata(batch.outcomes.numpy())


def main():
    root = Path(__file__).resolve().parents[1]
    sources = [str(p.relative_to(root)) for p in sorted((root/'src/pbpf').rglob('*.py'))]
    io = WorkerIO('association', sources + ['scripts/run_apbpf_association_worker.py', 'scripts/run_rbr_prediction_gate.py'])
    spec = importlib.util.spec_from_file_location('apbpf_association_prediction_helper', root/'scripts/run_rbr_prediction_gate.py')
    helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper; spec.loader.exec_module(helper)
    training = load_training(io, 'belief')
    execution = json.loads(io.artifact('execution_cache', 'execution-index.json').read_text())
    results = {}
    for domain, dataset in DOMAINS.items():
        entry = execution['training_caches'][domain]['full']
        path = io.artifact('execution_cache', entry['path']); checksum = file_sha(path)
        if checksum != entry['sha256']:
            raise ValueError('cache differs from declared execution index')
        payload = json.loads(path.read_text())
        values, populations, strata, bindings = {}, {}, {}, []
        for seed in io.config['protocol']['seeds']:
            io.check_sources()
            cell = training[domain, seed]
            if cell['entry']['cache_sha256'] != checksum:
                raise ValueError('belief checkpoint trained against another execution cache')
            values[seed], populations[seed], masks = replay(payload, cell, helper=helper)
            if strata and any(not np.array_equal(strata[k], mask) for k, mask in masks.items()):
                raise ValueError('outcome strata changed across fixed seeds')
            strata = masks
            output = io.outputs/f'{domain}-seed{seed}-controls.npz'
            np.savez_compressed(output, **values[seed])
            bindings.append({'seed': seed, 'controls_sha256': file_sha(output), 'controls': output.name,
                             'training_report_sha256': cell['training_report_sha256'],
                             'belief_checkpoint_sha256': file_sha(cell['checkpoints']['belief'])})
        report = aggregate_predictions(values, populations, bootstrap_seed=201701,
                                        replicates=io.config['protocol']['bootstrap_draws'])
        report.update(cache_sha256=checksum, primary_population=populations[1701],
                      prediction_bindings=bindings, strata={})
        for name, candidate_mask in strata.items():
            if name == 'full':
                continue
            if not candidate_mask.any():
                report['strata'][name] = {'candidates': 0, 'status': 'empty_prespecified_stratum'}
                continue
            mask = np.repeat(candidate_mask, 6)
            subset = {seed: {arm: array[mask] for arm, array in items.items()} for seed, items in values.items()}
            subset_pop = {seed: [row for row, keep in zip(population, candidate_mask, strict=True) if keep]
                          for seed, population in populations.items()}
            report['strata'][name] = aggregate_predictions(subset, subset_pop, bootstrap_seed=201701,
                replicates=io.config['protocol']['bootstrap_draws'])
            report['strata'][name]['status'] = 'descriptive_only_not_gate_population'
        comparisons = report['comparisons']
        report['negative_controls_worse_than_aligned'] = all(comparisons[k]['mean_nll_gap'] > 0
            for k in ('semantics_masked', 'wrong_candidate', 'random_latent'))
        results[dataset] = report
    evidence = {'schema': 'apbpf-stage-association-v1', 'population': 'full_locked_population',
                'domains': results, 'bootstrap_draws': io.config['protocol']['bootstrap_draws'],
                'scope': 'exploratory full-population controls; no favorable stratum selection'}
    (io.outputs/'association.json').write_text(json.dumps(evidence, indent=2)+'\n')
    io.finish({'actual_counterfactual_replay': True, 'domains': list(results),
               'controls': list(CONTROLS), 'full_primary_population': True,
               'scope': 'fixed-checkpoint association and pair-preserving controls; gates downstream'})


if __name__ == '__main__':
    main()
