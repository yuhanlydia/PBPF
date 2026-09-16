import copy
import json

import numpy as np
import pytest

from pbpf.apbpf.stage_prediction import aggregate_predictions, BASELINES, DOMAINS
from pbpf.apbpf.config import ROOT, digest, resolve_config
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_cache import build_stage_cache
from pbpf.apbpf.stages import _validate_gate_decision
from test_stage_cache import population
from test_stage_training import module
from test_hard_bank_stage_workers import fixture_stage, artifact_inventory, complete, invoke


def probs(n, p):
    return np.tile([p]+[(1-p)/4]*4, (n, 1))


def test_seed_loss_aggregation_keeps_source_clusters_and_does_not_ensemble_probabilities():
    population = [{'candidate_id': 'a', 'source_component_id': 's1'},
                  {'candidate_id': 'b', 'source_component_id': 's2'}]
    values = {seed: {'labels': np.zeros(12, dtype=int), 'aligned': probs(12, p), 'other': probs(12, .5)}
              for seed, p in zip([1701,1702,1703], [.8,.4,.1])}
    report = aggregate_predictions(values, {s: population for s in values}, bootstrap_seed=1, replicates=100)
    expected = np.mean(np.log(np.array([.8,.4,.1])/.5))
    assert report['comparisons']['other']['mean_nll_gap'] == pytest.approx(expected)
    assert report['comparisons']['other']['clusters'] == 2
    assert expected != pytest.approx(np.log(np.mean([.8,.4,.1])/.5))
    changed = copy.deepcopy(values); changed[1703]['labels'][0] = 1
    with pytest.raises(ValueError, match='labels differ'):
        aggregate_predictions(changed, {s: population for s in values}, bootstrap_seed=1)
    with pytest.raises(ValueError, match='all three'):
        aggregate_predictions({1701: values[1701]}, {1701: population}, bootstrap_seed=1)


@pytest.mark.parametrize('worse_domain', [False, True])
def test_actual_fairness_worker_rejects_a_failing_domain_and_keeps_all_seeds(tmp_path, worse_domain):
    config = resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory').config
    run = tmp_path/digest(config); inputs = {}
    for kind in ('baselines', 'belief'):
        attempt, value = fixture_stage(run, 'train_'+kind, digest(config)); entries = []
        for domain, dataset in DOMAINS.items():
            for seed in config['protocol']['seeds']:
                directory = f'{domain}-seed{seed}'; output = attempt/'outputs'/directory; output.mkdir()
                checkpoint_arms = ['belief'] if kind == 'belief' else [k for k in BASELINES if k != 'tuned_dirichlet']
                arms = {}
                for arm in checkpoint_arms:
                    p = output/f'{arm}.pt'; p.write_bytes(b'contract fixture checkpoint; not a real model')
                    arms[arm] = {'checkpoint': p.name, 'checkpoint_sha256': file_sha(p)}
                if kind == 'baselines':
                    arms['tuned_dirichlet'] = {'alpha': .1}
                    (output/'tuned_dirichlet.json').write_text(json.dumps(arms['tuned_dirichlet']))
                primary = [{'candidate_id': f'{domain}/{i}/{j}', 'source_component_id': f'{domain}/{i}'}
                           for i in range(500) for j in range(8)]
                settings = {'kind': kind, 'seed': seed, 'protocol': config['protocol'], 'steps': 1000,
                    'batch_size': 64, 'learning_rate': .0003, 'feature_dim': 256, 'hidden_dim': 192,
                    'latent_dim': 32, 'expected_is_public': domain == 'rbr', 'encoder': 'frozen_hash_text',
                    'checkpoint_selection_split': 'development', 'primary_used_for_training_or_selection': False}
                counts = {'train': 1696 if domain == 'codearc' else 2568, 'development': 3200, 'test': 4000}
                report = {'schema': 'apbpf-stage-training-v1', 'dataset': dataset, 'config': settings,
                          'primary_population': primary, 'arms': arms, 'counts': counts}
                (output/'training.json').write_text(json.dumps(report))
                probability = .4 if worse_domain and domain == 'codearc' else .9
                predictions = {'aligned': probs(24000, probability)} if kind == 'belief' else {k: probs(24000, .5) for k in BASELINES}
                np.savez_compressed(output/'primary-predictions.npz', labels=np.zeros(24000, dtype=int), **predictions)
                entries.append({'domain': domain, 'seed': seed, 'directory': directory, 'counts': counts, 'cache_sha256': 'a'*64})
        (attempt/'outputs/training-index.json').write_text(json.dumps({'kind': kind, 'runs': entries}))
        value['artifacts'] = artifact_inventory(attempt); inputs['train_'+kind] = complete(attempt, value)
    _, request, result = invoke(run, 'baseline_fairness_gate', config, inputs, 'run_apbpf_fairness_worker.py')
    assert result['gate']['passed'] is (not worse_domain)
    assert set(result['gate']['metrics']['baselines']) == set(BASELINES.values())
    for report in result['gate']['metrics']['domains'].values():
        assert report['seeds'] == [1701,1702,1703]
        assert report['source_components'] == 500
    _validate_gate_decision(result, {**request, 'backend': 'real'}, config)


def test_checkpoint_replay_reproduces_aligned_and_rejects_population_drift(tmp_path):
    worker = module('association_replay_under_test', 'run_apbpf_association_worker.py')
    trainer = module('association_training_fixture', 'run_apbpf_train_worker.py')
    helper = module('association_helper_fixture', 'run_rbr_prediction_gate.py')
    payload = build_stage_cache(*population(), domain='rbr')
    protocol = {'difficulty_dim':8, 'diagnosis_dim':24, 'particles':8, 'lambda_assoc':1., 'lambda_inv':.1, 'association_margin':.03}
    output = tmp_path/'train'
    report = trainer.train_models(payload, output, kind='belief', seed=1701, steps=1, batch_size=8,
        learning_rate=.0003, feature_dim=16, hidden_dim=16, protocol=protocol, helper=helper)
    with np.load(output/'primary-predictions.npz') as archive:
        predictions = {k:archive[k] for k in archive.files}
    cell = {'report': report, 'predictions': predictions, 'checkpoints': {'belief': output/'belief.pt'}}
    values, pop, strata = worker.replay(payload, cell, helper=helper)
    assert set(values) == set(worker.CONTROLS) | {'labels'}
    assert np.array_equal(values['aligned'], predictions['aligned'])
    assert strata['constant_visible'].all()
    assert np.array_equal(values['outcome_shuffled'], values['aligned'])
    changed = copy.deepcopy(payload); changed['records'][-1]['source_component_id'] = 'different'
    with pytest.raises(ValueError, match='population'):
        worker.replay(changed, cell, helper=helper)


@pytest.mark.parametrize('stage', ['association_gate', 'pair_invariance_gate'])
@pytest.mark.parametrize('bad', [False, True])
def test_prediction_gate_decisions_agree_with_locked_runner_rules(stage, bad):
    worker = module('prediction_gate_under_test', 'run_apbpf_prediction_gate_worker.py')
    config = resolve_config(ROOT/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory').config
    report = {'seeds':[1701,1702,1703], 'source_components':500, 'primary_candidates':4000,
              'comparisons': {'outcome_shuffled':{'mean_nll_gap':.04, 'ci95':[.01,.07]},
                              'joint_reversed':{'mean_nll_gap':.001}, 'presentation_permuted':{'mean_nll_gap':.001}}}
    evidence = {'schema':'apbpf-stage-association-v1', 'population':'full_locked_population',
                'bootstrap_draws':10000, 'domains':{d:copy.deepcopy(report) for d in DOMAINS.values()}}
    if bad:
        if stage == 'association_gate': evidence['domains']['codearc_replay']['comparisons']['outcome_shuffled']['ci95'][0] = -.01
        else: evidence['domains']['codearc_replay']['comparisons']['presentation_permuted']['mean_nll_gap'] = .02
    gate = worker.decision(evidence, config, stage)
    assert gate['passed'] is (not bad)
    _validate_gate_decision({'gate':gate}, {'backend':'real','stage':stage}, config)
