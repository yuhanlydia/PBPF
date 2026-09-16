import copy

import numpy as np
import pytest
import torch

from pbpf.apbpf.repair_materials import build_repair_materials
from pbpf.apbpf.repair_packets import public_posteriors, select_repair_rows
from pbpf.belief.model import NeuralBeliefModel
from test_repair_materials import fixture


def selection(cache):
    rows = [r for r in cache['records'] if r['split'] == 'test']
    return {'seed': 1701, 'cache_sha256': 'cache', 'belief_checkpoint_sha256': 'model',
            'candidate_order': [r['task_id'] for r in rows], 'sources': [rows[0]['source_component_id']],
            'selections': {'particle': [rows[2]['task_id']]}}


def test_repair_packets_keep_selected_source_and_ignore_future_changes(tmp_path):
    cache, public, private = fixture('rbr')
    targets, _, context = build_repair_materials(cache, public, private, domain='rbr')
    report = selection(cache)
    rows = select_repair_rows(cache, targets, context, report, cache_sha256='cache', checkpoint_sha256='model', seed=1701)
    assert len(rows) == 25
    assert rows[-1]['task_id'] == report['selections']['particle'][0]
    model = NeuralBeliefModel(16, 32, 16, difficulty_dim=8)
    checkpoint = tmp_path/'belief.pt'
    torch.save({'model': model.state_dict(), 'seed': 1701, 'encoder': 'frozen_hash_text',
                'apbpf': True, 'difficulty_dim': 8, 'diagnosis_dim': 24, 'latent_dim': 32,
                'feature_dim': 16, 'hidden_dim': 16, 'particles': 8, 'expected_is_public': True}, checkpoint)
    a, weights = public_posteriors(rows, checkpoint, seed=1701)
    assert a.shape == (25, 8, 24)
    np.testing.assert_allclose(np.exp(weights).sum(1), 1, atol=1e-5)
    changed = copy.deepcopy(cache)
    for row in changed['records']:
        for test in row['tests'][4:]:
            test.update(outcome='PASS', actual='PRIVATE_NEW_EXECUTION')
        row['outcomes'][4:] = ['PASS']*6
    targets2, _, context2 = build_repair_materials(changed, public, private, domain='rbr')
    rows2 = select_repair_rows(changed, targets2, context2, report, cache_sha256='cache', checkpoint_sha256='model', seed=1701)
    b, weights2 = public_posteriors(rows2, checkpoint, seed=1701)
    assert rows == rows2
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(weights, weights2)
    rows2[0]['tests'].append(rows2[0]['tests'][0])
    with pytest.raises(ValueError, match='exactly four'):
        public_posteriors(rows2, checkpoint, seed=1701)


@pytest.mark.parametrize('damage', ['seed', 'checkpoint', 'candidate', 'source'])
def test_repair_packet_rejects_selection_identity_drift(damage):
    cache, public, private = fixture('rbr')
    targets, _, context = build_repair_materials(cache, public, private, domain='rbr')
    report = selection(cache)
    if damage == 'seed': report['seed'] = 1702
    elif damage == 'checkpoint': report['belief_checkpoint_sha256'] = 'wrong'
    elif damage == 'candidate': report['selections']['particle'][0] = 'task0/0'
    else: report['sources'][0] = 'wrong'
    with pytest.raises(ValueError):
        select_repair_rows(cache, targets, context, report, cache_sha256='cache', checkpoint_sha256='model', seed=1701)
