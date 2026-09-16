import importlib
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from pbpf.belief.features import BeliefBatch


def api():
    return importlib.import_module('pbpf.belief.semantic_features')


def test_public_context_never_reads_execution_or_expected_output():
    row = dict(task_text='task', candidate='code', tests=[dict(input='input', expected='SECRET_A',
        actual='SECRET_B', stderr='SECRET_C', outcome='SECRET_D')], outcomes=['SECRET_E'])
    pairs = api().public_contexts(row)
    assert pairs['task'] == [('task', None)]
    assert pairs['candidate'] == [('code', None)]
    assert pairs['tests'] == [('input', 'Code:\ncode\nTask:\ntask')]
    assert 'SECRET' not in str(pairs)


def test_pooling_excludes_context_and_padding():
    hidden = torch.tensor([[[99.,99.], [1.,3.], [3.,5.], [100.,100.], [1000.,1000.]]])
    mask = torch.tensor([[0,1,1,0,0]], dtype=torch.bool)
    torch.testing.assert_close(api().pool_inputs(hidden, mask), torch.tensor([[2.,4.]]))
    with pytest.raises(ValueError):
        api().pool_inputs(hidden, torch.zeros_like(mask))


def test_projection_fits_only_training_rows():
    train = {'task':np.arange(24,dtype='float32').reshape(4,6),
             'candidate':np.ones((4,6),dtype='float32'), 'tests':np.arange(48,dtype='float32').reshape(4,2,6)}
    dev = {k:np.full_like(v,10000.) for k,v in train.items()}
    transform = api().fit_projection(train, 3, seed=17)
    np.testing.assert_allclose(transform['task_mean'], train['task'].mean(0))
    before = {k:v.copy() for k,v in transform.items()}
    out = api().apply_projection(dev, transform)
    for k in transform:
        np.testing.assert_array_equal(transform[k], before[k])
    assert out['tests'].shape == (4,2,3)
    np.testing.assert_allclose(np.linalg.norm(out['tests'],axis=-1), 1., atol=1e-6)


def test_training_view_keeps_pairs_and_is_reproducible():
    tests = torch.arange(20).reshape(2,10,1).float()
    labels = tests[:,:,0].long() % 5
    b = BeliefBatch(torch.zeros(2,1),torch.zeros(2,1),tests,labels)
    a = api().training_view(b,17)
    again = api().training_view(b,17)
    assert torch.equal(a.tests,again.tests)
    assert torch.equal(a.outcomes,a.tests[:,:,0].long()%5)
    assert not torch.equal(a.tests,b.tests)
    torch.testing.assert_close(a.tests.sort(dim=1).values,b.tests)
    assert torch.equal(b.tests,tests)


def test_feature_cache_rejects_wrong_candidate_order_and_dataset(tmp_path):
    p=tmp_path/'features.npz'
    meta=dict(schema='pbpf-frozen-features-v1',dataset_sha256='a'*64,
              ids={'train':['a','b'],'development':['c']})
    arrays={f'{s}_{k}':np.ones((n,2,3) if k=='tests' else (n,3),dtype='float32')
            for s,n in [('train',2),('development',1)] for k in ('task','candidate','tests')}
    np.savez(p, metadata=json.dumps(meta), **arrays)
    with pytest.raises(ValueError,match='identit'):
        api().load_features(p,'a'*64,{'train':['b','a'],'development':['c']},3)
    with pytest.raises(ValueError,match='dataset'):
        api().load_features(p,'b'*64,meta['ids'],3)
    loaded, _=api().load_features(p,'a'*64,meta['ids'],3)
    assert loaded['train']['tests'].shape==(2,2,3)
