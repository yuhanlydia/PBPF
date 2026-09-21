"""Frozen public semantic features and paired training-history views."""

import json

import numpy as np
import torch

from .features import BeliefBatch


def public_contexts(row):
    context = 'Code:\n' + row['candidate'] + '\nTask:\n' + row['task_text']
    return dict(task=[(row['task_text'], None)], candidate=[(row['candidate'], None)],
                tests=[(case['input'], context) for case in row['tests']])


def pool_inputs(hidden, mask):
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not mask.any(-1).all():
        raise ValueError('each example needs at least one input token to pool')
    weights = mask.to(torch.float32)
    return (hidden.float() * weights[..., None]).sum(1) / weights.sum(1, keepdim=True)


def fit_projection(train, dimension, *, seed):
    if type(dimension) is not int or dimension < 1:
        raise ValueError('positive projection dimension required')
    width = train['task'].shape[-1]
    if any(not np.isfinite(v).all() or v.shape[-1] != width for v in train.values()):
        raise ValueError('finite compatible training embeddings required')
    rng = np.random.default_rng(seed)
    result = {'projection': (rng.standard_normal((width, dimension)) / np.sqrt(dimension)).astype('float32')}
    for name in ('task', 'candidate', 'tests'):
        result[name + '_mean'] = train[name].reshape(-1, width).mean(0)
    return result


def apply_projection(values, transform):
    result = {}
    for name, array in values.items():
        projected = (array - transform[name + '_mean']) @ transform['projection']
        projected /= np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-8)
        if not np.isfinite(projected).all():
            raise ValueError('nonfinite projected embeddings')
        result[name] = projected.astype('float32')
    return result


def load_features(path, dataset_sha256, ids, feature_dim):
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata']))
        if metadata.get('schema') != 'pbpf-frozen-features-v1':
            raise ValueError('unsupported feature cache schema')
        if metadata.get('dataset_sha256') != dataset_sha256:
            raise ValueError('feature dataset fingerprint mismatch')
        if metadata.get('ids') != ids:
            raise ValueError('feature candidate identities/order mismatch')
        result = {}
        for split in ('train', 'development'):
            result[split] = {}
            for name in ('task', 'candidate', 'tests'):
                value = cache[f'{split}_{name}']
                rank = 3 if name == 'tests' else 2
                if (value.ndim != rank or value.shape[0] != len(ids[split])
                        or value.shape[-1] != feature_dim or not np.isfinite(value).all()
                        or value.dtype != np.float32):
                    raise ValueError('invalid frozen feature shape/dtype/values')
                result[split][name] = value.copy()
    return result, metadata


def training_view(batch, seed):
    """Uniformly permute training test/outcome pairs, never individual labels."""
    generator = torch.Generator(device=batch.tests.device).manual_seed(seed)
    order = torch.rand(batch.outcomes.shape, device=batch.tests.device, generator=generator).argsort(-1)
    tests = batch.tests.gather(1, order[..., None].expand_as(batch.tests))
    return BeliefBatch(batch.task, batch.candidate, tests,
                      batch.outcomes.gather(1, order), batch.diff)
