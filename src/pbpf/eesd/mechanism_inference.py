"""Conditional, paired source inference for the adopted mechanism amendment.

This module accepts aligned arrays, not cache files. Callers must validate seals
and identity reconstruction. Fixed seeds are averaged, never independently
resampled. Bootstrap p values are approximate, not exact randomization p values.
"""
from __future__ import annotations

import hashlib
import numpy as np

from pbpf.runner.stats import holm_locked

METRICS = ('nll', 'brier', 'accuracy', 'ece')
REPLICATES = 10_000


def stable_seed(domain: str, model: str, population: str = 'all') -> int:
    if any(not isinstance(x, str) or not x or '|' in x for x in (domain, model, population)):
        raise ValueError('nonempty canonical RNG keys without | required')
    return int.from_bytes(hashlib.sha256(
        f'314159|{domain}|{model}|{population}'.encode('utf-8')).digest()[:8], 'big')


def unsupported(reason: str) -> dict:
    if not isinstance(reason, str) or not reason:
        raise ValueError('unsupported requires an explicit reason')
    return {'status': 'unsupported', 'reason': reason}


class PairedMechanismData:
    """One fixed contrast on an already aligned source × seed population.

    ``comparators`` contains one arm or several fixed permutation realizations.
    Their metrics (not probabilities) are averaged equally. Query IDs must be
    unique within a source/seed, with identities shared by all probability arrays.
    Query counts may vary; each source and each seed receive equal weight.
    """
    def __init__(self, *, labels, proposed, comparators, sources, seeds, query_ids,
                 expected_seeds=(1701, 1702, 1703)):
        y = np.asarray(labels)
        n = len(y) if y.ndim == 1 else 0
        if not n or y.dtype.kind not in 'iu' or np.any(y < 0):
            raise ValueError('nonempty integer labels required')
        if not comparators or not isinstance(comparators, dict):
            raise ValueError('at least one named comparator required')
        if any(not isinstance(k, str) or not k for k in comparators):
            raise ValueError('nonempty comparator names required')
        if len(sources) != n or len(seeds) != n or len(query_ids) != n:
            raise ValueError('identity and prediction lengths differ')
        if any(not isinstance(x, str) or not x for x in sources):
            raise ValueError('nonempty string source IDs required')
        if any(not isinstance(x, str) or not x for x in query_ids):
            raise ValueError('nonempty string query IDs required')
        expected_seeds = tuple(expected_seeds)
        if (not expected_seeds or len(set(expected_seeds)) != len(expected_seeds)
                or any(type(x) is not int for x in expected_seeds)
                or any(type(x) is not int for x in seeds)
                or set(seeds) != set(expected_seeds)):
            raise ValueError('exact expected seed population required')
        keys = list(zip(sources, seeds, query_ids))
        if len(set(keys)) != n:
            raise ValueError('duplicate source/seed/query identity')
        self.sources = tuple(sorted(set(sources)))
        self.seeds = tuple(sorted(expected_seeds))
        self.comparator_names = tuple(sorted(comparators))
        source_arr, seed_arr = np.asarray(sources), np.asarray(seeds)
        groups = [[np.flatnonzero((source_arr == source) & (seed_arr == seed))
                   for source in self.sources] for seed in self.seeds]
        if any(not len(indices) for group in groups for indices in group):
            raise ValueError('every source must have all expected seeds')
        arms = [proposed] + [comparators[name] for name in self.comparator_names]
        classes = None
        self.identical_predictions = True
        proposed_array = np.asarray(proposed, dtype=float)
        # Arm × seed × source × additive metric/bin statistic.
        self._additive = np.empty((len(arms), len(self.seeds), len(self.sources), 3))
        self._bin_residual = np.empty((len(arms), len(self.seeds), len(self.sources), 10))
        for a, probabilities in enumerate(arms):
            p = np.asarray(probabilities, dtype=float)
            if (p.ndim != 2 or p.shape[0] != n or p.shape[1] < 2
                    or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
                    or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10)
                    or np.any(y >= p.shape[1])):
                raise ValueError('invalid categorical probabilities')
            if classes is not None and classes != p.shape[1]:
                raise ValueError('paired arms require the same class space')
            classes = p.shape[1]
            self.identical_predictions &= np.array_equal(p, proposed_array)
            confidence = p.max(axis=1)
            correct = (p.argmax(axis=1) == y).astype(float)
            nll = -np.log(p[np.arange(n), y].clip(1e-12, 1))
            brier = np.square(p - np.eye(p.shape[1])[y]).sum(axis=1)
            values = np.column_stack((nll, brier, correct))
            # These sufficient statistics exactly recompute weighted bin ECE:
            # bin mass × (bin accuracy - confidence) is sum w*(correct-conf).
            # Absolute values are applied ONLY after source resampling/averaging.
            edges = np.linspace(0, 1, 11)
            bins = np.searchsorted(edges, confidence, side='right') - 1
            bins = np.minimum(bins, 9)
            for s, group in enumerate(groups):
                for c, indices in enumerate(group):
                    self._additive[a, s, c] = values[indices].mean(axis=0)
                    self._bin_residual[a, s, c] = np.bincount(
                        bins[indices], weights=correct[indices] - confidence[indices],
                        minlength=10) / len(indices)

    def _metrics_batch(self, indices):
        # indices: draw × source; output draw × arm × metric.
        additive = self._additive[:, :, indices, :].mean(axis=3).mean(axis=1)
        ece = np.abs(self._bin_residual[:, :, indices, :].mean(axis=3)).sum(axis=-1).mean(axis=1)
        return np.concatenate((additive, ece[..., None]), axis=-1).transpose(1, 0, 2)

    def metrics(self, source_indices=None) -> dict:
        indices = np.arange(len(self.sources)) if source_indices is None else np.asarray(source_indices)
        if (indices.ndim != 1 or len(indices) != len(self.sources)
                or indices.dtype.kind not in 'iu' or np.any(indices < 0)
                or np.any(indices >= len(self.sources))):
            raise ValueError('one whole-source draw of the original population size required')
        arms = self._metrics_batch(indices[None, :])[0]
        def record(values):
            return dict(zip(METRICS, map(float, values)))
        return {'proposed': record(arms[0]), 'comparator': record(arms[1:].mean(axis=0)),
                'comparators': {name: record(arms[i + 1]) for i, name in enumerate(self.comparator_names)}}


def bootstrap_gain(data: PairedMechanismData, *, domain, model, population='all') -> dict:
    """10k whole-source paired draws; pointwise intervals and centered p+1."""
    if len(data.sources) < 2:
        return unsupported('fewer than two independent source clusters')
    seed = stable_seed(domain, model, population)
    point = data.metrics()
    orientation = np.asarray([1., 1., -1., 1.])
    observed = np.asarray([[point['comparators'][name][m] - point['proposed'][m]
                            for m in METRICS] for name in data.comparator_names]).mean(axis=0) * orientation
    rng = np.random.Generator(np.random.PCG64(seed))
    gains = np.empty((REPLICATES, len(METRICS)))
    for start in range(0, REPLICATES, 128):
        stop = min(start + 128, REPLICATES)
        indices = rng.integers(0, len(data.sources), size=(stop - start, len(data.sources)))
        arms = data._metrics_batch(indices)
        gains[start:stop] = (arms[:, 1:] - arms[:, :1]).mean(axis=1) * orientation
    # Exact equality of every prediction is an exact null, independent of any
    # floating-point reduction order in metric or permutation averaging.
    if data.identical_predictions:
        observed.fill(0.)
        gains.fill(0.)
    results = {}
    for i, metric in enumerate(METRICS):
        p = (1 + np.count_nonzero(np.abs(gains[:, i] - observed[i]) >= abs(observed[i]))) / (REPLICATES + 1)
        results[metric] = {'status': 'supported', 'gain': float(observed[i]),
                           'proposed': point['proposed'][metric], 'comparator': point['comparator'][metric],
                           'ci95': np.quantile(gains[:, i], [.025, .975]).tolist(), 'p_raw': float(p)}
    return {'status': 'supported', 'metrics': results, 'replicates': REPLICATES,
            'sources': len(data.sources), 'seeds': list(data.seeds), 'rng_seed': seed,
            'numpy_version': np.__version__, 'interval_scope': 'pointwise percentile, not simultaneous',
            'p_method': 'approximate centered two-sided source bootstrap with plus-one correction',
            'inference_scope': 'conditional on fixed generation seeds and sealed selected parameters'}


def family_report(results: dict, *, family) -> dict:
    """Require the full inventory; untestable slots block ALL family decisions."""
    family = list(family)
    if not family or len(set(family)) != len(family) or set(results) != set(family):
        raise ValueError('complete prespecified family required, without duplicate IDs')
    missing, values = {}, {}
    for key in family:
        result = results[key]
        if result.get('status') == 'unsupported':
            reason = result.get('reason')
            if not isinstance(reason, str) or not reason:
                raise ValueError('unsupported hypothesis requires a reason')
            missing[key] = reason
        elif result.get('status') == 'supported':
            value = result.get('p_raw')
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('valid raw p required')
            if value < 1 / (REPLICATES + 1):
                raise ValueError('raw p below locked 10000-draw plus-one resolution')
            values[key] = value
        else:
            raise ValueError('explicit supported/unsupported hypothesis status required')
    adjusted = None if missing else holm_locked(values, family=family)
    return {'complete': not missing, 'family': family, 'family_size': len(family),
            'unsupported': missing, 'raw_p': values, 'adjusted_p': adjusted,
            'reject': None if missing else {key: adjusted[key] <= .05 for key in family},
            'alpha': .05, 'method': 'Holm on approximate p values; not exact FWER control'}
