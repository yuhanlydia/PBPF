"""Formal task-macro statistics, separate from the legacy hierarchical APIs."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from pbpf.statistics import BootstrapInterval
from .shard import atomic_write, canonical_bytes, digest


@dataclass(frozen=True)
class PlanLock:
    """External trusted expected identity, never loaded from the plan file."""
    seed: int
    upstream_identity: str
    task_keys: tuple
    source_clusters: tuple
    strata: tuple
    expected_digest: str

    def __post_init__(self):
        for name in ("task_keys", "source_clusters", "strata"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (type(self.seed) is not int or self.seed < 0 or not self.task_keys
                or len(self.task_keys) != len(self.source_clusters) or len(self.task_keys) != len(self.strata)
                or any(type(key) is not str or not key for values in (self.task_keys, self.source_clusters, self.strata) for key in values)
                or any(type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                       for value in (self.upstream_identity, self.expected_digest))):
            raise ValueError("complete external bootstrap PlanLock required")

    @classmethod
    def from_plan(cls, plan):
        """Use only a plan freshly computed from trusted fingerprint inputs."""
        return cls(plan.seed, plan.upstream_identity, plan.task_keys, plan.source_clusters, plan.strata, plan.digest)


@dataclass(frozen=True)
class ClusterPlan:
    task_keys: tuple
    source_clusters: tuple
    strata: tuple
    draws: tuple
    seed: int
    upstream_identity: str

    def __post_init__(self):
        if (type(self.seed) is not int or self.seed < 0 or not isinstance(self.upstream_identity, str)
                or len(self.upstream_identity) != 64 or any(c not in "0123456789abcdef" for c in self.upstream_identity)):
            raise ValueError("exact seed and explicit immutable upstream identity required")
        for name in ("task_keys", "source_clusters", "strata"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "draws", tuple(tuple(draw) for draw in self.draws))
        if (not self.task_keys or len(set(self.task_keys)) != len(self.task_keys)
                or len(self.task_keys) != len(self.source_clusters) or len(self.task_keys) != len(self.strata)):
            raise ValueError("unique locked task inventory and aligned cluster/strata vectors required")
        cluster_strata = {}
        for cluster, stratum in zip(self.source_clusters, self.strata):
            if cluster in cluster_strata and cluster_strata[cluster] != stratum:
                raise ValueError("source cluster cannot cross locked strata")
            cluster_strata[cluster] = stratum
        expected = {s: list(cluster_strata.values()).count(s) for s in set(self.strata)}
        if len(self.draws) != 10000:
            raise ValueError("formal plan requires exactly 10000 common replicates")
        for draw in self.draws:
            if any(c not in cluster_strata for c in draw) or {s: sum(cluster_strata[c] == s for c in draw) for s in expected} != expected:
                raise ValueError("bootstrap plan violates intact stratified source clusters")

    @classmethod
    def create(cls, *, task_keys, source_clusters, strata, seed, upstream_identity):
        if len(task_keys) != len(source_clusters) or len(task_keys) != len(strata):
            raise ValueError("unaligned plan inventory")
        groups = {}
        for cluster, stratum in zip(source_clusters, strata):
            groups.setdefault(stratum, set()).add(cluster)
        rng = np.random.default_rng(seed)
        draws = [[] for _ in range(10000)]
        for stratum in sorted(groups):
            clusters = sorted(groups[stratum])
            choices = rng.integers(0, len(clusters), size=(10000, len(clusters)))
            for draw, indices in zip(draws, choices):
                draw.extend(clusters[index] for index in indices)
        return cls(tuple(task_keys), tuple(source_clusters), tuple(strata), tuple(map(tuple, draws)), seed, upstream_identity)

    def to_dict(self):
        return {"schema": "pbpf-common-cluster-plan-v1", "task_keys": self.task_keys,
            "source_clusters": self.source_clusters, "strata": self.strata, "draws": self.draws, "seed": self.seed,
            "upstream_identity": self.upstream_identity}

    @property
    def digest(self):
        return digest(self.to_dict())

    def task_indices(self, draw):
        return [i for cluster in draw for i, source in enumerate(self.source_clusters) if source == cluster]

    def save(self, path):
        payload = canonical_bytes(dict(self.to_dict(), digest=self.digest))
        path = Path(path)
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("immutable common bootstrap plan conflict")
        else:
            atomic_write(path, payload, create_once=True)

    @classmethod
    def load(cls, path, *, lock):
        if type(lock) is not PlanLock:
            raise ValueError("external immutable PlanLock required")
        row = json.loads(Path(path).read_bytes())
        recorded_digest = row.pop("digest", None)
        if recorded_digest != digest(row):
            raise ValueError("bootstrap plan digest mismatch")
        if row.pop("schema") != "pbpf-common-cluster-plan-v1":
            raise ValueError("bootstrap plan schema mismatch")
        result = cls(**row)
        if PlanLock.from_plan(result) != lock:
            raise ValueError("bootstrap plan differs from external immutable lock")
        regenerated = cls.create(task_keys=lock.task_keys, source_clusters=lock.source_clusters,
            strata=lock.strata, seed=lock.seed, upstream_identity=lock.upstream_identity)
        if result.draws != regenerated.draws:
            raise ValueError("same-seed bootstrap draws were replaced")
        return result


def formal_interval(treatment, control, plan, *, statistic, one_sided=False):
    left, right = np.asarray(treatment, dtype=float), np.asarray(control, dtype=float)
    if (left.shape != (len(plan.task_keys),) or left.shape != right.shape
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("finite paired task-macro values must match common plan")
    if statistic not in {"difference", "log_nll_ratio"}:
        raise ValueError("unsupported formal statistic")
    if statistic == "log_nll_ratio" and (np.any(left < 0) or np.any(right < 0)):
        raise ValueError("nonnegative task NLL required for log ratio")
    def measure(a, b):
        if statistic == "difference":
            return float((a-b).mean())
        numerator, denominator = float(b.mean()), float(a.mean())
        if numerator == denominator == 0:
            raise ValueError("both aggregate means zero: log ratio undefined")
        if denominator == 0:
            return math.inf
        return -math.inf if numerator == 0 else math.log(numerator/denominator)
    samples = np.asarray([measure(left[indices], right[indices])
        for indices in (plan.task_indices(draw) for draw in plan.draws)])
    # Discrete order-statistic quantiles do not interpolate inf-inf into NaN.
    lower, upper = np.quantile(samples, [.05, .95] if one_sided else [.025, .975], method="inverted_cdf")
    return BootstrapInterval(measure(left, right), float(lower), float(upper), len(samples))


def reduce_metrics(records, *, expected_keys=None):
    """Tests -> candidate -> task -> the three fixed seeds, with no sample pooling."""
    records = list(records)
    groups, seen, metadata = {}, set(), {}
    for row in records:
        key = (row["task"], row["seed"], row["arm"], row["candidate"], row["test"])
        if key in seen:
            raise ValueError("duplicate metric key")
        seen.add(key)
        if type(row["seed"]) is not int or row["seed"] not in (1701, 1702, 1703):
            raise ValueError("formal seed inventory mismatch")
        value = np.asarray([row["nll"], row["brier"]], dtype=float)
        if not np.isfinite(value).all() or np.any(value < 0) or any(isinstance(row[k], bool) for k in ("nll", "brier")):
            raise ValueError("finite nonnegative metrics required")
        meta = (row["source_cluster"], row["stratum"])
        if row["task"] in metadata and metadata[row["task"]] != meta:
            raise ValueError("task cluster/stratum changed")
        metadata[row["task"]] = meta
        groups.setdefault(key[:4], []).append(value)
    if not seen or (expected_keys is not None and (len(expected_keys) != len(set(map(tuple, expected_keys))) or seen != set(map(tuple, expected_keys)))):
        raise ValueError("metric expected inventory mismatch")
    candidates = {}
    for (task, seed, arm, candidate), values in groups.items():
        candidates.setdefault((task, seed, arm), []).append(np.mean(values, axis=0))
    seeds = {}
    for (task, seed, arm), values in candidates.items():
        seeds.setdefault((task, arm), {})[seed] = np.mean(values, axis=0)
    output = {}
    for (task, arm), values in seeds.items():
        if set(values) != {1701, 1702, 1703}:
            raise ValueError("each task requires exactly the three fixed seeds")
        means = np.mean(list(values.values()), axis=0)
        output.setdefault(arm, {})[task] = dict(nll=float(means[0]), brier=float(means[1]),
            source_cluster=metadata[task][0], stratum=metadata[task][1])
    if len({tuple(sorted(tasks)) for tasks in output.values()}) != 1:
        raise ValueError("paired arms have different task inventory")
    return output


def holm_locked(p_values, *, family):
    family = list(family)
    if not family or len(family) != len(set(family)) or set(p_values) != set(family):
        raise ValueError("complete prespecified secondary family required")
    if any(isinstance(p, bool) or not np.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("valid p values required")
    adjusted, previous = {}, 0.
    for rank, name in enumerate(sorted(family, key=lambda name: (p_values[name], name))):
        previous = max(previous, min(1., (len(family)-rank)*p_values[name]))
        adjusted[name] = previous
    return adjusted
