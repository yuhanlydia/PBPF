from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    replicates: int


def paired_cluster_bootstrap(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    clusters: Sequence[Hashable],
    rng: np.random.Generator,
    replicates: int = 10_000,
    confidence: float = 0.95,
) -> BootstrapInterval:
    left = np.asarray(treatment, dtype=np.float64)
    right = np.asarray(control, dtype=np.float64)
    cluster_values = np.asarray(clusters, dtype=object)
    if left.shape != right.shape or left.ndim != 1 or len(left) != len(cluster_values):
        raise ValueError("paired values and cluster labels must have equal vector shape")
    if len(left) == 0 or replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("non-empty values, positive replicates, and valid confidence required")
    unique = np.asarray(list(dict.fromkeys(clusters)), dtype=object)
    indices = {cluster: np.flatnonzero(cluster_values == cluster) for cluster in unique}
    differences = left - right
    samples = np.empty(replicates, dtype=np.float64)
    for iteration in range(replicates):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sampled_indices = np.concatenate([indices[cluster] for cluster in selected])
        samples[iteration] = differences[sampled_indices].mean()
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return BootstrapInterval(float(differences.mean()), float(lower), float(upper), replicates)


def paired_seed_task_bootstrap(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    source_clusters: Sequence[Hashable],
    rng: np.random.Generator,
    replicates: int = 10_000,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Hierarchical paired bootstrap with equal source, task, and seed units."""
    left = np.asarray(treatment, dtype=np.float64)
    right = np.asarray(control, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("treatment and control must have equal [seeds, tasks] shape")
    if left.shape[0] == 0 or left.shape[1] != len(source_clusters):
        raise ValueError("source cluster labels must match a non-empty task axis")
    if replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("positive replicates and valid confidence are required")
    sources = tuple(dict.fromkeys(source_clusters))
    tasks_by_source = {
        source: np.asarray(
            [index for index, value in enumerate(source_clusters) if value == source],
            dtype=np.int64,
        )
        for source in sources
    }
    differences = left - right

    def equal_unit_mean(values: np.ndarray) -> float:
        source_means = [
            float(values[:, task_indices].mean())
            for task_indices in tasks_by_source.values()
        ]
        return float(np.mean(source_means))

    samples = np.empty(replicates, dtype=np.float64)
    seed_count = left.shape[0]
    for iteration in range(replicates):
        sampled_seeds = rng.integers(0, seed_count, size=seed_count)
        sampled_sources = rng.choice(np.asarray(sources, dtype=object), size=len(sources), replace=True)
        source_means = []
        for source in sampled_sources:
            available_tasks = tasks_by_source[source]
            sampled_tasks = rng.choice(
                available_tasks, size=len(available_tasks), replace=True
            )
            task_means = []
            for task_index in sampled_tasks:
                task_means.append(
                    float(differences[sampled_seeds, int(task_index)].mean())
                )
            source_means.append(float(np.mean(task_means)))
        samples[iteration] = float(np.mean(source_means))
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return BootstrapInterval(
        equal_unit_mean(differences), float(lower), float(upper), replicates
    )


def prediction_gate(
    *,
    pbpf_nll: float,
    baseline_nll: float,
    improvement_ci_lower: float,
    pbpf_brier: float,
    baseline_brier: float,
    shuffled_gain_removed: float,
    random_gain_removed: float,
) -> bool:
    relative_gain = (baseline_nll - pbpf_nll) / baseline_nll
    return bool(
        relative_gain >= 0.05
        and improvement_ci_lower > 0
        and pbpf_brier <= baseline_brier
        and shuffled_gain_removed >= 0.8
        and random_gain_removed >= 0.8
    )


def repair_gate(
    *,
    pbpf_pass_at_1: float,
    baseline_pass_at_1: float,
    improvement_ci_lower: float,
    retains_future_nll_advantage: bool,
) -> bool:
    return bool(
        pbpf_pass_at_1 - baseline_pass_at_1 >= 0.03
        and improvement_ci_lower > 0
        and retains_future_nll_advantage
    )
