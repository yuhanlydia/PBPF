#!/usr/bin/env python3
"""Evaluate axiomatic EESD trust against independent hidden correction behavior.

This is an evaluation-only script. It never changes relevance, training weights,
candidate selection, or stopping. Public before/after executions construct each
trust score; hidden executions are read only after scoring to measure credibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pbpf.eesd.evidence import (
    correction_benefit_posterior,
    effective_mass,
    posterior_benefit_probability,
    posterior_update_weight,
)
from pbpf.registry import OUTCOMES


def to_indices(values):
    result = []
    for value in values:
        if isinstance(value, str):
            if value not in OUTCOMES:
                raise ValueError(f"unknown outcome: {value}")
            value = OUTCOMES.index(value)
        if type(value) is not int or not 0 <= value < len(OUTCOMES):
            raise ValueError("invalid outcome")
        result.append(value)
    return np.asarray(result, dtype=np.int64)


def soft_log_loss(target, probability):
    target = np.asarray(target, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if target.shape != probability.shape or target.ndim != 1 or not len(target):
        raise ValueError("target/probability mismatch")
    p = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return float(-(target * np.log(p) + (1.0 - target) * np.log(1.0 - p)).mean())


def brier(target, probability):
    target = np.asarray(target, dtype=float)
    probability = np.asarray(probability, dtype=float)
    return float(np.square(target - probability).mean())


def rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(x, y):
    x, y = rankdata(x), rankdata(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def paired_source_bootstrap(target, proposed, baseline, sources, *, seed, replicates):
    target = np.asarray(target, dtype=float)
    proposed = np.asarray(proposed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    sources = np.asarray(sources)
    if not (target.shape == proposed.shape == baseline.shape == sources.shape):
        raise ValueError("paired bootstrap arrays must align")
    unique = np.unique(sources)
    if len(unique) < 2:
        raise ValueError("at least two independent source clusters required")
    def loss(p):
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
        return -(target * np.log(p) + (1.0 - target) * np.log(1.0 - p))
    delta = loss(baseline) - loss(proposed)
    cluster = np.asarray([float(delta[sources == key].mean()) for key in unique])
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for i in range(replicates):
        draws[i] = float(rng.choice(cluster, size=len(cluster), replace=True).mean())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "gain": float(cluster.mean()),
        "ci95": [float(lo), float(hi)],
        "clusters": int(len(unique)),
        "replicates": int(replicates),
    }


def public_scores(row, relevance, *, alpha, mass_rule):
    before = to_indices(row["before_outcomes"])
    after = to_indices(row["after_outcomes"])
    posterior = correction_benefit_posterior(
        before, after, relevance, alpha=alpha, mass_rule=mass_rule
    )
    return {
        "benefit_probability": posterior_benefit_probability(posterior),
        "update_weight": posterior_update_weight(posterior),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corrections-shapley", type=Path, required=True)
    p.add_argument("--hidden-eval", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--bootstrap-seed", type=int, default=314159)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = p.parse_args()
    if args.alpha <= 0 or args.bootstrap_replicates < 1:
        raise ValueError("alpha and bootstrap count must be positive")

    rows = [
        json.loads(line)
        for line in args.corrections_shapley.read_text().splitlines()
        if line.strip()
    ]
    evaluation = json.loads(args.hidden_eval.read_text())
    if not rows or evaluation.get("schema") != "eesd-independent-correction-eval-v1":
        raise ValueError("Shapley corrections and independent hidden evaluation required")
    hidden = {row["trajectory_id"]: row for row in evaluation.get("records", [])}
    if set(hidden) != {row["trajectory_id"] for row in rows}:
        raise ValueError("hidden evaluation must cover exactly the scored trajectory population")

    target, hidden_net, sources = [], [], []
    benefit_probability = {
        "shapley_effective": [],
        "shapley_fixed": [],
        "lexical_effective": [],
        "scalar_confidence": [],
        "final_correctness": [],
    }
    update_weight = {
        "shapley_effective": [],
        "shapley_fixed": [],
        "lexical_effective": [],
    }
    shapley_mass, lexical_mass = [], []

    for row in rows:
        hidden_row = hidden[row["trajectory_id"]]
        before_hidden = to_indices(hidden_row["before_hidden"])
        after_hidden = to_indices(hidden_row["after_hidden"])
        before_pass = before_hidden == 0
        after_pass = after_hidden == 0
        fixes = int(((~before_pass) & after_pass).sum())
        regressions = int((before_pass & (~after_pass)).sum())
        target.append(1.0 if fixes > regressions else 0.0 if fixes < regressions else 0.5)
        hidden_net.append((fixes - regressions) / len(before_hidden))
        sources.append(row["source_component_id"])

        shapley = np.asarray(row["relevance"], dtype=float)
        legacy = row.get("legacy_relevance")
        if legacy is None:
            raise ValueError("Shapley artifact must preserve legacy relevance for paired comparison")
        legacy = np.asarray(legacy, dtype=float)

        for name, relevance, mass_rule in (
            ("shapley_effective", shapley, "effective"),
            ("shapley_fixed", shapley, "fixed"),
            ("lexical_effective", legacy, "effective"),
        ):
            scores = public_scores(row, relevance, alpha=args.alpha, mass_rule=mass_rule)
            benefit_probability[name].append(scores["benefit_probability"])
            update_weight[name].append(scores["update_weight"])
        after = to_indices(row["after_outcomes"])
        benefit_probability["scalar_confidence"].append(float((after == 0).mean()))
        benefit_probability["final_correctness"].append(float((after == 0).all()))
        shapley_mass.append(effective_mass(shapley))
        lexical_mass.append(effective_mass(legacy))

    target = np.asarray(target, dtype=float)
    hidden_net = np.asarray(hidden_net, dtype=float)
    sources = np.asarray(sources)
    benefit_probability = {
        key: np.asarray(value, dtype=float) for key, value in benefit_probability.items()
    }
    update_weight = {
        key: np.asarray(value, dtype=float) for key, value in update_weight.items()
    }
    hidden_positive_net = np.maximum(hidden_net, 0.0)

    probability_metrics = {
        key: {
            "soft_nll": soft_log_loss(target, value),
            "brier": brier(target, value),
            "spearman_hidden_net_gain": spearman(value, hidden_net),
        }
        for key, value in benefit_probability.items()
    }
    weight_metrics = {
        key: {
            "mse_hidden_positive_net": float(np.square(value - hidden_positive_net).mean()),
            "mae_hidden_positive_net": float(np.abs(value - hidden_positive_net).mean()),
            "spearman_hidden_net_gain": spearman(value, hidden_net),
        }
        for key, value in update_weight.items()
    }
    comparisons = {
        baseline: paired_source_bootstrap(
            target,
            benefit_probability["shapley_effective"],
            benefit_probability[baseline],
            sources,
            seed=args.bootstrap_seed,
            replicates=args.bootstrap_replicates,
        )
        for baseline in (
            "shapley_fixed",
            "lexical_effective",
            "scalar_confidence",
            "final_correctness",
        )
    }
    def mass_summary(values):
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.quantile(values, 0.5)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    report = {
        "schema": "eesd-axiomatic-credibility-v1",
        "population": len(rows),
        "sources": len(set(sources.tolist())),
        "alpha": args.alpha,
        "target": "hidden fix-count greater than regression-count; ties have soft target 0.5",
        "benefit_probability_metrics": probability_metrics,
        "update_weight_metrics": weight_metrics,
        "paired_nll_gain_of_shapley_effective_probability": comparisons,
        "effective_mass": {
            "shapley": mass_summary(shapley_mass),
            "legacy_lexical": mass_summary(lexical_mass),
        },
        "hidden_net_gain_mean": float(hidden_net.mean()),
        "hidden_evidence_used_for_training_or_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
