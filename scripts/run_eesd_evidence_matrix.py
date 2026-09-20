#!/usr/bin/env python3
"""Run the locked EED mechanism/ablation matrix on one execution cache.

The cache must expose public test inputs and ordered categorical outcomes. Future
outcomes are used only as evaluator labels. This runner is therefore appropriate
for public-input/private-outcome protocols; fully hidden-input benchmarks require
a separately declared query-free relevance rule.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from pbpf.eesd.evidence import (
    cosine_relevance_weights,
    dirichlet_predict,
    effective_evidence_weights,
    effective_mass,
    fixed_mass_weights,
    mass_rescaled_weights,
    score_predictions,
    source_cluster_bootstrap_gain,
)
from pbpf.real_gate import FrozenTextEncoder
from pbpf.registry import OUTCOMES


@lru_cache(maxsize=100_000)
def _encode(text: str) -> np.ndarray:
    return FrozenTextEncoder(256)(text)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _outcome_index(value: str, *, binary: bool) -> int:
    if value not in OUTCOMES:
        raise ValueError(f"unknown outcome: {value}")
    index = OUTCOMES.index(value)
    return int(index != 0) if binary else index


def _rows(payload, split: str):
    rows = [row for row in payload["records"] if row.get("split") == split]
    if not rows:
        raise ValueError(f"cache has no rows for split={split}")
    return rows


def _validate_rows(rows, visible: int) -> None:
    for row in rows:
        tests, outcomes = row.get("tests"), row.get("outcomes")
        if (
            not isinstance(tests, list)
            or not isinstance(outcomes, list)
            or len(tests) != len(outcomes)
            or len(tests) <= visible
            or any(not isinstance(t, dict) or not isinstance(t.get("input"), str) for t in tests)
        ):
            raise ValueError("records require ordered tests/outcomes and public input strings")
        if "source_component_id" not in row and "problem_id" not in row:
            raise ValueError("source-component identity required for clustered statistics")


def build_examples(rows, history_size: int, strength: float, *, binary: bool):
    examples = []
    classes = 2 if binary else len(OUTCOMES)
    for row in rows:
        if len(row["tests"]) <= history_size:
            raise ValueError("history size leaves no future outcome")
        visible = np.asarray(
            [_outcome_index(x, binary=binary) for x in row["outcomes"][:history_size]],
            dtype=np.int64,
        )
        history = np.stack([_encode(t["input"]) for t in row["tests"][:history_size]])
        cluster = row.get("source_component_id", row.get("problem_id"))
        for test, outcome in zip(row["tests"][history_size:], row["outcomes"][history_size:], strict=True):
            raw = cosine_relevance_weights(_encode(test["input"]), history, strength)
            examples.append(
                {
                    "visible": visible,
                    "weights": raw,
                    "label": _outcome_index(outcome, binary=binary),
                    "cluster": cluster,
                    "mass": effective_mass(raw),
                    "classes": classes,
                }
            )
    return examples


def predictions(examples, alpha: float, rule: str, *, global_mass=None, mass_overrides=None):
    rows = []
    masses = []
    if mass_overrides is not None and len(mass_overrides) != len(examples):
        raise ValueError("mass override length mismatch")
    for index, example in enumerate(examples):
        raw = example["weights"]
        if rule == "ordinary":
            weights = np.ones(len(raw), dtype=float)
            mass = float(len(raw))
        elif rule == "fixed":
            weights = fixed_mass_weights(raw)
            mass = float(len(raw))
        elif rule == "effective":
            weights = effective_evidence_weights(raw)
            mass = float(weights.sum())
        elif rule == "global":
            mass = float(mass_overrides[index]) if mass_overrides is not None else float(global_mass)
            weights = mass_rescaled_weights(raw, mass)
        else:
            raise ValueError(f"unknown mass rule: {rule}")
        rows.append(
            dirichlet_predict(
                example["visible"], alpha, weights=weights, classes=example["classes"]
            )
        )
        masses.append(mass)
    return np.asarray(rows), np.asarray(masses)


def labels_clusters(examples):
    return (
        np.asarray([x["label"] for x in examples], dtype=np.int64),
        np.asarray([x["cluster"] for x in examples]),
    )


def select_arm(example_cache, split_key, history_size, alphas, strengths, rule, *, masses=None, ece_bins=10):
    scores = []
    for strength in strengths:
        examples = example_cache(split_key, history_size, strength, False)
        labels, _ = labels_clusters(examples)
        mass_grid = [None] if masses is None else list(masses)
        for alpha in alphas:
            for mass in mass_grid:
                p, _ = predictions(examples, alpha, rule, global_mass=mass)
                metrics = score_predictions(labels, p, ece_bins=ece_bins)
                scores.append(
                    {
                        "alpha": float(alpha),
                        "strength": float(strength),
                        "mass": None if mass is None else float(mass),
                        **metrics,
                    }
                )
    return min(
        scores,
        key=lambda x: (
            x["nll"],
            x["strength"],
            x["alpha"],
            -1 if x["mass"] is None else x["mass"],
        ),
    ), scores


def evaluate(examples, selection, rule, *, ece_bins, global_mass=None, mass_overrides=None):
    p, masses = predictions(
        examples,
        selection["alpha"],
        rule,
        global_mass=global_mass if global_mass is not None else selection.get("mass"),
        mass_overrides=mass_overrides,
    )
    y, c = labels_clusters(examples)
    return {
        "probabilities": p,
        "masses": masses,
        "labels": y,
        "clusters": c,
        "metrics": score_predictions(y, p, ece_bins=ece_bins),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/eesd_iclr2027.yaml"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-split", default="development")
    parser.add_argument("--assessment-split", default="primary")
    parser.add_argument("--visible", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if config.get("schema") != "eesd-iclr2027-v1":
        raise ValueError("unexpected EESD config schema")
    evidence = config["evidence"]
    if args.visible < 1:
        raise ValueError("visible history must be positive")
    payload = json.loads(args.cache.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("cache must contain a records list")

    validation_rows = _rows(payload, args.validation_split)
    assessment_rows = _rows(payload, args.assessment_split)
    _validate_rows(validation_rows, args.visible)
    _validate_rows(assessment_rows, args.visible)

    args.output.mkdir(parents=True, exist_ok=False)
    alphas = [float(x) for x in evidence["alpha_grid"]]
    strengths = [float(x) for x in evidence["strength_grid"]]
    global_masses = [float(x) for x in evidence["global_mass_grid"]]
    ece_bins = int(evidence["ece_bins"])
    bootstrap_seed = int(config["bootstrap_seed"])
    bootstrap_reps = int(config["bootstrap_replicates"])

    cached = {}
    def example_cache(split_key, history_size, strength, binary):
        key = split_key, int(history_size), float(strength), bool(binary)
        if key not in cached:
            rows = validation_rows if split_key == "validation" else assessment_rows
            cached[key] = build_examples(rows, history_size, strength, binary=binary)
        return cached[key]

    ordinary_sel, ordinary_grid = select_arm(
        example_cache, "validation", args.visible, alphas, [0.0], "ordinary", ece_bins=ece_bins
    )
    fixed_sel, fixed_grid = select_arm(
        example_cache, "validation", args.visible, alphas, strengths, "fixed", ece_bins=ece_bins
    )
    eed_sel, eed_grid = select_arm(
        example_cache, "validation", args.visible, alphas, strengths, "effective", ece_bins=ece_bins
    )
    global_sel, global_grid = select_arm(
        example_cache, "validation", args.visible, alphas, strengths, "global",
        masses=global_masses, ece_bins=ece_bins
    )

    ordinary = evaluate(
        example_cache("assessment", args.visible, 0.0, False), ordinary_sel, "ordinary", ece_bins=ece_bins
    )
    fixed = evaluate(
        example_cache("assessment", args.visible, fixed_sel["strength"], False),
        fixed_sel, "fixed", ece_bins=ece_bins
    )
    eed = evaluate(
        example_cache("assessment", args.visible, eed_sel["strength"], False),
        eed_sel, "effective", ece_bins=ece_bins
    )
    global_arm = evaluate(
        example_cache("assessment", args.visible, global_sel["strength"], False),
        global_sel, "global", ece_bins=ece_bins
    )

    # Same-alpha/same-strength controls isolate mass while holding relevance fixed.
    fixed_parameter_examples = example_cache("assessment", args.visible, fixed_sel["strength"], False)
    eed_at_fixed = evaluate(fixed_parameter_examples, fixed_sel, "effective", ece_bins=ece_bins)
    eed_parameter_examples = example_cache("assessment", args.visible, eed_sel["strength"], False)
    fixed_at_eed = evaluate(eed_parameter_examples, eed_sel, "fixed", ece_bins=ece_bins)

    # Constant mean effective mass: same EED alpha/strength but no query adaptivity.
    validation_eed_examples = example_cache("validation", args.visible, eed_sel["strength"], False)
    mean_effective_mass = float(np.mean([x["mass"] for x in validation_eed_examples]))
    constant_mean = evaluate(
        eed_parameter_examples,
        eed_sel,
        "global",
        ece_bins=ece_bins,
        global_mass=mean_effective_mass,
    )

    # Alignment control: keep the exact assessment mass multiset and permute it.
    aligned_masses = np.asarray([x["mass"] for x in eed_parameter_examples], dtype=float)
    permutation_metrics = []
    for seed in evidence["permutation_seeds"]:
        rng = np.random.default_rng(int(seed))
        permuted = rng.permutation(aligned_masses)
        arm = evaluate(
            eed_parameter_examples,
            eed_sel,
            "global",
            ece_bins=ece_bins,
            mass_overrides=permuted,
        )
        permutation_metrics.append({"seed": int(seed), **arm["metrics"]})

    # Full same-parameter factorial; no assessment-based reselection.
    factorial = []
    for strength in strengths:
        examples = example_cache("assessment", args.visible, strength, False)
        labels, _ = labels_clusters(examples)
        for alpha in alphas:
            pf, _ = predictions(examples, alpha, "fixed")
            pe, _ = predictions(examples, alpha, "effective")
            mf = score_predictions(labels, pf, ece_bins=ece_bins)
            me = score_predictions(labels, pe, ece_bins=ece_bins)
            factorial.append(
                {
                    "alpha": alpha,
                    "strength": strength,
                    "fixed_nll": mf["nll"],
                    "effective_nll": me["nll"],
                    "nll_gain": mf["nll"] - me["nll"],
                    "fixed_brier": mf["brier"],
                    "effective_brier": me["brier"],
                    "accuracy_agreement": float(np.mean(pf.argmax(1) == pe.argmax(1))),
                }
            )

    # History-size sweep: retune on validation for each declared history length.
    history_sweep = []
    min_tests = min(len(r["tests"]) for r in validation_rows + assessment_rows)
    for history_size in evidence["history_sizes"]:
        history_size = int(history_size)
        if history_size >= min_tests:
            continue
        fsel, _ = select_arm(
            example_cache, "validation", history_size, alphas, strengths, "fixed", ece_bins=ece_bins
        )
        esel, _ = select_arm(
            example_cache, "validation", history_size, alphas, strengths, "effective", ece_bins=ece_bins
        )
        fa = evaluate(
            example_cache("assessment", history_size, fsel["strength"], False),
            fsel, "fixed", ece_bins=ece_bins
        )
        ea = evaluate(
            example_cache("assessment", history_size, esel["strength"], False),
            esel, "effective", ece_bins=ece_bins
        )
        history_sweep.append(
            {
                "history_size": history_size,
                "fixed_selected": fsel,
                "effective_selected": esel,
                "fixed_metrics": fa["metrics"],
                "effective_metrics": ea["metrics"],
            }
        )

    # Binary outcome taxonomy sensitivity using the selected multiclass EED hyperparameters.
    binary_examples = example_cache("assessment", args.visible, eed_sel["strength"], True)
    binary_fixed = evaluate(binary_examples, eed_sel, "fixed", ece_bins=ece_bins)
    binary_eed = evaluate(binary_examples, eed_sel, "effective", ece_bins=ece_bins)

    # Concentration-bin mechanism analysis at identical alpha/strength.
    same_fixed_probs = fixed_at_eed["probabilities"]
    same_eed_probs = eed["probabilities"]
    y = eed["labels"]
    losses_fixed = -np.log(same_fixed_probs[np.arange(len(y)), y].clip(1e-12, 1.0))
    losses_eed = -np.log(same_eed_probs[np.arange(len(y)), y].clip(1e-12, 1.0))
    masses = eed["masses"]
    if args.visible == 4:
        edges = np.asarray(evidence["concentration_bins_for_n4"], dtype=float)
    else:
        edges = np.quantile(masses, np.linspace(0, 1, 5))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
    concentration_bins = []
    for i in range(len(edges) - 1):
        mask = (masses >= edges[i]) & (masses < edges[i + 1])
        concentration_bins.append(
            {
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "count": int(mask.sum()),
                "mean_mass": float(masses[mask].mean()) if mask.any() else None,
                "fixed_nll": float(losses_fixed[mask].mean()) if mask.any() else None,
                "effective_nll": float(losses_eed[mask].mean()) if mask.any() else None,
                "nll_gain": float((losses_fixed[mask] - losses_eed[mask]).mean()) if mask.any() else None,
            }
        )

    bootstraps = {
        "tuned_effective_vs_tuned_fixed": source_cluster_bootstrap_gain(
            eed["labels"], eed["probabilities"], fixed["probabilities"], eed["clusters"],
            seed=bootstrap_seed, replicates=bootstrap_reps,
        ),
        "effective_vs_fixed_at_fixed_params": source_cluster_bootstrap_gain(
            eed_at_fixed["labels"], eed_at_fixed["probabilities"], fixed["probabilities"],
            eed_at_fixed["clusters"], seed=bootstrap_seed, replicates=bootstrap_reps,
        ),
        "effective_vs_fixed_at_effective_params": source_cluster_bootstrap_gain(
            eed["labels"], eed["probabilities"], fixed_at_eed["probabilities"], eed["clusters"],
            seed=bootstrap_seed, replicates=bootstrap_reps,
        ),
        "effective_vs_tuned_global_mass": source_cluster_bootstrap_gain(
            eed["labels"], eed["probabilities"], global_arm["probabilities"], eed["clusters"],
            seed=bootstrap_seed, replicates=bootstrap_reps,
        ),
    }

    report = {
        "schema": "eesd-evidence-matrix-v1",
        "claim_status": (
            "locked-primary-assessment" if args.assessment_split in {"primary", "test"}
            else "exploratory-development-assessment"
        ),
        "dataset": args.dataset,
        "model": args.model,
        "cache_sha256": sha(args.cache),
        "config_sha256": sha(args.config),
        "validation_split": args.validation_split,
        "assessment_split": args.assessment_split,
        "visible": args.visible,
        "counts": {
            "validation_records": len(validation_rows),
            "assessment_records": len(assessment_rows),
            "assessment_examples": int(len(eed["labels"])),
            "assessment_sources": int(len(np.unique(eed["clusters"]))),
        },
        "selected": {
            "ordinary": ordinary_sel,
            "fixed": fixed_sel,
            "effective": eed_sel,
            "global_mass": global_sel,
            "constant_mean_effective_mass": mean_effective_mass,
        },
        "metrics": {
            "ordinary": ordinary["metrics"],
            "fixed": fixed["metrics"],
            "effective": eed["metrics"],
            "global_mass": global_arm["metrics"],
            "effective_at_fixed_params": eed_at_fixed["metrics"],
            "fixed_at_effective_params": fixed_at_eed["metrics"],
            "constant_mean_mass": constant_mean["metrics"],
            "permuted_mass": {
                "runs": permutation_metrics,
                "mean_nll": float(np.mean([x["nll"] for x in permutation_metrics])),
                "std_nll": float(np.std([x["nll"] for x in permutation_metrics], ddof=1)),
            },
            "binary_fixed_at_effective_params": binary_fixed["metrics"],
            "binary_effective": binary_eed["metrics"],
        },
        "argmax": {
            "fixed_vs_effective_tuned_agreement": float(
                np.mean(fixed["probabilities"].argmax(1) == eed["probabilities"].argmax(1))
            ) if len(fixed["probabilities"]) == len(eed["probabilities"]) else None,
            "same_effective_params_agreement": float(
                np.mean(fixed_at_eed["probabilities"].argmax(1) == eed["probabilities"].argmax(1))
            ),
        },
        "bootstraps": bootstraps,
        "factorial": factorial,
        "history_sweep": history_sweep,
        "concentration_bins": concentration_bins,
        "selection_grids": {
            "ordinary": ordinary_grid,
            "fixed": fixed_grid,
            "effective": eed_grid,
            "global_mass": global_grid,
        },
    }
    write_json(args.output / "report.json", report)
    np.savez_compressed(
        args.output / "predictions.npz",
        labels=eed["labels"],
        clusters=eed["clusters"],
        ordinary=ordinary["probabilities"],
        fixed=fixed["probabilities"],
        effective=eed["probabilities"],
        global_mass=global_arm["probabilities"],
        fixed_at_effective_params=fixed_at_eed["probabilities"],
        effective_at_fixed_params=eed_at_fixed["probabilities"],
        effective_mass=eed["masses"],
    )
    write_json(
        args.output / "complete.json",
        {
            "report_sha256": sha(args.output / "report.json"),
            "predictions_sha256": sha(args.output / "predictions.npz"),
        },
    )
    print(json.dumps({"status": "complete", "report": report["metrics"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
