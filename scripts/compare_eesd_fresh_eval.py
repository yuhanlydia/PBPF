#!/usr/bin/env python3
"""Compare fresh-policy evaluations with paired source-level gain/regression counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path):
    value = json.loads(path.read_text())
    if value.get("schema") != "eesd-fresh-policy-eval-v1":
        raise ValueError("fresh EESD report required")
    rows = {r["source_component_id"]: bool(r["all_tests_pass"]) for r in value["records"]}
    if len(rows) != len(value["records"]):
        raise ValueError("duplicate source_component_id")
    return value, rows


def canonical_domain(value):
    return "runbugrun" if value == "rbr" else value


def bootstrap(diff, *, seed=314159, replicates=10000):
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for i in range(replicates):
        draws[i] = rng.choice(diff, size=len(diff), replace=True).mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return [float(lo), float(hi)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--method", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base_report, base = load(args.baseline)
    method_report, method = load(args.method)
    if (base_report.get("execution_profile", "sandbox")
            != method_report.get("execution_profile", "sandbox")
            or base_report.get("execution_lock_sha256")
            != method_report.get("execution_lock_sha256")):
        raise ValueError("paired evaluations require the same execution profile and lock")
    if canonical_domain(base_report.get("domain")) != canonical_domain(method_report.get("domain")):
        raise ValueError("paired evaluations differ in domain")
    for key in ("model", "revision", "task_manifest_sha256"):
        if base_report.get(key) != method_report.get(key):
            raise ValueError(f"paired evaluations differ in {key}")
    if set(base) != set(method):
        raise ValueError("paired source populations differ")
    keys = sorted(base)
    b = np.asarray([base[k] for k in keys], dtype=int)
    m = np.asarray([method[k] for k in keys], dtype=int)
    diff = m - b
    regressions = int(np.sum((b == 1) & (m == 0)))
    fixes = int(np.sum((b == 0) & (m == 1)))
    baseline_correct = int(b.sum())
    baseline_wrong = int((1 - b).sum())
    summary = {
        "schema": "eesd-fresh-policy-comparison-v1",
        "sources": len(keys),
        "baseline_pass_at_1": float(b.mean()),
        "method_pass_at_1": float(m.mean()),
        "absolute_gain": float(diff.mean()),
        "gain_ci95": bootstrap(diff),
        "fixes": fixes,
        "regressions": regressions,
        "net_gain_count": fixes - regressions,
        "regression_rate_on_previously_correct": regressions / baseline_correct if baseline_correct else None,
        "fix_rate_on_previously_wrong": fixes / baseline_wrong if baseline_wrong else None,
        "baseline_adapter_sha256": base_report.get("adapter_sha256"),
        "method_adapter_sha256": method_report.get("adapter_sha256"),
        "domain": canonical_domain(base_report.get("domain")),
        "model": base_report.get("model"),
        "revision": base_report.get("revision"),
        "task_manifest_sha256": base_report.get("task_manifest_sha256"),
        "execution_profile": base_report.get("execution_profile", "sandbox"),
        "execution_lock_sha256": base_report.get("execution_lock_sha256"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
