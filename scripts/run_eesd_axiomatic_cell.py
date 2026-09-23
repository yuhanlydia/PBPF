#!/usr/bin/env python3
"""Run one matched-budget axiomatic EESD downstream validation cell.

This wrapper reuses an existing correction bank. It replaces only the relevance/trust
computation, trains one eesd_full adapter with the supplied sealed response-token
budget, generates one fresh greedy primary candidate per source, evaluates it, and
compares against an existing no_update report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def call(command, *, root):
    print(json.dumps({"command": command}), flush=True)
    subprocess.check_call(command, cwd=root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corrections", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/experiments/eesd_iclr2027.yaml"))
    p.add_argument("--domain", choices=["rbr", "codearc"], required=True)
    p.add_argument("--public-root", type=Path, required=True)
    p.add_argument("--evaluator-root", type=Path, required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--baseline-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--response-token-budget", type=int, required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--anchor-beta", type=float, default=0.03)
    p.add_argument("--previous-adapter", type=Path)
    args = p.parse_args()
    if args.response_token_budget < 1 or args.alpha <= 0 or args.anchor_beta < 0:
        raise ValueError("invalid budget/prior/anchor")

    root = Path(__file__).resolve().parents[1]
    for path in (
        args.corrections, args.model_config, args.config, args.baseline_report
    ):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    if not args.public_root.resolve().is_dir() or not args.evaluator_root.resolve().is_dir():
        raise FileNotFoundError("public/evaluator root missing")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    attributed = output / "shapley-relevance"
    command = [
        sys.executable, "scripts/score_eesd_shapley_relevance.py",
        "--input", str(args.corrections.resolve()),
        "--model-config", str(args.model_config.resolve()),
        "--output", str(attributed),
        "--mode", "exact",
    ]
    if args.previous_adapter:
        command += ["--adapter", str(args.previous_adapter.resolve())]
    call(command, root=root)

    scored = output / "scored"
    call([
        sys.executable, "scripts/score_eesd_corrections.py",
        "--input", str(attributed / "corrections-shapley.jsonl"),
        "--config", str(args.config.resolve()),
        "--output", str(scored),
        "--alpha", str(args.alpha),
    ], root=root)

    training = output / "training"
    command = [
        sys.executable, "scripts/run_eesd_weighted_sft.py",
        "--input", str(scored / "scored-corrections.jsonl"),
        "--model-config", str(args.model_config.resolve()),
        "--rule", "eesd_full",
        "--output", str(training),
        "--seed", str(args.seed),
        "--response-token-budget", str(args.response_token_budget),
        "--anchor-beta", str(args.anchor_beta),
    ]
    if args.previous_adapter:
        command += ["--previous-adapter", str(args.previous_adapter.resolve())]
    call(command, root=root)

    bank = output / "fresh-primary-bank"
    generator = (
        "scripts/generate_apbpf_rbr_bank.py"
        if args.domain == "rbr"
        else "scripts/generate_apbpf_codearc_bank.py"
    )
    call([
        sys.executable, generator,
        "--public-root", str(args.public_root.resolve()),
        "--output", str(bank),
        "--family", args.family,
        "--split", "primary",
        "--candidates", "1",
        "--seed", str(args.seed),
        "--greedy",
        "--adapter", str(training / "adapter"),
    ], root=root)

    evaluation = output / "fresh-eval"
    call([
        sys.executable, "scripts/evaluate_eesd_fresh_bank.py",
        "--domain", args.domain,
        "--evaluator-root", str(args.evaluator_root.resolve()),
        "--bank", str(bank),
        "--output", str(evaluation),
    ], root=root)

    comparison = output / "vs-no-update.json"
    call([
        sys.executable, "scripts/compare_eesd_fresh_eval.py",
        "--baseline", str(args.baseline_report.resolve()),
        "--method", str(evaluation / "report.json"),
        "--output", str(comparison),
    ], root=root)

    report = {
        "schema": "eesd-axiomatic-cell-v1",
        "domain": args.domain,
        "family": args.family,
        "seed": args.seed,
        "response_token_budget": args.response_token_budget,
        "alpha": args.alpha,
        "prior": "symmetric_jeffreys_dirichlet",
        "relevance": "exact_evidence_shapley_edit_logprob_contrast",
        "effective_mass": "renyi2_effective_support",
        "trust_rule": "posterior_excess_benefit_confidence",
        "anchor_beta": args.anchor_beta,
        "comparison": json.loads(comparison.read_text()),
        "attribution_report": json.loads((attributed / "report.json").read_text()),
        "scoring_summary": json.loads((scored / "summary.json").read_text()),
        "training_report": json.loads((training / "training-report.json").read_text()),
        "fresh_report": json.loads((evaluation / "report.json").read_text()),
    }
    (output / "axiomatic-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
