#!/usr/bin/env python3
"""Run the closed-loop EESD recursive study for equal-weight SD vs full EESD.

Round 0 experience is shared. From round 1 onward, each arm collects new
experience using its own current adapter, then updates from that adapter. Every
round is evaluated by one fresh greedy primary candidate per source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml


RULES = ("equal_weight", "eesd_full")


def call(command, *, root: Path):
    print(json.dumps({"command": command}), flush=True)
    subprocess.check_call(command, cwd=root)


def generation_script(domain: str) -> str:
    return (
        "scripts/generate_apbpf_rbr_bank.py"
        if domain == "rbr"
        else "scripts/generate_apbpf_codearc_bank.py"
    )


def generate_bank(
    *,
    root: Path,
    domain: str,
    public_root: Path,
    family: str,
    split: str,
    output: Path,
    seed: int,
    adapter: Path | None,
    greedy: bool,
):
    if (output / "complete.json").exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        generation_script(domain),
        "--public-root", str(public_root),
        "--output", str(output),
        "--family", family,
        "--split", split,
        "--candidates", "1",
        "--seed", str(seed),
    ]
    if greedy:
        command.append("--greedy")
    if adapter is not None:
        command += ["--adapter", str(adapter)]
    call(command, root=root)


def evaluate_primary(
    *,
    root: Path,
    domain: str,
    evaluator_root: Path,
    bank: Path,
    output: Path,
):
    if (output / "report.json").exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    call(
        [
            sys.executable,
            "scripts/evaluate_eesd_fresh_bank.py",
            "--domain", domain,
            "--evaluator-root", str(evaluator_root),
            "--bank", str(bank),
            "--output", str(output),
        ],
        root=root,
    )


def compare_reports(root: Path, baseline: Path, method: Path, output: Path):
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    call(
        [
            sys.executable,
            "scripts/compare_eesd_fresh_eval.py",
            "--baseline", str(baseline),
            "--method", str(method),
            "--output", str(output),
        ],
        root=root,
    )


def collect_experience(
    *,
    root: Path,
    domain: str,
    public_root: Path,
    model_config: Path,
    family: str,
    output: Path,
    seed: int,
    adapter: Path | None,
    config: Path,
    alpha: float,
    uncertainty_penalty: float,
    relevance_strength: float,
):
    """Generate current-policy originals, corrections, and all locked trust weights."""
    train_bank = output / "original-banks" / "train"
    development_bank = output / "original-banks" / "development"
    generate_bank(
        root=root, domain=domain, public_root=public_root, family=family,
        split="train", output=train_bank, seed=seed, adapter=adapter, greedy=False,
    )
    generate_bank(
        root=root, domain=domain, public_root=public_root, family=family,
        split="development", output=development_bank, seed=seed, adapter=adapter, greedy=False,
    )

    public_bank = output / "public-corrections"
    if not (public_bank / "audit.json").exists():
        public_bank.parent.mkdir(parents=True, exist_ok=True)
        call(
            [
                sys.executable,
                "scripts/prepare_eesd_recursive_corrections.py",
                "--domain", domain,
                "--public-root", str(public_root),
                "--bank", str(train_bank),
                "--bank", str(development_bank),
                "--output", str(public_bank),
            ],
            root=root,
        )

    generated = output / "generated-corrections"
    if not (generated / "report.json").exists():
        generated.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "scripts/generate_eesd_corrections.py",
            "--public-bank", str(public_bank / "public-corrections.jsonl"),
            "--model-config", str(model_config),
            "--domain", domain,
            "--output", str(generated),
            "--relevance-strength", str(relevance_strength),
        ]
        if adapter is not None:
            command += ["--adapter", str(adapter)]
        call(command, root=root)

    scored = output / "scored"
    if not (scored / "summary.json").exists():
        scored.parent.mkdir(parents=True, exist_ok=True)
        call(
            [
                sys.executable,
                "scripts/score_eesd_corrections.py",
                "--input", str(generated / "corrections.jsonl"),
                "--config", str(config),
                "--output", str(scored),
                "--alpha", str(alpha),
                "--uncertainty-penalty", str(uncertainty_penalty),
            ],
            root=root,
        )
    return scored / "scored-corrections.jsonl"


def train_next(
    *,
    root: Path,
    model_config: Path,
    scored: Path,
    rule: str,
    previous_adapter: Path | None,
    output: Path,
    seed: int,
    max_steps: int,
    anchor_beta: float,
):
    if (output / "training-report.json").exists():
        return output / "adapter"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/run_eesd_weighted_sft.py",
        "--input", str(scored),
        "--model-config", str(model_config),
        "--rule", rule,
        "--output", str(output),
        "--seed", str(seed),
        "--max-steps", str(max_steps),
        "--anchor-beta", str(anchor_beta),
    ]
    if previous_adapter is not None:
        command += ["--previous-adapter", str(previous_adapter)]
    call(command, root=root)
    return output / "adapter"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/experiments/eesd_iclr2027.yaml"))
    p.add_argument("--domain", choices=["rbr", "codearc"], required=True)
    p.add_argument("--public-root", type=Path, required=True)
    p.add_argument("--evaluator-root", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--uncertainty-penalty", type=float, default=0.5)
    p.add_argument("--anchor-beta", type=float, default=0.03)
    p.add_argument("--relevance-strength", type=float, default=16.0)
    args = p.parse_args()
    if args.rounds < 1 or args.max_steps < 1:
        raise ValueError("round and training budgets must be positive")

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("schema") != "eesd-iclr2027-v1":
        raise ValueError("locked EESD config required")
    if args.seed not in [int(x) for x in cfg["seeds"]]:
        raise ValueError("recursive seed must be one of the locked seeds")
    public_root = args.public_root.resolve()
    evaluator_root = args.evaluator_root.resolve()
    model_config = args.model_config.resolve()
    if not public_root.is_dir() or not evaluator_root.is_dir() or not model_config.is_file():
        raise FileNotFoundError("recursive roots/model config are missing")
    args.output.mkdir(parents=True, exist_ok=True)

    # Round-0 fresh evaluation: one shared base-policy result.
    base_bank = args.output / "round0" / "base" / "primary-bank"
    generate_bank(
        root=root, domain=args.domain, public_root=public_root, family=args.family,
        split="primary", output=base_bank, seed=args.seed, adapter=None, greedy=True,
    )
    base_eval = args.output / "round0" / "base" / "fresh-eval"
    evaluate_primary(
        root=root, domain=args.domain, evaluator_root=evaluator_root,
        bank=base_bank, output=base_eval,
    )

    # Shared base-policy experience produces both round-1 updates.
    shared = args.output / "round0" / "shared-experience"
    shared_scored = collect_experience(
        root=root, domain=args.domain, public_root=public_root,
        model_config=model_config, family=args.family, output=shared,
        seed=args.seed, adapter=None, config=args.config.resolve(),
        alpha=args.alpha, uncertainty_penalty=args.uncertainty_penalty,
        relevance_strength=args.relevance_strength,
    )

    current = {}
    summaries = []
    for rule in RULES:
        current[rule] = train_next(
            root=root, model_config=model_config, scored=shared_scored, rule=rule,
            previous_adapter=None,
            output=args.output / "round1" / rule / "training",
            seed=args.seed, max_steps=args.max_steps, anchor_beta=args.anchor_beta,
        )

    previous_eval = {rule: base_eval / "report.json" for rule in RULES}
    base_report = base_eval / "report.json"

    for round_index in range(1, args.rounds + 1):
        for rule in RULES:
            # Round 1 has already been trained from the shared round-0 experience.
            if round_index > 1:
                experience = args.output / f"round{round_index-1}" / rule / "experience"
                scored = collect_experience(
                    root=root, domain=args.domain, public_root=public_root,
                    model_config=model_config, family=args.family, output=experience,
                    seed=args.seed, adapter=current[rule], config=args.config.resolve(),
                    alpha=args.alpha, uncertainty_penalty=args.uncertainty_penalty,
                    relevance_strength=args.relevance_strength,
                )
                current[rule] = train_next(
                    root=root, model_config=model_config, scored=scored, rule=rule,
                    previous_adapter=current[rule],
                    output=args.output / f"round{round_index}" / rule / "training",
                    seed=args.seed, max_steps=args.max_steps, anchor_beta=args.anchor_beta,
                )

            bank = args.output / f"round{round_index}" / rule / "primary-bank"
            generate_bank(
                root=root, domain=args.domain, public_root=public_root, family=args.family,
                split="primary", output=bank, seed=args.seed, adapter=current[rule], greedy=True,
            )
            evaluation = args.output / f"round{round_index}" / rule / "fresh-eval"
            evaluate_primary(
                root=root, domain=args.domain, evaluator_root=evaluator_root,
                bank=bank, output=evaluation,
            )
            current_report = evaluation / "report.json"
            vs_base = args.output / f"round{round_index}" / rule / "vs-base.json"
            vs_previous = args.output / f"round{round_index}" / rule / "vs-previous.json"
            compare_reports(root, base_report, current_report, vs_base)
            compare_reports(root, previous_eval[rule], current_report, vs_previous)
            previous_eval[rule] = current_report

        equal = json.loads(
            (args.output / f"round{round_index}" / "equal_weight" / "fresh-eval" / "report.json").read_text()
        )
        eesd = json.loads(
            (args.output / f"round{round_index}" / "eesd_full" / "fresh-eval" / "report.json").read_text()
        )
        head_to_head = args.output / f"round{round_index}" / "eesd-vs-equal.json"
        compare_reports(
            root,
            args.output / f"round{round_index}" / "equal_weight" / "fresh-eval" / "report.json",
            args.output / f"round{round_index}" / "eesd_full" / "fresh-eval" / "report.json",
            head_to_head,
        )
        summaries.append({
            "round": round_index,
            "equal_weight_pass_at_1": equal["fresh_all_tests_pass_at_1"],
            "eesd_pass_at_1": eesd["fresh_all_tests_pass_at_1"],
            "eesd_vs_equal": json.loads(head_to_head.read_text()),
        })

    report = {
        "schema": "eesd-recursive-study-v1",
        "domain": args.domain,
        "family": args.family,
        "model_config": str(model_config),
        "seed": args.seed,
        "rounds": args.rounds,
        "rules": list(RULES),
        "round0_pass_at_1": json.loads(base_report.read_text())["fresh_all_tests_pass_at_1"],
        "summary": summaries,
        "experience_policy": (
            "round0 shared; later rounds collect arm-specific current-policy originals and corrections"
        ),
        "evaluation_policy": "one fresh greedy primary candidate per source; all tests; no selection",
    }
    (args.output / "recursive-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
