#!/usr/bin/env python3
"""Resumable orchestrator for the locked EESD mechanism and distillation matrix."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from pbpf.eesd.distillation import TRAIN_RULES


def run(command, *, cwd: Path, env=None):
    print(json.dumps({"command": command}), flush=True)
    subprocess.check_call(command, cwd=cwd, env=env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/experiments/eesd_iclr2027.yaml"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["mechanism", "prepare-corrections", "generate-corrections", "score", "train", "fresh", "transfer"], required=True)
    p.add_argument("--rules", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=None, help="one seed; omit to run all locked seeds")
    p.add_argument("--max-steps", type=int, default=200)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load(args.manifest.read_text())
    if manifest.get("schema") != "eesd-cache-manifest-v1":
        raise ValueError("unexpected EESD cache manifest schema")
    locked = yaml.safe_load(args.config.read_text())
    if locked.get("schema") != "eesd-iclr2027-v1":
        raise ValueError("unexpected EESD experiment config")
    seeds = [args.seed] if args.seed is not None else [int(x) for x in locked["seeds"]]
    args.output.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    if args.stage == "mechanism":
        seen = set()
        for cell in manifest.get("mechanism_cells", []):
            key = cell["dataset"], cell["model"]
            if key in seen:
                raise ValueError(f"duplicate mechanism cell: {key}")
            seen.add(key)
            cache = Path(cell["cache"]).resolve()
            if not cache.exists():
                raise FileNotFoundError(cache)
            directory = args.output / "mechanism" / cell["dataset"] / cell["model"]
            if (directory / "complete.json").exists():
                print(json.dumps({"status": "skip_complete", "cell": key}), flush=True)
                continue
            directory.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    python,
                    "scripts/run_eesd_evidence_matrix.py",
                    "--config", str(args.config.resolve()),
                    "--cache", str(cache),
                    "--dataset", cell["dataset"],
                    "--model", cell["model"],
                    "--validation-split", cell.get("validation_split", "development"),
                    "--assessment-split", cell.get("assessment_split", "primary"),
                    "--visible", str(cell.get("visible", 4)),
                    "--output", str(directory.resolve()),
                ],
                cwd=root,
            )
        return

    cells = manifest.get("correction_cells", [])
    if not cells:
        raise ValueError("manifest has no correction_cells")

    if args.stage == "prepare-corrections":
        for cell in cells:
            cache = Path(cell["cache"]).resolve()
            if not cache.exists():
                raise FileNotFoundError(cache)
            name = f"{cell['dataset']}/{cell['model']}/round{cell['round']}"
            directory = args.output / "public-corrections" / name
            if (directory / "audit.json").exists():
                print(json.dumps({"status": "skip_complete", "cell": name}), flush=True)
                continue
            directory.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    python,
                    "scripts/prepare_eesd_public_corrections.py",
                    "--cache", str(cache),
                    "--output", str(directory.resolve()),
                    "--visible", str(cell.get("visible", 4)),
                ],
                cwd=root,
            )
        return

    if args.stage == "generate-corrections":
        for cell in cells:
            name = f"{cell['dataset']}/{cell['model']}/round{cell['round']}"
            public_bank = args.output / "public-corrections" / name / "public-corrections.jsonl"
            if not public_bank.exists():
                raise FileNotFoundError(f"prepare-corrections stage missing: {public_bank}")
            directory = args.output / "generated-corrections" / name
            if (directory / "report.json").exists():
                print(json.dumps({"status": "skip_complete", "cell": name}), flush=True)
                continue
            directory.parent.mkdir(parents=True, exist_ok=True)
            command = [
                python,
                "scripts/generate_eesd_corrections.py",
                "--public-bank", str(public_bank.resolve()),
                "--model-config", str((root / cell["model_config"]).resolve()),
                "--domain", cell["domain"],
                "--output", str(directory.resolve()),
                "--relevance-strength", str(cell.get("relevance_strength", 16.0)),
            ]
            if cell.get("previous_adapter"):
                command += ["--adapter", str(Path(cell["previous_adapter"]).resolve())]
            run(command, cwd=root, env=dict(os.environ))
        return

    if args.stage == "score":
        for cell in cells:
            name = f"{cell['dataset']}/{cell['model']}/round{cell['round']}"
            source = args.output / "generated-corrections" / name / "corrections.jsonl"
            if not source.exists():
                raise FileNotFoundError(f"generate-corrections stage missing: {source}")
            directory = args.output / "corrections" / name
            if (directory / "summary.json").exists():
                print(json.dumps({"status": "skip_complete", "cell": name}), flush=True)
                continue
            directory.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    python,
                    "scripts/score_eesd_corrections.py",
                    "--input", str(source),
                    "--config", str(args.config.resolve()),
                    "--output", str(directory.resolve()),
                    "--alpha", str(cell.get("alpha", 0.1)),
                    "--uncertainty-penalty", str(cell.get("uncertainty_penalty", 0.5)),
                ],
                cwd=root,
            )
        return

    if args.stage == "fresh":
        rules = args.rules or [
            "no_update", "equal_weight", "final_correctness",
            "fixed_mass_dirichlet", "eesd_full",
        ]
        unknown = sorted(set(rules) - set(TRAIN_RULES))
        if unknown:
            raise ValueError(f"invalid fresh-eval rules: {unknown}")
        for cell in manifest.get("fresh_cells", []):
            name = f"{cell['dataset']}/{cell['model']}"
            public_root = Path(cell["public_root"]).resolve()
            evaluator_root = Path(cell["evaluator_root"]).resolve()
            if not public_root.is_dir() or not evaluator_root.is_dir():
                raise FileNotFoundError(f"fresh cell roots missing for {name}")
            for seed in seeds:
                base_report = None
                for rule in rules:
                    bank = args.output / "fresh-banks" / name / rule / f"seed{seed}"
                    if not (bank / "complete.json").exists():
                        bank.parent.mkdir(parents=True, exist_ok=True)
                        script = (
                            "scripts/generate_apbpf_rbr_bank.py"
                            if cell["domain"] == "rbr"
                            else "scripts/generate_apbpf_codearc_bank.py"
                        )
                        command = [
                            python, script,
                            "--public-root", str(public_root),
                            "--output", str(bank.resolve()),
                            "--family", cell["family"],
                            "--split", "primary",
                            "--candidates", "1",
                            "--seed", str(seed),
                            "--greedy",
                        ]
                        if rule != "no_update":
                            adapter = (
                                args.output / "training" / cell["source_dataset"] / cell["model"]
                                / f"round{cell['source_round']}" / rule / f"seed{seed}" / "adapter"
                            )
                            if not adapter.is_dir():
                                raise FileNotFoundError(f"training adapter missing: {adapter}")
                            command += ["--adapter", str(adapter.resolve())]
                        run(command, cwd=root, env=dict(os.environ))
                    evaluation = args.output / "fresh-eval" / name / rule / f"seed{seed}"
                    if not (evaluation / "report.json").exists():
                        evaluation.parent.mkdir(parents=True, exist_ok=True)
                        run(
                            [
                                python, "scripts/evaluate_eesd_fresh_bank.py",
                                "--domain", cell["domain"],
                                "--evaluator-root", str(evaluator_root),
                                "--bank", str(bank.resolve()),
                                "--output", str(evaluation.resolve()),
                            ],
                            cwd=root,
                        )
                    report = evaluation / "report.json"
                    if rule == "no_update":
                        base_report = report
                    else:
                        if base_report is None or not base_report.exists():
                            raise FileNotFoundError("base fresh evaluation must run before comparisons")
                        comparison = (
                            args.output / "fresh-comparison" / name / rule / f"seed{seed}.json"
                        )
                        if not comparison.exists():
                            comparison.parent.mkdir(parents=True, exist_ok=True)
                            run(
                                [
                                    python, "scripts/compare_eesd_fresh_eval.py",
                                    "--baseline", str(base_report.resolve()),
                                    "--method", str(report.resolve()),
                                    "--output", str(comparison.resolve()),
                                ],
                                cwd=root,
                            )
        return

    if args.stage == "transfer":
        rules = args.rules or [
            "no_update", "equal_weight", "final_correctness",
            "fixed_mass_dirichlet", "eesd_full",
        ]
        unknown = sorted(set(rules) - set(TRAIN_RULES))
        if unknown:
            raise ValueError(f"invalid transfer rules: {unknown}")
        for cell in manifest.get("transfer_cells", []):
            name = f"{cell['dataset']}/{cell['model']}"
            for seed in seeds:
                for rule in rules:
                    directory = args.output / "transfer" / name / rule / f"seed{seed}"
                    if (directory / "report.json").exists():
                        print(json.dumps({"status": "skip_complete", "cell": name, "rule": rule, "seed": seed}), flush=True)
                        continue
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        python,
                        "scripts/run_eesd_evalplus_transfer.py",
                        "--dataset", cell["dataset"],
                        "--model-config", str((root / cell["model_config"]).resolve()),
                        "--output", str(directory.resolve()),
                        "--evaluate",
                    ]
                    if rule != "no_update":
                        adapter = (
                            args.output / "training" / cell["source_dataset"] / cell["model"]
                            / f"round{cell['source_round']}" / rule / f"seed{seed}" / "adapter"
                        )
                        if not adapter.is_dir():
                            raise FileNotFoundError(f"training adapter missing: {adapter}")
                        command += ["--adapter", str(adapter.resolve())]
                    run(command, cwd=root, env=dict(os.environ))
        return

    rules = args.rules or [rule for rule in TRAIN_RULES if rule != "no_update"]
    unknown = sorted(set(rules) - set(TRAIN_RULES))
    if unknown or "no_update" in rules:
        raise ValueError(f"invalid train rules: {unknown or ['no_update']}")
    for cell in cells:
        name = f"{cell['dataset']}/{cell['model']}/round{cell['round']}"
        scored = args.output / "corrections" / name / "scored-corrections.jsonl"
        if not scored.exists():
            raise FileNotFoundError(f"score stage missing: {scored}")
        previous = cell.get("previous_adapter")
        for seed in seeds:
            for rule in rules:
                directory = args.output / "training" / name / rule / f"seed{seed}"
                if (directory / "training-report.json").exists():
                    print(json.dumps({"status": "skip_complete", "cell": name, "rule": rule, "seed": seed}), flush=True)
                    continue
                directory.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    python,
                    "scripts/run_eesd_weighted_sft.py",
                    "--input", str(scored.resolve()),
                    "--model-config", str((root / cell["model_config"]).resolve()),
                    "--rule", rule,
                    "--output", str(directory.resolve()),
                    "--seed", str(seed),
                    "--max-steps", str(args.max_steps),
                    "--anchor-beta", str(cell.get("anchor_beta", 0.03)),
                ]
                if previous:
                    command += ["--previous-adapter", str(Path(previous).resolve())]
                run(command, cwd=root, env=dict(os.environ))


if __name__ == "__main__":
    main()
