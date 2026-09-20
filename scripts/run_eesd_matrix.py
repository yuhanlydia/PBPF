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
    p.add_argument("--stage", choices=["mechanism", "score", "train"], required=True)
    p.add_argument("--rules", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--max-steps", type=int, default=200)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load(args.manifest.read_text())
    if manifest.get("schema") != "eesd-cache-manifest-v1":
        raise ValueError("unexpected EESD cache manifest schema")
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

    if args.stage == "score":
        for cell in cells:
            source = Path(cell["corrections"]).resolve()
            if not source.exists():
                raise FileNotFoundError(source)
            name = f"{cell['dataset']}/{cell['model']}/round{cell['round']}"
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
        for rule in rules:
            directory = args.output / "training" / name / rule / f"seed{args.seed}"
            if (directory / "training-report.json").exists():
                print(json.dumps({"status": "skip_complete", "cell": name, "rule": rule}), flush=True)
                continue
            directory.parent.mkdir(parents=True, exist_ok=True)
            command = [
                python,
                "scripts/run_eesd_weighted_sft.py",
                "--input", str(scored.resolve()),
                "--model-config", str((root / cell["model_config"]).resolve()),
                "--rule", rule,
                "--output", str(directory.resolve()),
                "--seed", str(args.seed),
                "--max-steps", str(args.max_steps),
                "--anchor-beta", str(cell.get("anchor_beta", 0.03)),
            ]
            if previous:
                command += ["--previous-adapter", str(Path(previous).resolve())]
            run(command, cwd=root, env=dict(os.environ))


if __name__ == "__main__":
    main()
