#!/usr/bin/env python3
"""Score original->correction trajectories for every locked EESD training rule.

Input JSONL fields:
  trajectory_id, source_component_id, prompt, correction,
  before_outcomes, after_outcomes, relevance

Outcomes may be integer registry indices or strings from pbpf.registry.OUTCOMES.
The script never drops a trajectory; zero-weight rows remain in the scored file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from pbpf.eesd.distillation import TRAIN_RULES, score_trajectory
from pbpf.registry import OUTCOMES


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_indices(values):
    result = []
    for value in values:
        if isinstance(value, str):
            if value not in OUTCOMES:
                raise ValueError(f"unknown outcome: {value}")
            value = OUTCOMES.index(value)
        if type(value) is not int or not 0 <= value < len(OUTCOMES):
            raise ValueError("outcome must be a registry string or integer index")
        result.append(value)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/experiments/eesd_iclr2027.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--uncertainty-penalty", type=float, default=0.5)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("schema") != "eesd-iclr2027-v1":
        raise ValueError("unexpected EESD config")
    utility = [float(x) for x in cfg["distillation"]["transition_utility"]]
    if len(utility) != 4:
        raise ValueError("transition utility must contain four entries")

    rows = []
    with args.input.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "trajectory_id", "source_component_id", "prompt", "correction",
                "before_outcomes", "after_outcomes", "relevance",
            }
            if required - set(row):
                raise ValueError(f"line {line_no} missing required trajectory fields")
            before = to_indices(row["before_outcomes"])
            after = to_indices(row["after_outcomes"])
            if len(before) != len(after) or len(before) != len(row["relevance"]):
                raise ValueError(f"line {line_no} trajectory vector lengths differ")
            scored = score_trajectory(
                before, after, row["relevance"], alpha=args.alpha, utility=utility,
                uncertainty_penalty=args.uncertainty_penalty,
            )
            rows.append({
                **row,
                "before_outcomes": before,
                "after_outcomes": after,
                "training_weights": scored.pop("weights"),
                "eesd_diagnostics": scored,
            })

    if not rows:
        raise ValueError("no correction trajectories found")
    if len({row["trajectory_id"] for row in rows}) != len(rows):
        raise ValueError("trajectory_id must be unique")

    args.output.mkdir(parents=True, exist_ok=False)
    scored_path = args.output / "scored-corrections.jsonl"
    with scored_path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    summary = {
        "schema": "eesd-scored-corrections-v1",
        "input_sha256": sha(args.input),
        "config_sha256": sha(args.config),
        "alpha": args.alpha,
        "uncertainty_penalty": args.uncertainty_penalty,
        "trajectories": len(rows),
        "sources": len({row["source_component_id"] for row in rows}),
        "rules": {},
    }
    for rule in TRAIN_RULES:
        weights = [float(row["training_weights"][rule]) for row in rows]
        summary["rules"][rule] = {
            "positive": sum(w > 0 for w in weights),
            "zero": sum(w == 0 for w in weights),
            "mean_weight": sum(weights) / len(weights),
            "max_weight": max(weights),
        }
    summary["scored_sha256"] = sha(scored_path)
    (args.output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
