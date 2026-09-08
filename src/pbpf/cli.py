from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Sequence

from .config import canonical_config_hash, dependency_status, load_experiment, validate_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pbpf-run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="check dependencies without importing optional ML stacks")
    plan = subparsers.add_parser("plan", help="validate and print an execution plan")
    plan.add_argument("config", type=Path)
    smoke = subparsers.add_parser("smoke", help="execute the deterministic no-download arm")
    smoke.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(dependency_status(), sort_keys=True))
        return 0
    if args.command == "smoke":
        from .orchestrator import run_deterministic_smoke

        if args.output is not None:
            report = run_deterministic_smoke(args.output)
        else:
            with tempfile.TemporaryDirectory(prefix="pbpf-smoke-") as directory:
                report = run_deterministic_smoke(Path(directory) / "report")
        print(json.dumps(report, sort_keys=True))
        return 0
    config = validate_experiment(load_experiment(args.config))
    print(
        json.dumps(
            {
                "config_hash": canonical_config_hash(config),
                "formal": bool(config["formal"]),
                "name": config["name"],
                "profile": config["profile"],
                "status": "validated; no run executed",
            },
            sort_keys=True,
        )
    )
    return 0
