"""Independent A-PBPF command family; real execution is the default."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

from .config import ROOT, resolve_config
from .stages import GATE_STAGES, STAGES, PipelineError, doctor, report_run, run_pipeline


def _parser():
    parser = argparse.ArgumentParser(prog="pbpf-apbpf", allow_abbrev=False)
    parser.add_argument("command", choices=("doctor", "run", "verify", "report", "rerun-stage"))
    parser.add_argument("--config", default=str(ROOT / "configs/experiments/apbpf_iclr2027.yaml"))
    parser.add_argument("--profile", default="local_24gb")
    parser.add_argument("--site", help="real worker argv/provisioning YAML; defaults to PBPF_APBPF_SITE")
    parser.add_argument("--output-root", default="runs/apbpf")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-exploratory", action="store_true")
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--fail-smoke-gate", choices=GATE_STAGES, help="explicit synthetic gate-stop control; fake backend only")
    return parser


def main(argv=None):
    parser = _parser()
    values = list(sys.argv[1:] if argv is None else argv)
    options = [token.split("=", 1)[0] for token in values if token.startswith("--")]
    if len(options) != len(set(options)):
        parser.error("duplicate options are not allowed")
    args = parser.parse_args(values)
    if (args.command == "rerun-stage") != (args.stage is not None):
        parser.error("--stage is required only for rerun-stage")
    if args.command not in {"run", "rerun-stage"} and (args.resume or args.continue_exploratory or args.fail_smoke_gate):
        parser.error("execution flags apply only to run/rerun-stage")
    try:
        os.umask(0o077)
        resolved = resolve_config(args.config, args.profile, site=args.site or os.environ.get("PBPF_APBPF_SITE"))
        directory = Path(args.output_root).resolve() / resolved.fingerprint
        if args.command == "doctor":
            response = doctor(resolved)
            print(json.dumps(response, sort_keys=True))
            return 0 if response["ready"] else 2
        if args.command in {"run", "rerun-stage"}:
            response = run_pipeline(resolved, directory, resume=args.resume,
                                    continue_exploratory=args.continue_exploratory,
                                    rerun_stage=args.stage, fail_smoke_gate=args.fail_smoke_gate)
        else:
            response = report_run(resolved, directory, require_complete=args.command == "verify")
        print(json.dumps(response, sort_keys=True))
        return 0
    except (PipelineError, ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return getattr(exc, "exit_code", 2)


if __name__ == "__main__":
    raise SystemExit(main())
