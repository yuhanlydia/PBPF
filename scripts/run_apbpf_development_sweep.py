#!/usr/bin/env python3
"""Run a predeclared, development-only A-PBPF diagnostic parameter sweep."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from pbpf.apbpf.development import development_cache
from pbpf.real_gate import validate_rbr_cache


VARIANTS = {
    "expanded_default": [],
    "association10": ["--association-weight", "10"],
    "association10_invariance1": ["--association-weight", "10", "--invariance-weight", "1"],
    "richer_semantics": ["--association-weight", "10", "--invariance-weight", "1",
                         "--feature-dim", "1024", "--hidden-dim", "256"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1701)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    payload = json.loads(args.source_cache.read_text())
    cache = development_cache(payload)
    validate_rbr_cache(cache)
    cache_path = args.output / "development_cache.json"
    cache_path.write_text(json.dumps(cache, sort_keys=True) + "\n")
    manifest = {"schema": "apbpf-development-sweep-v1", "evaluation_role": "development_assessment_only",
                "steps": args.steps, "seed": args.seed, "variants": VARIANTS,
                "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                "source_cache_sha256": hashlib.sha256(args.source_cache.read_bytes()).hexdigest(),
                "counts": cache["counts"], "problem_counts": cache["problem_counts"],
                "selection_policy": "assess association, invariance and baseline fairness on development only; no confirmatory test claim",
                "source_sha256": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in [root / "scripts/run_rbr_prediction_gate.py", *sorted((root / "src/pbpf").rglob("*.py"))]}}
    (args.output / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    queue = list(VARIANTS.items()); running = {}; finished = {}
    try:
        while queue or running:
            for gpu in args.gpus:
                if gpu in running or not queue:
                    continue
                name, flags = queue.pop(0)
                command = ["bash", "local/sandbox.sh", str(root / ".venv/bin/python"), "-u",
                           "scripts/run_rbr_prediction_gate.py", "--cache", str(cache_path),
                           "--output", str(args.output / f"{name}.json"), "--apbpf", "--expected-is-public",
                           "--seed", str(args.seed), "--steps", str(args.steps), *flags]
                log = (args.output / f"{name}.log").open("w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                           PYTHONUNBUFFERED="1")
                env.pop("PYTHONPATH", None); env.pop("PYTHONHOME", None)
                process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                running[gpu] = (name, process, log, command)
                print(json.dumps({"started": name, "gpu": gpu, "pid": process.pid}), flush=True)
            for gpu, (name, process, log, command) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                result = {"returncode": code, "command": command}
                if code == 0:
                    report = json.loads((args.output / f"{name}.json").read_text())
                    result.update(gate=report["gate"], metrics=report["metrics"],
                                  association=report["cluster_bootstrap"]["outcome_shuffled"],
                                  evaluation_role=report["evaluation_role"])
                finished[name] = result
                del running[gpu]
                print(json.dumps({"finished": name, "returncode": code}), flush=True)
            state = {"scope": "development-only nonconfirmatory", "completed": finished,
                     "running": {name: {"pid": process.pid, "gpu": gpu}
                                 for gpu, (name, process, _, _) in running.items()},
                     "pending": [name for name, _ in queue]}
            temporary = args.output / "status.tmp"
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(args.output / "status.json")
            if running:
                time.sleep(10)
    finally:
        for _, process, log, _ in running.values():
            if process.poll() is None:
                process.terminate()
            log.close()
    if any(r["returncode"] for r in finished.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
