#!/usr/bin/env python3
"""Build and execute one real-model, real-evaluator PBPF pilot task."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from pbpf.bank import TrajectoryBank
from pbpf.config import load_experiment
from pbpf.manifest import CandidateVersion, TaskRecord, canonical_hash
from pbpf.models import TransformersRepairBackend
from pbpf.orchestrator import run_arm
from pbpf.registry import OUTCOMES
from pbpf.repair import RepairBudget
from pbpf.sandbox import LocalPythonSandbox


def clean_code(text: str) -> str:
    import re
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


class OutcomePredictor:
    def predict(self, public_task, candidate, history):
        row = np.full(len(OUTCOMES), 0.05, dtype=np.float64)
        row[OUTCOMES.index("PASS")] = 0.8
        return {test_id: row for test_id in public_task["test_order"]}


class StatusScorer:
    def predict_success(self, task, candidate, history, *, request_text):
        return (sum(item["outcome"] == "PASS" for item in history) + 1.0) / (len(history) + 2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilots/local_real_7b.yaml"))
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    config = load_experiment(args.config)
    rows = pq.read_table(args.parquet).to_pylist()
    row = next(item for item in rows if int(item["id"]) == 6584)
    expected = subprocess.run(
        ["/usr/bin/python3", "-I", "-c", row["fixed_code"]],
        text=True, capture_output=True, timeout=5, check=True,
    ).stdout
    task = TaskRecord(
        task_id="runbugrun-6584",
        dataset="runbugrun",
        public_prompt=(
            "Repair this Python program. It must print the 1 through 9 multiplication table, "
            "one product per line in nested i-then-j order, formatted i x j=result using the "
            "literal lowercase x. Return only the complete corrected Python source.\n\n"
            + row["buggy_code"]
        ),
        test_order=("hidden-stdio-0",),
        group_ids=(str(row["problem_id"]), str(row["user_id"])),
        gold_solution=row["fixed_code"],
    )
    backend = TransformersRepairBackend(
        config["model"]["id"], config["model"]["revision"],
        quantization={"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"},
        success_scorer=StatusScorer(),
        generation={"max_new_tokens": args.max_new_tokens, "do_sample": True,
                    "temperature": 0.8, "top_p": 0.95},
    ).load(local_files_only=True)
    torch.manual_seed(0)
    visible = backend.count_tokens(task.public_prompt)
    samples = []
    for index in range(6):
        started = time.perf_counter()
        text, logprob = backend.generate_with_logprob(task.public_prompt)
        code = clean_code(text)
        elapsed = time.perf_counter() - started
        samples.append(CandidateVersion.create(
            f"sample-{index}", task.task_id, code,
            provenance="model_sample", generation_logprob=logprob,
            sampling_temperature=0.8, sampling_top_p=0.95,
            model_id=config["model"]["id"], model_revision=config["model"]["revision"],
            generation_trace_hash=canonical_hash({"code": code, "stored_logprob": logprob}),
            visible_tokens=visible, generated_tokens=backend.generated_tokens_for_last_call(),
            wall_seconds=elapsed, gpu_hours=backend.gpu_hours_for_last_call(), model_calls=1,
        ))
    mutant_codes = (samples[0].code + "\n# deterministic-mutant-0", samples[1].code + "\n# deterministic-mutant-1")
    mutant_registry = {f"append-comment-{i}": canonical_hash({"code": code}) for i, code in enumerate(mutant_codes)}
    registry_hash = canonical_hash(dict(sorted(mutant_registry.items())))
    mutants = tuple(CandidateVersion.create(
        f"mutant-{i}", task.task_id, code, provenance="deterministic_mutant",
        generation_logprob=samples[i].generation_logprob, source_ref=f"append-comment-{i}",
        mutation_registry_hash=registry_hash, generation_trace_hash=samples[i].generation_trace_hash,
        source_candidate_hash=samples[i].content_hash, model_id=config["model"]["id"],
        model_revision=config["model"]["revision"],
    ) for i, code in enumerate(mutant_codes))
    bank = TrajectoryBank((task,), tuple(samples) + mutants, (), tuple(sorted(mutant_registry.items())))
    sandbox = LocalPythonSandbox(
        {"hidden-stdio-0": {"input": "", "output": expected}},
        python_executable="/usr/bin/python3",
    )
    truth = {(c.content_hash, "hidden-stdio-0"): sandbox.execute(c.code, "hidden-stdio-0").outcome for c in bank.candidates}
    hidden = int(backend.model.config.hidden_size)
    particle_rng = np.random.default_rng(7)
    particles = tuple(
        {
            "id": f"particle-{i}",
            "soft_prompt": particle_rng.normal(0.0, 1e-3, size=(1, hidden)).astype(np.float32),
        }
        for i in range(8)
    )
    uniform = lambda observation, particles, weights: np.zeros(len(particles), dtype=np.float64)
    report = run_arm(
        config=config, bank=bank, predictor=OutcomePredictor(), scorer=StatusScorer(),
        sequence_backend=backend, sandbox=sandbox, prefix_cutoffs={task.task_id: 0},
        future_outcomes_by_candidate=truth, particles=particles,
        previous_log_weights=np.full(8, -np.log(8)), log_likelihood=uniform,
        log_transition=uniform, log_proposal=uniform,
        budget=RepairBudget(**config["budget"]), gate_evidence=None,
        rng=np.random.default_rng(0), artifact_path=args.output,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
