#!/usr/bin/env python3
"""Run a multi-task real-Qwen PBPF screen with HumanEval's executable tests."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from evalplus.data import get_human_eval_plus

from pbpf.bank import TrajectoryBank
from pbpf.config import load_experiment
from pbpf.manifest import CandidateVersion, TaskRecord, canonical_hash
from pbpf.models import TransformersRepairBackend
from pbpf.orchestrator import run_arm
from pbpf.registry import OUTCOMES
from pbpf.repair import RepairBudget
from pbpf.sandbox import LocalPythonSandbox


def _clean(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


class Predictor:
    def predict(self, task, candidate, history):
        row = np.full(len(OUTCOMES), 0.1 / (len(OUTCOMES) - 1))
        row[OUTCOMES.index("PASS")] = 0.9
        return {test_id: row for test_id in task["test_order"]}


class Scorer:
    def predict_success(self, task, candidate, history, *, request_text):
        return (sum(item["outcome"] == "PASS" for item in history) + 1) / (len(history) + 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    config = load_experiment(args.config)
    if config["data"]["id"] != "evalplus":
        raise ValueError("EvalPlus screen requires a config whose data.id is evalplus")
    if config["baseline"]["name"] != "pbpf_soft_prompt" or config["arms"] != ["pbpf_soft_prompt"]:
        raise ValueError("EvalPlus diagnostic currently implements only the pbpf_soft_prompt arm")
    from pbpf.config import validate_experiment
    validate_experiment(config, for_execution=True)
    scorer = Scorer()
    backend = TransformersRepairBackend(
        config["model"]["id"], config["model"]["revision"],
        quantization={"load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"},
        success_scorer=scorer,
        generation={"max_new_tokens": args.max_new_tokens, "do_sample": True,
                    "temperature": 0.8, "top_p": 0.95},
    ).load(local_files_only=True)
    torch.manual_seed(0)
    dataset = get_human_eval_plus()
    summaries = []
    selected = list(dataset.values())[args.start_index : args.start_index + args.num_tasks]
    for task_index, raw in enumerate(selected, start=args.start_index):
        generation_prompt = (
            "Complete the following Python function. Return only complete executable Python "
            "source with no Markdown or explanation.\n\n" + raw["prompt"]
        )
        task = TaskRecord(
            raw["task_id"], "evalplus", generation_prompt, ("humaneval-base",),
            (raw["task_id"],), gold_solution=raw["prompt"] + raw["canonical_solution"],
        )
        visible = backend.count_tokens(generation_prompt)
        samples = []
        for sample_index in range(6):
            started = time.perf_counter()
            text, logprob = backend.generate_with_logprob(generation_prompt)
            completion = _clean(text)
            code = completion if f"def {raw['entry_point']}" in completion else raw["prompt"] + completion
            elapsed = time.perf_counter() - started
            samples.append(CandidateVersion.create(
                f"sample-{sample_index}", task.task_id, code,
                provenance="model_sample", generation_logprob=logprob,
                sampling_temperature=0.8, sampling_top_p=0.95,
                model_id=config["model"]["id"], model_revision=config["model"]["revision"],
                generation_trace_hash=canonical_hash({"code": code, "stored_logprob": logprob}),
                visible_tokens=visible, generated_tokens=backend.generated_tokens_for_last_call(),
                wall_seconds=elapsed, gpu_hours=backend.gpu_hours_for_last_call(), model_calls=1,
            ))
        mutant_codes = tuple(samples[i].code + f"\n# deterministic-mutant-{i}\n" for i in range(2))
        registry = {f"append-comment-{i}": canonical_hash({"code": code}) for i, code in enumerate(mutant_codes)}
        registry_hash = canonical_hash(dict(sorted(registry.items())))
        mutants = tuple(CandidateVersion.create(
            f"mutant-{i}", task.task_id, code, provenance="deterministic_mutant",
            generation_logprob=samples[i].generation_logprob, source_ref=f"append-comment-{i}",
            mutation_registry_hash=registry_hash, generation_trace_hash=samples[i].generation_trace_hash,
            source_candidate_hash=samples[i].content_hash, model_id=config["model"]["id"],
            model_revision=config["model"]["revision"],
        ) for i, code in enumerate(mutant_codes))
        bank = TrajectoryBank((task,), tuple(samples) + mutants, (), tuple(sorted(registry.items())))
        harness = raw["test"] + f"\ncheck({raw['entry_point']})\n"
        sandbox = LocalPythonSandbox(
            {"humaneval-base": {"harness": harness}},
            timeout_seconds=3,
            python_executable="/usr/bin/python3",
        )
        truth = {(item.content_hash, "humaneval-base"): sandbox.execute(item.code, "humaneval-base").outcome for item in bank.candidates}
        rng = np.random.default_rng(7000 + task_index)
        hidden = int(backend.model.config.hidden_size)
        particles = tuple({"id": f"particle-{i}", "soft_prompt": rng.normal(0, 1e-3, (1, hidden)).astype(np.float32)} for i in range(8))
        flat = lambda observation, particles, weights: np.zeros(len(particles), dtype=np.float64)
        task_output = args.output / raw["task_id"].replace("/", "-")
        report = run_arm(
            config=config, bank=bank, predictor=Predictor(), scorer=scorer,
            sequence_backend=backend, sandbox=sandbox, prefix_cutoffs={task.task_id: 0},
            future_outcomes_by_candidate=truth, particles=particles,
            previous_log_weights=np.full(8, -np.log(8)), log_likelihood=flat,
            log_transition=flat, log_proposal=flat, budget=RepairBudget(**config["budget"]),
            gate_evidence=None, rng=rng, artifact_path=task_output,
        )
        summaries.append({"task_id": task.task_id, "initial_passes": sum(value == "PASS" for value in truth.values()),
                          "round_pass_at_1": report["round_pass_at_1"], "final_pass_at_1": report["final_pass_at_1"],
                          "artifact": str(task_output)})
        print(json.dumps(summaries[-1], sort_keys=True), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
