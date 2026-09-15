#!/usr/bin/env python3
"""Train and evaluate the real RunBugRun PBPF prediction gate.

This is a local scientific gate, not the sealed ICLR evaluator.  It uses the
official immutable v0.0.1 files, keeps problem IDs disjoint across fitting and
evaluation, executes every candidate, and compares matched predictions made
with and without the first four outcomes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import gzip
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from pbpf.belief.features import BeliefBatch
from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, classify_execution, compare_predictions, html_to_text
from pbpf.registry import OUTCOMES
from pbpf.train_belief import train_belief_step


FILES = {
    "train": ("python_train0.jsonl.gz", "python_train1.jsonl.gz", "python_train2.jsonl.gz"),
    "test": ("python_test0.jsonl.gz",),
}
EXPECTED_MD5 = {
    "python_train0.jsonl.gz": "de4df37fe33753acc6310bd0c48d5e41",
    "python_train1.jsonl.gz": "fbafa0786576e3b49bc795de2b304b4c",
    "python_train2.jsonl.gz": "57c614050be4745c2c2337e78e410a2a",
    "python_test0.jsonl.gz": "375f056565af619146c12e0e63e3974c",
    "tests_all.jsonl.gz": "6eaac2b6a535aa8b5213489c6511cb26",
}
DESCRIPTION_SOURCE = "IBM Project CodeNet problem_descriptions.tar.gz"
DESCRIPTION_ARCHIVE_SHA256 = "8b631ae168ba84dce69c7d8e1b6c632256e0155858c2664c6493a5f001b45fdd"
DATASET_SCHEMA = "pbpf-rbr-real-gate-v2"


def _md5(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _rank(seed: int, *values: object) -> bytes:
    return hashlib.sha256((str(seed) + "\0" + "\0".join(map(str, values))).encode()).digest()


def _load_jsonl(path: Path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def _output_matches(actual: str, expected: str, *, tolerance: float = 1e-4) -> bool:
    if actual.rstrip() == expected.rstrip():
        return True
    actual_lines, expected_lines = actual.rstrip().splitlines(), expected.rstrip().splitlines()
    if len(actual_lines) != len(expected_lines):
        return False
    for left, right in zip(actual_lines, expected_lines):
        left_items, right_items = left.split(), right.split()
        if len(left_items) != len(right_items):
            return False
        for a, b in zip(left_items, right_items):
            if a == b:
                continue
            try:
                if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance):
                    return False
            except ValueError:
                return False
    return True


def _execute(payload):
    code, cases, timeout = payload
    outcomes = []
    try:
        compile(code, "candidate.py", "exec")
    except (SyntaxError, ValueError, TypeError):
        return ["COMPILE_ERROR"] * len(cases)
    with tempfile.TemporaryDirectory(prefix="pbpf-rbr-") as directory:
        source = Path(directory) / "candidate.py"
        source.write_text(code)
        for case in cases:
            try:
                result = subprocess.run(
                    ["/usr/bin/python3", "-I", str(source)], input=case["input"], text=True,
                    capture_output=True, timeout=timeout, cwd=directory,
                    env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"},
                )
                outcome = classify_execution(result.returncode, result.stdout, result.stderr, case["output"])
                if result.returncode == 0:
                    outcome = "PASS" if _output_matches(result.stdout, case["output"]) else "WRONG_OUTPUT"
            except subprocess.TimeoutExpired:
                outcome = classify_execution(None, "", "", case["output"], timed_out=True)
            outcomes.append(outcome)
    return outcomes


def prepare(root: Path, descriptions_root: Path, descriptions_archive: Path, cache: Path, *,
            train_problems: int, dev_problems: int,
            test_problems: int, candidates_per_problem: int, tests_per_candidate: int,
            workers: int, timeout: float, seed: int) -> dict:
    for name, expected in EXPECTED_MD5.items():
        path = root / name
        if not path.is_file() or _md5(path) != expected:
            raise ValueError(f"missing or checksum-mismatched official file: {path}")
    if (not descriptions_archive.is_file()
            or _sha256(descriptions_archive) != DESCRIPTION_ARCHIVE_SHA256):
        raise ValueError(f"missing or checksum-mismatched CodeNet descriptions: {descriptions_archive}")
    descriptions = {
        path.stem: html_to_text(path.read_text(errors="replace"))[:6000]
        for path in descriptions_root.glob("p*.html")
    }
    descriptions = {key: value for key, value in descriptions.items() if value.strip()}
    if not descriptions:
        raise ValueError(f"no CodeNet problem descriptions found under {descriptions_root}")
    tests = defaultdict(list)
    for row in _load_jsonl(root / "tests_all.jsonl.gz"):
        if len(row["input"]) <= 8192 and len(row["output"]) <= 4096:
            tests[row["problem_id"]].append(row)
    for values in tests.values():
        values.sort(key=lambda row: _rank(seed, "test", row["id"]))

    bugs = {}
    problem_sets = {}
    for split, names in FILES.items():
        rows = []
        for name in names:
            rows.extend(_load_jsonl(root / name))
        rows = [row for row in rows if row["problem_id"] in descriptions
                and len(row["buggy_code"]) <= 8192
                and len(tests[row["problem_id"]]) >= tests_per_candidate]
        bugs[split] = rows
        problem_sets[split] = {row["problem_id"] for row in rows}
    train_only = problem_sets["train"] - problem_sets["test"]
    test_only = problem_sets["test"] - problem_sets["train"]
    ranked_train = sorted(train_only, key=lambda value: _rank(seed, "train-problem", value))
    ranked_test = sorted(test_only, key=lambda value: _rank(seed, "test-problem", value))
    selected = {
        "train": set(ranked_train[:train_problems]),
        "development": set(ranked_train[train_problems:train_problems + dev_problems]),
        "test": set(ranked_test[:test_problems]),
    }
    if any(len(selected[name]) != cap for name, cap in (("train", train_problems), ("development", dev_problems), ("test", test_problems))):
        raise ValueError("requested caps exceed source-disjoint eligible problems")

    jobs = []
    for target_split, source_split in (("train", "train"), ("development", "train"), ("test", "test")):
        grouped = defaultdict(list)
        for row in bugs[source_split]:
            if row["problem_id"] in selected[target_split]:
                grouped[row["problem_id"]].append(row)
        for problem_id in sorted(grouped):
            rows = sorted(grouped[problem_id], key=lambda row: _rank(seed, "bug", row["id"]))[:candidates_per_problem]
            cases = tests[problem_id][:tests_per_candidate]
            for row in rows:
                jobs.append((target_split, row, cases))
    print(json.dumps({"phase": "execute", "jobs": len(jobs), "workers": workers}), flush=True)
    payloads = [(code, cases, timeout) for _, row, cases in jobs for code in (row["buggy_code"], row["fixed_code"])]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(_execute, payloads, chunksize=1))
    records = []
    paired_outcomes = zip(outcomes[::2], outcomes[1::2])
    rejected_fixed = 0
    for (split, row, cases), (result, fixed_result) in zip(jobs, paired_outcomes):
        if any(value != "PASS" for value in fixed_result):
            rejected_fixed += 1
            continue
        if all(value == "PASS" for value in result):
            continue
        records.append({
            "task_id": str(row["id"]), "problem_id": row["problem_id"], "split": split,
            "task_text": descriptions[row["problem_id"]],
            "candidate": row["buggy_code"],
            "tests": [{"id": str(case["id"]), "input": case["input"]} for case in cases],
            "outcomes": result,
        })
    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": DATASET_SCHEMA, "seed": seed, "tests_per_candidate": tests_per_candidate,
        "official_md5": EXPECTED_MD5, "records": records,
        "problem_descriptions": {"source": DESCRIPTION_SOURCE,
                                 "archive_sha256": DESCRIPTION_ARCHIVE_SHA256,
                                 "maximum_characters": 6000},
        "counts": dict(Counter(row["split"] for row in records)),
        "problem_counts": {split: len({row["problem_id"] for row in records if row["split"] == split}) for split in selected},
        "rejected_fixed_candidates": rejected_fixed,
    }
    cache.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def _tensorize(rows, encoder, device):
    task = np.stack([encoder(row["task_text"]) for row in rows])
    candidate = np.stack([encoder(row["candidate"]) for row in rows])
    tests = np.stack([[encoder(case["input"]) for case in row["tests"]] for row in rows])
    outcomes = np.asarray([[OUTCOMES.index(value) for value in row["outcomes"]] for row in rows])
    return BeliefBatch(
        torch.tensor(task, device=device), torch.tensor(candidate, device=device),
        torch.tensor(tests, device=device), torch.tensor(outcomes, dtype=torch.long, device=device),
    )


def _subset(batch, indices):
    return BeliefBatch(batch.task[indices], batch.candidate[indices], batch.tests[indices], batch.outcomes[indices])


@torch.no_grad()
def _predict(model, batch, *, particles, seed, mode, problem_ids=None):
    generator = torch.Generator(device=batch.task.device).manual_seed(seed)
    source = batch
    if mode == "shuffled":
        outcomes = batch.outcomes.clone()
        outcomes[:, :4] = outcomes[:, :4].flip(1)
        source = BeliefBatch(batch.task, batch.candidate, batch.tests, outcomes)
    elif mode == "wrong_candidate":
        if problem_ids is None or len(problem_ids) != len(batch.task):
            raise ValueError("wrong-candidate control requires one problem ID per candidate")
        groups = defaultdict(list)
        for index, problem_id in enumerate(problem_ids):
            groups[problem_id].append(index)
        donor = list(range(len(problem_ids)))
        for indices in groups.values():
            if len(indices) > 1:
                for offset, index in enumerate(indices):
                    donor[index] = indices[(offset + 1) % len(indices)]
        outcomes = batch.outcomes.clone()
        outcomes[:, :4] = batch.outcomes[torch.tensor(donor, device=batch.task.device), :4]
        source = BeliefBatch(batch.task, batch.candidate, batch.tests, outcomes)
    trace = model.filter(source, particles=particles, visible_steps=4, ess_fraction=0.5, generator=generator)
    z, weights = trace.latents[:, 3], trace.log_weights[:, 3]
    if mode == "baseline":
        prior = model.root(batch.task, batch.candidate)
        noise = torch.randn((len(batch.task), particles, model.latent_dim), device=batch.task.device, generator=generator)
        z = prior.mean[:, None] + prior.std[:, None] * noise
        weights = batch.task.new_full((len(batch.task), particles), -math.log(particles))
    elif mode == "random":
        radius = torch.sqrt((weights.exp() * z.square().sum(-1)).sum(-1, keepdim=True))
        z = torch.randn(z.shape, device=z.device, generator=generator)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8) * radius[:, None]
    future = batch.tests[:, 4:]
    return model.future_predict(batch.task, batch.candidate, future, z, weights).exp().reshape(-1, 5).cpu().numpy()


def _history_rate(batch):
    visible = batch.outcomes[:, :4].cpu().numpy()
    rows = []
    for outcomes in visible:
        counts = np.bincount(outcomes, minlength=5).astype(np.float64) + 1.0
        rows.extend([counts / counts.sum()] * (batch.outcomes.shape[1] - 4))
    return np.asarray(rows)


def _cluster_bootstrap(labels, predictions, problem_ids, *, comparator="baseline", seed, replicates=10_000):
    labels = np.asarray(labels)
    per_example = {}
    for name, probabilities in predictions.items():
        per_example[name] = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0))
    width = len(labels) // len(problem_ids)
    if width * len(problem_ids) != len(labels):
        raise ValueError("future examples must have a fixed count per candidate")
    groups = defaultdict(list)
    for index, problem_id in enumerate(problem_ids):
        groups[problem_id].extend(range(index * width, (index + 1) * width))
    keys = sorted(groups)
    gains = np.asarray([
        np.mean(per_example[comparator][groups[key]] - per_example["pbpf"][groups[key]])
        for key in keys
    ])
    rng = np.random.default_rng(seed)
    draws = gains[rng.integers(0, len(gains), size=(replicates, len(gains)))].mean(1)
    return {
        "unit": "problem_id", "problems": len(keys), "replicates": replicates,
        "comparator": comparator, "mean_nll_gain": float(gains.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def train_and_evaluate(payload: dict, output: Path, *, feature_dim: int, latent_dim: int,
                       hidden_dim: int, particles: int, steps: int, batch_size: int,
                       learning_rate: float, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrozenTextEncoder(feature_dim)
    rows = payload["records"]
    by_split = {name: [row for row in rows if row["split"] == name] for name in ("train", "development", "test")}
    batches = {name: _tensorize(values, encoder, device) for name, values in by_split.items()}
    model = NeuralBeliefModel(feature_dim, latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    best, best_state, history = float("inf"), None, []
    for step in range(1, steps + 1):
        indices = torch.tensor(rng.integers(0, len(by_split["train"]), size=batch_size), device=device)
        metrics = train_belief_step(model, _subset(batches["train"], indices), optimizer, particles=particles)
        if step == 1 or step % max(1, steps // 20) == 0:
            model.eval()
            labels = batches["development"].outcomes[:, 4:].reshape(-1).cpu().numpy()
            predictions = {mode: _predict(model, batches["development"], particles=particles, seed=seed + 50_000, mode=mode)
                           for mode in ("baseline", "pbpf")}
            validation = compare_predictions(labels, predictions)
            row = {"step": step, **metrics, "validation": validation}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if validation["pbpf"]["nll"] < best:
                best = validation["pbpf"]["nll"]
                best_state = copy.deepcopy(model.state_dict())
                for key, value in tuple(best_state.items()):
                    if isinstance(value, torch.Tensor):
                        best_state[key] = value.cpu()
            model.train()
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    batch = batches["test"]
    labels = batch.outcomes[:, 4:].reshape(-1).cpu().numpy()
    problem_ids = [row["problem_id"] for row in by_split["test"]]
    predictions = {mode: _predict(model, batch, particles=particles, seed=seed + 100_000, mode=mode,
                                  problem_ids=problem_ids)
                   for mode in ("baseline", "pbpf", "shuffled", "wrong_candidate", "random")}
    predictions["history_rate"] = _history_rate(batch)
    metrics = compare_predictions(labels, predictions)
    bootstrap = {
        comparator: _cluster_bootstrap(labels, predictions, problem_ids, comparator=comparator,
                                       seed=seed + 200_000 + index)
        for index, comparator in enumerate(("baseline", "history_rate"))
    }
    wrong_eligible = sum(count > 1 for count in Counter(problem_ids).values())
    future_width = batch.outcomes.shape[1] - 4
    nonconstant_candidates = np.asarray([len(set(row["outcomes"][:4])) > 1 for row in by_split["test"]])
    multi_candidate = np.asarray([Counter(problem_ids)[problem_id] > 1 for problem_id in problem_ids])
    subset_metrics = {}
    for name, candidate_mask in (("nonconstant_visible", nonconstant_candidates),
                                 ("multi_candidate_problem", multi_candidate)):
        example_mask = np.repeat(candidate_mask, future_width)
        subset_metrics[name] = {
            "candidates": int(candidate_mask.sum()), "future_examples": int(example_mask.sum()),
            "metrics": compare_predictions(labels[example_mask],
                {arm: values[example_mask] for arm, values in predictions.items()}),
        }
    controls_worse = all(metrics["pbpf"]["nll"] < metrics[name]["nll"]
                         for name in ("shuffled", "wrong_candidate", "random"))
    paired_positive = all(row["ci95"][0] > 0 for row in bootstrap.values())
    report = {
        "schema": "pbpf-rbr-gate-b-result-v1", "device": str(device), "seed": seed,
        "records": {name: len(values) for name, values in by_split.items()},
        "problems": {name: len({row["problem_id"] for row in values}) for name, values in by_split.items()},
        "future_examples": len(labels), "best_validation_nll": best, "metrics": metrics,
        "gate": {
            "relative_nll_gain": (metrics["baseline"]["nll"] - metrics["pbpf"]["nll"]) / metrics["baseline"]["nll"],
            "passes_point_estimate": metrics["pbpf"]["nll"] < min(metrics["baseline"]["nll"], metrics["history_rate"]["nll"]),
            "paired_lower_bound_above_zero": paired_positive,
            "controls_worse_than_pbpf": controls_worse,
            "full_gate_passes": paired_positive and controls_worse,
        },
        "cluster_bootstrap": bootstrap,
        "wrong_candidate_multi_candidate_problems": wrong_eligible,
        "subset_metrics": subset_metrics,
        "training_history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    torch.save({"model": best_state, "feature_dim": feature_dim, "latent_dim": latent_dim,
                "hidden_dim": hidden_dim, "seed": seed}, output.with_suffix(".pt"))
    print(json.dumps({key: value for key, value in report.items() if key != "training_history"}, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/pbpf_external/runbugrun-v0.0.1"))
    parser.add_argument("--descriptions-root", type=Path,
                        default=Path("/root/pbpf_external/project_codenet/problem_descriptions"))
    parser.add_argument("--descriptions-archive", type=Path,
                        default=Path("/root/pbpf_external/project_codenet/problem_descriptions.tar.gz"))
    parser.add_argument("--cache", type=Path, default=Path("results/rbr_real_gate_dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("results/rbr_gate_b_result.json"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train-problems", type=int, default=160)
    parser.add_argument("--dev-problems", type=int, default=32)
    parser.add_argument("--test-problems", type=int, default=96)
    parser.add_argument("--candidates-per-problem", type=int, default=4)
    parser.add_argument("--tests-per-candidate", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    payload = prepare(args.data_root, args.descriptions_root, args.descriptions_archive, args.cache,
        train_problems=args.train_problems,
        dev_problems=args.dev_problems, test_problems=args.test_problems,
        candidates_per_problem=args.candidates_per_problem, tests_per_candidate=args.tests_per_candidate,
        workers=args.workers, timeout=args.timeout, seed=args.seed) if args.prepare or not args.cache.exists() else json.loads(args.cache.read_text())
    if payload.get("schema") != DATASET_SCHEMA or any("task_text" not in row for row in payload.get("records", ())):
        raise ValueError("dataset cache predates CodeNet descriptions; rerun with --prepare")
    train_and_evaluate(payload, args.output, feature_dim=args.feature_dim, latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim, particles=args.particles, steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed)


if __name__ == "__main__":
    main()
