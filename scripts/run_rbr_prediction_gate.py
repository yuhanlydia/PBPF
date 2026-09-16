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
from pbpf.real_gate import (FrozenTextEncoder, classify_execution, compare_predictions, html_to_text,
                           RBR_CACHE_SCHEMA, bounded_execution_record, validate_rbr_cache,
                           public_test_text, clustered_nll_gap, association_strata)
from pbpf.apbpf.counterfactual import outcome_derangement, joint_permutation
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
DATASET_SCHEMA = RBR_CACHE_SCHEMA


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
    except (SyntaxError, ValueError, TypeError) as error:
        return [bounded_execution_record(case, stderr=str(error), outcome="COMPILE_ERROR") for case in cases]
    with tempfile.TemporaryDirectory(prefix="pbpf-rbr-") as directory:
        source = Path(directory) / "candidate.py"
        source.write_text(code)
        for case in cases:
            actual, stderr, returncode, timed_out = "", "", None, False
            try:
                result = subprocess.run(
                    ["/usr/bin/python3", "-I", str(source)], input=case["input"], text=True,
                    capture_output=True, timeout=timeout, cwd=directory,
                    env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"},
                )
                outcome = classify_execution(result.returncode, result.stdout, result.stderr, case["output"])
                actual, stderr, returncode = result.stdout, result.stderr, result.returncode
                if result.returncode == 0:
                    outcome = "PASS" if _output_matches(result.stdout, case["output"]) else "WRONG_OUTPUT"
            except subprocess.TimeoutExpired as error:
                actual, stderr, timed_out = error.stdout or "", error.stderr or "", True
                outcome = classify_execution(None, "", "", case["output"], timed_out=True)
            outcomes.append(bounded_execution_record(case, actual=actual, stderr=stderr,
                returncode=returncode, timed_out=timed_out, outcome=outcome))
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
        if any(value["outcome"] != "PASS" for value in fixed_result):
            rejected_fixed += 1
            continue
        records.append({
            "task_id": str(row["id"]), "problem_id": row["problem_id"], "split": split,
            "source_component_id": row["problem_id"],
            "task_text": descriptions[row["problem_id"]],
            "candidate": row["buggy_code"],
            "tests": [{"id": str(case["id"]), "input": case["input"], **execution}
                      for case, execution in zip(cases, result, strict=True)],
            "outcomes": [value["outcome"] for value in result],
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
        "population_policy": "all selected buggy candidates, including all-pass; fixed-program validity screen",
        "execution_fields_visibility": "evaluator-only unless explicitly whitelisted by protocol",
    }
    cache.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def _tensorize(rows, encoder, device, *, expected_is_public=False):
    task = np.stack([encoder(row["task_text"]) for row in rows])
    candidate = np.stack([encoder(row["candidate"]) for row in rows])
    tests = np.stack([[encoder(public_test_text(case, expected_is_public=expected_is_public))
                       for case in row["tests"]] for row in rows])
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
    mode = {"pbpf": "aligned", "shuffled": "outcome_shuffled", "random": "random_latent"}.get(mode, mode)
    if mode not in {"baseline", "aligned", "outcome_shuffled", "joint_reversed", "presentation_permuted",
                    "orderless", "semantics_masked", "wrong_candidate", "random_latent"}:
        raise ValueError(f"unknown prediction control: {mode}")
    if mode == "outcome_shuffled":
        outcomes = torch.as_tensor(outcome_derangement(batch.outcomes.cpu().numpy(), 4, seed),
                                   device=batch.task.device, dtype=torch.long)
        source = BeliefBatch(batch.task, batch.candidate, batch.tests, outcomes)
    elif mode in {"joint_reversed", "presentation_permuted", "orderless"}:
        indices = np.tile(np.arange(batch.tests.shape[1]), (len(batch.task), 1))
        if mode == "joint_reversed":
            indices[:, :4] = indices[:, :4][:, ::-1]
        elif mode == "presentation_permuted":
            indices, _ = joint_permutation(indices, batch.outcomes.cpu().numpy(), 4, seed)
        else:
            # Canonical ordering is based solely on public semantic features,
            # never on outcomes, IDs, or evaluator-hidden execution strings.
            for i, tests in enumerate(batch.tests[:, :4].cpu().numpy()):
                indices[i, :4] = sorted(range(4), key=lambda j: tuple(tests[j].tolist()))
        order = torch.as_tensor(indices, device=batch.task.device)
        tests = batch.tests.gather(1, order[..., None].expand_as(batch.tests))
        outcomes = batch.outcomes.gather(1, order)
        source = BeliefBatch(batch.task, batch.candidate, tests, outcomes)
    elif mode == "semantics_masked":
        tests = batch.tests.clone()
        tests[:, :4] = 0
        source = BeliefBatch(batch.task, batch.candidate, tests, batch.outcomes)
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
    # Explicit arrays keep common noise aligned even when controls trigger
    # different resampling decisions or numbers of resampled candidates.
    noise = torch.randn((len(batch.task), particles, model.latent_dim), device=batch.task.device,
                        dtype=batch.task.dtype, generator=generator)
    uniforms = torch.rand((len(batch.task), 4), device=batch.task.device,
                          dtype=batch.task.dtype, generator=generator)
    trace = model.filter(source, particles=particles, visible_steps=4, ess_fraction=0.5, generator=generator,
                         proposal_noise=noise, resampling_uniforms=uniforms)
    z, weights = trace.latents[:, 3], trace.log_weights[:, 3]
    if mode == "baseline":
        prior = model.root(batch.task, batch.candidate)
        z = prior.mean[:, None] + prior.std[:, None] * noise
        weights = batch.task.new_full((len(batch.task), particles), -math.log(particles))
    elif mode == "random_latent":
        radius = torch.sqrt((weights.exp() * z.square().sum(-1)).sum(-1, keepdim=True))
        z = torch.randn(z.shape, device=z.device, generator=generator)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8) * radius[:, None]
    future = batch.tests[:, 4:]
    return model.future_predict(batch.task, batch.candidate, future, z, weights).exp().reshape(-1, 5).cpu().numpy()


def _history_rate(batch, alpha=1.0):
    visible = batch.outcomes[:, :4].cpu().numpy()
    rows = []
    for outcomes in visible:
        counts = np.bincount(outcomes, minlength=5).astype(np.float64) + alpha
        rows.extend([counts / counts.sum()] * (batch.outcomes.shape[1] - 4))
    return np.asarray(rows)


def _cluster_bootstrap(labels, predictions, problem_ids, *, comparator="baseline", seed, replicates=10_000):
    if not problem_ids:
        raise ValueError("source clusters are required")
    width = len(labels) // len(problem_ids)
    if width * len(problem_ids) != len(labels):
        raise ValueError("future examples must have a fixed count per candidate")
    aligned = predictions["aligned"] if "aligned" in predictions else predictions["pbpf"]
    result = clustered_nll_gap(labels, aligned, predictions[comparator], np.repeat(problem_ids, width),
                               seed=seed, replicates=replicates)
    return {**result, "comparator": comparator, "mean_nll_gain": result["mean_nll_gap"]}


class _DeterministicPredictor(torch.nn.Module):
    """Matched semantic inputs, deterministic pair-aware and exchangeable controls."""

    def __init__(self, feature_dim, hidden_dim, latent_dim, arm):
        super().__init__()
        self.arm = arm
        self.pair = torch.nn.Sequential(torch.nn.Linear(feature_dim + 5, hidden_dim), torch.nn.Tanh())
        width = 4 * hidden_dim if arm == "pair_aware" else hidden_dim
        bottleneck = latent_dim if arm == "no_particle_bottleneck" else hidden_dim
        self.context = torch.nn.Sequential(torch.nn.Linear(2 * feature_dim + width, bottleneck), torch.nn.Tanh())
        self.head = torch.nn.Sequential(torch.nn.Linear(bottleneck + feature_dim, hidden_dim),
                                        torch.nn.Tanh(), torch.nn.Linear(hidden_dim, 5))

    def forward(self, batch):
        pairs = self.pair(torch.cat([batch.tests[:, :4],
            torch.nn.functional.one_hot(batch.outcomes[:, :4], 5).to(batch.task.dtype)], -1))
        history = pairs.flatten(1) if self.arm == "pair_aware" else pairs.mean(1)
        context = self.context(torch.cat([batch.task, batch.candidate, history], -1))
        future = batch.tests[:, 4:]
        return self.head(torch.cat([context[:, None].expand(-1, future.shape[1], -1), future], -1))


def _fit_strong_baselines(batches, *, feature_dim, hidden_dim, latent_dim, steps,
                          batch_size, learning_rate, seed):
    """Fit only train; choose checkpoints/smoothing only on development."""
    predictions, metadata = {}, {}
    for offset, arm in enumerate(("pair_aware", "deep_sets", "no_particle_bottleneck")):
        torch.manual_seed(seed + offset)
        model = _DeterministicPredictor(feature_dim, hidden_dim, latent_dim, arm).to(batches["train"].task.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
        rng = np.random.default_rng(seed)
        best, state = float("inf"), None
        for step in range(1, steps + 1):
            indices = torch.tensor(rng.integers(0, len(batches["train"].task), size=batch_size),
                                   device=batches["train"].task.device)
            batch = _subset(batches["train"], indices)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch).reshape(-1, 5), batch.outcomes[:, 4:].reshape(-1))
            loss.backward()
            optimizer.step()
            if step == 1 or step % max(1, steps // 20) == 0:
                with torch.no_grad():
                    nll = float(torch.nn.functional.cross_entropy(model(batches["development"]).reshape(-1, 5),
                        batches["development"].outcomes[:, 4:].reshape(-1)))
                if nll < best:
                    best, state = nll, copy.deepcopy(model.state_dict())
        model.load_state_dict(state)
        with torch.no_grad():
            predictions[arm] = model(batches["test"]).softmax(-1).reshape(-1, 5).cpu().numpy()
        metadata[arm] = {"steps": steps, "parameters": sum(p.numel() for p in model.parameters()),
                         "development_nll": best, "checkpoint_selection_split": "development"}
    labels = batches["development"].outcomes[:, 4:].reshape(-1).cpu().numpy()
    grid = (.01, .1, .25, .5, 1., 2., 5., 10.)
    alpha = min(grid, key=lambda a: compare_predictions(labels, {"baseline": _history_rate(batches["development"], a)})["baseline"]["nll"])
    predictions["tuned_dirichlet"] = _history_rate(batches["test"], alpha)
    metadata["tuned_dirichlet"] = {"alpha": alpha, "grid": grid, "selection_split": "development"}
    return predictions, metadata


def _apbpf_step(model, batch, optimizer, *, particles, association_weight,
                invariance_weight, association_margin, shuffle_seed):
    from pbpf.train_belief import train_apbpf_step
    return train_apbpf_step(model, batch, optimizer, particles=particles, visible_steps=4,
        association_weight=association_weight, invariance_weight=invariance_weight,
        margin=association_margin, shuffle_seed=shuffle_seed)


def train_and_evaluate(payload: dict, output: Path, *, feature_dim: int, latent_dim: int,
                       hidden_dim: int, particles: int, steps: int, batch_size: int,
                       learning_rate: float, seed: int, apbpf: bool = False,
                       difficulty_dim: int = 8, association_weight: float = 1.,
                       invariance_weight: float = .1, association_margin: float = .03,
                       expected_is_public: bool = False, strong_baselines: bool = True,
                       bootstrap_replicates: int = 10000) -> dict:
    validate_rbr_cache(payload)
    if output.exists() or output.with_suffix(".pt").exists():
        raise FileExistsError("gate outputs are create-once; use a new output path")
    if min(steps, batch_size, particles) < 1:
        raise ValueError("steps, batch size, and particles must be positive")
    config = {"feature_dim": feature_dim, "latent_dim": latent_dim, "hidden_dim": hidden_dim,
              "particles": particles, "steps": steps, "batch_size": batch_size,
              "learning_rate": learning_rate, "seed": seed, "apbpf": apbpf,
              "difficulty_dim": difficulty_dim if apbpf else None,
              "diagnosis_dim": latent_dim - difficulty_dim if apbpf else None,
              "association_weight": association_weight, "invariance_weight": invariance_weight,
              "association_margin": association_margin, "expected_is_public": expected_is_public,
              "strong_baselines": strong_baselines, "bootstrap_replicates": bootstrap_replicates}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrozenTextEncoder(feature_dim)
    rows = payload["records"]
    by_split = {name: [row for row in rows if row["split"] == name] for name in ("train", "development", "test")}
    if any(not values for values in by_split.values()):
        raise ValueError("train, development, and test splits must all be nonempty")
    # Seal the supplied full test inventory and configuration before fitting or
    # computing any model's held-out metrics. No ambiguity stratum is selectable.
    population = [{"task_id": row["task_id"],
                   "source_component_id": row.get("source_component_id", row["problem_id"])}
                  for row in by_split["test"]]
    population_hash = hashlib.sha256(json.dumps(population, sort_keys=True).encode()).hexdigest()
    cache_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".population.json").open("x") as stream:
        stream.write(json.dumps({"schema": "apbpf-association-population-lock-v1", "population": population,
            "population_sha256": population_hash, "cache_sha256": cache_hash, "config_sha256": config_hash},
            sort_keys=True, indent=2) + "\n")
    batches = {name: _tensorize(values, encoder, device, expected_is_public=expected_is_public)
               for name, values in by_split.items()}
    options = {"difficulty_dim": difficulty_dim} if apbpf else {}
    model = NeuralBeliefModel(feature_dim, latent_dim, hidden_dim, **options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    best, best_state, history = float("inf"), None, []
    for step in range(1, steps + 1):
        indices = torch.tensor(rng.integers(0, len(by_split["train"]), size=batch_size), device=device)
        if apbpf:
            metrics = _apbpf_step(model, _subset(batches["train"], indices), optimizer,
                particles=particles, association_weight=association_weight,
                invariance_weight=invariance_weight, association_margin=association_margin,
                shuffle_seed=seed + step)
        else:
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
    problem_ids = [row.get("source_component_id", row["problem_id"]) for row in by_split["test"]]
    predictions = {mode: _predict(model, batch, particles=particles, seed=seed + 100_000, mode=mode,
                                  problem_ids=problem_ids)
                   for mode in ("baseline", "aligned", "outcome_shuffled", "joint_reversed",
                                "presentation_permuted", "orderless", "semantics_masked",
                                "wrong_candidate", "random_latent")}
    predictions["history_rate"] = _history_rate(batch)
    baseline_metadata = {}
    if strong_baselines:
        strong_predictions, baseline_metadata = _fit_strong_baselines(batches, feature_dim=feature_dim,
            hidden_dim=hidden_dim, latent_dim=latent_dim, steps=steps, batch_size=batch_size,
            learning_rate=learning_rate, seed=seed)
        predictions.update(strong_predictions)
    metrics = compare_predictions(labels, predictions)
    future_width = batch.outcomes.shape[1] - 4
    clusters = np.repeat(problem_ids, future_width)
    bootstrap = {comparator: clustered_nll_gap(labels, predictions["aligned"], values, clusters,
        seed=seed + 200000, replicates=bootstrap_replicates)
        for comparator, values in predictions.items() if comparator != "aligned"}
    strata = {}
    for name, candidate_mask in association_strata(batch.outcomes.cpu().numpy()).items():
        example_mask = np.repeat(candidate_mask, future_width)
        strata[name] = {
            "candidates": int(candidate_mask.sum()), "future_examples": int(example_mask.sum()),
            "status": "evaluated" if example_mask.any() else "empty_prespecified_stratum",
            "metrics": compare_predictions(labels[example_mask],
                {arm: values[example_mask] for arm, values in predictions.items()}) if example_mask.any() else None,
            "association_gap": clustered_nll_gap(labels[example_mask], predictions["aligned"][example_mask],
                predictions["outcome_shuffled"][example_mask], clusters[example_mask], seed=seed + 200000,
                replicates=bootstrap_replicates) if example_mask.any() else None,
        }
    association = bootstrap["outcome_shuffled"]
    association_passes = association["mean_nll_gap"] >= .03 and association["ci95"][0] > 0
    reversal_degradation = max(0., bootstrap["joint_reversed"]["mean_nll_gap"])
    invariance_passes = (association["mean_nll_gap"] > 0 and
        reversal_degradation <= .25 * association["mean_nll_gap"] and
        max(0., bootstrap["presentation_permuted"]["mean_nll_gap"]) <= .25 * association["mean_nll_gap"])
    controls_worse = all(bootstrap[name]["mean_nll_gap"] > 0
                         for name in ("semantics_masked", "wrong_candidate", "random_latent"))
    required_baselines = ("pair_aware", "deep_sets", "tuned_dirichlet", "no_particle_bottleneck")
    baseline_passes = strong_baselines and all(bootstrap[name]["mean_nll_gap"] >= .02
                                              and bootstrap[name]["ci95"][0] > 0 for name in required_baselines)
    report = {
        "schema": "apbpf-rbr-association-gate-v1", "device": str(device), "seed": seed,
        "config": config, "config_sha256": config_hash,
        "cache_sha256": cache_hash,
        "locked_population": population,
        "population_sha256": population_hash,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "baseline_matching": "same semantic inputs, train/dev splits, optimizer, steps, batch size, and checkpoint schedule; parameter counts reported",
        "primary_estimand": "full-population NLL(outcome_shuffled) - NLL(aligned)",
        "visibility": {"input": "public", "expected": "public" if expected_is_public else "hidden",
                       "actual_stderr_returncode": "cached evaluator-only; not predictor features"},
        "records": {name: len(values) for name, values in by_split.items()},
        "problems": {name: len({row["problem_id"] for row in values}) for name, values in by_split.items()},
        "future_examples": len(labels), "best_validation_nll": best, "metrics": metrics,
        "gate": {
            "association_passes": association_passes, "association_threshold": .03,
            "pair_invariance_passes": invariance_passes, "max_degradation_fraction": .25,
            "controls_worse_than_aligned": controls_worse,
            "baseline_fairness_passes": baseline_passes, "baseline_margin": .02,
            "baseline_status": "evaluated" if strong_baselines else "missing_fail_closed",
            "particles_necessary_claim_allowed": bool(apbpf and baseline_passes),
            "full_gate_passes": bool(apbpf and association_passes and invariance_passes and controls_worse and baseline_passes),
        },
        "cluster_bootstrap": bootstrap,
        "wrong_candidate_multi_candidate_problems": sum(count > 1 for count in Counter(problem_ids).values()),
        "wrong_candidate_singletons_unchanged": sum(count == 1 for count in Counter(problem_ids).values()),
        "strata": strata, "strong_baselines": baseline_metadata,
        "training_history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    with output.with_suffix(".pt").open("xb") as stream:
        torch.save({"model": best_state, **config, "config_sha256": config_hash,
                    "shuffle_seed_schedule": "seed + training_step",
                    "training_eligibility_mask": association_strata(batches["train"].outcomes.cpu().numpy())["eligible"].tolist()}, stream)
    with output.with_suffix(".checksums.json").open("x") as stream:
        stream.write(json.dumps({path.name: _sha256(path) for path in
            (output, output.with_suffix(".pt"), output.with_suffix(".population.json"))}, sort_keys=True) + "\n")
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
    parser.add_argument("--apbpf", action="store_true", help="enable factored association-trained belief")
    parser.add_argument("--difficulty-dim", type=int, default=8)
    parser.add_argument("--association-weight", type=float, default=1.)
    parser.add_argument("--invariance-weight", type=float, default=.1)
    parser.add_argument("--association-margin", type=float, default=.03)
    parser.add_argument("--expected-is-public", action="store_true",
                        help="explicitly declare expected outputs public for this protocol")
    parser.add_argument("--skip-strong-baselines", action="store_true",
                        help="exploratory only: baseline fairness fails closed")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    payload = prepare(args.data_root, args.descriptions_root, args.descriptions_archive, args.cache,
        train_problems=args.train_problems,
        dev_problems=args.dev_problems, test_problems=args.test_problems,
        candidates_per_problem=args.candidates_per_problem, tests_per_candidate=args.tests_per_candidate,
        workers=args.workers, timeout=args.timeout, seed=args.seed) if args.prepare or not args.cache.exists() else json.loads(args.cache.read_text())
    validate_rbr_cache(payload)
    train_and_evaluate(payload, args.output, feature_dim=args.feature_dim, latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim, particles=args.particles, steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed, apbpf=args.apbpf,
        difficulty_dim=args.difficulty_dim, association_weight=args.association_weight,
        invariance_weight=args.invariance_weight, association_margin=args.association_margin,
        expected_is_public=args.expected_is_public, strong_baselines=not args.skip_strong_baselines,
        bootstrap_replicates=args.bootstrap_replicates)


if __name__ == "__main__":
    main()
