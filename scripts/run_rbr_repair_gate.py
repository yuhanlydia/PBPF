#!/usr/bin/env python3
"""Train a frozen-Qwen soft prefix and run a matched RunBugRun repair gate."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pbpf.belief.features import BeliefBatch
from pbpf.belief.model import NeuralBeliefModel
from pbpf.conditioning.mixture import sample_components_once, whole_sequence_mixture_loss
from pbpf.conditioning.soft_prompt import SoftPrefixProjector
from pbpf.real_gate import FrozenTextEncoder, fit_histogram_latent, histogram_latent, render_repair_prompt
from pbpf.registry import OUTCOMES


MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
BUG_FILES = ("python_train0.jsonl.gz", "python_train1.jsonl.gz", "python_train2.jsonl.gz")


def _rank(seed, value):
    return hashlib.sha256(f"{seed}\0{value}".encode()).digest()


def _clean(text):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def _features(rows, encoder, device):
    task = np.stack([encoder(row["task_text"]) for row in rows])
    candidate = np.stack([encoder(row["candidate"]) for row in rows])
    tests = np.stack([[encoder(case["input"]) for case in row["tests"]] for row in rows])
    outcomes = np.asarray([[OUTCOMES.index(value) for value in row["outcomes"]] for row in rows])
    return BeliefBatch(torch.tensor(task, device=device), torch.tensor(candidate, device=device),
        torch.tensor(tests, device=device), torch.tensor(outcomes, dtype=torch.long, device=device))


@torch.no_grad()
def _posterior(rows, checkpoint, *, batch_size, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = NeuralBeliefModel(saved["feature_dim"], saved["latent_dim"], saved["hidden_dim"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    encoder = FrozenTextEncoder(saved["feature_dim"])
    latents, weights = [], []
    generator = torch.Generator(device=device).manual_seed(saved["seed"] + 880_000)
    for start in range(0, len(rows), batch_size):
        batch = _features(rows[start:start + batch_size], encoder, device)
        trace = model.filter(batch, particles=8, visible_steps=4, generator=generator)
        latents.append(trace.latents[:, 3].cpu())
        weights.append(trace.log_weights[:, 3].cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(latents), torch.cat(weights)


def _load_fixed_codes(root, wanted):
    result = {}
    for name in BUG_FILES:
        with gzip.open(root / name, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                key = str(row["id"])
                if key in wanted:
                    result[key] = row["fixed_code"]
        if len(result) == len(wanted):
            break
    missing = wanted - set(result)
    if missing:
        raise ValueError(f"training/development fixed code missing for {len(missing)} records")
    return result


def _load_actor():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        local_files_only=True, device_map={"": "cuda:0"},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, tokenizer


def _prompt_ids(tokenizer, row):
    visible = render_repair_prompt(row["candidate"], row["tests"], row["outcomes"], visible=4,
                                   task_text=row["task_text"])
    messages = [
        {"role": "system", "content": "You repair Python programs and output only complete source code."},
        {"role": "user", "content": visible},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def _training_item(tokenizer, row, fixed, *, max_sequence_tokens):
    prompt = _prompt_ids(tokenizer, row)
    target = tokenizer(fixed, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt) + len(target) + 8 > max_sequence_tokens:
        return None
    return torch.tensor(prompt + target), torch.tensor([-100] * len(prompt) + target)


def _mixture_nll(model, projector, ids, labels, z, log_weights):
    components = len(z)
    ids = ids[None].expand(components, -1).to("cuda")
    labels = labels[None].expand(components, -1).to("cuda")
    mask = torch.ones_like(ids)
    embeddings = model.get_input_embeddings()(ids).detach()
    conditioned = projector.prepend(z.to("cuda"), inputs_embeds=embeddings, attention_mask=mask, labels=labels)
    # Keep the full vocabulary tensor in actor dtype. Casting all logits to
    # float32 adds ~1.8 GB at the 1,536-token cap on Qwen and can OOM before CE;
    # PyTorch's fused CE performs its own stable accumulation.
    logits = model(**conditioned, use_cache=False).logits[:, :-1]
    shifted = conditioned["labels"][:, 1:]
    target_mask = shifted[0] != -100
    targets = shifted[0, target_mask]
    selected_logits = logits[:, target_mask]
    pieces = []
    # Convert only a bounded target-token slice to float32. Full-vocabulary
    # fp32 CE is stable but adds gigabytes; pure bf16 CE yielded NaN gradients.
    for start in range(0, len(targets), 64):
        stop = start + 64
        piece = selected_logits[:, start:stop].float().transpose(1, 2)
        expected = targets[start:stop][None].expand(components, -1)
        pieces.append(-F.cross_entropy(piece, expected, reduction="none"))
    token_log_probs = torch.cat(pieces, 1)
    return whole_sequence_mixture_loss(token_log_probs[None], log_weights[None].to("cuda"))


@torch.no_grad()
def _validation_nll(model, projector, items, particles, log_weights):
    """Score held-out fixed repairs, including an exact posterior mixture."""
    totals = {"no_latent": 0.0, "pbpf": 0.0}
    target_tokens = 0
    projector.eval()
    for index, _row, ids, labels in items:
        tokens = int((labels != -100).sum())
        totals["no_latent"] += float(_mixture_nll(
            model, projector, ids, labels, torch.zeros_like(particles[index, :1]),
            torch.zeros(1, dtype=log_weights.dtype)))
        totals["pbpf"] += float(_mixture_nll(
            model, projector, ids, labels, particles[index], log_weights[index]))
        target_tokens += tokens
    projector.train()
    if target_tokens == 0:
        raise ValueError("no development sequence fits the explicit context cap")
    return {name: value / target_tokens for name, value in totals.items()} | {
        "target_tokens": target_tokens, "items": len(items)}


def train_projector(payload, checkpoint, data_root, output, *, steps, learning_rate,
                    max_sequence_tokens, seed, maximum_token_rms=0.02, maximum_delta_rms=0.002):
    rows = [row for row in payload["records"] if row["split"] in {"train", "development"}]
    train_rows = [row for row in rows if row["split"] == "train"]
    train_indices = np.asarray([index for index, row in enumerate(rows) if row["split"] == "train"])
    fixed = _load_fixed_codes(data_root, {row["task_id"] for row in rows})
    particles, log_weights = _posterior(rows, checkpoint, batch_size=64, device="cuda")
    posterior_mean = (log_weights.exp()[..., None] * particles).sum(1).numpy()
    latent_center = posterior_mean[train_indices].mean(0)
    latent_scale = posterior_mean[train_indices].std(0)
    latent_scale = np.maximum(latent_scale, 0.1)
    particles = (particles - torch.tensor(latent_center)[None, None]) / torch.tensor(latent_scale)[None, None]
    posterior_mean = (log_weights.exp()[..., None] * particles).sum(1).numpy()
    coefficients = fit_histogram_latent([row["outcomes"][:4] for row in train_rows],
        posterior_mean[train_indices], ridge=1e-3)
    model, tokenizer = _load_actor()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    projector = SoftPrefixProjector(saved["latent_dim"], model.config.hidden_size, hidden_dim=128,
                                    maximum_token_rms=maximum_token_rms,
                                    maximum_delta_rms=maximum_delta_rms).cuda()
    resume = torch.load(output, map_location="cpu", weights_only=False) if output.exists() else None
    if resume is not None:
        if resume.get("schema") not in {"pbpf-rbr-soft-prefix-v1", "pbpf-rbr-soft-prefix-v2"} or resume.get("belief_checkpoint_sha256") != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
            raise ValueError("projector resume checkpoint identity mismatch")
        projector.load_state_dict(resume["projector"])
    else:
        torch.nn.init.zeros_(projector.network[-1].weight)
        torch.nn.init.zeros_(projector.network[-1].bias)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=learning_rate, weight_decay=0.01)
    prepared = []
    for index, row in enumerate(rows):
        item = _training_item(tokenizer, row, fixed[row["task_id"]], max_sequence_tokens=max_sequence_tokens)
        if item is not None:
            prepared.append((index, row, *item))
    train_items = [item for item in prepared if item[1]["split"] == "train"]
    if not train_items:
        raise ValueError("no training sequence fits the explicit context cap")
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 991_000)
    history = list(resume.get("history", ())) if resume else []
    start_step = int(resume.get("step", 0)) if resume else 0
    dev_items = [item for item in prepared if item[1]["split"] == "development"]
    best_validation = float(resume.get("best_validation_nll", float("inf"))) if resume else float("inf")
    best_step = int(resume.get("best_step", 0)) if resume else 0
    best_state = copy.deepcopy(resume.get("best_projector", projector.state_dict())) if resume else copy.deepcopy(projector.state_dict())
    started = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        index, row, ids, labels = train_items[int(rng.integers(len(train_items)))]
        selected = sample_components_once(log_weights[index:index + 1], num_sequences=2, generator=generator)[0]
        z = particles[index, selected]
        selected_weights = torch.full((2,), -math.log(2), dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = _mixture_nll(model, projector, ids, labels, z, selected_weights)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0))
        optimizer.step()
        if step == 1 or step % 10 == 0:
            record = {"step": step, "loss": float(loss.detach()), "gradient_norm": norm,
                      "task_id": row["task_id"], "elapsed_seconds": time.perf_counter() - started}
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        if step % 250 == 0 or step == steps:
            validation = _validation_nll(model, projector, dev_items, particles, log_weights)
            validation["step"] = step
            history.append({"validation": validation, "step": step})
            print(json.dumps({"validation": validation}, sort_keys=True), flush=True)
            if validation["pbpf"] < best_validation:
                best_validation, best_step = validation["pbpf"], step
                best_state = copy.deepcopy(projector.state_dict())
                for key, value in best_state.items():
                    best_state[key] = value.cpu()
        del loss, z, selected, ids, labels
        if step % 10 == 0:
            torch.cuda.empty_cache()
        if step % 25 == 0:
            torch.save({"schema": "pbpf-rbr-soft-prefix-v2", "projector": projector.state_dict(),
                "best_projector": best_state, "best_validation_nll": best_validation, "best_step": best_step,
                "histogram_coefficients": coefficients, "belief_checkpoint": str(checkpoint),
                "belief_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                "actor": {"id": MODEL_ID, "revision": MODEL_REVISION}, "step": step,
                "training_items": len(train_items), "history": history, "seed": seed,
                "latent_center": latent_center, "latent_scale": latent_scale,
                "maximum_token_rms": maximum_token_rms,
                "maximum_delta_rms": maximum_delta_rms}, output)
    projector.load_state_dict(best_state)
    torch.save({"schema": "pbpf-rbr-soft-prefix-v2", "projector": projector.state_dict(),
        "best_projector": best_state, "best_validation_nll": best_validation, "best_step": best_step,
        "histogram_coefficients": coefficients, "belief_checkpoint": str(checkpoint),
        "belief_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "actor": {"id": MODEL_ID, "revision": MODEL_REVISION}, "step": steps,
        "training_items": len(train_items), "history": history, "seed": seed,
        "latent_center": latent_center, "latent_scale": latent_scale,
        "maximum_token_rms": maximum_token_rms,
        "maximum_delta_rms": maximum_delta_rms}, output)
    del model, projector
    torch.cuda.empty_cache()


def _generate(model, tokenizer, projector, row, latent, *, max_new_tokens):
    ids = torch.tensor([_prompt_ids(tokenizer, row)], device="cuda")
    embeddings = model.get_input_embeddings()(ids)
    kwargs = projector.prepend(torch.tensor(latent, dtype=torch.float32, device="cuda")[None],
        inputs_embeds=embeddings, attention_mask=torch.ones_like(ids))
    with torch.inference_mode():
        output = model.generate(**kwargs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id, use_cache=True)
    return _clean(tokenizer.decode(output[0], skip_special_tokens=True))


def _match(actual, expected):
    if actual.rstrip() == expected.rstrip():
        return True
    left, right = actual.rstrip().split(), expected.rstrip().split()
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a == b:
            continue
        try:
            if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-4):
                return False
        except ValueError:
            return False
    return True


def _execute(code, cases, timeout=2.0):
    try:
        compile(code, "candidate.py", "exec")
    except Exception:
        return ["COMPILE_ERROR"] * len(cases)
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="pbpf-repair-") as directory:
        source = Path(directory) / "candidate.py"
        source.write_text(code)
        for case in cases:
            try:
                result = subprocess.run(["/usr/bin/python3", "-I", str(source)], input=case["input"],
                    text=True, capture_output=True, timeout=timeout, cwd=directory,
                    env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"})
                outcome = "RUNTIME_EXCEPTION" if result.returncode else (
                    "PASS" if _match(result.stdout, case["output"]) else "WRONG_OUTPUT")
            except subprocess.TimeoutExpired:
                outcome = "TIMEOUT"
            outcomes.append(outcome)
    return outcomes


def _evaluator_cases(data_root, rows):
    wanted = {case["id"] for row in rows for case in row["tests"]}
    found = {}
    with gzip.open(data_root / "tests_all.jsonl.gz", "rt") as stream:
        for line in stream:
            case = json.loads(line)
            key = str(case["id"])
            if key in wanted:
                found[key] = case
    if set(found) != wanted:
        raise ValueError("evaluator test inventory is incomplete")
    return [[found[case["id"]] for case in row["tests"]] for row in rows]


def evaluate(payload, belief_checkpoint, projector_checkpoint, data_root, output, *, tasks, seed,
             max_new_tokens):
    rows = [row for row in payload["records"] if row["split"] == "test"]
    rows = sorted(rows, key=lambda row: _rank(seed, row["task_id"]))[:tasks]
    evaluator_cases = _evaluator_cases(data_root, rows)
    particles, log_weights = _posterior(rows, belief_checkpoint, batch_size=64, device="cuda")
    saved = torch.load(projector_checkpoint, map_location="cpu", weights_only=False)
    particles = ((particles - torch.tensor(saved["latent_center"])[None, None]) /
                 torch.tensor(saved["latent_scale"])[None, None])
    model, tokenizer = _load_actor()
    projector = SoftPrefixProjector(particles.shape[-1], model.config.hidden_size, hidden_dim=128,
                                    maximum_token_rms=float(saved["maximum_token_rms"]),
                                    maximum_delta_rms=float(saved["maximum_delta_rms"])).cuda()
    projector.load_state_dict(saved["projector"])
    projector.eval()
    heuristic = histogram_latent([row["outcomes"][:4] for row in rows], saved["histogram_coefficients"])
    rng = np.random.default_rng(seed)
    reports = []
    arms = ("no_latent", "random_latent", "heuristic", "posterior_mean", "posterior_map", "pbpf")
    for index, (row, cases) in enumerate(zip(rows, evaluator_cases)):
        probabilities = log_weights[index].exp().numpy()
        particle_index = int(rng.choice(len(probabilities), p=probabilities / probabilities.sum()))
        pbpf = particles[index, particle_index].numpy()
        posterior_mean = (log_weights[index].exp()[..., None] * particles[index]).sum(0).numpy()
        posterior_map = particles[index, int(log_weights[index].argmax())].numpy()
        random = rng.normal(size=pbpf.shape)
        random *= np.linalg.norm(pbpf) / max(np.linalg.norm(random), 1e-12)
        latents = {"no_latent": np.zeros_like(pbpf), "random_latent": random,
                   "heuristic": heuristic[index], "posterior_mean": posterior_mean,
                   "posterior_map": posterior_map, "pbpf": pbpf}
        for arm in arms:
            torch.manual_seed(seed + index)
            started = time.perf_counter()
            code = _generate(model, tokenizer, projector, row, latents[arm], max_new_tokens=max_new_tokens)
            generation_seconds = time.perf_counter() - started
            outcomes = _execute(code, cases)
            report = {"task_id": row["task_id"], "problem_id": row["problem_id"], "arm": arm,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "code": code,
                "outcomes": outcomes, "visible_pass_fraction": outcomes[:4].count("PASS") / 4,
                "future_pass_fraction": outcomes[4:].count("PASS") / len(outcomes[4:]),
                "solved": all(value == "PASS" for value in outcomes),
                "generated_tokens": len(tokenizer(code, add_special_tokens=False)["input_ids"]),
                "generation_seconds": generation_seconds,
                "particle_index": particle_index if arm == "pbpf" else None}
            reports.append(report)
            print(json.dumps({key: report[key] for key in ("task_id", "arm", "solved", "future_pass_fraction", "generated_tokens", "generation_seconds")}, sort_keys=True), flush=True)
    summary = {}
    for arm in arms:
        values = [row for row in reports if row["arm"] == arm]
        summary[arm] = {"solved": sum(row["solved"] for row in values), "tasks": len(values),
            "success_rate": sum(row["solved"] for row in values) / len(values),
            "future_pass_fraction": float(np.mean([row["future_pass_fraction"] for row in values])),
            "generated_tokens": sum(row["generated_tokens"] for row in values),
            "generation_seconds": sum(row["generation_seconds"] for row in values)}
    result = {"schema": "pbpf-rbr-repair-gate-v1", "seed": seed, "arms": list(arms),
        "matched": {"same_tasks": True, "same_prompt": True, "same_actor": MODEL_ID,
                    "same_revision": MODEL_REVISION, "same_prefix_tokens": 8,
                    "same_decode_policy": "greedy", "max_new_tokens": max_new_tokens},
        "summary": summary, "tasks": reports,
        "claim_status": "pilot-no-confirmatory-claim"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/rbr_real_gate_dataset.json"))
    parser.add_argument("--data-root", type=Path, default=Path("/root/pbpf_external/runbugrun-v0.0.1"))
    parser.add_argument("--belief-checkpoint", type=Path, default=Path("results/rbr_gate_b_seed1703_v3.pt"))
    parser.add_argument("--projector-checkpoint", type=Path, default=Path("results/rbr_soft_prefix.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/rbr_repair_gate.json"))
    parser.add_argument("--train-projector", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--maximum-token-rms", type=float, default=0.02)
    parser.add_argument("--maximum-delta-rms", type=float, default=0.002)
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2701)
    args = parser.parse_args()
    payload = json.loads(args.data.read_text())
    if args.train_projector or not args.projector_checkpoint.exists():
        train_projector(payload, args.belief_checkpoint, args.data_root, args.projector_checkpoint,
            steps=args.steps, learning_rate=args.learning_rate,
            max_sequence_tokens=args.max_sequence_tokens, seed=args.seed,
            maximum_token_rms=args.maximum_token_rms,
            maximum_delta_rms=args.maximum_delta_rms)
    evaluate(payload, args.belief_checkpoint, args.projector_checkpoint, args.data_root, args.output,
             tasks=args.tasks, seed=args.seed, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
