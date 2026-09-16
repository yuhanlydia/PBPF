#!/usr/bin/env python3
"""Train a frozen-Qwen soft prefix and run a matched RunBugRun repair gate."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pbpf.apbpf.stages import RUNNER_LINEAGE_SCHEMA, load_stage_attempt
from pbpf.belief.features import BeliefBatch
from pbpf.belief.model import NeuralBeliefModel
from pbpf.conditioning.mixture import sample_components_once, whole_sequence_mixture_loss
from pbpf.conditioning.soft_prompt import SoftPrefixProjector
from pbpf.real_gate import FrozenTextEncoder, fit_histogram_latent, render_repair_prompt
from pbpf.registry import OUTCOMES


MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
BUG_FILES = ("python_train0.jsonl.gz", "python_train1.jsonl.gz", "python_train2.jsonl.gz")
CONDITIONING_ARMS = (
    "no_latent",
    "mean",
    "map",
    "random",
    "sample_once",
    "token_remix_fault",
)
PROJECTOR_CHECKPOINT_SCHEMAS = frozenset({
    "pbpf-rbr-soft-prefix-v1",
    "pbpf-rbr-soft-prefix-v2",
    "pbpf-rbr-soft-prefix-v3",
    "apbpf-rbr-diagnosis-prefix-v4",
})


@dataclass(frozen=True)
class ContinuationTrace:
    """Immutable conditioning identity for one generated repair continuation.

    ``sample_once`` has one component index and component fingerprint for the
    complete continuation.  ``token_remix_fault`` intentionally has one index
    per output token and is therefore never eligible for a confirmatory claim.
    """

    task_id: str
    arm: str
    continuation: int
    posterior_fingerprint: str
    conditioning_fingerprint: str
    component_fingerprint: str
    component_indices: tuple[int, ...]
    component_mode: str
    immutable_component: bool

    def json(self):
        return asdict(self) | {"component_indices": list(self.component_indices)}


def _fingerprint(value):
    """Hash canonical JSON without relying on mutable ndarray byte layout."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _posterior_fingerprint(particles, log_weights):
    return _fingerprint({"particles": np.asarray(particles, dtype=np.float64).tolist(),
                         "log_weights": np.asarray(log_weights, dtype=np.float64).tolist()})


def _continuation_trace(*, task_id, arm, continuation, particles, log_weights, latent,
                        component_indices, component_mode, immutable_component):
    """Bind selected conditioning to an exact posterior and continuation id."""
    posterior = _posterior_fingerprint(particles, log_weights)
    component_indices = tuple(int(index) for index in component_indices)
    component = _fingerprint({"posterior_fingerprint": posterior, "indices": component_indices,
                              "latent": np.asarray(latent, dtype=np.float64).tolist(),
                              "mode": component_mode})
    trace = ContinuationTrace(task_id, arm, int(continuation), posterior, "", component,
                              component_indices, component_mode, bool(immutable_component))
    conditioning = _fingerprint({"task_id": trace.task_id, "arm": trace.arm,
                                 "continuation": trace.continuation,
                                 "posterior_fingerprint": trace.posterior_fingerprint,
                                 "component_fingerprint": trace.component_fingerprint,
                                 "component_indices": trace.component_indices,
                                 "component_mode": trace.component_mode,
                                 "immutable_component": trace.immutable_component})
    return ContinuationTrace(trace.task_id, trace.arm, trace.continuation, trace.posterior_fingerprint,
                             conditioning, trace.component_fingerprint, trace.component_indices,
                             trace.component_mode, trace.immutable_component)


def _check_projector_checkpoint(saved, checkpoint, *, allow_partial=False,
                                data_payload_sha256=None):
    """Accept frozen historical checkpoints while requiring identity continuity."""
    if saved.get("schema") not in PROJECTOR_CHECKPOINT_SCHEMAS:
        raise ValueError("unsupported projector checkpoint schema")
    required = {"projector", "belief_checkpoint_sha256", "latent_center", "latent_scale",
                "maximum_token_rms", "maximum_delta_rms"}
    missing = sorted(required - set(saved))
    if missing:
        raise ValueError(f"projector checkpoint is missing required fields: {', '.join(missing)}")
    actual = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if saved["belief_checkpoint_sha256"] != actual:
        raise ValueError("projector checkpoint identity does not match the belief checkpoint")
    belief = torch.load(checkpoint, map_location="cpu", weights_only=False)
    difficulty_dim = belief.get("difficulty_dim")
    expected_dim = int(belief["latent_dim"] if difficulty_dim is None
                       else belief["latent_dim"] - difficulty_dim)
    if saved.get("conditioning_dim", expected_dim) != expected_dim:
        raise ValueError("projector conditioning dimension does not match the belief diagnosis slice")
    if difficulty_dim is not None:
        if (saved.get("schema") != "apbpf-rbr-diagnosis-prefix-v4"
                or saved.get("conditioning_source") != "diagnosis_component"
                or saved.get("belief_difficulty_dim") != difficulty_dim):
            raise ValueError("factored A-PBPF repair requires a diagnosis-only v4 projector")
        if type(saved.get("complete")) is not bool:
            raise ValueError("v4 projector checkpoint must declare complete status")
        if (data_payload_sha256 is not None
                and saved.get("data_payload_sha256") != data_payload_sha256):
            raise ValueError("v4 projector checkpoint was trained on a different data payload")
        if not allow_partial and not saved["complete"]:
            raise ValueError("partial v4 projector checkpoint cannot be used for evaluation")


def _projector_needs_training(output, checkpoint, steps, resume_fingerprint):
    if not Path(output).exists():
        return True
    saved = torch.load(output, map_location="cpu", weights_only=False)
    _check_projector_checkpoint(saved, checkpoint, allow_partial=True)
    completed = int(saved.get("step", 0))
    if completed > steps or ("target_steps" in saved and int(saved["target_steps"]) != steps):
        raise ValueError("projector checkpoint target steps differ; use a new checkpoint path")
    if saved.get("schema") == "apbpf-rbr-diagnosis-prefix-v4":
        if saved.get("resume_fingerprint") != resume_fingerprint:
            raise ValueError("projector checkpoint training identity differs; use a new checkpoint path")
        return not (saved.get("complete") is True and completed == steps)
    return completed < steps


def _atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.pending-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _upstream_gate(paths, *, belief_checkpoint=None, continue_exploratory=False):
    """Require both association and selection decisions for a repair claim.

    A confirmatory decision must be an intact A-PBPF runner attempt. Its
    transitive lineage binds the selection gate to the exact association-gate
    completion used upstream. Legacy compact decisions remain exploratory only.
    """
    if paths is None:
        return {"status": "not_provided", "confirmatory": False,
                "reason": "no upstream A-PBPF gate artifact was supplied", "gate_sha256": None}
    if isinstance(paths, (str, Path)):
        paths = [paths]
    required = {"association_gate", "selection_gate"}
    decisions = {}
    loaded_attempts = {}
    fingerprints = set()
    for path in paths:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
        decision = payload.get("gate") if isinstance(payload.get("gate"), dict) else payload
        stage = payload.get("stage") or decision.get("stage")
        if stage is None and payload.get("schema") == "apbpf-rbr-association-gate-v1":
            stage = "association_gate"
        if stage not in required or stage in decisions:
            raise ValueError("upstream artifacts must uniquely identify association_gate and selection_gate")
        passed = decision.get("passed")
        if passed is None and stage == "association_gate":
            passed = decision.get("full_gate_passes")
        if passed is None and decision.get("status") in {"passed", "pass", "failed", "fail"}:
            passed = decision.get("status") in {"passed", "pass"}
        if type(passed) is not bool:
            raise ValueError("upstream gate artifact must declare a boolean pass decision")
        fingerprint = payload.get("fingerprint")
        dependencies = payload.get("dependencies")
        loaded = None
        runner_error = None
        if payload.get("schema") == "apbpf-stage-result-v1":
            try:
                loaded = load_stage_attempt(Path(path).resolve().parent)
                if ((loaded.directory / "result.json").resolve() != Path(path).resolve()
                        or loaded.value != payload):
                    raise ValueError("gate path is not the loaded attempt result")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                runner_error = str(exc)
                loaded = None
        lineage = payload.get("runner_lineage")
        runner_envelope = (payload.get("schema") == "apbpf-stage-result-v1"
                           and payload.get("stage") == stage
                           and isinstance(fingerprint, str) and re.fullmatch(r"[a-f0-9]{64}", fingerprint)
                           and isinstance(dependencies, dict)
                           and set(dependencies) == {stage.removesuffix("_gate")}
                           and all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)
                                   for value in dependencies.values())
                           and loaded is not None
                           and isinstance(lineage, dict)
                           and lineage.get("schema") == RUNNER_LINEAGE_SCHEMA)
        if runner_envelope:
            fingerprints.add(fingerprint)
            loaded_attempts[stage] = loaded
        confirmatory_attempt = bool(runner_envelope
            and loaded.identity.get("backend") == "real"
            and loaded.identity.get("confirmatory") is True
            and loaded.identity.get("failed_gates") == []
            and loaded.identity.get("claim_status") == "prospective-gated")
        decisions[stage] = {"passed": passed, "reason": decision.get("reason", ""),
                            "sha256": hashlib.sha256(raw).hexdigest(), "path": str(path),
                            "runner_envelope": bool(runner_envelope),
                            "confirmatory_attempt": confirmatory_attempt,
                            "runner_error": runner_error,
                            "completion_sha256": loaded.checksum if runner_envelope else None,
                            "fingerprint": fingerprint if runner_envelope else None,
                            "dependencies": dependencies if runner_envelope else None,
                            "runner_lineage": lineage if runner_envelope else None}
    missing = sorted(required - set(decisions))
    valid_envelopes = (not missing and len(fingerprints) == 1
                       and all(value["runner_envelope"] for value in decisions.values()))
    confirmatory_envelopes = (valid_envelopes
                              and all(value["confirmatory_attempt"] for value in decisions.values()))
    exact_ancestor = (valid_envelopes
                      and decisions["selection_gate"]["runner_lineage"]["ancestor_completion_sha256"]
                      .get("association_gate")
                      == decisions["association_gate"]["completion_sha256"])
    belief_binding = {"bound": False, "reason": "belief checkpoint was not supplied",
                      "train_belief_completion_sha256": None, "checkpoint_sha256": None,
                      "artifact_path": None}
    if valid_envelopes and exact_ancestor and belief_checkpoint is not None:
        selection = loaded_attempts["selection_gate"]
        target = decisions["selection_gate"]["runner_lineage"]["ancestor_completion_sha256"].get("train_belief")
        belief_binding["train_belief_completion_sha256"] = target
        attempts = []
        if target is not None:
            run_root = selection.directory.parents[2]
            for directory in sorted((run_root / "stages" / "train_belief").glob(
                    "attempt-[0-9][0-9][0-9][0-9][0-9][0-9]")):
                try:
                    candidate = load_stage_attempt(directory)
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                if candidate.checksum == target:
                    attempts.append(candidate)
        checkpoint = Path(belief_checkpoint)
        if len(attempts) != 1:
            belief_binding["reason"] = "exact train_belief ancestor attempt is missing or ambiguous"
        elif not checkpoint.is_file():
            belief_binding["reason"] = "belief checkpoint file is missing"
        else:
            with checkpoint.open("rb") as stream:
                checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            artifacts = [artifact for artifact in attempts[0].value["artifacts"]
                         if artifact["sha256"] == checkpoint_sha256]
            belief_binding.update({
                "bound": bool(artifacts),
                "reason": "" if artifacts else "checkpoint is not an artifact of the exact train_belief ancestor",
                "checkpoint_sha256": checkpoint_sha256,
                "artifact_path": artifacts[0]["path"] if artifacts else None,
            })
    valid_lineage = bool(confirmatory_envelopes and exact_ancestor and belief_binding["bound"])
    passed = valid_lineage and all(value["passed"] for value in decisions.values())
    identity = _fingerprint({name: value.get("completion_sha256") or value["sha256"]
                             for name, value in sorted(decisions.items())})
    failure_reason = (f"missing {missing}" if missing else
                      "gate artifacts are not intact same-fingerprint runner attempts"
                      if not valid_envelopes else
                      "gate attempts are smoke or carry exploratory/failed-gate lineage"
                      if not confirmatory_envelopes else
                      "selection gate does not descend from the supplied association gate completion"
                      if not exact_ancestor else
                      belief_binding["reason"] if not belief_binding["bound"] else
                      "at least one gate failed")
    if not passed and not continue_exploratory:
        raise RuntimeError(f"upstream A-PBPF gates are incomplete/failed ({failure_reason}); repair is blocked")
    return {"status": "passed" if passed else "failed_or_incomplete_exploratory_override",
            "confirmatory": passed, "reason": "" if passed else failure_reason,
            "gate_sha256": identity, "gates": decisions, "belief_binding": belief_binding,
            "fingerprint": next(iter(fingerprints)) if valid_lineage else None,
            "continue_exploratory": bool(continue_exploratory)}


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
    difficulty_dim = saved.get("difficulty_dim")
    model = NeuralBeliefModel(saved["feature_dim"], saved["latent_dim"], saved["hidden_dim"],
                              difficulty_dim=difficulty_dim).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    encoder = FrozenTextEncoder(saved["feature_dim"])
    latents, weights = [], []
    generator = torch.Generator(device=device).manual_seed(saved["seed"] + 880_000)
    for start in range(0, len(rows), batch_size):
        batch = _features(rows[start:start + batch_size], encoder, device)
        trace = model.filter(batch, particles=8, visible_steps=4, generator=generator)
        posterior = trace.latents[:, 3]
        # A-PBPF actor conditioning is diagnosis-only. Difficulty remains useful
        # for outcome prediction but is never allowed to become a repair shortcut.
        if difficulty_dim is not None:
            posterior = posterior[..., difficulty_dim:]
        latents.append(posterior.cpu())
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


def _projector_training_identity(payload, checkpoint, data_root, *, steps, learning_rate,
                                 max_sequence_tokens, seed, maximum_token_rms,
                                 maximum_delta_rms):
    rows = [row for row in payload["records"] if row["split"] in {"train", "development"}]
    fixed = _load_fixed_codes(data_root, {row["task_id"] for row in rows})
    identity = {
        "schema": "apbpf-projector-training-identity-v1",
        "data_payload_sha256": _fingerprint(payload),
        "fixed_code_sha256": _fingerprint({key: fixed[key] for key in sorted(fixed)}),
        "belief_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "actor": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "steps": int(steps), "learning_rate": float(learning_rate),
        "max_sequence_tokens": int(max_sequence_tokens), "seed": int(seed),
        "maximum_token_rms": float(maximum_token_rms),
        "maximum_delta_rms": float(maximum_delta_rms),
        "particles": 8, "visible_steps": 4, "posterior_batch_size": 64,
        "optimizer": {"name": "AdamW", "weight_decay": 0.01},
        "projector_hidden_dim": 128,
    }
    return identity, fixed


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


def _mixture_nll(model, projector, ids, labels, z, log_weights, *, conditioned=True):
    components = len(z) if conditioned else 1
    ids = ids[None].expand(components, -1).to("cuda")
    labels = labels[None].expand(components, -1).to("cuda")
    mask = torch.ones_like(ids)
    embeddings = model.get_input_embeddings()(ids).detach()
    actor_inputs = (projector.prepend(z.to("cuda"), inputs_embeds=embeddings,
                                     attention_mask=mask, labels=labels)
                    if conditioned else
                    {"inputs_embeds": embeddings, "attention_mask": mask, "labels": labels})
    # Keep the full vocabulary tensor in actor dtype. Casting all logits to
    # float32 adds ~1.8 GB at the 1,536-token cap on Qwen and can OOM before CE;
    # PyTorch's fused CE performs its own stable accumulation.
    # We compute the exact mixture loss below. Passing labels here additionally
    # computes the actor's unused full-vocabulary FP32 CE, which can exhaust a
    # 24 GB GPU during eight-particle validation without changing the logits.
    forward_inputs = {key: value for key, value in actor_inputs.items() if key != "labels"}
    logits = model(**forward_inputs, use_cache=False).logits[:, :-1]
    shifted = actor_inputs["labels"][:, 1:]
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
            torch.zeros(1, dtype=log_weights.dtype), conditioned=False))
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
    resume_identity, fixed = _projector_training_identity(
        payload, checkpoint, data_root, steps=steps, learning_rate=learning_rate,
        max_sequence_tokens=max_sequence_tokens, seed=seed,
        maximum_token_rms=maximum_token_rms, maximum_delta_rms=maximum_delta_rms)
    resume_fingerprint = _fingerprint(resume_identity)
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
    difficulty_dim = saved.get("difficulty_dim")
    conditioning_source = "joint_latent" if difficulty_dim is None else "diagnosis_component"
    checkpoint_schema = ("pbpf-rbr-soft-prefix-v3" if difficulty_dim is None
                         else "apbpf-rbr-diagnosis-prefix-v4")
    conditioning_contract = ("one-immutable-component-per-continuation-v1"
                             if difficulty_dim is None else
                             "one-immutable-diagnosis-component-per-continuation-v2")
    projector = SoftPrefixProjector(particles.shape[-1], model.config.hidden_size, hidden_dim=128,
                                    maximum_token_rms=maximum_token_rms,
                                    maximum_delta_rms=maximum_delta_rms).cuda()
    resume = torch.load(output, map_location="cpu", weights_only=False) if output.exists() else None
    if resume is not None:
        _check_projector_checkpoint(resume, checkpoint, allow_partial=True,
                                    data_payload_sha256=resume_identity["data_payload_sha256"])
        if (resume.get("schema") == "apbpf-rbr-diagnosis-prefix-v4"
                and resume.get("resume_fingerprint") != resume_fingerprint):
            raise ValueError("projector resume identity changed; use a new checkpoint path")
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
    if resume and start_step < steps:
        required_resume = {"optimizer", "numpy_rng_state", "component_rng_state", "target_steps", "complete"}
        if not required_resume.issubset(resume) or resume["complete"] is not False:
            raise ValueError("partial projector checkpoint lacks exact optimizer/RNG resume state")
        if int(resume["target_steps"]) != steps:
            raise ValueError("partial projector target steps differ; use a new checkpoint path")
        optimizer.load_state_dict(resume["optimizer"])
        rng.bit_generator.state = resume["numpy_rng_state"]
        generator.set_state(resume["component_rng_state"])
    dev_items = [item for item in prepared if item[1]["split"] == "development"]
    best_validation = float(resume.get("best_validation_nll", float("inf"))) if resume else float("inf")
    best_step = int(resume.get("best_step", 0)) if resume else 0
    best_state = copy.deepcopy(resume.get("best_projector", projector.state_dict())) if resume else copy.deepcopy(projector.state_dict())
    started = time.perf_counter()

    def save_projector(step, *, complete):
        _atomic_torch_save({"schema": checkpoint_schema, "projector": projector.state_dict(),
            "best_projector": best_state, "best_validation_nll": best_validation, "best_step": best_step,
            "histogram_coefficients": coefficients, "belief_checkpoint": str(checkpoint),
            "belief_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "actor": {"id": MODEL_ID, "revision": MODEL_REVISION}, "step": step,
            "target_steps": steps, "complete": bool(complete),
            "resume_identity": resume_identity, "resume_fingerprint": resume_fingerprint,
            "data_payload_sha256": resume_identity["data_payload_sha256"],
            "optimizer": optimizer.state_dict(), "numpy_rng_state": rng.bit_generator.state,
            "component_rng_state": generator.get_state(),
            "training_items": len(train_items), "history": history, "seed": seed,
            "latent_center": latent_center, "latent_scale": latent_scale,
            "conditioning_dim": int(particles.shape[-1]),
            "conditioning_source": conditioning_source,
            "belief_difficulty_dim": difficulty_dim,
            "maximum_token_rms": maximum_token_rms,
            "maximum_delta_rms": maximum_delta_rms,
            "conditioning_contract": conditioning_contract}, output)

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
            save_projector(step, complete=False)
    projector.load_state_dict(best_state)
    save_projector(steps, complete=True)
    del model, projector
    torch.cuda.empty_cache()


def _generate(model, tokenizer, projector, row, latent, *, max_new_tokens, conditioned=True):
    """Generate one continuation from one fixed latent component."""
    ids = torch.tensor([_prompt_ids(tokenizer, row)], device="cuda")
    embeddings = model.get_input_embeddings()(ids)
    kwargs = (projector.prepend(torch.tensor(latent, dtype=torch.float32, device="cuda")[None],
              inputs_embeds=embeddings, attention_mask=torch.ones_like(ids))
              if conditioned else
              {"inputs_embeds": embeddings, "attention_mask": torch.ones_like(ids)})
    with torch.inference_mode():
        output = model.generate(**kwargs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id, use_cache=True)
    return _clean(tokenizer.decode(output[0], skip_special_tokens=True))


def _generate_token_remix_fault(model, tokenizer, projector, row, particles, component_indices, *, max_new_tokens):
    """Intentionally wrong control: replace the conditioning component per token.

    This recomputes the prefix-conditioned context for every next token.  It is
    deliberately slower than normal generation and must never be represented as
    an implementation of posterior sampling or a matched serving method.
    """
    if not component_indices:
        raise ValueError("token-remix fault requires at least one component index")
    prompt = torch.tensor([_prompt_ids(tokenizer, row)], device="cuda")
    generated = []
    for position in range(max_new_tokens):
        component = int(component_indices[position % len(component_indices)])
        latent = torch.as_tensor(particles[component], dtype=torch.float32, device="cuda")[None]
        suffix = torch.tensor([generated], device="cuda", dtype=prompt.dtype) if generated else prompt[:, :0]
        ids = torch.cat((prompt, suffix), dim=1)
        embeddings = model.get_input_embeddings()(ids)
        conditioned = projector.prepend(latent, inputs_embeds=embeddings, attention_mask=torch.ones_like(ids))
        with torch.inference_mode():
            token = int(model(**conditioned, use_cache=False).logits[0, -1].argmax())
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
    return _clean(tokenizer.decode(generated, skip_special_tokens=True))


def _arm_conditioning(*, arm, particles, log_weights, rng, task_id, continuation, max_new_tokens):
    """Select an explicit conditioning control and its immutable trace."""
    if arm not in CONDITIONING_ARMS:
        raise ValueError(f"unknown conditioning arm: {arm}")
    probabilities = np.exp(np.asarray(log_weights, dtype=np.float64))
    probabilities /= probabilities.sum()
    particles = np.asarray(particles, dtype=np.float64)
    if arm == "no_latent":
        latent, indices, mode, immutable = np.zeros_like(particles[0]), (), "none", True
    elif arm == "mean":
        latent, indices, mode, immutable = probabilities @ particles, (), "posterior_mean", True
    elif arm == "map":
        index = int(np.asarray(log_weights).argmax())
        latent, indices, mode, immutable = particles[index], (index,), "posterior_map", True
    elif arm == "random":
        direction = rng.normal(size=particles.shape[1])
        radius = math.sqrt(float(np.sum(probabilities * np.square(particles).sum(axis=1))))
        latent, indices, mode, immutable = direction * radius / max(np.linalg.norm(direction), 1e-12), (), "matched_norm_random", True
    elif arm == "sample_once":
        index = int(rng.choice(len(probabilities), p=probabilities))
        latent, indices, mode, immutable = particles[index], (index,), "posterior_component_sample_once", True
    else:
        # Draw a schedule before any actor call; its non-singleton schedule is
        # the audit evidence that this is the intentionally faulty ablation.
        indices = tuple(int(value) for value in rng.choice(len(probabilities), size=max_new_tokens, p=probabilities))
        latent, mode, immutable = particles[indices[0]], "tokenwise_component_remix_fault", False
    trace = _continuation_trace(task_id=task_id, arm=arm, continuation=continuation,
                                particles=particles, log_weights=log_weights, latent=latent,
                                component_indices=indices, component_mode=mode,
                                immutable_component=immutable)
    return latent, trace


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
             max_new_tokens, upstream_gate=None):
    rows = [row for row in payload["records"] if row["split"] == "test"]
    rows = sorted(rows, key=lambda row: _rank(seed, row["task_id"]))[:tasks]
    evaluator_cases = _evaluator_cases(data_root, rows)
    particles, log_weights = _posterior(rows, belief_checkpoint, batch_size=64, device="cuda")
    saved = torch.load(projector_checkpoint, map_location="cpu", weights_only=False)
    _check_projector_checkpoint(saved, belief_checkpoint,
                                data_payload_sha256=_fingerprint(payload))
    particles = ((particles - torch.tensor(saved["latent_center"])[None, None]) /
                 torch.tensor(saved["latent_scale"])[None, None])
    model, tokenizer = _load_actor()
    projector = SoftPrefixProjector(particles.shape[-1], model.config.hidden_size, hidden_dim=128,
                                    maximum_token_rms=float(saved["maximum_token_rms"]),
                                    maximum_delta_rms=float(saved["maximum_delta_rms"])).cuda()
    projector.load_state_dict(saved["projector"])
    projector.eval()
    rng = np.random.default_rng(seed)
    reports = []
    arms = CONDITIONING_ARMS
    for index, (row, cases) in enumerate(zip(rows, evaluator_cases)):
        for arm in arms:
            torch.manual_seed(seed + index)
            latent, trace = _arm_conditioning(arm=arm, particles=particles[index].numpy(),
                log_weights=log_weights[index].numpy(), rng=rng, task_id=row["task_id"],
                continuation=index + 1, max_new_tokens=max_new_tokens)
            started = time.perf_counter()
            if arm == "token_remix_fault":
                code = _generate_token_remix_fault(model, tokenizer, projector, row, particles[index].numpy(),
                    trace.component_indices, max_new_tokens=max_new_tokens)
            else:
                code = _generate(model, tokenizer, projector, row, latent,
                                 max_new_tokens=max_new_tokens, conditioned=arm != "no_latent")
            generation_seconds = time.perf_counter() - started
            outcomes = _execute(code, cases)
            report = {"task_id": row["task_id"], "problem_id": row["problem_id"], "arm": arm,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "code": code,
                "outcomes": outcomes, "visible_pass_fraction": outcomes[:4].count("PASS") / 4,
                "future_pass_fraction": outcomes[4:].count("PASS") / len(outcomes[4:]),
                "solved": all(value == "PASS" for value in outcomes),
                "generated_tokens": len(tokenizer(code, add_special_tokens=False)["input_ids"]),
                "generation_seconds": generation_seconds,
                "particle_index": trace.component_indices[0] if len(trace.component_indices) == 1 else None,
                "conditioning_trace": trace.json()}
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
    result = {"schema": "pbpf-rbr-repair-gate-v2", "seed": seed, "arms": list(arms),
        "matched": {"same_tasks": True, "same_prompt": True, "same_actor": MODEL_ID,
                    "same_revision": MODEL_REVISION,
                    "prefix_tokens_by_arm": {arm: (0 if arm == "no_latent" else 8) for arm in arms},
                    "same_decode_policy": "greedy", "max_new_tokens": max_new_tokens,
                    "conditioning_contract": "one-immutable-diagnosis-component-per-continuation-v2",
                    "conditioning_source": saved.get("conditioning_source", "joint_latent")},
        "summary": summary, "tasks": reports,
        "upstream_gate": upstream_gate or {"status": "not_provided", "confirmatory": False},
        "claim_status": ("prospective-confirmatory-repair" if upstream_gate and upstream_gate["confirmatory"]
                         else "exploratory-after-failed-upstream-gate" if upstream_gate
                         and upstream_gate["status"] != "not_provided"
                         else "pilot-no-confirmatory-claim")}
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
    parser.add_argument("--upstream-gate", type=Path, action="append",
                        help="repeat for immutable association_gate and selection_gate decisions")
    parser.add_argument("--continue-exploratory", action="store_true",
                        help="permit a failed upstream gate only for explicitly nonconfirmatory repair artifacts")
    args = parser.parse_args()
    payload = json.loads(args.data.read_text())
    upstream_gate = _upstream_gate(args.upstream_gate, belief_checkpoint=args.belief_checkpoint,
                                   continue_exploratory=args.continue_exploratory)
    projector_identity, _ = _projector_training_identity(
        payload, args.belief_checkpoint, args.data_root, steps=args.steps,
        learning_rate=args.learning_rate, max_sequence_tokens=args.max_sequence_tokens,
        seed=args.seed, maximum_token_rms=args.maximum_token_rms,
        maximum_delta_rms=args.maximum_delta_rms)
    if args.train_projector or _projector_needs_training(
            args.projector_checkpoint, args.belief_checkpoint, args.steps,
            _fingerprint(projector_identity)):
        train_projector(payload, args.belief_checkpoint, args.data_root, args.projector_checkpoint,
            steps=args.steps, learning_rate=args.learning_rate,
            max_sequence_tokens=args.max_sequence_tokens, seed=args.seed,
            maximum_token_rms=args.maximum_token_rms,
            maximum_delta_rms=args.maximum_delta_rms)
    evaluate(payload, args.belief_checkpoint, args.projector_checkpoint, args.data_root, args.output,
             tasks=args.tasks, seed=args.seed, max_new_tokens=args.max_new_tokens,
             upstream_gate=upstream_gate)


if __name__ == "__main__":
    main()
