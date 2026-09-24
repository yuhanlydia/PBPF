#!/usr/bin/env python3
"""Replace lexical EESD relevance by counterfactual evidence attribution.

For each stored correction trajectory, score the *edited-token preference* of the
correction over the original program under subsets of the same public executions.
Exact Shapley uses all 2^4 coalitions; LOO is a cheaper diagnostic. Only stored
public evidence is used. Hidden tests/outcomes never enter attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from pbpf.eesd.evidence import effective_mass, normalize_relevance
from pbpf.eesd.shapley_relevance import (
    absolute_relevance,
    build_causal_labels,
    coalition_masks,
    edited_token_masks,
    exact_shapley_values,
    leave_one_out_values,
    select_trajectory_indices,
    subset_execution_messages,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(directory: Path) -> str:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError("adapter must be a directory")
    digest = hashlib.sha256()
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    if not files:
        raise ValueError("adapter directory is empty")
    for path in files:
        rel = path.relative_to(root).as_posix().encode()
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def render_prompt_ids(tokenizer, messages, model_id: str) -> list[int]:
    kwargs = {"enable_thinking": False} if "Qwen3" in model_id else {}
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    return tokenizer(
        rendered, add_special_tokens=False, return_token_type_ids=False
    )["input_ids"]


def response_ids(tokenizer, text: str) -> list[int]:
    values = tokenizer(text, add_special_tokens=False, return_token_type_ids=False)["input_ids"]
    if not values:
        raise ValueError("candidate response tokenized to an empty sequence")
    return list(map(int, values))


def mean_selected_logprob(model, prompt_ids, candidate_ids, target_mask) -> float:
    """Teacher-forced mean log-probability on selected candidate tokens only."""
    import torch

    mask = np.asarray(target_mask, dtype=bool)
    if mask.shape != (len(candidate_ids),):
        raise ValueError("target mask must match candidate token count")
    if not mask.any():
        return 0.0
    full = list(prompt_ids) + list(candidate_ids)
    labels = build_causal_labels(len(prompt_ids), candidate_ids, mask)
    input_tensor = torch.tensor([full], dtype=torch.long, device=model.device)
    label_tensor = torch.tensor([labels.tolist()], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_tensor,
            labels=label_tensor,
            use_cache=False,
            return_dict=True,
        )
    value = -float(outputs.loss.detach().float().cpu())
    if not np.isfinite(value):
        raise ValueError("nonfinite teacher-forced edit log-probability")
    return value


def coalition_value(
    model,
    tokenizer,
    messages,
    model_id,
    original_ids,
    correction_ids,
    original_mask,
    correction_mask,
) -> float:
    prompt = render_prompt_ids(tokenizer, messages, model_id)
    correction_score = mean_selected_logprob(
        model, prompt, correction_ids, correction_mask
    )
    original_score = mean_selected_logprob(
        model, prompt, original_ids, original_mask
    )
    # Score the edit itself rather than the copied code. Positive values mean
    # this coalition makes the observed correction more likely than the original.
    return correction_score - original_score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="existing corrections.jsonl")
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--mode", choices=["exact", "loo"], default="exact")
    p.add_argument("--zero-tolerance", type=float, default=1e-12)
    p.add_argument("--sample-size", type=int, help="uniform no-replacement signal-gate subset")
    p.add_argument("--sample-seed", type=int, default=1701)
    args = p.parse_args()
    if not np.isfinite(args.zero_tolerance) or args.zero_tolerance < 0:
        raise ValueError("zero-tolerance must be finite and nonnegative")

    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    if not rows or any(
        row.get("schema") != "eesd-correction-trajectory-v1" for row in rows
    ):
        raise ValueError("EESD correction trajectories required")
    population = len(rows)
    selected_indices = select_trajectory_indices(population, args.sample_size, seed=args.sample_seed)
    rows = [rows[i] for i in selected_indices]

    cfg = yaml.safe_load(args.model_config.read_text())
    model_id, revision = cfg.get("model_id"), cfg.get("revision")
    if (
        not isinstance(model_id, str)
        or not isinstance(revision, str)
        or len(revision) != 40
    ):
        raise ValueError("pinned model config required")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
    )
    adapter_hash = None
    if args.adapter:
        from peft import PeftModel
        adapter_hash = tree_sha(args.adapter)
        model = PeftModel.from_pretrained(
            model, str(args.adapter), is_trainable=False
        )
    model.eval()

    args.output.mkdir(parents=True, exist_ok=False)
    output_path = args.output / "corrections-shapley.jsonl"
    fallbacks = 0
    changed_tokens = []
    masses = []
    game_spans = []
    attribution_l1 = []

    with output_path.open("x") as stream:
        for index, row in enumerate(rows):
            if (
                row.get("model_id") != model_id
                or row.get("model_revision") != revision
            ):
                raise ValueError(
                    "trajectory model identity differs from attribution model"
                )
            if row.get("adapter_sha256") != adapter_hash:
                raise ValueError(
                    "trajectory adapter identity differs from attribution model"
                )

            before = row.get("before_outcomes")
            after = row.get("after_outcomes")
            if (
                not isinstance(before, list)
                or not isinstance(after, list)
                or len(before) != len(after)
            ):
                raise ValueError("trajectory before/after outcomes missing")
            players = len(before)
            if players != 4:
                raise ValueError(
                    "current locked EESD attribution requires exactly four public executions"
                )
            if not isinstance(row.get("messages"), list):
                raise ValueError(
                    "stored generation messages are required for counterfactual attribution"
                )

            original_ids = response_ids(tokenizer, row["original"])
            correction_ids = response_ids(tokenizer, row["correction"])
            original_mask, correction_mask = edited_token_masks(
                original_ids, correction_ids
            )
            changed_tokens.append(
                int(original_mask.sum() + correction_mask.sum())
            )

            values = {}
            for mask in coalition_masks(players, args.mode):
                keep = [
                    i for i in range(players)
                    if mask & (1 << i)
                ]
                messages = subset_execution_messages(row["messages"], keep)
                values[mask] = coalition_value(
                    model,
                    tokenizer,
                    messages,
                    model_id,
                    original_ids,
                    correction_ids,
                    original_mask,
                    correction_mask,
                )

            signed = (
                exact_shapley_values(values, players)
                if args.mode == "exact"
                else leave_one_out_values(values, players)
            )
            relevance = absolute_relevance(
                signed, zero_tolerance=args.zero_tolerance
            )
            if np.allclose(
                signed,
                0.0,
                rtol=0.0,
                atol=args.zero_tolerance,
            ):
                fallbacks += 1
            probabilities = normalize_relevance(relevance)
            mass = effective_mass(relevance)
            masses.append(mass)
            game_span = float(max(values.values()) - min(values.values()))
            game_spans.append(game_span)
            attribution_l1.append(float(np.abs(signed).sum()))

            record = {
                **row,
                "legacy_relevance": row.get("relevance"),
                "legacy_relevance_kernel": row.get("relevance_kernel"),
                "relevance": relevance.tolist(),
                "relevance_kernel": {
                    "type": (
                        "exact_evidence_shapley_edit_logprob_contrast"
                        if args.mode == "exact"
                        else "leave_one_evidence_out_edit_logprob_contrast"
                    ),
                    "players": players,
                    "coalitions": len(values),
                    "value": (
                        "mean_logp(correction_edit_tokens)"
                        "-mean_logp(original_edit_tokens)"
                    ),
                    "target": "edited_tokens_only",
                    "signed_attribution": signed.tolist(),
                    "normalized_relevance": probabilities.tolist(),
                    "effective_mass": mass,
                    "game_span": game_span,
                    "attribution_l1": float(np.abs(signed).sum()),
                    "zero_attribution_policy": "uniform_relevance",
                },
                "evidence_attribution": {
                    "mode": args.mode,
                    "coalition_values": {
                        str(k): values[k] for k in sorted(values)
                    },
                    "original_edit_tokens": int(original_mask.sum()),
                    "correction_edit_tokens": int(correction_mask.sum()),
                },
            }
            stream.write(
                json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            )
            stream.flush()
            print(json.dumps({
                "scored": index + 1,
                "total": len(rows),
                "task_id": row.get("task_id"),
                "effective_mass": mass,
                "game_span": game_span,
                "signed_attribution": signed.tolist(),
            }, sort_keys=True), flush=True)

    report = {
        "schema": "eesd-shapley-relevance-v1",
        "input_sha256": sha(args.input),
        "model_config_sha256": sha(args.model_config),
        "model_id": model_id,
        "model_revision": revision,
        "adapter_sha256": adapter_hash,
        "mode": args.mode,
        "value_function": (
            "contrastive mean teacher-forced log-probability "
            "on edit tokens only"
        ),
        "population_trajectories": population,
        "trajectories": len(rows),
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "selected_input_indices": selected_indices,
        "uniform_fallbacks": fallbacks,
        "mean_changed_tokens": float(np.mean(changed_tokens)),
        "mean_effective_mass": float(np.mean(masses)),
        "min_effective_mass": float(np.min(masses)),
        "max_effective_mass": float(np.max(masses)),
        "mean_game_span": float(np.mean(game_spans)),
        "median_game_span": float(np.median(game_spans)),
        "mean_attribution_l1": float(np.mean(attribution_l1)),
        "output_sha256": sha(output_path),
        "hidden_evidence_used": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
