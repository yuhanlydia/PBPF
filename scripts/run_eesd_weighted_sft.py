#!/usr/bin/env python3
"""Train one locked self-distillation arm from scored EESD correction trajectories.

The supervised pull is scaled by the precomputed trajectory weight. Selected EESD
arms additionally use forward KL from the frozen previous policy to the student on
response tokens. The explicit eed_no_anchor ablation sets this KL coefficient to zero.

This runner trains LoRA parameters only and keeps the quantized base model frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import yaml


ANCHORED_RULES = {
    "fixed_mass_dirichlet",
    "eed_mean_no_uncertainty",
    "eesd_full",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path, rule: str):
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") not in {"train", "development"}:
                raise ValueError("scored correction split must be train or development")
            weights = row.get("training_weights", {})
            if rule not in weights:
                raise ValueError(f"training rule {rule} absent from scored correction")
            weight = float(weights[rule])
            if not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError("trajectory weight must lie in [0,1]")
            rows.append({**row, "_weight": weight})
    if not rows:
        raise ValueError("no scored correction rows")
    if not any(row["_weight"] > 0 for row in rows):
        raise ValueError("selected rule has no positive-weight trajectories")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--rule", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous-adapter", type=Path)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--anchor-beta", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    args = p.parse_args()

    from pbpf.eesd.distillation import TRAIN_RULES
    if args.rule not in TRAIN_RULES or args.rule == "no_update":
        raise ValueError("train one of the seven update rules; no_update evaluates the previous policy directly")
    if (
        args.max_steps < 1
        or args.gradient_accumulation < 1
        or args.max_length < 64
        or args.learning_rate <= 0
        or args.anchor_beta < 0
        or args.lora_r < 1
        or args.lora_alpha < 1
    ):
        raise ValueError("invalid training hyperparameters")

    model_cfg = yaml.safe_load(args.model_config.read_text())
    model_id, revision = model_cfg.get("model_id"), model_cfg.get("revision")
    if not isinstance(model_id, str) or not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("model config requires pinned model_id and 40-character revision")

    rows = load_rows(args.input, args.rule)
    args.output.mkdir(parents=True, exist_ok=False)
    anchor_beta = args.anchor_beta if args.rule in ANCHORED_RULES else 0.0

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            quantization_config=quant,
        )

    student_base = prepare_model_for_kbit_training(load_base())
    student_base.gradient_checkpointing_enable()
    if args.previous_adapter:
        student = PeftModel.from_pretrained(
            student_base, str(args.previous_adapter), is_trainable=True
        )
    else:
        student = get_peft_model(
            student_base,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )
    student.train()

    reference = None
    if anchor_beta > 0:
        reference_base = load_base()
        if args.previous_adapter:
            reference = PeftModel.from_pretrained(
                reference_base, str(args.previous_adapter), is_trainable=False
            )
        else:
            reference = reference_base
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)

    def tokenize(row):
        prompt = row["prompt"]
        correction = row["correction"]
        if not isinstance(prompt, str) or not isinstance(correction, str) or not correction.strip():
            raise ValueError("prompt/correction must be nonempty strings")
        kwargs = {"enable_thinking": False} if "Qwen3" in model_id else {}
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(
            correction + (tokenizer.eos_token or ""),
            add_special_tokens=False,
        )["input_ids"]
        room = args.max_length - len(prompt_ids)
        if room <= 0:
            raise ValueError("prompt exceeds max_length before response")
        completion_ids = completion_ids[:room]
        if not completion_ids:
            raise ValueError("correction has no trainable tokens after truncation")
        ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return (
            torch.tensor([ids], dtype=torch.long, device=student.device),
            torch.tensor([labels], dtype=torch.long, device=student.device),
        )

    positive = [row for row in rows if row["split"] == "train" and row["_weight"] > 0]
    development = [row for row in rows if row["split"] == "development"]
    if not positive:
        raise ValueError("no positive-weight training rows")
    if not development:
        raise ValueError("development correction rows are required but never used for gradients")
    rng = random.Random(args.seed)
    order = list(range(len(positive)))
    rng.shuffle(order)
    cursor = 0
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    micro_steps = 0
    totals = {"loss": 0.0, "ce": 0.0, "kl": 0.0, "weight": 0.0}

    while optimizer_steps < args.max_steps:
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        row = positive[order[cursor]]
        cursor += 1
        input_ids, labels = tokenize(row)
        attention_mask = torch.ones_like(input_ids)
        output = student(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        shift_logits = output.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels.ne(-100)
        if not mask.any():
            raise ValueError("trajectory produced no response-token loss")
        ce_tokens = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]).float(),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape_as(shift_labels)
        ce = ce_tokens[mask].mean()

        kl = torch.zeros((), device=ce.device, dtype=torch.float32)
        if reference is not None:
            with torch.no_grad():
                ref_logits = reference(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits[:, :-1, :]
            # Compute only at response positions. Full-vocabulary forward KL keeps the
            # previous policy as the anchor without selecting favored tokens.
            s = shift_logits[mask].float()
            r = ref_logits[mask].float()
            r_log = F.log_softmax(r, dim=-1)
            s_log = F.log_softmax(s, dim=-1)
            kl = (r_log.exp() * (r_log - s_log)).sum(dim=-1).mean()

        weight = float(row["_weight"])
        loss = weight * ce + anchor_beta * kl
        (loss / args.gradient_accumulation).backward()
        micro_steps += 1
        totals["loss"] += float(loss.detach())
        totals["ce"] += float(ce.detach())
        totals["kl"] += float(kl.detach())
        totals["weight"] += weight

        if micro_steps % args.gradient_accumulation == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if optimizer_steps % 10 == 0 or optimizer_steps == args.max_steps:
                print(
                    json.dumps(
                        {
                            "optimizer_step": optimizer_steps,
                            "micro_steps": micro_steps,
                            "mean_loss": totals["loss"] / micro_steps,
                            "mean_ce": totals["ce"] / micro_steps,
                            "mean_kl": totals["kl"] / micro_steps,
                            "mean_weight": totals["weight"] / micro_steps,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    adapter_dir = args.output / "adapter"
    student.save_pretrained(adapter_dir)
    report = {
        "schema": "eesd-weighted-sft-v1",
        "rule": args.rule,
        "model_id": model_id,
        "revision": revision,
        "input_sha256": sha(args.input),
        "model_config_sha256": sha(args.model_config),
        "previous_adapter": str(args.previous_adapter) if args.previous_adapter else None,
        "seed": args.seed,
        "positive_training_trajectories": len(positive),
        "development_trajectories": len(development),
        "all_trajectories": len(rows),
        "sources": len({row["source_component_id"] for row in positive}),
        "max_steps": args.max_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "anchor_beta": anchor_beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "mean_loss": totals["loss"] / micro_steps,
        "mean_ce": totals["ce"] / micro_steps,
        "mean_kl": totals["kl"] / micro_steps,
        "mean_training_weight": totals["weight"] / micro_steps,
    }
    (args.output / "training-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
