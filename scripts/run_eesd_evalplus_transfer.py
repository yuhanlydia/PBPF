#!/usr/bin/env python3
"""Generate and optionally evaluate HumanEval+/MBPP+ with a base or EESD adapter.

Generation writes the official EvalPlus custom sample schema (task_id, solution).
With --evaluate, the script delegates code post-processing and execution to the
installed EvalPlus CLI, then summarizes one-sample pass@1 from its result JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import yaml


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(directory: Path) -> str:
    root = Path(directory)
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


def summarize_evalplus(path: Path) -> dict:
    result = json.loads(path.read_text())
    rows = result.get("eval")
    if not isinstance(rows, dict) or not rows:
        raise ValueError("EvalPlus result JSON has no eval rows")
    base, plus, total = 0, 0, 0
    for task_id, samples in rows.items():
        if not isinstance(samples, list) or len(samples) != 1:
            raise ValueError(f"expected exactly one sample for {task_id}")
        sample = samples[0]
        total += 1
        b = str(sample.get("base_status", "")).lower() == "pass"
        p = str(sample.get("plus_status", "")).lower() == "pass"
        base += int(b)
        plus += int(b and p)
    return {
        "tasks": total,
        "base_pass_at_1": base / total,
        "plus_pass_at_1": plus / total,
        "base_passes": base,
        "plus_passes": plus,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--evaluate", action="store_true")
    args = p.parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("positive generation budget required")

    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    if args.dataset == "humaneval":
        problems = get_human_eval_plus()
    else:
        problems = get_mbpp_plus()
    if not problems:
        raise ValueError("EvalPlus dataset is empty")

    cfg = yaml.safe_load(args.model_config.read_text())
    model_id, revision = cfg.get("model_id"), cfg.get("revision")
    if not isinstance(model_id, str) or not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("pinned model config required")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, local_files_only=True, device_map={"": 0},
        torch_dtype=torch.bfloat16, quantization_config=quant,
    )
    adapter_hash = None
    if args.adapter:
        from peft import PeftModel
        adapter_hash = tree_sha(args.adapter)
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()

    args.output.mkdir(parents=True, exist_ok=False)
    samples_path = args.output / "samples.jsonl"
    generated = []
    with samples_path.open("x") as stream:
        for index, (task_id, problem) in enumerate(problems.items()):
            user = (
                "Write a complete self-contained Python solution for the following task. "
                "Return only code and include the requested function signature.\n\n"
                + problem["prompt"]
            )
            kwargs = {"enable_thinking": False} if "Qwen3" in model_id else {}
            ids = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are an expert Python programmer."},
                    {"role": "user", "content": user},
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **kwargs,
            ).to(model.device)
            started = time.monotonic()
            with torch.inference_mode():
                out = model.generate(
                    ids,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )[0, ids.shape[1]:]
            raw = tokenizer.decode(out, skip_special_tokens=True)
            record = {"task_id": task_id, "solution": raw}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            generated.append(
                {
                    "task_id": task_id,
                    "tokens": int(len(out)),
                    "seconds": time.monotonic() - started,
                }
            )
            if (index + 1) % 20 == 0:
                print(json.dumps({"generated": index + 1, "total": len(problems)}), flush=True)

    report = {
        "schema": "eesd-evalplus-transfer-v1",
        "dataset": args.dataset,
        "model_id": model_id,
        "revision": revision,
        "adapter_sha256": adapter_hash,
        "tasks": len(problems),
        "samples_sha256": sha(samples_path),
        "generation_tokens": sum(x["tokens"] for x in generated),
        "generation_seconds": sum(x["seconds"] for x in generated),
        "decode": "greedy",
        "evalplus_summary": None,
    }

    if args.evaluate:
        sanitize = shutil.which("evalplus.sanitize")
        evaluate = shutil.which("evalplus.evaluate")
        if not sanitize or not evaluate:
            raise RuntimeError("EvalPlus console scripts are not installed")
        subprocess.check_call(
            [sanitize, "--samples", str(samples_path), "--dataset", args.dataset]
        )
        sanitized = samples_path.with_name(samples_path.stem + "-sanitized.jsonl")
        if not sanitized.exists():
            raise FileNotFoundError(f"EvalPlus sanitizer did not create {sanitized}")
        subprocess.check_call(
            [
                evaluate,
                "--dataset", args.dataset,
                "--samples", str(sanitized),
                "--i-just-wanna-run",
            ]
        )
        result_path = sanitized.with_name(sanitized.name.replace(".jsonl", ".eval_results.json"))
        if not result_path.exists():
            legacy = sanitized.with_name(sanitized.name.replace(".jsonl", "_eval_results.json"))
            result_path = legacy if legacy.exists() else result_path
        if not result_path.exists():
            raise FileNotFoundError("EvalPlus result JSON not found")
        report["sanitized_samples_sha256"] = sha(sanitized)
        report["evalplus_result_sha256"] = sha(result_path)
        report["evalplus_summary"] = summarize_evalplus(result_path)

    (args.output / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
