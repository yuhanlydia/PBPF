#!/usr/bin/env python3
"""Generate and publicly re-execute one correction per source for EESD.

Input is the public-only artifact emitted by prepare_eesd_public_corrections.py.
The model never receives future tests/outcomes or gold repairs. Relevance is a
frozen signed-hash cosine kernel between the original->correction code change and
each public execution descriptor; this is a controlled default, not a learned
semantic-dependence estimator.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

from pbpf.apbpf.code_extraction import extract_solution
from pbpf.apbpf.codearc_execution import execute_call
from pbpf.apbpf.rbr_execution import execute_stdin
from pbpf.apbpf.rbr_prompt import extract_program
from pbpf.eesd.evidence import cosine_relevance_weights
from pbpf.real_gate import FrozenTextEncoder


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


def user_prompt(row, domain: str) -> str:
    mode = "stdin/stdout Python program" if domain == "rbr" else "Python function solution"
    evidence = [
        {k: test[k] for k in ("id","input","expected","actual","stderr","outcome")}
        | ({"expected_error": test["expected_error"]} if "expected_error" in test else {})
        for test in row["tests"]
    ]
    payload = {
        "task": row["task_text"],
        "program": row["candidate"],
        "public_executions": evidence,
    }
    return (
        f"Repair the following {mode}. Use only the public execution evidence. "
        "Return only complete Python source code.\n" +
        json.dumps(payload, sort_keys=True, ensure_ascii=False)
    )


def patch_text(before: str, after: str) -> str:
    value = "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(), fromfile="before.py", tofile="after.py", lineterm=""
    ))
    return value if value.strip() else after


def execution_descriptor(test) -> str:
    return json.dumps(
        {k: test[k] for k in ("input","expected","actual","stderr","outcome")},
        sort_keys=True,
        ensure_ascii=False,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public-bank", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--domain", choices=["rbr","codearc"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--relevance-strength", type=float, default=16.0)
    p.add_argument("--timeout", type=float, default=6.0)
    args = p.parse_args()
    if args.max_new_tokens < 1 or args.relevance_strength < 0 or args.timeout <= 0:
        raise ValueError("invalid generation/relevance/execution budget")

    rows = [json.loads(line) for line in args.public_bank.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("schema") != "eesd-public-correction-row-v1" for row in rows):
        raise ValueError("public-only EESD correction bank required")
    forbidden = {"reference_code", "future_tests", "future_outcomes", "gold_patch"}
    if any(forbidden & set(row) for row in rows):
        raise ValueError("public bank contains forbidden private fields")

    model_cfg = yaml.safe_load(args.model_config.read_text())
    model_id, revision = model_cfg.get("model_id"), model_cfg.get("revision")
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

    encode = FrozenTextEncoder(256)
    execute = execute_stdin if args.domain == "rbr" else execute_call
    args.output.mkdir(parents=True, exist_ok=False)
    output_path = args.output / "corrections.jsonl"
    generated = []
    with output_path.open("x") as stream:
        for index, row in enumerate(rows):
            prompt = user_prompt(row, args.domain)
            kwargs = {"enable_thinking": False} if "Qwen3" in model_id else {}
            messages = [
                {"role": "system", "content": "You repair Python programs using public execution feedback."},
                {"role": "user", "content": prompt},
            ]
            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", **kwargs
            ).to(model.device)
            started = time.monotonic()
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )[0, input_ids.shape[1]:]
            raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
            correction = extract_program(raw) if args.domain == "rbr" else extract_solution(raw)
            empty_fallback = not correction.strip()
            if empty_fallback:
                correction = row["candidate"]

            after_tests = []
            for test in row["tests"]:
                result = execute(correction, test, timeout=args.timeout)
                after_tests.append(result)
            after_outcomes = [result["outcome"] for result in after_tests]

            change = patch_text(row["candidate"], correction)
            query = encode(change)
            history = np.stack([encode(execution_descriptor(test)) for test in row["tests"]])
            relevance = cosine_relevance_weights(query, history, args.relevance_strength)

            record = {
                "schema": "eesd-correction-trajectory-v1",
                "trajectory_id": hashlib.sha256(
                    f"{model_id}\0{revision}\0{adapter_hash}\0{row['task_id']}".encode()
                ).hexdigest(),
                "source_component_id": row["source_component_id"],
                "problem_id": row.get("problem_id"),
                "task_id": row["task_id"],
                "split": row["split"],
                "prompt": prompt,
                "original": row["candidate"],
                "correction": correction,
                "before_outcomes": row["outcomes"],
                "after_outcomes": after_outcomes,
                "relevance": relevance.tolist(),
                "relevance_kernel": {
                    "type": "frozen_signed_hash_cosine",
                    "dimension": 256,
                    "strength": args.relevance_strength,
                    "query": "unified_diff(original, correction)",
                    "history": "public execution descriptor only",
                },
                "model_id": model_id,
                "model_revision": revision,
                "adapter_sha256": adapter_hash,
                "raw_completion": raw,
                "generated_tokens": int(len(generated_ids)),
                "generation_seconds": time.monotonic() - started,
                "empty_generation_fallback_to_original": empty_fallback,
                "correction_code_sha256": hashlib.sha256(correction.encode()).hexdigest(),
                "public_after_tests": [
                    {k: value for k, value in result.items() if k in {
                        "outcome","stdout","stderr","returncode","timed_out"
                    }}
                    for result in after_tests
                ],
            }
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            generated.append(record)
            print(json.dumps({
                "generated": index + 1,
                "total": len(rows),
                "source": row["source_component_id"],
                "split": row["split"],
                "before_passes": row["outcomes"].count("PASS"),
                "after_passes": after_outcomes.count("PASS"),
            }, sort_keys=True), flush=True)

    report = {
        "schema": "eesd-correction-generation-v1",
        "public_bank_sha256": sha(args.public_bank),
        "model_config_sha256": sha(args.model_config),
        "model_id": model_id,
        "revision": revision,
        "adapter_sha256": adapter_hash,
        "domain": args.domain,
        "trajectories": len(generated),
        "split_counts": {
            split: sum(row["split"] == split for row in generated)
            for split in ("train","development")
        },
        "before_visible_pass_rate": float(np.mean([
            value == "PASS" for row in generated for value in row["before_outcomes"]
        ])),
        "after_visible_pass_rate": float(np.mean([
            value == "PASS" for row in generated for value in row["after_outcomes"]
        ])),
        "output_sha256": sha(output_path),
        "private_fields_available_to_generator": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
