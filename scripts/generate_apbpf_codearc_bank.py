#!/usr/bin/env python3
"""Generate a locked CodeARC candidate bank from public replay examples only.

No candidate is executed or selected here. Use a sandbox mounting only the
public input, immutable source/model files, and this generation output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from pbpf.apbpf.code_extraction import extract_solution


MODELS = {
    "qwen": ("Qwen/Qwen2.5-Coder-7B-Instruct", "c03e6d358207e414f1eca0bb1891e29f1db0e242"),
    "deepseek": ("deepseek-ai/deepseek-coder-6.7b-instruct", "e5d64addd26a6a1db0f9b863abf6ee3141936807"),
    "qwen25_1p5b": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "2e1fd397ee46e1388853d2af2c993145b0f1098a"),
    "qwen25_7b": ("Qwen/Qwen2.5-Coder-7B-Instruct", "c03e6d358207e414f1eca0bb1891e29f1db0e242"),
    "qwen3_8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "deepseek_6p7b": ("deepseek-ai/deepseek-coder-6.7b-instruct", "e5d64addd26a6a1db0f9b863abf6ee3141936807"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha(directory):
    """Stable digest for a LoRA adapter directory."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError('adapter must be a directory')
    digest = hashlib.sha256()
    files = [p for p in sorted(root.rglob('*')) if p.is_file()]
    if not files:
        raise ValueError('adapter directory is empty')
    for path in files:
        rel = path.relative_to(root).as_posix().encode()
        digest.update(len(rel).to_bytes(4, 'big'))
        digest.update(rel)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=MODELS, default="qwen")
    parser.add_argument("--adapter", type=Path, help="optional previous-round LoRA adapter")
    parser.add_argument("--split", choices=["train", "development", "primary"], required=True)
    parser.add_argument("--components", type=int, default=0, help="0 means all source components")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if min(args.candidates, args.max_new_tokens) < 1 or args.temperature <= 0 or min(args.offset, args.components) < 0:
        parser.error("invalid candidate/generation bounds")
    manifest = json.loads((args.public_root / "manifest.json").read_text())
    if sha(args.public_root / "tasks.jsonl") != manifest["public_tasks_sha256"]:
        raise ValueError("public task checksum mismatch")
    with (args.public_root / "tasks.jsonl").open() as stream:
        tasks = [json.loads(line) for line in stream]
    tasks = [row for row in tasks if row["split"] == args.split]
    # One representative per pre-locked source component avoids duplicate groups.
    grouped = {}
    for row in tasks:
        grouped.setdefault(row["source_component_id"], row)
    tasks = list(grouped.values())[args.offset:]
    if args.components:
        tasks = tasks[:args.components]
    if not tasks:
        raise ValueError("selected generation inventory is empty")
    for row in tasks:
        if set(row) != {"task_id", "source_component_id", "split", "protocol", "task_text", "visible_tests"}:
            raise ValueError("generation only accepts whitelisted public record fields")
        if [test["id"] for test in row["visible_tests"]] != ["0", "1", "2", "3"]:
            raise ValueError("generation requires exactly the four public invocations")
    model_id, revision = MODELS[args.family]
    adapter_sha256 = tree_sha(args.adapter) if args.adapter else None
    config = {"schema": "apbpf-codearc-generation-v1", "family": args.family, "model": model_id, "revision": revision,
              "adapter_sha256": adapter_sha256,
              "source_sha256": sha(__file__), "public_manifest_sha256": sha(args.public_root / "manifest.json"),
              "public_tasks_sha256": manifest["public_tasks_sha256"], "split": args.split,
              "components": len(tasks), "offset": args.offset, "candidates": args.candidates,
              "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": .95,
              "seed": args.seed, "task_ids": [row["task_id"] for row in tasks],
              "source_component_ids": [row["source_component_id"] for row in tasks],
              "population_selection": "pre-hidden fixed source order; no execution/quality filtering",
              "claim_status": "candidate-generation-no-efficacy-claim"}
    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text()) != config:
            raise ValueError("resume identity mismatch; use a new output directory")
    else:
        write_once(run_path, config)
    # Only now import and load the actor. The inventory is already immutable.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=True,
        device_map={"": 0}, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    started = time.monotonic()
    for index, row in enumerate(tasks):
        output = args.output / (row["task_id"].replace("/", "-") + ".json")
        if output.exists():
            old = json.loads(output.read_text())
            if old["task_id"] != row["task_id"] or len(old["candidates"]) != args.candidates:
                raise ValueError("invalid completed task on resume")
            continue
        task_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{row['task_id']}".encode()).digest()[:4], "big")
        torch.manual_seed(task_seed)
        template_kwargs = {"enable_thinking": False} if args.family == "qwen3_8b" else {}
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": "You write Python code that generalizes from observed input-output examples."},
            {"role": "user", "content": row["task_text"]}], tokenize=False, add_generation_prompt=True, **template_kwargs)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        begin = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=True, temperature=args.temperature, top_p=.95,
                num_return_sequences=args.candidates, max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id)
        continuations = generated[:, inputs.input_ids.shape[1]:]
        candidates = []
        for candidate_index, tokens in enumerate(continuations):
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            ids = tokens.cpu().tolist()
            stop = ids.index(tokenizer.eos_token_id) + 1 if tokenizer.eos_token_id in ids else len(ids)
            candidates.append({"candidate_id": f"{row['task_id']}/{args.family}/{candidate_index}",
                               "code": extract_solution(text), "raw_completion": text,
                               "generated_tokens": stop, "hit_token_cap": tokenizer.eos_token_id not in ids})
        result = {"task_id": row["task_id"], "source_component_id": row["source_component_id"],
                  "split": row["split"], "seed": task_seed, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "candidates": candidates, "elapsed_seconds": time.monotonic() - begin}
        temporary = output.with_suffix(".partial")
        write_once(temporary, result)
        os.replace(temporary, output)
        print(json.dumps({"completed": index + 1, "total": len(tasks), "task_id": row["task_id"],
                          "tokens": sum(c["generated_tokens"] for c in candidates),
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
    completion = {"schema": "apbpf-candidate-generation-complete-v1", "run_sha256": sha(run_path),
                  "files": {row["task_id"].replace("/", "-") + ".json":
                            sha(args.output / (row["task_id"].replace("/", "-") + ".json")) for row in tasks}}
    complete_path = args.output / "complete.json"
    if complete_path.exists():
        if json.loads(complete_path.read_text()) != completion:
            raise ValueError("completion manifest mismatch")
    else:
        write_once(complete_path, completion)


if __name__ == "__main__":
    main()
