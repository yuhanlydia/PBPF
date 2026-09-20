#!/usr/bin/env python3
"""Generate exactly eight repairs per fixed source from public inputs only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from pbpf.apbpf.rbr_prompt import prompt_for, extract_program
from pbpf.apbpf.codearc_bank import load_bank

MODELS = {
    'qwen': ('Qwen/Qwen2.5-Coder-7B-Instruct', 'c03e6d358207e414f1eca0bb1891e29f1db0e242'),
    'deepseek': ('deepseek-ai/deepseek-coder-6.7b-instruct', 'e5d64addd26a6a1db0f9b863abf6ee3141936807'),
    'qwen25_1p5b': ('Qwen/Qwen2.5-Coder-1.5B-Instruct', '2e1fd397ee46e1388853d2af2c993145b0f1098a'),
    'qwen25_7b': ('Qwen/Qwen2.5-Coder-7B-Instruct', 'c03e6d358207e414f1eca0bb1891e29f1db0e242'),
    'qwen3_8b': ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'),
    'deepseek_6p7b': ('deepseek-ai/deepseek-coder-6.7b-instruct', 'e5d64addd26a6a1db0f9b863abf6ee3141936807'),
    'seed_coder_8b': ('ByteDance-Seed/Seed-Coder-8B-Instruct', 'c62d428b84d52a7f1bb38d2aa72a79d6d5f5e614'),
    'qwen3_coder_30b': ('Qwen/Qwen3-Coder-30B-A3B-Instruct', 'b2cff646eb4bb1d68355c01b18ae02e7cf42d120'),
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
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--public-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--family', choices=MODELS, default='qwen')
    p.add_argument('--adapter', type=Path, help='optional previous-round LoRA adapter')
    p.add_argument('--split', choices=['train', 'development', 'primary'], required=True)
    p.add_argument('--components', type=int, default=0)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--max-input-tokens', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--candidates', type=int, default=8)
    args = p.parse_args()
    if min(args.components, args.offset) < 0 or min(args.max_input_tokens, args.max_new_tokens, args.candidates) < 1:
        raise ValueError('invalid generation limits')
    manifest = json.loads((args.public_root/'manifest.json').read_text())
    if manifest['schema'] != 'apbpf-rbr-generated-materialization-v1' or sha(args.public_root/'tasks.jsonl') != manifest['public_tasks_sha256']:
        raise ValueError('public materialization checksum/schema mismatch')
    with (args.public_root/'tasks.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    rows = [r for r in rows if r['split'] == args.split][args.offset:]
    if args.components:
        rows = rows[:args.components]
    if not rows or len({r['source_component_id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique source inventory required')
    model_id, revision = MODELS[args.family]
    adapter_sha256 = tree_sha(args.adapter) if args.adapter else None
    root = Path(__file__).resolve().parents[1]
    config = {'schema': 'apbpf-rbr-generation-v1', 'family': args.family, 'model': model_id, 'revision': revision,
              'source_sha256': sha(__file__), 'adapter_sha256': adapter_sha256,
              'prompt_source_sha256': sha(root/'src/pbpf/apbpf/rbr_prompt.py'),
              'inventory_source_sha256': sha(root/'src/pbpf/apbpf/codearc_bank.py'),
              'public_tasks_sha256': manifest['public_tasks_sha256'], 'public_manifest_sha256': sha(args.public_root/'manifest.json'),
              'split': args.split, 'seed': args.seed, 'offset': args.offset, 'candidates': args.candidates,
              'components': len(rows), 'task_ids': [r['task_id'] for r in rows],
              'source_component_ids': [r['source_component_id'] for r in rows],
              'max_input_tokens': args.max_input_tokens, 'max_new_tokens': args.max_new_tokens,
              'temperature': .8, 'top_p': .95, 'prompt_policy': 'adaptively cap public fields while retaining all four example slots',
              'population_selection': 'fixed pre-hidden source order; no outcome filtering',
              'claim_status': 'exploratory-predeclared-candidate-generation-only'}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'run.json').exists():
        if json.loads((args.output/'run.json').read_text()) != config:
            raise ValueError('resume identity changed')
    else:
        write_once(args.output/'run.json', config)
    if (args.output/'complete.json').exists():
        load_bank(args.output)
        print(json.dumps({'status': 'verified_complete_resume', 'components': len(rows)}), flush=True)
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=True,
        device_map={'': 0}, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16))
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval(); started = time.monotonic()
    for i, row in enumerate(rows):
        path = args.output/(row['task_id'].replace('/', '-')+'.json')
        checksum = path.with_suffix('.sha256')
        if path.exists():
            if not checksum.exists() or checksum.read_text().strip() != sha(path):
                raise ValueError('resumed candidate record checksum missing or mismatched')
            continue
        template_kwargs = {'enable_thinking': False} if args.family in {'qwen3_8b', 'qwen3_coder_30b'} else None
        prompt, metadata = prompt_for(tokenizer, row, max_input_tokens=args.max_input_tokens,
                                      chat_template_kwargs=template_kwargs)
        task_seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{row["task_id"]}'.encode()).digest()[:4], 'big')
        torch.manual_seed(task_seed)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
        if inputs.input_ids.shape[1] != metadata['input_tokens']:
            raise ValueError('token count changed after prompt construction')
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=True, temperature=.8, top_p=.95,
                num_return_sequences=args.candidates, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        candidates = []
        for j, tokens in enumerate(generated[:, inputs.input_ids.shape[1]:]):
            text = tokenizer.decode(tokens, skip_special_tokens=True); ids = tokens.cpu().tolist()
            count = ids.index(tokenizer.eos_token_id)+1 if tokenizer.eos_token_id in ids else len(ids)
            candidates.append({'candidate_id': f'{row["task_id"]}/{args.family}/{j}', 'code': extract_program(text),
                               'raw_completion': text, 'generated_tokens': count, 'hit_token_cap': tokenizer.eos_token_id not in ids})
        value = {k: row[k] for k in ('task_id', 'source_component_id', 'split')}
        value.update(seed=task_seed, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), candidates=candidates,
                     prompt_metadata=metadata)
        temporary = path.with_suffix('.partial'); write_once(temporary, value); os.replace(temporary, path)
        with checksum.open('x') as stream: stream.write(sha(path)+'\n')
        print(json.dumps({'completed': i+1, 'total': len(rows), 'task_id': row['task_id'],
                          'elapsed_seconds': time.monotonic()-started}), flush=True)
    complete = {'schema': 'apbpf-candidate-generation-complete-v1', 'run_sha256': sha(args.output/'run.json'),
                'files': {r['task_id'].replace('/', '-')+'.json': sha(args.output/(r['task_id'].replace('/', '-')+'.json')) for r in rows}}
    path = args.output/'complete.json'
    if path.exists():
        if json.loads(path.read_text()) != complete: raise ValueError('completion manifest mismatch')
    else:
        write_once(path, complete)


if __name__ == '__main__':
    main()
