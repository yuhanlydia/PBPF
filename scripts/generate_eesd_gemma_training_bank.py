#!/usr/bin/env python3
"""Generate one Gemma 3 4B candidate per disjoint public train source."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT = Path('/root/PBPF')
MODEL = Path('/root/PBPF-models/gemma3_4b')
PROOF = Path('/root/eesd-migration-20260921/gemma3-4b-model-proof.json')
PROMPT_SOURCE = Path('/root/eesd-migration-20260921/generate_gemma_bank.py')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish(path, value):
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('x') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()


def source_rows(domain):
    if domain == 'runbugrun':
        public = ROOT / 'runs/eesd-data/rbr-public'
    elif domain == 'codearc':
        public = ROOT / 'runs/eesd-data/codearc-public'
    else:
        public = ROOT / 'runs/eesd-data/replay-training-reserve-20260921' / domain / 'public'
    manifest_path = public / 'manifest.json'
    tasks_path = public / 'tasks.jsonl'
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get('public_tasks_sha256')
    if expected is None:
        expected = manifest['tasks']['sha256']
    if expected != sha(tasks_path):
        raise ValueError('public training inventory checksum mismatch')
    if domain.endswith('_replay') and (manifest.get('schema') != 'eesd-replay-public-train-v1'
                                       or manifest.get('counts') != {'train': 200}):
        raise ValueError('Replay training reserve manifest differs')
    rows = [json.loads(line) for line in tasks_path.read_text().splitlines()
            if json.loads(line)['split'] == 'train']
    if domain == 'codearc':
        grouped = {}
        for row in rows:
            component = row['source_component_id']
            if component not in grouped or row['task_id'] < grouped[component]['task_id']:
                grouped[component] = row
        rows = list(grouped.values())
    expected_count = {'runbugrun': 321, 'codearc': 212,
                      'apps_replay': 200, 'codecontests_replay': 200}[domain]
    if len(rows) != expected_count or len({row['source_component_id'] for row in rows}) != expected_count:
        raise ValueError('unexpected Gemma training source population')
    return public, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--domain', choices=['runbugrun','codearc','apps_replay','codecontests_replay'], required=True)
    p.add_argument('--seed', type=int, choices=[1701], required=True)
    p.add_argument('--offset', type=int, required=True)
    p.add_argument('--components', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    public, all_rows = source_rows(args.domain)
    if args.offset < 0 or args.components < 1 or args.offset + args.components > len(all_rows):
        p.error('invalid training shard bounds')
    rows = all_rows[args.offset:args.offset + args.components]
    proof = json.loads(PROOF.read_text())
    if (proof['model_id'] != 'google/gemma-3-4b-it'
            or proof['revision'] != '093f9f388b31de276ce2de164bdc2081324b9767'
            or proof['snapshot'] != str(MODEL)):
        raise ValueError('Gemma model proof differs')
    for file in proof['files']:
        path = MODEL / file['file']
        if path.stat().st_size != file['bytes']:
            raise ValueError('Gemma model weight size differs from verified proof')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    spec = importlib.util.spec_from_file_location('eesd_gemma_prompt_source', PROMPT_SOURCE)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    prepared = [(row, *source.prompt(tokenizer, row, args.domain)) for row in rows]
    if args.preflight_only:
        print(json.dumps({'status': 'Gemma-train-preflight', 'domain': args.domain,
                          'components': len(prepared), 'max_input_tokens':
                          max(value[3]['input_tokens'] for value in prepared)}), flush=True)
        return
    max_new_tokens = 512 if args.domain == 'codearc' else 1024
    run = {'schema': f'apbpf-{"rbr" if args.domain == "runbugrun" else "codearc"}-generation-v1'
           if args.domain in ('runbugrun','codearc') else 'eesd-gemma-replay-train-generation-v1',
           'domain': args.domain, 'split': 'train', 'seed': args.seed,
           'family': 'gemma3_4b', 'model': proof['model_id'], 'revision': proof['revision'],
           'model_proof_sha256': sha(PROOF), 'generator_source_sha256': sha(__file__),
           'prompt_source_sha256': sha(PROMPT_SOURCE), 'public_manifest_sha256': sha(public / 'manifest.json'),
           'public_tasks_sha256': sha(public / 'tasks.jsonl'),
           'task_ids': [row['task_id'] for row in rows],
           'source_component_ids': [row['source_component_id'] for row in rows],
           'offset': args.offset, 'components': len(rows), 'candidates': 1,
           'max_input_tokens': 4096, 'max_new_tokens': max_new_tokens,
           'quantization': 'bitsandbytes-4bit-fp4-bfloat16',
           'claim_status': 'train-candidate-generation-only'}
    args.output.mkdir(parents=True, exist_ok=True)
    run_file = args.output / 'run.json'
    if run_file.exists():
        if json.loads(run_file.read_text()) != run:
            raise ValueError('Gemma training bank resume identity changed')
    elif any(args.output.iterdir()):
        raise ValueError('orphan Gemma training bank artifacts')
    else:
        publish(run_file, run)
    if list(args.output.glob('*.partial')):
        raise ValueError('partial Gemma training record requires audit')
    from pbpf.apbpf.codearc_bank import load_bank
    if (args.output / 'complete.json').is_file():
        load_bank(args.output)
        print(json.dumps({'status': 'verified-complete-resume'}), flush=True)
        return
    pending = []
    for row, rendered, ids, metadata in prepared:
        path = args.output / (row['task_id'].replace('/', '-') + '.json')
        if path.is_file():
            if not path.with_suffix('.sha256').is_file() or path.with_suffix('.sha256').read_text().strip() != sha(path):
                raise ValueError('Gemma training record checksum differs')
        else:
            pending.append((row, rendered, ids, metadata, path))
    if pending:
        import torch
        from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration
        model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, local_files_only=True,
            device_map={'': 0}, dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='fp4'))
        model.eval()
        from pbpf.apbpf.rbr_prompt import extract_program
        from pbpf.apbpf.code_extraction import extract_solution
        started = time.monotonic()
        for index, (row, rendered, ids, metadata, path) in enumerate(pending, 1):
            seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{row["task_id"]}'.encode()).digest()[:4], 'big')
            torch.manual_seed(seed)
            inputs = tokenizer(rendered, add_special_tokens=(args.domain == 'codearc'),
                               return_tensors='pt').to(model.device)
            if inputs.input_ids.shape[1] != len(ids):
                raise ValueError('Gemma training prompt token identity differs')
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=True, temperature=.8, top_p=.95,
                    max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
            tokens = generated[0, inputs.input_ids.shape[1]:].cpu().tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            code = extract_solution(text) if args.domain == 'codearc' else extract_program(text)
            record = {'task_id': row['task_id'], 'source_component_id': row['source_component_id'],
                      'split': 'train', 'seed': seed, 'prompt_sha256': metadata['rendered_prompt_sha256'],
                      'prompt_metadata': metadata,
                      'candidates': [{'candidate_id': f'{row["task_id"]}/gemma3_4b/0',
                                      'code': code, 'raw_completion': text,
                                      'generated_tokens': len(tokens),
                                      'hit_token_cap': len(tokens) >= max_new_tokens}]}
            publish(path, record)
            with path.with_suffix('.sha256').open('x') as stream:
                stream.write(sha(path) + '\n')
            print(json.dumps({'completed': index, 'total': len(pending),
                              'elapsed_seconds': time.monotonic() - started}), flush=True)
    files = {row['task_id'].replace('/', '-') + '.json':
             sha(args.output / (row['task_id'].replace('/', '-') + '.json')) for row in rows}
    publish(args.output / 'complete.json', {'schema': 'eesd-gemma-training-bank-complete-v1',
                                           'run_sha256': sha(run_file), 'files': files})
    load_bank(args.output)
    print(json.dumps({'status': 'Gemma-training-bank-complete', 'components': len(files)}), flush=True)


if __name__ == '__main__':
    main()
