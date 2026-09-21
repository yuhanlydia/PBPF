"""Cache frozen CodeBERT features of public train/development fields only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from pbpf.belief.semantic_features import public_contexts, pool_inputs, fit_projection, apply_projection
from pbpf.real_gate import validate_rbr_cache


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--revision', default='3b0952feddeffad0063f274080e3c23d75e7eb39')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-dim', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--projection-seed', type=int, default=91701)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('feature outputs are create-once')
    if args.batch_size < 1:
        raise ValueError('positive encoding batch size required')
    payload = validate_rbr_cache(json.loads(args.cache.read_text()))
    rows = {s: [r for r in payload['records'] if r['split'] == s] for s in ('train', 'development')}
    if any(not v for v in rows.values()):
        raise ValueError('both training and development required')
    model_files = ['config.json', 'pytorch_model.bin', 'merges.txt', 'vocab.json',
                   'tokenizer_config.json', 'special_tokens_map.json']
    hashes = {}
    for name in model_files:
        record = args.model_dir / '.cache/huggingface/download' / (name + '.metadata')
        if record.read_text().splitlines()[0] != args.revision:
            raise ValueError('download revision metadata mismatch')
        hashes[name] = sha(args.model_dir / name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32).to(device).eval()
    if model.config.model_type != 'roberta':
        raise ValueError('this input-token span protocol is locked to RoBERTa/CodeBERT')
    model.requires_grad_(False)
    references = {s: [public_contexts(row) for row in values] for s, values in rows.items()}
    unique = list(dict.fromkeys(pair for values in references.values() for row in values
                               for pairs in row.values() for pair in pairs))
    index = {key: i for i, key in enumerate(unique)}
    vectors, truncations = [], dict(input=0, context=0, single_text=0)
    with torch.inference_mode():
        for start in range(0, len(unique), args.batch_size):
            encodings, spans = [], []
            for text, context in unique[start:start + args.batch_size]:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                cap = 128 if context is not None else 510
                truncations['input' if context is not None else 'single_text'] += int(len(tokens) > cap)
                first = tokens[:cap] or [tokenizer.unk_token_id]
                second = None
                if context is not None:
                    second = tokenizer.encode(context, add_special_tokens=False)
                    truncations['context'] += int(len(second) > 380)
                    second = second[:380]
                ids = tokenizer.build_inputs_with_special_tokens(first, second)
                if ids[1:1 + len(first)] != first or len(ids) > 512:
                    raise ValueError('unexpected CodeBERT token layout')
                encodings.append(dict(input_ids=ids, attention_mask=[1] * len(ids)))
                spans.append(len(first))
            encoded = tokenizer.pad(encodings, padding=True, return_tensors='pt').to(device)
            mask = torch.zeros_like(encoded['attention_mask'], dtype=torch.bool)
            for i, length in enumerate(spans):
                mask[i, 1:1 + length] = True
            pooled = pool_inputs(model(**encoded).last_hidden_state, mask)
            vectors.extend(pooled.cpu().numpy())
            if start == 0 or (start // args.batch_size) % 50 == 0:
                print(json.dumps(dict(encoded=min(start + args.batch_size, len(unique)), total=len(unique))), flush=True)
    vectors = np.asarray(vectors, dtype='float32')
    raw = {}
    for split, refs in references.items():
        raw[split] = {name: np.stack([vectors[[index[key] for key in row[name]]] for row in refs])
                      for name in ('task', 'candidate', 'tests')}
        for name in ('task', 'candidate'):
            raw[split][name] = raw[split][name][:, 0]
    transform = fit_projection(raw['train'], args.feature_dim, seed=args.projection_seed)
    metadata = dict(schema='pbpf-frozen-features-v1', dataset_sha256=sha(args.cache),
        model_id='microsoft/codebert-base', model_revision=args.revision, model_file_sha256=hashes,
        ids={s:[r['task_id'] for r in v] for s,v in rows.items()}, test_encoded=False,
        public_fields=['task_text', 'candidate', 'test.input'], pooling='input_token_mean_with_code_task_context',
        input_token_cap=128, context_token_cap=380, single_token_cap=510, truncations=truncations,
        projection='fixed_gaussian_after_training_only_per_role_centering_then_l2',
        projection_seed=args.projection_seed, feature_dim=args.feature_dim,
        device=str(device), torch_version=torch.__version__,
        preparation_source_sha256={str(Path(__file__).name):sha(Path(__file__)),
            'semantic_features.py':sha(Path(__file__).resolve().parents[1]/'src/pbpf/belief/semantic_features.py')})
    arrays = {f'{s}_{name}':array for s,values in raw.items()
              for name,array in apply_projection(values,transform).items()}
    arrays.update({f'transform_{key}':value for key,value in transform.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata,sort_keys=True), **arrays)
    args.output.with_suffix('.json').write_text(json.dumps({**metadata,'feature_file_sha256':sha(args.output)},indent=2)+'\n')
    print(json.dumps(dict(phase='complete',output=str(args.output),sha256=sha(args.output),truncations=truncations)),flush=True)


if __name__ == '__main__':
    main()
