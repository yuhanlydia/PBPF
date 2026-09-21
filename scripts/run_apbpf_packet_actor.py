#!/usr/bin/env python3
"""Train a frozen-Qwen repair prefix or generate from an actor-only packet.

Run inside apbpf_generator_sandbox.sh. The actor receives only its packet at
/input and its own outputs at /output, never evaluator tests or gate labels.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.code_extraction import extract_solution
from pbpf.conditioning.mixture import sample_components_once
from pbpf.conditioning.soft_prompt import SoftPrefixProjector


def load_packet(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['schema'] != 'apbpf-repair-actor-packet-v1':
        raise ValueError('actor-only repair packet required')
    if set(manifest['files']) != {'actor-rows.jsonl', 'posteriors.npz'}:
        raise ValueError('unexpected actor packet inventory')
    if any(file_sha(directory/name) != checksum for name, checksum in manifest['files'].items()):
        raise ValueError('actor packet checksum mismatch')
    rows = [json.loads(line) for line in (directory/'actor-rows.jsonl').read_text().splitlines()]
    if len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate actor candidate')
    allowed = {'task_id', 'problem_id', 'source_component_id', 'task_text', 'candidate', 'split', 'tests', 'outcomes', 'reference_code'}
    for row in rows:
        if (set(row) - allowed or row['split'] not in {'train', 'development', 'primary'}
                or len(row['tests']) != 4 or len(row['outcomes']) != 4
                or ('reference_code' in row) != (row['split'] != 'primary')):
            raise ValueError('actor packet violates public/target boundary')
        for i, test in enumerate(row['tests']):
            if (set(test) - {'id', 'input', 'expected', 'actual', 'stderr', 'outcome', 'expected_error'}
                    or test['id'] != str(i) or test['outcome'] != row['outcomes'][i]):
                raise ValueError('actor tests must be the exact four public observations')
    counts = {s: sum(r['split'] == s for r in rows) for s in ('train', 'development', 'primary')}
    owners = {}
    for row in rows:
        source = row['source_component_id']
        if source in owners and owners[source] != row['split']:
            raise ValueError('actor source crosses training/assessment splits')
        owners[source] = row['split']
    if counts != manifest['counts'] or any(n == 0 for n in counts.values()):
        raise ValueError('actor split counts differ')
    if (counts['primary'] != len({r['source_component_id'] for r in rows if r['split'] == 'primary'})
            or counts['primary'] != manifest['primary_sources']):
        raise ValueError('exactly one selected primary candidate per source required')
    with np.load(directory/'posteriors.npz', allow_pickle=False) as data:
        z, logw = data['diagnosis'].copy(), data['log_weights'].copy()
    if (z.shape != (len(rows), 8, 24) or logw.shape != (len(rows), 8)
            or not np.isfinite(z).all() or not np.isfinite(logw).all()
            or not np.allclose(np.exp(logw).sum(1), 1., atol=1e-5)):
        raise ValueError('invalid diagnosis-only posterior arrays')
    return manifest, rows, z, logw


def prompt_ids(tokenizer, row, domain, cap=4096):
    if len(row['tests']) != 4 or len(row['outcomes']) != 4:
        raise ValueError('repair prompt accepts four public observations only')
    task = ('Repair the Python function solution; preserve its callable interface.' if domain == 'codearc'
            else 'Repair the Python stdin/stdout program.')
    evidence = [{k: t[k] for k in ('id', 'input', 'expected', 'actual', 'stderr', 'outcome')}
                | ({'expected_error': t['expected_error']} if 'expected_error' in t else {}) for t in row['tests']]
    payload = {'problem': row['task_text'], 'candidate': row['candidate'], 'public_tests': evidence}
    messages = [{'role': 'system', 'content': 'You repair Python programs. Return only complete Python source code.'},
                {'role': 'user', 'content': task+'\n'+json.dumps(payload, sort_keys=True, ensure_ascii=False)}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return ids[-cap:], len(ids)


def training_item(tokenizer, row, domain, maximum):
    if row['split'] == 'primary' or 'reference_code' not in row:
        raise ValueError('primary row cannot become a supervised training item')
    ids, original = prompt_ids(tokenizer, row, domain)
    target = tokenizer(row['reference_code'], add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
    if len(ids) + len(target) + 8 > maximum:
        return None, {'prompt_tokens': len(ids), 'original_prompt_tokens': original, 'target_tokens': len(target)}
    return (torch.tensor(ids+target), torch.tensor([-100]*len(ids)+target)), None


def helper_module():
    path = Path(__file__).with_name('run_rbr_repair_gate.py')
    spec = importlib.util.spec_from_file_location('packet_actor_legacy_math', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_save(value, path):
    temporary = path.with_suffix('.partial')
    torch.save(value, temporary)
    os.replace(temporary, path)


def train(args, identity, rows, z, logw, helper):
    output = args.output
    checkpoint = output/'projector.pt'
    resume = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint.exists() else None
    if resume and resume['identity'] != identity:
        raise ValueError('projector resume identity differs')
    if resume and resume['complete']:
        return
    torch.manual_seed(args.seed)
    training_indices = [i for i, r in enumerate(rows) if r['split'] == 'train']
    means = (np.exp(logw)[..., None]*z).sum(1)
    center, scale = means[training_indices].mean(0), np.maximum(means[training_indices].std(0), .1)
    normalized = torch.tensor((z-center[None, None])/scale[None, None])
    weights = torch.tensor(logw)
    model, tokenizer = helper._load_actor()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    projector = SoftPrefixProjector(24, model.config.hidden_size, hidden_dim=128,
        maximum_token_rms=.02, maximum_delta_rms=.002).cuda()
    torch.nn.init.zeros_(projector.network[-1].weight)
    torch.nn.init.zeros_(projector.network[-1].bias)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=.0003, weight_decay=.01)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed+991000)
    items, excluded = [], []
    # Choose the lexicographically first candidate per original development
    # source before token-length checks; validation covers sources, not repeats.
    dev_ids = {}
    for row in rows:
        if row['split'] == 'development':
            source = row['source_component_id']
            dev_ids[source] = min(dev_ids.get(source, row['task_id']), row['task_id'])
    for i, row in enumerate(rows):
        if row['split'] == 'primary' or (row['split'] == 'development' and row['task_id'] != dev_ids[row['source_component_id']]):
            continue
        value, reason = training_item(tokenizer, row, args.domain, args.max_sequence_tokens)
        if value is None:
            excluded.append({'task_id': row['task_id'], 'split': row['split'], 'reason': 'sequence_token_cap', **reason})
        else:
            items.append((i, row, *value))
    train_items = [item for item in items if item[1]['split'] == 'train']
    dev_items = [item for item in items if item[1]['split'] == 'development']
    if not train_items or not dev_items:
        raise ValueError('no fitting or development items satisfy the fixed context cap')
    audit = {'train_items': len(train_items), 'development_items': len(dev_items),
             'development_sources_before_token_cap': len(dev_ids), 'excluded': excluded,
             'selection': 'all training candidates; lexicographically first development candidate per source; fixed token cap',
             'primary_used_for_training_or_checkpoint_selection': False}
    (output/'training-population.json').write_text(json.dumps(audit, indent=2)+'\n')
    history, best, best_step, best_state = [], float('inf'), 0, None
    start = 0
    if resume:
        projector.load_state_dict(resume['projector'])
        optimizer.load_state_dict(resume['optimizer'])
        rng.bit_generator.state = resume['numpy_rng_state']
        generator.set_state(resume['component_rng_state'])
        history, best, best_step, best_state = resume['history'], resume['best_validation_nll'], resume['best_step'], resume['best_projector']
        start = resume['step']
    started = time.monotonic()

    def save(step, complete):
        atomic_save({'schema': 'apbpf-packet-projector-v1', 'identity': identity, 'step': step,
            'complete': complete, 'projector': projector.state_dict(), 'best_projector': best_state,
            'best_validation_nll': best, 'best_step': best_step, 'optimizer': optimizer.state_dict(),
            'numpy_rng_state': rng.bit_generator.state, 'component_rng_state': generator.get_state(),
            'history': history, 'latent_center': center, 'latent_scale': scale,
            'actor_hidden_size': model.config.hidden_size}, checkpoint)
        print(json.dumps({'step': step, 'complete': complete, 'elapsed_seconds': time.monotonic()-started,
                          'best_validation_nll': best if math.isfinite(best) else None}), flush=True)

    for step in range(start+1, args.steps+1):
        index, row, ids, labels = train_items[int(rng.integers(len(train_items)))]
        selected = sample_components_once(weights[index:index+1], num_sequences=2, generator=generator)[0]
        optimizer.zero_grad(set_to_none=True)
        loss = helper._mixture_nll(model, projector, ids, labels, normalized[index, selected],
                                  torch.full((2,), -math.log(2)))
        if not torch.isfinite(loss):
            raise ValueError('nonfinite projector training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError('nonfinite projector gradient')
        optimizer.step()
        if step == 1 or step % 10 == 0:
            record = {'step': step, 'loss': float(loss.detach()), 'gradient_norm': float(norm), 'task_id': row['task_id']}
            history.append(record)
            print(json.dumps(record), flush=True)
        del loss
        if step % 250 == 0 or step == args.steps:
            validation = helper._validation_nll(model, projector, dev_items, normalized, weights)
            if not math.isfinite(validation['pbpf']):
                raise ValueError('nonfinite development validation')
            history.append({'step': step, 'validation': validation})
            if validation['pbpf'] < best:
                best, best_step = validation['pbpf'], step
                best_state = {k: v.detach().cpu().clone() for k, v in projector.state_dict().items()}
            print(json.dumps({'step': step, 'validation': validation}), flush=True)
        if step % 25 == 0:
            save(step, False)
        if step % 10 == 0:
            torch.cuda.empty_cache()
    if best_state is None:
        raise ValueError('no finite development checkpoint')
    projector.load_state_dict(best_state)
    save(args.steps, True)


def generate(args, identity, rows, z, logw, helper):
    checkpoint = args.output/'projector.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not saved['complete'] or saved['identity'] != identity:
        raise ValueError('complete exactly bound projector required')
    model, tokenizer = helper._load_actor()
    projector = SoftPrefixProjector(24, model.config.hidden_size, hidden_dim=128,
        maximum_token_rms=.02, maximum_delta_rms=.002).cuda()
    projector.load_state_dict(saved['projector'])
    projector.eval()
    z = (z-saved['latent_center'][None, None])/saved['latent_scale'][None, None]
    bank = args.output/'generated'
    bank.mkdir(exist_ok=True)
    files = {}
    for index, row in enumerate(rows):
        if row['split'] != 'primary':
            continue
        ids, original_length = prompt_ids(tokenizer, row, args.domain)
        prompt = torch.tensor([ids], device='cuda')
        for arm in helper.CONDITIONING_ARMS:
            name = hashlib.sha256(f'{args.seed}:{row["task_id"]}:{arm}'.encode()).hexdigest()+'.json'
            path = bank/name
            binding = {'task_id': row['task_id'], 'source_component_id': row['source_component_id'],
                       'problem_id': row['problem_id'], 'arm': arm, 'seed': args.seed,
                       'projector_sha256': file_sha(checkpoint), 'actor_packet_sha256': identity['packet_manifest_sha256']}
            if path.exists():
                old = json.loads(path.read_text())
                if old['binding'] != binding or old['code_sha256'] != hashlib.sha256(old['code'].encode()).hexdigest():
                    raise ValueError('completed repair continuation changed')
                files[name] = file_sha(path)
                continue
            seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{row["task_id"]}:{arm}'.encode()).digest()[:8], 'big')
            latent, trace = helper._arm_conditioning(arm=arm, particles=z[index], log_weights=logw[index],
                rng=np.random.default_rng(seed), task_id=row['task_id'], continuation=index+1, max_new_tokens=args.max_new_tokens)
            started = time.monotonic()
            with torch.inference_mode():
                if arm == 'token_remix_fault':
                    tokens = []
                    for position in range(args.max_new_tokens):
                        chosen = z[index, trace.component_indices[position]]
                        suffix = torch.tensor([tokens], device='cuda', dtype=prompt.dtype) if tokens else prompt[:, :0]
                        current = torch.cat((prompt, suffix), 1)
                        inputs = projector.prepend(torch.tensor(chosen, device='cuda')[None],
                            inputs_embeds=model.get_input_embeddings()(current), attention_mask=torch.ones_like(current))
                        token = int(model(**inputs, use_cache=False).logits[0, -1].argmax())
                        tokens.append(token)
                        if token == tokenizer.eos_token_id:
                            break
                else:
                    embeddings = model.get_input_embeddings()(prompt)
                    inputs = ({'inputs_embeds': embeddings, 'attention_mask': torch.ones_like(prompt)} if arm == 'no_latent'
                              else projector.prepend(torch.tensor(latent, device='cuda', dtype=torch.float32)[None],
                                   inputs_embeds=embeddings, attention_mask=torch.ones_like(prompt)))
                    tokens = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                            pad_token_id=tokenizer.eos_token_id, use_cache=True)[0].tolist()
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            code = extract_solution(raw) if args.domain == 'codearc' else helper._clean(raw)
            report = {'binding': binding, 'code': code, 'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                      'raw_completion': raw, 'generated_token_ids': tokens, 'generated_tokens': len(tokens),
                      'prompt_tokens': len(ids), 'original_prompt_tokens': original_length,
                      'prompt_truncated': original_length > len(ids), 'conditioning_trace': trace.json(),
                      'generation_seconds': time.monotonic()-started, 'decode_policy': 'greedy'}
            temporary = path.with_suffix('.partial')
            temporary.write_text(json.dumps(report, sort_keys=True)+'\n')
            temporary.replace(path)
            files[name] = file_sha(path)
            print(json.dumps({'generated': len(files), 'task_id': row['task_id'], 'arm': arm,
                              'tokens': len(tokens), 'seconds': report['generation_seconds']}), flush=True)
    expected = sum(r['split'] == 'primary' for r in rows)*len(helper.CONDITIONING_ARMS)
    if len(files) != expected:
        raise ValueError('incomplete repair arm/source inventory')
    (args.output/'generation-complete.json').write_text(json.dumps({'identity': identity, 'files': files,
        'generated': expected, 'scope': 'public-only generation; no hidden evaluation performed'}, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--domain', choices=['rbr', 'codearc'], required=True)
    p.add_argument('--mode', choices=['train', 'generate'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--max-sequence-tokens', type=int, default=1536)
    p.add_argument('--max-new-tokens', type=int, default=512)
    args = p.parse_args()
    if min(args.steps, args.max_sequence_tokens, args.max_new_tokens) < 1:
        raise ValueError('positive training and token budgets required')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, rows, z, logw = load_packet(args.input)
    if manifest['seed'] != args.seed:
        raise ValueError('repair packet seed mismatch')
    helper = helper_module()
    root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), Path(helper.__file__), *sorted((root/'src/pbpf').rglob('*.py'))]
    identity = {'schema': 'apbpf-packet-actor-run-v1', 'domain': args.domain, 'seed': args.seed,
                'packet_manifest_sha256': file_sha(args.input/'manifest.json'),
                'actor': {'id': helper.MODEL_ID, 'revision': helper.MODEL_REVISION},
                'steps': args.steps, 'max_sequence_tokens': args.max_sequence_tokens,
                'max_new_tokens': args.max_new_tokens, 'max_prompt_tokens': 4096,
                'source_sha256': {str(path.relative_to(root)): file_sha(path) for path in source_paths},
                'scope': 'supportive exploratory repair; all failed upstream gates remain binding'}
    run = args.output/'run.json'
    if run.exists() and json.loads(run.read_text()) != identity:
        raise ValueError('actor run identity differs')
    if not run.exists():
        run.write_text(json.dumps(identity, indent=2)+'\n')
    if args.mode == 'train':
        train(args, identity, rows, z, logw, helper)
    else:
        generate(args, identity, rows, z, logw, helper)


if __name__ == '__main__':
    main()
