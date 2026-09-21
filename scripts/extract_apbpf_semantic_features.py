#!/usr/bin/env python3
"""Extract frozen code-model features from a public, label-free text inventory.

Run in apbpf_generator_sandbox.sh: only /input, /output and the pinned model cache
are visible. Every text is encoded with the same fixed settings; no label fitting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    manifest=json.loads((args.input/'manifest.json').read_text())
    if manifest['schema']!='apbpf-public-text-inventory-v1' or sha(args.input/'texts.jsonl')!=manifest['texts_sha256']:
        raise ValueError('invalid public text inventory')
    with (args.input/'texts.jsonl').open() as stream:
        rows=[json.loads(line) for line in stream]
    if len(rows)!=manifest['texts'] or len({r['text_sha256'] for r in rows})!=len(rows):
        raise ValueError('public text inventory count/uniqueness mismatch')
    for row in rows:
        if set(row)!={'text','text_sha256'} or hashlib.sha256(row['text'].encode()).hexdigest()!=row['text_sha256']:
            raise ValueError('extractor accepts only verified public text content')
    model_id='Qwen/Qwen2.5-Coder-7B-Instruct';revision='c03e6d358207e414f1eca0bb1891e29f1db0e242'
    config={'schema':'apbpf-frozen-semantic-extraction-v1','model':model_id,'revision':revision,
            'input_manifest_sha256':sha(args.input/'manifest.json'),'input_texts_sha256':manifest['texts_sha256'],
            'source_sha256':sha(__file__),'dimension':512,'projection_seed':1701,'max_input_tokens':512,
            'pooling':'attention-mask mean of last decoder hidden state','normalization':'L2 after fixed Gaussian projection',
            'quantization':'bitsandbytes 4-bit default FP4; bfloat16 compute','batch_size':args.batch_size,
            'evaluation_role':manifest['evaluation_role'],'expected_is_public':manifest['expected_is_public'],
            'source_cache_sha256':manifest['source_cache_sha256'],
            'source_payload_sha256':manifest['source_payload_sha256'],'texts':len(rows)}
    args.output.mkdir(parents=True,exist_ok=True)
    run=args.output/'run.json'
    if run.exists():
        if json.loads(run.read_text())!=config:
            raise ValueError('resume identity mismatch')
    else:
        with run.open('x') as stream: json.dump(config,stream,indent=2)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,local_files_only=True,
        device_map={'':0},torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    generator=torch.Generator().manual_seed(1701)
    projection=(torch.randn(model.config.hidden_size,512,generator=generator)/model.config.hidden_size**.5).to(model.device)
    # Length bucketing reduces padding; output rows retain SHA inventory order.
    order=sorted(range(len(rows)),key=lambda i:(len(rows[i]['text']),rows[i]['text_sha256']))
    started=time.monotonic();chunks=[]
    for start in range(0,len(order),args.batch_size):
        indices=order[start:start+args.batch_size]
        path=args.output/f'chunk-{start:08d}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as old:
                if not np.array_equal(old['indices'],indices) or old['vectors'].shape!=(len(indices),512):
                    raise ValueError('completed chunk identity mismatch')
        else:
            inputs=tokenizer([rows[i]['text'] for i in indices],padding=True,truncation=True,
                             max_length=512,return_tensors='pt').to(model.device)
            with torch.inference_mode():
                hidden=model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state.float()
                mask=inputs.attention_mask[...,None]
                pooled=(hidden*mask).sum(1)/mask.sum(1)
                vectors=pooled@projection
                vectors=vectors/vectors.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            data=vectors.cpu().numpy()
            if not np.isfinite(data).all(): raise ValueError('nonfinite semantic feature')
            temporary=path.with_suffix('.partial')
            with temporary.open('wb') as stream:
                np.savez(stream,indices=np.asarray(indices),vectors=data)
            os.replace(temporary,path)
        chunks.append(path)
        if start%(args.batch_size*32)==0:
            print(json.dumps({'completed':min(start+args.batch_size,len(rows)),'total':len(rows),
                              'elapsed_seconds':time.monotonic()-started}),flush=True)
    vectors=np.empty((len(rows),512),dtype=np.float32)
    for path in chunks:
        with np.load(path,allow_pickle=False) as data: vectors[data['indices']]=data['vectors']
    vector_path=args.output/'vectors.npy'
    if vector_path.exists():
        if not np.array_equal(np.load(vector_path,allow_pickle=False),vectors):
            raise ValueError('final vectors mismatch on resume')
    else:
        with vector_path.open('xb') as stream: np.save(stream,vectors,allow_pickle=False)
    result={**config,'schema':'apbpf-frozen-semantic-cache-v1','vectors_sha256':sha(vector_path),
            'text_sha256':[r['text_sha256'] for r in rows],'run_sha256':sha(run),
            'chunk_sha256':{p.name:sha(p) for p in chunks}}
    path=args.output/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=result: raise ValueError('final manifest mismatch')
    else:
        with path.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({'complete':True,'texts':len(rows),'dimension':512}),flush=True)


if __name__=='__main__':
    main()
