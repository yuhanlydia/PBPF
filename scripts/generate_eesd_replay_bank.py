#!/usr/bin/env python3
"""Generate a sealed public-only Replay synthesis bank; never execute candidates."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from pbpf.apbpf.rbr_prompt import extract_program, generation_stop_kwargs, decode_completion


def task_seed(seed,task_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{task_id}'.encode()).digest()[:4],'big')


def decode_one(model,tokenizer,prepared,family,seed,torch):
    torch.manual_seed(seed)
    ids=torch.tensor([prepared['ids']],dtype=torch.long,device=model.device)
    kwargs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=True,
                num_return_sequences=1,max_new_tokens=1024,temperature=.8,top_p=.95,
                pad_token_id=tokenizer.pad_token_id)
    kwargs.update(generation_stop_kwargs(tokenizer,family))
    with torch.inference_mode():
        result=model.generate(**kwargs)
    text,count,hit_cap=decode_completion(tokenizer,result[0,ids.shape[1]:].cpu().tolist(),family)
    return dict(code=extract_program(text),raw_completion=text,generated_tokens=count,hit_token_cap=hit_cap)


def write_json_once(path,value):
    """An interrupted publication leaves evidence and must not be re-sampled."""
    partial=path.with_suffix(path.suffix+'.partial')
    with partial.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.link(partial,path)  # Atomic create, never overwrite an existing record.
    partial.unlink()


@contextmanager
def writer_lock(output):
    output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/(output.name+'.writer.lock')).open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def load_tokenizer(model):
    from transformers import AutoTokenizer
    from huggingface_hub import try_to_load_from_cache
    for name,expected in model['tokenizer_files_sha256'].items():
        path=try_to_load_from_cache(model['model_id'],name,revision=model['revision'])
        if not isinstance(path,str) or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('local tokenizer file differs from admitted version: '+name)
    tokenizer=AutoTokenizer.from_pretrained(model['model_id'],revision=model['revision'],local_files_only=True)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()!=model['chat_template_sha256']:
        raise ValueError('local chat template differs from admission')
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
    return tokenizer


def load_model(model):
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    loaded=AutoModelForCausalLM.from_pretrained(model['model_id'],revision=model['revision'],local_files_only=True,
        device_map={'':0},torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='fp4',bnb_4bit_use_double_quant=False))
    loaded.eval()
    return loaded,torch


def execute(args):
    from pbpf.eesd.replay_generation import (load_public_split,prepare_prompt,build_expected,
        verify_record,verify_replay_bank,record_filename,sha)
    preflight=load_public_split(args.bundle,args.admission_sha256,args.domain,args.split,args.family)
    tokenizer=load_tokenizer(preflight['model'])
    prepared={row['task_id']:prepare_prompt(tokenizer,row,preflight['audits'][row['task_id']],args.family)
              for row in preflight['rows']}
    run=build_expected(preflight,args.family,args.seed,Path(__file__))
    if args.preflight_only:
        print(json.dumps(dict(status='public-token-preflight-only',domain=args.domain,family=args.family,
            split=args.split,components=len(prepared),run=run)),flush=True)
        return
    with writer_lock(args.output):
        args.output.mkdir(exist_ok=True)
        if list(args.output.glob('*.partial')):
            raise ValueError('incomplete publication: preserve partial artifacts; repair before resuming')
        run_path=args.output/'run.json'
        if run_path.exists():
            if json.loads(run_path.read_text())!=run:
                raise ValueError('resume identity changed')
        else:
            if any(args.output.iterdir()):
                raise ValueError('orphan bank artifacts without run identity')
            write_json_once(run_path,run)
        if (args.output/'complete.json').exists():
            verify_replay_bank(args.output,run,preflight=preflight)
            print(json.dumps({'status':'verified-complete-resume'}),flush=True)
            return
        allowed={'run.json'}
        pending=[]
        for row in preflight['rows']:
            path=args.output/record_filename(row['task_id'])
            checksum=path.with_suffix('.sha256')
            allowed.update((path.name,checksum.name))
            if path.exists():
                verify_record(path,run,row,preflight['audits'][row['task_id']])
            elif checksum.exists():
                raise ValueError('orphan checksum: candidate publication incomplete')
            else:
                pending.append(row)
        if {p.name for p in args.output.iterdir()}-allowed:
            raise ValueError('unexpected artifacts in partial bank')
        if pending:
            model,torch=load_model(preflight['model'])
            started=time.monotonic()
            for i,row in enumerate(pending):
                task=row['task_id'];seed=task_seed(args.seed,task)
                candidate=decode_one(model,tokenizer,prepared[task],args.family,seed,torch)
                candidate['candidate_id']=f'{task}/{args.family}/0'
                record={key:row[key] for key in ('task_id','source_component_id','domain','split')}
                record.update(seed=seed,prompt_sha256=prepared[task]['metadata']['rendered_prompt_sha256'],
                    prompt_metadata=prepared[task]['metadata'],candidates=[candidate])
                path=args.output/record_filename(task)
                write_json_once(path,record)
                with path.with_suffix('.sha256').open('x') as stream:
                    stream.write(sha(path)+'\n');stream.flush();os.fsync(stream.fileno())
                verify_record(path,run,row,preflight['audits'][task])
                print(json.dumps(dict(completed=len(preflight['rows'])-len(pending)+i+1,total=len(preflight['rows']),
                    task_id=task,elapsed_seconds=time.monotonic()-started)),flush=True)
        files={record_filename(row['task_id']):sha(args.output/record_filename(row['task_id'])) for row in preflight['rows']}
        write_json_once(args.output/'complete.json',dict(schema='eesd-replay-generation-complete-v1',run_sha256=sha(run_path),files=files))
        verify_replay_bank(args.output,run,preflight=preflight)
        print(json.dumps(dict(status='candidate-generation-complete-not-executed',components=len(files))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--admission-sha256',required=True)
    parser.add_argument('--domain',choices=['apps_replay','codecontests_replay'],required=True)
    parser.add_argument('--family',choices=['qwen25_7b','deepseek_6p7b','seed_coder_8b','starcoder2_15b'],required=True)
    parser.add_argument('--split',choices=['development','primary'],required=True)
    parser.add_argument('--seed',type=int,choices=[1701,1702,1703],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    execute(parser.parse_args())

if __name__=='__main__':
    main()
