#!/usr/bin/env python3
"""Predeclare equal response-token budgets from the equal-weight control schedule.

This is a public-data-only CPU plan. It does not train, inspect hidden outcomes,
or increase the reference control's requested optimizer-step workload.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import yaml

ROOT=Path(__file__).resolve().parents[1]
TRAINER=ROOT/'scripts/run_eesd_weighted_sft.py'
SFT=runpy.run_path(str(TRAINER))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_budget(lengths, *, seed, steps, accumulation):
    return sum(tokens for _,tokens in SFT['training_schedule'](lengths,seed=seed,
        max_steps=steps,gradient_accumulation=accumulation,response_token_budget=None))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='sealed scored-corrections.jsonl')
    p.add_argument('--scoring-summary',type=Path,required=True)
    p.add_argument('--model-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True)
    p.add_argument('--reference-steps',type=int,default=200)
    p.add_argument('--gradient-accumulation',type=int,default=16)
    p.add_argument('--max-length',type=int,default=4608)
    args=p.parse_args()
    if (min(args.reference_steps,args.gradient_accumulation,args.max_length)<1
            or args.max_length < 64 or len(set(args.seeds))!=len(args.seeds)):
        p.error('positive bounds, max_length >= 64 and distinct seeds required')
    summary=json.loads(args.scoring_summary.read_text())
    if summary.get('schema')!='eesd-scored-corrections-v1' or summary.get('scored_sha256')!=sha(args.input):
        raise ValueError('scored correction seal mismatch')
    cfg=yaml.safe_load(args.model_config.read_text())
    model,revision=cfg.get('model_id'),cfg.get('revision')
    if not isinstance(model,str) or not isinstance(revision,str) or len(revision)!=40:
        raise ValueError('pinned model config required')
    rows=SFT['load_rows'](args.input,'equal_weight')
    trajectory_ids=[r.get('trajectory_id') for r in rows]
    if (any(not isinstance(value,str) or not value for value in trajectory_ids)
            or len(set(trajectory_ids))!=len(rows)):
        raise ValueError('trajectory_id must be nonempty and unique')
    split_counts={split:sum(r['split']==split for r in rows)
                  for split in sorted({r['split'] for r in rows})}
    if (summary.get('trajectories')!=len(rows)
            or summary.get('sources')!=len({r['source_component_id'] for r in rows})
            or summary.get('split_counts')!=split_counts):
        raise ValueError('scoring summary population mismatch')
    if any(r['_weight']!=1 for r in rows):
        raise ValueError('reference control must assign equal unit trajectory weights')
    if any(r.get('model_id')!=model or r.get('model_revision')!=revision for r in rows):
        raise ValueError('correction/model identity mismatch')
    train=[r for r in rows if r['split']=='train']
    dev=[r for r in rows if r['split']=='development']
    if not train or not dev or {r['source_component_id'] for r in train}&{r['source_component_id'] for r in dev}:
        raise ValueError('require disjoint nonempty train and development populations')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,revision=revision,local_files_only=True)
    encoded=[SFT['encode_training_row'](tokenizer,r,max_length=args.max_length,model_id=model) for r in rows]
    lengths=[len(e['completion_ids']) for r,e in zip(rows,encoded) if r['split']=='train']
    report={'schema':'eesd-reference-token-budget-v1','status':'plan_only_no_training',
        'policy':'same exact response-token count as equal_weight reference step schedule for each seed; use across all update arms',
        'input_sha256':sha(args.input),'scoring_summary_sha256':sha(args.scoring_summary),
        'model_config_sha256':sha(args.model_config),'trainer_source_sha256':sha(TRAINER),
        'planner_source_sha256':sha(__file__),'model_id':model,'revision':revision,
        'reference_steps':args.reference_steps,'gradient_accumulation':args.gradient_accumulation,
        'max_length':args.max_length,'training_trajectories':len(train),
        'development_trajectories':len(dev),'response_truncated_rows':sum(e['response_truncated'] for e in encoded),
        'budgets':{str(seed):reference_budget(lengths,seed=seed,steps=args.reference_steps,
                     accumulation=args.gradient_accumulation) for seed in args.seeds},
        'hidden_outcomes_used':False,
        'seal_scope':'scored input digest and scoring summary population only; upstream must verify generation ancestry',
        'adoption_status':'proposed budget rule; not evidence of adoption or completed training',
        'limitation':'Budgets may differ across seeds/cells; the matched comparison is across arms within each cell and seed.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'output':str(args.output),'budgets':report['budgets'],'status':report['status']}))

if __name__=='__main__':main()
