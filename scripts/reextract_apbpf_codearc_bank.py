#!/usr/bin/env python3
"""Reextract every candidate from raw completions; never read execution labels."""
import argparse
import copy
import json
from pathlib import Path

from pbpf.apbpf.code_extraction import extract_solution
from pbpf.apbpf.codearc_bank import file_sha, load_bank


def write(path, value):
    with path.open('x') as stream: json.dump(value,stream,indent=2,ensure_ascii=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('reextracted banks are create-once')
    run,rows,parent=load_bank(args.input)
    root=Path(__file__).resolve().parents[1]
    transform={'policy':'first parseable module-level solution definition plus import-only blocks; truncated solution preferred over usage; longest block only if no solution definition',
               'source_sha256':file_sha(root/'src/pbpf/apbpf/code_extraction.py'),
               'entrypoint_sha256':file_sha(__file__),'parent_complete_sha256':parent}
    revised=copy.deepcopy(run);revised['code_reextraction']=transform
    args.output.mkdir(parents=True)
    write(args.output/'run.json',revised)
    files={};changes=[]
    for row in rows:
        value=copy.deepcopy(row)
        for candidate in value['candidates']:
            original=candidate['code'];candidate['code']=extract_solution(candidate['raw_completion'])
            candidate['extraction_changed']=candidate['code']!=original
            if candidate['extraction_changed']: changes.append(candidate['candidate_id'])
        name=row['task_id'].replace('/','-')+'.json';write(args.output/name,value)
        files[name]=file_sha(args.output/name)
    report={'groups':len(rows),'candidates':sum(len(r['candidates']) for r in rows),
            'changed_candidates':changes,'change_count':len(changes),**transform}
    write(args.output/'reextraction.json',report)
    write(args.output/'complete.json',{'schema':'apbpf-candidate-generation-complete-v1',
        'run_sha256':file_sha(args.output/'run.json'),'files':files,
        'reextraction_sha256':file_sha(args.output/'reextraction.json')})
    print(json.dumps({k:v for k,v in report.items() if k!='changed_candidates'}),flush=True)


if __name__=='__main__': main()
