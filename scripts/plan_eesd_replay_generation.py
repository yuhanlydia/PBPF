#!/usr/bin/env python3
"""Plan the independent Replay generation matrix without launching commands."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

DOMAINS=('apps_replay','codecontests_replay')
FAMILIES=('qwen25_7b','deepseek_6p7b','seed_coder_8b','starcoder2_15b')
SEEDS=(1701,1702,1703)
SPLITS={'development':200,'primary':500}
SOURCE_FILES=('scripts/generate_eesd_replay_bank.py','scripts/plan_eesd_replay_generation.py',
              'src/pbpf/eesd/replay_generation.py','src/pbpf/eesd/replay_prompt.py',
              'src/pbpf/apbpf/rbr_prompt.py')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':')).encode()


def build_plan(root,bundle,admission_sha256,output_root,*,loader=None):
    """Validate admitted public populations, then construct shell-free argv."""
    root,bundle,output_root=(Path(p).resolve() for p in (root,bundle,output_root))
    if sha(bundle/'admission.json')!=admission_sha256:
        raise ValueError('admission checksum mismatch')
    if loader is None:
        from pbpf.eesd.replay_generation import load_public_split
        loader=load_public_split
    sources={name:sha(root/name) for name in SOURCE_FILES}
    populations={domain:{} for domain in DOMAINS}
    preflights={}
    global_tasks,global_sources=set(),set()
    for domain in DOMAINS:
        for split,count in SPLITS.items():
            expected=None
            for family in FAMILIES:
                data=loader(bundle,admission_sha256,domain,split,family)
                rows=data['rows']
                identities=[{'task_id':r['task_id'],'source_component_id':r['source_component_id']} for r in rows]
                if (len(rows)!=count or any(r['domain']!=domain or r['split']!=split for r in rows)
                    or len({r['task_id'] for r in rows})!=count
                    or len({r['source_component_id'] for r in rows})!=count):
                    raise ValueError('incomplete or duplicate planned population')
                if expected is not None and identities!=expected:
                    raise ValueError('models have different ordered public populations')
                if expected is None:
                    expected=identities
                    tasks={r['task_id'] for r in rows};components={r['source_component_id'] for r in rows}
                    if tasks & global_tasks or components & global_sources:
                        raise ValueError('task/source reused across domains or splits')
                    global_tasks.update(tasks);global_sources.update(components)
                    populations[domain][split]={'components':count,'ordered_identities':identities,
                        'sha256':hashlib.sha256(canonical(identities)).hexdigest()}
                preflights[(domain,split,family)]=data
    cells=[]
    for domain in DOMAINS:
        for family in FAMILIES:
            for seed in SEEDS:
                splits={}
                for split,count in SPLITS.items():
                    preflight=preflights[(domain,split,family)]
                    output=output_root/'replay-mechanism-banks'/domain/family/f'seed{seed}'/split
                    argv=[str(root/'.venv/bin/python'),str(root/'scripts/generate_eesd_replay_bank.py'),
                          '--bundle',str(bundle),'--admission-sha256',admission_sha256,
                          '--domain',domain,'--family',family,'--split',split,'--seed',str(seed),'--output',str(output)]
                    splits[split]={'components':count,'population_sha256':populations[domain][split]['sha256'],
                        'bindings':preflight['bindings'],'model':preflight['model'],'output':str(output),'argv':argv}
                cells.append({'domain':domain,'family':family,'seed':seed,'splits':splits})
    # A source mutation during planning invalidates this prospective plan.
    if sources!={name:sha(root/name) for name in SOURCE_FILES} or sha(bundle/'admission.json')!=admission_sha256:
        raise ValueError('source/admission changed during planning')
    return {'schema':'eesd-replay-generation-matrix-v1','status':'planned-generation-only-not-launched',
            'bundle':str(bundle),'admission_sha256':admission_sha256,'output_root':str(output_root),
            'sources':sources,'populations':populations,'cells':cells,'cell_count':24,'command_count':48,
            'planned_candidates':16800,'decode':{'candidates':1,'do_sample':True,'temperature':.8,
                'top_p':.95,'max_new_tokens':1024,'max_input_tokens':4096,'truncation':False}}


def write_manifest_once(path,manifest):
    """Publish a fully serialized manifest atomically, without replacing a file."""
    path=Path(path)
    data=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--admission-sha256',required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--manifest-out',type=Path,required=True)
    args=parser.parse_args()
    if args.manifest_out.exists(): raise FileExistsError(args.manifest_out)
    root=Path(__file__).resolve().parents[1]
    plan=build_plan(root,args.bundle,args.admission_sha256,args.output_root)
    write_manifest_once(args.manifest_out,plan)
    print(json.dumps({'status':plan['status'],'cells':24,'commands':48,'manifest':str(args.manifest_out.resolve()),'sha256':sha(args.manifest_out)}))

if __name__=='__main__':main()
