#!/usr/bin/env python3
"""Admit the separately declared APPS/CodeContests Replay populations."""
import argparse
from pathlib import Path
from pbpf.eesd.replay_admission import verify_inputs, publish_bundle
from pbpf.eesd.replay_materialization import select_population, project_views


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('joint-receipt','public-candidates','evaluator-candidates','token-receipt','token-results','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    public,private,tokens,quarantine,evidence=verify_inputs(root,args.joint_receipt,args.public_candidates,args.evaluator_candidates,args.token_receipt,args.token_results)
    selected=select_population(public,tokens,quarantine)
    public_view,private_view=project_views(selected,private)
    selected_ids={r['task_id'] for r in public_view}
    selected_tokens=[r for r in tokens if r['task_id'] in selected_ids]
    publish_bundle(args.output,public_view,private_view,selected_tokens,evidence)
    print(f'Admitted 2 domains × (200 development + 500 primary): {args.output}')

if __name__=='__main__':
    main()
