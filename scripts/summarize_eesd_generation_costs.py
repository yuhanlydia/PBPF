#!/usr/bin/env python3
"""Write a point-in-time token ledger for sealed records in running banks."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from pbpf.eesd.generation_ledger import summarize_bank


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--banks',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[summarize_bank(path.parent) for path in sorted(args.banks.rglob('run.json'))]
    report={'schema':'eesd-generation-ledger-v1','created_at':datetime.now(timezone.utc).isoformat(),
            'banks':rows,'totals':{key:sum(row[key] for row in rows) for key in
                ['verified_sources','verified_candidates','input_tokens','generated_tokens','token_cap_hits','clipped_prompts']},
            'cost_usd':None,'scientific_scoring_complete':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output':str(args.output),'totals':report['totals']}))

if __name__=='__main__':
    main()
