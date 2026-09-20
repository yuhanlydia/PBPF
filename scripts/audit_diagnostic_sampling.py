"""Separate sampling noise, history corruption, and particle-count sensitivity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inspect_diagnostic_state as state
from run_association_debug import source_hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    hashes=source_hashes()
    for p in (Path(__file__),Path(state.__file__)):
        hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for relative,digest in hashes.items():
        raw=(root/relative).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('source changed during audit snapshot')
        target=args.output/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    configs=[dict(kind='anchor',particles=64,sampling_seed=51701,counterfactual_seed=51701)]
    configs += [dict(kind='sampling_only',particles=64,sampling_seed=s,counterfactual_seed=51701) for s in (71701,81701,91701)]
    configs += [dict(kind='corruption_only',particles=64,sampling_seed=51701,counterfactual_seed=s) for s in (71701,81701,91701)]
    configs += [dict(kind='particle_budget',particles=256,sampling_seed=s,counterfactual_seed=51701) for s in (51701,71701)]
    manifest=dict(scope='development_only_posthoc_checkpoint_audit',test_evaluated=False,
                  source_sha256=hashes,configs=configs,runs=[str(p) for p in args.runs],
                  checkpoint_selection='Frozen original selections; no reselection during audit')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]
    for run in args.runs:
        measurements=[]
        for config in configs:
            result=state.inspect(run,config['particles'],config['sampling_seed'],device,config['counterfactual_seed'])
            measurements.append(dict(kind=config['kind'],**result))
        rows.append(dict(run=str(run),measurements=measurements))
        print(json.dumps(dict(run=str(run),gaps=[r['gap_full'] for r in measurements])),flush=True)
        (args.output/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'result.json').write_text(json.dumps(dict(**manifest,rows=rows),indent=2)+'\n')
    checks={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n')


if __name__=='__main__':
    main()
