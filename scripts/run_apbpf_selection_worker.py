#!/usr/bin/env python3
"""Fit and evaluate full-bank utility selectors with trained baseline controls."""
import argparse
import json
from pathlib import Path

from pbpf.apbpf.assessment_bundle import load_bundle
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_prediction import DOMAINS, load_training
from pbpf.apbpf.stage_selection import fit_selection, aggregate_selection, NEURAL_BASELINES
from pbpf.apbpf.worker_io import WorkerIO


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=1000);args=parser.parse_args()
    if args.steps<1:raise ValueError('positive utility training budget required')
    root=Path(__file__).resolve().parents[1]
    sources=[str(p.relative_to(root)) for p in sorted((root/'src/pbpf').rglob('*.py'))]
    io=WorkerIO('selection',sources+['scripts/run_apbpf_selection_worker.py'])
    bundle=load_bundle(io,'association_gate')
    manifest=json.loads(io.artifact('association_gate','assessment-bundle/manifest.json').read_text())
    if manifest['origin_dependencies']['train_belief']!=io.request['dependencies']['train_belief']:
        raise ValueError('selection bundle does not descend from the exact trained belief dependency')
    belief,baselines=load_training(io,'belief'),load_training(io,'baselines')
    # Binding the utility target to the hard-bank audit prevents stale/relabelled success targets.
    hidden=json.loads(io.artifact('hard_bank','hidden-bank.json').read_text())
    hidden_labels={candidate:label for g in hidden for candidate,label in zip(g['candidate_ids'],g['hidden_labels'],strict=True)}
    if len(hidden_labels)!=8000:raise ValueError('both complete hard-bank candidate inventories required')
    domains={}
    for domain,dataset in DOMAINS.items():
        path=bundle[domain]['cache'];payload=json.loads(path.read_text());checksum=file_sha(path)
        primary=[r for r in payload['records'] if r['split']=='test']
        if len(primary)!=4000 or len({r['source_component_id'] for r in primary})!=500:
            raise ValueError('selection requires complete500-by8 primary population')
        for row in primary:
            if hidden_labels[row['task_id']]!=int(all(o=='PASS' for o in row['outcomes'][4:])):
                raise ValueError('utility target differs from exact hard-bank hidden evaluation')
        reports={}
        for seed in io.config['protocol']['seeds']:
            io.check_sources();a,b=belief[domain,seed],baselines[domain,seed]
            if (a['entry']['cache_sha256']!=checksum or b['entry']['cache_sha256']!=checksum
                    or file_sha(a['checkpoints']['belief'])!=file_sha(bundle[domain]['models'][seed]['checkpoint'])):
                raise ValueError('selection data/model provenance differs from original fitted models')
            reports[seed]=fit_selection(payload,a['checkpoints']['belief'],{k:b['checkpoints'][k] for k in NEURAL_BASELINES},
                io.outputs/f'{domain}-seed{seed}',seed=seed,cache_sha256=checksum,steps=args.steps)
        domains[dataset]=aggregate_selection(reports)
    evidence={'schema':'apbpf-stage-selection-v1','comparator':'strongest_cross_fitted_deterministic',
              'domains':domains,'scope':'full-primary exploratory utility selection; failed upstream gates remain binding'}
    (io.outputs/'selection.json').write_text(json.dumps(evidence,indent=2)+'\n')
    io.finish({'actual_utility_training':True,'primary_groups':1000,'domains':list(domains),'scope':evidence['scope']})


if __name__=='__main__':
    main()
