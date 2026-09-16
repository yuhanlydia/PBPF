#!/usr/bin/env python3
"""Apply the unchanged selected-Pass@1 gate to all source-cross-fitted evidence."""
import json
import shutil

from pbpf.apbpf.stage_prediction import DOMAINS
from pbpf.apbpf.worker_io import WorkerIO


def decision(evidence,config):
    if (evidence['schema']!='apbpf-stage-selection-v1' or evidence['comparator']!='strongest_cross_fitted_deterministic'
            or set(evidence['domains'])!=set(DOMAINS.values())):
        raise ValueError('selection gate requires both domains and the strongest cross-fitted comparator')
    domains={}
    for domain,row in evidence['domains'].items():
        if (row['seeds']!=config['protocol']['seeds'] or row['bootstrap_draws']!=10000
                or row['primary_sources']!=500 or row['primary_candidates']!=4000):
            raise ValueError('selection population or bootstrap protocol differs')
        domains[domain]={'absolute_selected_pass1_advantage':row['absolute_selected_pass1_advantage'],
                         'clustered_lower_bound':row['ci95'][0]}
    threshold=config['gates']['selection']['minimum_absolute_selected_pass1_advantage']
    passed=all(row['absolute_selected_pass1_advantage']>=threshold and row['clustered_lower_bound']>0 for row in domains.values())
    return {'passed':passed,'metrics':{'comparator':'strongest_cross_fitted_deterministic','domains':domains},
            'reason':'Both domains meet the fixed selected-Pass@1 margin and positive CI' if passed else
                     'At least one domain fails the fixed selected-Pass@1 margin or positive CI'}


def main():
    io=WorkerIO('selection_gate',['scripts/run_apbpf_selection_gate_worker.py','src/pbpf/apbpf/stage_prediction.py'])
    evidence=json.loads(io.artifact('selection','selection.json').read_text());gate=decision(evidence,io.config)
    (io.outputs/'gate-evidence.json').write_text(json.dumps(gate,indent=2)+'\n')
    # Repair may read only direct dependencies. Preserve the actual selected
    # candidates and source order, rather than inventing a new selection rule.
    directory=io.outputs/'selection-reports';directory.mkdir()
    for domain in DOMAINS:
        for seed in io.config['protocol']['seeds']:
            source=io.artifact('selection',f'{domain}-seed{seed}/results.json')
            shutil.copyfile(source,directory/f'{domain}-seed{seed}.json')
    (directory/'origin.json').write_text(json.dumps({
        'selection_completion_sha256':io.request['dependencies']['selection'],
        'visibility':'evaluator-only; contains assessment labels; build an actor-only packet before generation'},indent=2)+'\n')
    io.finish({'actual_selection_evidence':True,'scope':'exploratory fixed selection gate'},gate=gate)


if __name__=='__main__':
    main()
