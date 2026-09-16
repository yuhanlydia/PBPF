#!/usr/bin/env python3
"""Gate complete query replay while retaining all failed and fixed-budget results."""
import argparse
import json

from pbpf.apbpf.assessment_bundle import forward_bundle
from pbpf.apbpf.stage_prediction import DOMAINS
from pbpf.apbpf.worker_io import WorkerIO


def decision(evidence, config, stage):
    active = stage == 'active_testing_gate'
    if (evidence['schema']!='apbpf-stage-query-replay-v1'
            or evidence['stage']!=('active_testing' if active else 'oracle_headroom')
            or evidence['public_pool']!=4 or evidence['oracle_gate_budget']!=2
            or set(evidence['domains'])!=set(DOMAINS.values())):
        raise ValueError('query evidence protocol/domain mismatch')
    domains = {}
    for name, report in evidence['domains'].items():
        if report['seeds']!=config['protocol']['seeds'] or report['source_components']!=500 or report['candidates_per_seed']!=4000:
            raise ValueError('all three seeds and all primary candidates required')
        if active:
            fixed = report['summary']['4']['diagnostic_mi']
            domains[name] = {'fixed_budget':4,'all_policies_execute_exact_budget':report['all_policies_execute_exact_budget'],
                'nll_advantage_at_four_tests':min(fixed['against_fixed']['mean_nll_gap'],fixed['against_random']['mean_nll_gap']),
                'matched_hidden_quality':report['stopping']['matched_hidden_quality'],
                'test_reduction':report['stopping']['test_reduction']}
        else:
            oracle = report['summary']['2']['oracle']
            domains[name] = {'nll_advantage_over_fixed':oracle['against_fixed']['mean_nll_gap'],
                             'nll_advantage_over_random':oracle['against_random']['mean_nll_gap']}
    if active:
        rule = config['gates']['active_testing']
        passed = all((r['matched_hidden_quality'] is True and r['test_reduction']>=rule['minimum_test_reduction_at_matched_hidden_quality'])
                     or (r['all_policies_execute_exact_budget'] is True and r['nll_advantage_at_four_tests']>=rule['minimum_nll_advantage_at_four_tests'])
                     for r in domains.values())
    else:
        threshold = config['gates']['oracle_headroom']['minimum_nll_advantage_over_fixed_and_random']
        passed = all(r['nll_advantage_over_fixed']>=threshold and r['nll_advantage_over_random']>=threshold for r in domains.values())
    return {'passed':passed,'metrics':{'domains':domains,'oracle_gate_budget':2,
                                     'budget_semantics':'cached public observations, no physical runtime savings'},
            'reason':'Both domains meet the fixed query gate' if passed else 'At least one domain fails the fixed query gate'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['oracle_headroom_gate','active_testing_gate'],required=True)
    args = parser.parse_args()
    io = WorkerIO(args.stage,['scripts/run_apbpf_query_gate_worker.py','src/pbpf/apbpf/assessment_bundle.py','src/pbpf/apbpf/stage_prediction.py'])
    dependency = args.stage.removesuffix('_gate')
    evidence = json.loads(io.artifact(dependency,'query-results.json').read_text())
    gate = decision(evidence,io.config,args.stage)
    (io.outputs/'gate-evidence.json').write_text(json.dumps(gate,indent=2)+'\n')
    if args.stage=='oracle_headroom_gate':
        forward_bundle(io,dependency)
    io.finish({'actual_query_evidence':True,'scope':'exploratory fixed query gate'},gate=gate)


if __name__=='__main__':
    main()
