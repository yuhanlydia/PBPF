#!/usr/bin/env python3
"""Run real oracle-subset or active-query SMC replay from declared bundles."""
import argparse
import json
from pathlib import Path

import numpy as np

from pbpf.apbpf.assessment_bundle import forward_bundle, load_bundle
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_prediction import DOMAINS
from pbpf.apbpf.stage_queries import replay_queries, summarize_queries, paired_loss_gap
from pbpf.apbpf.stopping import lock_threshold, stopped_budgets
from pbpf.apbpf.worker_io import WorkerIO
from pbpf.real_gate import validate_rbr_cache

THRESHOLDS = [0., .001, .003, .01, .03, .1, .3, 1.]


def stopping_arrays(records):
    information = [[r['budgets'][str(b)]['diagnostic_mi']['remaining_information'] for b in (1,2,3)] for r in records]
    nll = np.array([[r['budgets'][str(b)]['diagnostic_mi']['future_nll'] for b in (1,2,3,4)] for r in records])
    return information, nll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['oracle_headroom', 'active_testing'], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = [str(p.relative_to(root)) for p in sorted((root/'src/pbpf').rglob('*.py'))]
    io = WorkerIO(args.stage, sources+['scripts/run_apbpf_query_worker.py'])
    active = args.stage == 'active_testing'
    dependency = 'oracle_headroom_gate' if active else 'association_gate'
    bundle = load_bundle(io, dependency)
    manifest = json.loads(io.artifact(dependency, 'assessment-bundle/manifest.json').read_text())
    if active and manifest['origin_dependencies']['train_belief'] != io.request['dependencies']['train_belief']:
        raise ValueError('forwarded query models are not the exact current belief-training dependency')
    domains = {}
    for domain, dataset in DOMAINS.items():
        cell = bundle[domain]; payload = validate_rbr_cache(json.loads(cell['cache'].read_text()))
        if payload.get('evaluation_role') != 'exploratory_locked_primary_assessment' or payload['dataset'] != dataset:
            raise ValueError('query replay requires the full original-split assessment cache')
        primary = [r for r in payload['records'] if r['split']=='test']
        development = [r for r in payload['records'] if r['split']=='development']
        if (len(primary)!=4000 or len({r['source_component_id'] for r in primary})!=500
                or {r['source_component_id'] for r in primary} & {r['source_component_id'] for r in development}):
            raise ValueError('query population differs from disjoint full primary/development sources')
        records, thresholds, stopped, stopped_sources, budgets_used, curves = {}, {}, [], [], [], {}
        for seed in io.config['protocol']['seeds']:
            io.check_sources()
            model = cell['models'][seed]
            expected = [{'candidate_id':r['task_id'],'source_component_id':r['source_component_id']} for r in primary]
            if expected != model['report']['primary_population']:
                raise ValueError('query population differs from training checkpoint evidence')
            checkpoint = model['checkpoint']
            if active:
                # Lock threshold on original development before primary replay.
                calibration = []
                with (io.outputs/f'{domain}-seed{seed}-calibration.jsonl').open('x') as stream:
                    for row in replay_queries(development, checkpoint, seed=seed, mode='active'):
                        calibration.append(row); stream.write(json.dumps(row)+'\n')
                information, nll = stopping_arrays(calibration)
                lock = lock_threshold(information, nll.mean(-1), THRESHOLDS)
                lock.update(calibration_sources=sorted({r['source'] for r in calibration}),
                            checkpoint_sha256=file_sha(checkpoint), cache_sha256=file_sha(cell['cache']))
                lock_path = io.outputs/f'{domain}-seed{seed}-threshold-lock.json'
                lock_path.write_text(json.dumps(lock,indent=2)+'\n'); thresholds[str(seed)] = lock
            scored = []
            with (io.outputs/f'{domain}-seed{seed}-primary.jsonl').open('x') as stream:
                for row in replay_queries(primary, checkpoint, seed=seed, mode='active' if active else 'oracle'):
                    scored.append(row); stream.write(json.dumps(row)+'\n')
                    if len(scored)%64 == 0:
                        stream.flush(); print(json.dumps({'domain':domain,'seed':seed,'primary_completed':len(scored),'total':len(primary)}),flush=True)
            records[seed] = scored
            if active:
                information, nll = stopping_arrays(scored)
                selected_budgets = stopped_budgets(information, lock['threshold'])
                stopped.extend(nll[np.arange(len(nll)), selected_budgets-1].tolist())
                budgets_used.extend(selected_budgets.tolist()); stopped_sources.extend(r['source'] for r in scored)
                curves[str(seed)] = []
                for threshold in [*THRESHOLDS, None]:
                    curve_budgets = stopped_budgets(information, threshold)
                    curve_nll = nll[np.arange(len(nll)), curve_budgets-1]
                    curves[str(seed)].append({'threshold':threshold, 'mean_tests':float(curve_budgets.mean()),
                        'nll':float(curve_nll.mean()), 'selected_by_development':threshold==lock['threshold'],
                        'against_same_policy_four':paired_loss_gap(nll[:,-1],curve_nll,[r['source'] for r in scored],draws=io.config['protocol']['bootstrap_draws'])})
        report = summarize_queries(records, mode='active' if active else 'oracle', draws=io.config['protocol']['bootstrap_draws'])
        report['cache_sha256'] = file_sha(cell['cache'])
        if active:
            matched = {}
            for reference in ('fixed','random','diagnostic_mi'):
                fixed = [r['budgets']['4'][reference]['future_nll'] for seed in (1701,1702,1703) for r in records[seed]]
                matched[reference] = paired_loss_gap(fixed, stopped, stopped_sources, draws=io.config['protocol']['bootstrap_draws'])
            report['stopping'] = {'thresholds':thresholds, 'assessment_curves':curves, 'test_reduction':float(1-np.mean(budgets_used)/4),
                'mean_tests':float(np.mean(budgets_used)), 'against_four_test_references':matched,
                'matched_hidden_quality':all(x['mean_nll_gap']>=0 and x['ci95'][0]>=0 for x in matched.values()),
                'quality_rule':'no worse than fixed, random and same-policy four-test NLL, with nonnegative clustered lower bounds',
                'budget_semantics':'logical cached observations only; no physical runtime savings claimed'}
        domains[dataset] = report
    evidence = {'schema':'apbpf-stage-query-replay-v1', 'stage':args.stage, 'domains':domains,
                'oracle_gate_budget':2, 'public_pool':4, 'future_targets':6,
                'oracle_protocol':'canonical public subsets; random comparator is a uniformly sampled canonical subset',
                'active_protocol':'adaptive sequential order, original joint SMC posterior; nuisance-marginal diagnostic MI selector',
                'scope':'exploratory complete-population cached replay; no new physical test-execution savings'}
    (io.outputs/'query-results.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if not active:
        forward_bundle(io, dependency)
    io.finish({'actual_smc_replay':True,'domains':list(domains),'public_pool':4,
               'scope':evidence['scope'],'oracle_gate_budget':2})


if __name__ == '__main__':
    main()
