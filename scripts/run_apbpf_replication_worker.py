#!/usr/bin/env python3
"""Freshly fit replication-family models and bind the full two-by-two matrix."""
import json
from pathlib import Path

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.config import resolve_config
from pbpf.apbpf.stage_prediction import DOMAINS
from pbpf.apbpf.stage_selection import aggregate_selection
from pbpf.apbpf.worker_io import WorkerIO


def make_cell(association, selection, *, dataset, family, checksum, bindings, scope):
    if (association['seeds'] != [1701, 1702, 1703] or selection['seeds'] != association['seeds']
            or association['source_components'] != 500 or association['primary_candidates'] != 4000
            or selection['primary_sources'] != 500 or selection['primary_candidates'] != 4000
            or selection['bootstrap_draws'] != 10000
            or association['cache_sha256'] != checksum or selection['cache_sha256'] != checksum
            or selection['comparator'] != 'strongest_cross_fitted_deterministic'):
        raise ValueError('full shared-cache populations, seeds and comparator required')
    return {'domain': dataset, 'family': family, 'seeds': association['seeds'],
            'primary_sources': 500, 'primary_candidates': 4000, 'bootstrap_draws': 10000,
            'association_gap': association['comparisons']['outcome_shuffled']['mean_nll_gap'],
            'association_ci95': association['comparisons']['outcome_shuffled']['ci95'],
            'selection_advantage': selection['absolute_selected_pass1_advantage'],
            'selection_ci95': selection['ci95'], 'cache_sha256': checksum,
            'bindings': bindings, 'scope': scope}


def main():
    root = Path(__file__).resolve().parents[1]
    sources = [str(p.relative_to(root)) for p in (root/'src/pbpf').rglob('*.py')]
    sources += ['scripts/'+name for name in ('run_apbpf_replication_worker.py',
        'run_local_stage_training_diagnostic.py', 'run_apbpf_train_worker.py',
        'run_apbpf_association_worker.py', 'run_rbr_prediction_gate.py', 'run_local_selection_diagnostic.py')]
    io = WorkerIO('replication', sources)
    local = resolve_config(root/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory').config
    if any(local[key] != io.config[key] for key in ('protocol', 'models', 'datasets')):
        raise ValueError('fixed local fitting helpers do not match the declared protocol')
    models = {m['role']: m for m in io.config['models'].values()}
    primary_path = io.artifact('association_gate', 'primary-association.json')
    primary = json.loads(primary_path.read_text())
    if (primary['schema'] != 'apbpf-stage-association-v1' or set(primary['domains']) != set(DOMAINS.values())
            or primary['bootstrap_draws'] != 10000 or primary['population'] != 'full_locked_population'):
        raise ValueError('complete stage-bound primary association required')
    directory = io.directory('association_gate', 'replication-inputs')
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest['schema'] != 'apbpf-bound-replication-inputs-v1'
            or manifest['fingerprint'] != io.request['fingerprint'] or set(manifest['domains']) != set(DOMAINS)):
        raise ValueError('root-declared replication inputs from this run required')
    cells = []
    python = str(root/'.venv/bin/python')
    for domain, dataset in DOMAINS.items():
        association = primary['domains'][dataset]
        reports, report_bindings = {}, {}
        for seed in io.config['protocol']['seeds']:
            path = io.artifact('selection_gate', f'selection-reports/{domain}-seed{seed}.json')
            reports[seed] = json.loads(path.read_text())
            if (reports[seed]['cache_sha256'] != association['cache_sha256']
                    or reports[seed]['dataset'] != dataset):
                raise ValueError('primary selection and association cache or dataset differs')
            report_bindings[str(seed)] = file_sha(path)
        selection = aggregate_selection(reports)
        selection['cache_sha256'] = association['cache_sha256']
        cells.append(make_cell(association, selection, dataset=dataset,
            family=models['primary']['family'], checksum=association['cache_sha256'],
            bindings={'association': file_sha(primary_path), 'selection_by_seed': report_bindings},
            scope='same-run primary-family stages; no refitting or independent-repeat claim'))
        entry = manifest['domains'][domain]
        paths = {}
        for kind in ('cache', 'proof'):
            if entry[kind] != f'{domain}/{kind}.json':
                raise ValueError('unexpected replication input artifact path')
            path = io.artifact('association_gate', 'replication-inputs/'+entry[kind])
            if file_sha(path) != entry[kind+'_sha256']:
                raise ValueError('replication input digest mismatch')
            paths[kind] = path
        proof = json.loads(paths['proof'].read_text())
        if (proof['generator'] != models['replication'] or proof['domain'] != domain
                or proof['cache_sha256'] != entry['cache_sha256']):
            raise ValueError('replication generator/domain/cache proof mismatch')
        training, selecting = io.outputs/f'{domain}-training', io.outputs/f'{domain}-selection'
        io.execute(domain+'-training', [python, '-u', 'scripts/run_local_stage_training_diagnostic.py',
            '--cache', str(paths['cache']), '--cache-proof', str(paths['proof']), '--output', str(training)])
        io.execute(domain+'-selection', [python, '-u', 'scripts/run_local_selection_diagnostic.py',
            '--cache', str(paths['cache']), '--training-root', str(training), '--output', str(selecting)])
        association = json.loads((training/'comparison.json').read_text())
        selection = json.loads((selecting/'results.json').read_text())
        cells.append(make_cell(association, selection, dataset=dataset,
            family=models['replication']['family'], checksum=entry['cache_sha256'],
            bindings={'cache_proof': entry['proof_sha256'], 'association': file_sha(training/'comparison.json'),
                      'selection': file_sha(selecting/'results.json')},
            scope='fresh three-seed fitting and assessment in this stage; root-declared candidate execution reuse'))
    result = {'schema': 'apbpf-stage-replication-v1', 'cells': cells,
              'root_manifest_sha256': manifest['root_manifest_sha256'],
              'fresh_replication_fitting': True, 'fresh_replication_candidate_execution': False,
              'scope': 'exploratory two-domain two-family assessment; failed upstream gates remain binding'}
    (io.outputs/'replication.json').write_text(json.dumps(result, indent=2)+'\n')
    io.finish({'actual_replication_fitting': True, 'complete_domain_family_matrix': True,
               'replication_candidate_execution_reused': True, 'claim_status': 'exploratory-predeclared'})


if __name__ == '__main__':
    main()
