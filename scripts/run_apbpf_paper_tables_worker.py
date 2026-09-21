#!/usr/bin/env python3
"""Export auditable exploratory tables, preserving every failed gate."""
import csv
import json
from pathlib import Path

from pbpf.apbpf.stage_prediction import DOMAINS
from pbpf.apbpf.stages import DEPENDENCIES
from pbpf.apbpf.worker_io import WorkerIO


def flatten(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f'{prefix}.{key}' if prefix else key)
    else:
        yield prefix, json.dumps(value, sort_keys=True, allow_nan=False)


def gate_rows(results, *, claim_status):
    if claim_status != 'exploratory-predeclared':
        raise ValueError('paper tables require the permanently exploratory claim identity')
    if set(results) != set(DEPENDENCIES['paper_tables']):
        raise ValueError('tables require every declared gate and repair input')
    gates, measurements = [], []
    for name, result in results.items():
        if result['stage'] != name or result['schema'] != 'apbpf-stage-result-v1':
            raise ValueError('paper-table stage identity mismatch')
        if name == 'repair':
            continue
        gate = result['gate']
        if type(gate['passed']) is not bool:
            raise ValueError('explicit gate decision required')
        gates.append({'stage': name, 'passed': gate['passed'], 'reason': gate['reason']})
        for metric, value in flatten(gate['metrics']):
            measurements.append({'stage': name, 'passed': gate['passed'], 'metric': metric, 'value_json': value})
    return gates, measurements


def write_csv(path, rows, columns):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    io = WorkerIO('paper_tables', ['scripts/run_apbpf_paper_tables_worker.py',
                                  'src/pbpf/apbpf/stage_prediction.py', 'src/pbpf/apbpf/stages.py'])
    # WorkerIO has verified the exact same-run result/completion receipts.
    results = {name: json.loads(Path(item['result']).read_text()) for name, item in io.request['inputs'].items()}
    gates, measurements = gate_rows(results, claim_status=io.request['claim_status'])
    write_csv(io.outputs/'gate-decisions.csv', gates, ['stage', 'passed', 'reason'])
    write_csv(io.outputs/'gate-measurements.csv', measurements, ['stage', 'passed', 'metric', 'value_json'])
    selection_rows = []
    for domain, dataset in DOMAINS.items():
        for seed in io.config['protocol']['seeds']:
            value = json.loads(io.artifact('selection_gate', f'selection-reports/{domain}-seed{seed}.json').read_text())
            if value['seed'] != seed or value['population']['primary_sources'] != 500 or value['population']['primary_candidates'] != 4000:
                raise ValueError('full-population selection report required for tables')
            for arm, rate in value['selected_pass1'].items():
                selection_rows.append({'domain': dataset, 'seed': seed, 'arm': arm,
                                       'hidden_selected_pass1': rate, 'sources': 500})
    write_csv(io.outputs/'selection-by-seed.csv', selection_rows,
              ['domain', 'seed', 'arm', 'hidden_selected_pass1', 'sources'])
    repair = json.loads(io.artifact('repair', 'repair.json').read_text())
    if repair['schema'] != 'apbpf-stage-repair-v1' or set(repair['domains']) != set(DOMAINS.values()):
        raise ValueError('complete two-domain repair evidence required')
    repair_rows = []
    for domain, seeds in repair['domains'].items():
        if set(seeds) != {str(s) for s in io.config['protocol']['seeds']}:
            raise ValueError('all repair seeds required')
        for seed, arms in seeds.items():
            if set(arms) != set(io.config['ablations']['repair']):
                raise ValueError('all six repair arms required')
            for arm, values in arms.items():
                if values['sources'] != 500:
                    raise ValueError('repair table cannot omit primary sources')
                repair_rows.append({'domain': domain, 'seed': seed, 'arm': arm, **values})
    write_csv(io.outputs/'repair-by-seed.csv', repair_rows,
              ['domain', 'seed', 'arm', 'sources', 'hidden_successes', 'all_ten_successes', 'future_pass_fraction'])
    replication = json.loads(io.artifact('replication_gate', 'replication-evidence.json').read_text())
    columns = ['domain', 'family', 'association_gap', 'association_ci95', 'selection_advantage', 'selection_ci95',
               'primary_sources', 'primary_candidates', 'seeds', 'bootstrap_draws']
    replication_rows = [{k: json.dumps(row[k]) if isinstance(row[k], list) else row[k] for k in columns}
                        for row in replication['cells']]
    write_csv(io.outputs/'replication.csv', replication_rows, columns)
    history_rows = []
    for cell in replication['cells']:
        control = cell['prediction_ablations']['history_rate']
        if (control['cache_sha256'] != cell['cache_sha256'] or control['alpha'] != 1.
                or control['seeds'] != cell['seeds'] or control['source_components'] != 500
                or control['primary_candidates'] != 4000):
            raise ValueError('history-rate table requires the exact full replication population')
        gap = control['comparisons']['history_rate']
        history_rows.append({'domain': cell['domain'], 'family': cell['family'], 'alpha': 1.,
            'aligned_nll': control['metrics']['aligned']['nll'],
            'history_rate_nll': control['metrics']['history_rate']['nll'],
            'nll_gap': gap['mean_nll_gap'], 'ci95': json.dumps(gap['ci95']), 'sources': 500})
    write_csv(io.outputs/'history-rate-ablation.csv', history_rows,
              ['domain', 'family', 'alpha', 'aligned_nll', 'history_rate_nll', 'nll_gap', 'ci95', 'sources'])
    failed = [row['stage'] for row in gates if not row['passed']]
    summary = {'schema': 'apbpf-exploratory-tables-v1', 'failed_gates': failed,
               'main_table_eligible': False, 'claim_status': 'exploratory-predeclared',
               'dependencies': io.request['dependencies'], 'seeds': io.config['protocol']['seeds'],
               'population': 'all500 primary sources per domain/family, eight candidates; repair uses one selected candidate per source',
               'scope': 'descriptive reproducibility tables; no confirmatory or particle-necessity claim',
               'limitations': ['Gate metrics retain only the bounds supplied by the direct gate evidence; no missing intervals inferred.',
                              'Seed rows are repeated fits on the same sources, not independent new source samples.',
                              'Active-testing counts are cached public observations, not wall-clock savings.',
                              'Token-remix fault is an intentionally incorrect and slower control.']}
    (io.outputs/'table-provenance.json').write_text(json.dumps(summary, indent=2)+'\n')
    (io.outputs/'README.md').write_text('# Exploratory experiment tables\n\n'
        'These tables retain all failed gates and cannot support main-table or confirmatory claims.\n\n'
        +'Failed gates: '+(', '.join(failed) if failed else 'none; profile remains permanently exploratory')+'.\n\n'
        +'\n'.join('- '+item for item in summary['limitations'])+'\n')
    io.finish({'actual_evidence_exported': True, 'failed_gates': failed,
               'main_table_eligible': False, 'scope': summary['scope']})


if __name__ == '__main__':
    main()
