#!/usr/bin/env python3
"""Audit the exact stage-locked primary bank and all development sources."""
import json

from pbpf.apbpf.config import digest
from pbpf.apbpf.worker_io import WorkerIO


def main():
    io = WorkerIO('hard_bank', ['scripts/run_apbpf_hard_bank_worker.py',
        'scripts/build_apbpf_hard_bank.py', 'src/pbpf/apbpf/hard_bank.py'])
    index_path = io.artifact('execution_cache', 'execution-index.json')
    index = json.loads(index_path.read_text())
    if (index['schema'] != 'apbpf-generated-execution-cache-v1'
            or index['lock_stage_completion_sha256'] != io.request['dependencies']['hard_bank_lock']):
        raise ValueError('execution cache does not descend from this exact pre-hidden lock')
    development, hidden, domain_counts = [], [], {}
    for entry in index['evaluations']:
        if entry['split'] not in ('development', 'primary'):
            continue
        domain = entry['domain']
        rows = json.loads(io.artifact('execution_cache',
            entry['evaluation_directory']+'/bank_groups.json').read_text())
        if entry['split'] == 'development':
            if entry['phase'] != 'all':
                raise ValueError('development requires complete execution')
            development += [{**row, 'source_component_id': domain+'::'+row['source_component_id']} for row in rows]
            counts = domain_counts.setdefault(domain, {'development_groups': 0, 'mixed_groups': 0})
            counts['development_groups'] += len(rows)
            counts['mixed_groups'] += sum(0 < sum(row['hidden_labels']) < 8 for row in rows)
        else:
            if entry['phase'] != 'hidden':
                raise ValueError('primary must use only the locked hidden phase')
            hidden += rows
    if (set(domain_counts) != {'rbr', 'codearc'}
            or any(row['development_groups'] != 400 for row in domain_counts.values())
            or len(hidden) != 1000 or len({r['source_component_id'] for r in development}) != 800):
        raise ValueError('audit needs both complete development and primary populations')
    lock = io.artifact('hard_bank_lock', 'population-lock.json')
    mixed = [r['group_id'] for r in development if 0 < sum(r['hidden_labels']) < 8]
    # A failing pilot is evidence too: retain its true count, without relaxing 300.
    pilot = {'schema': 'apbpf-hard-bank-pilot-v2', 'mixed_groups': len(mixed),
        'total_development_groups': len(development),
        'all_source_component_ids': [r['source_component_id'] for r in development],
        'selected_development_group_ids': mixed,
        'selection_scope': 'development only; entire population retained in supporting evidence',
        'minimum_mixed_groups': 300, 'passes': len(mixed) >= 300, 'domains': domain_counts}
    pilot_path = io.outputs/'pilot.json'
    pilot_path.write_text(json.dumps({**pilot, 'content_sha256': digest(pilot)}, indent=2)+'\n')
    (io.outputs/'development-bank.json').write_text(json.dumps(development, indent=2)+'\n')
    hidden_path = io.outputs/'hidden-bank.json'; hidden_path.write_text(json.dumps(hidden, indent=2)+'\n')
    (io.outputs/'execution-provenance.json').write_bytes(index_path.read_bytes())
    io.execute('audit', [str(io.root/'.venv/bin/python'), 'scripts/build_apbpf_hard_bank.py',
        'audit', '--lock', str(lock), '--hidden-bank', str(hidden_path), '--pilot-report', str(pilot_path),
        '--output', str(io.outputs/'audit.json')])
    audit = json.loads((io.outputs/'audit.json').read_text())
    io.finish({'actual_hard_bank_audit': True, 'domains': domain_counts,
               'primary_groups': len(hidden), 'gate_conditions': audit['gate'],
               'scope': 'exploratory audit; subsequent gate recomputes the locked decision'})


if __name__ == '__main__':
    main()
