#!/usr/bin/env python3
"""Run supportive two-domain repair using only exact declared dependencies."""
import argparse
import json

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.repair_packets import select_repair_rows, write_packet
from pbpf.apbpf.stage_prediction import DOMAINS, load_training
from pbpf.apbpf.worker_io import WorkerIO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=['0', '1', '2'], default='0')
    args = parser.parse_args()
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sources = [str(p.relative_to(root)) for p in sorted((root/'src/pbpf').rglob('*.py'))]
    sources += ['scripts/run_apbpf_repair_worker.py', 'scripts/run_apbpf_packet_actor.py',
                'scripts/run_local_packet_repair.py', 'scripts/run_local_packet_repair_evaluation.py',
                'scripts/run_rbr_repair_gate.py', 'scripts/apbpf_generator_sandbox.sh']
    io = WorkerIO('repair', sources)
    if not io.outputs.is_relative_to(root/'local'):
        raise ValueError('repair outputs must reside under the generator-masked local tree')
    training = load_training(io, 'belief')
    origin = json.loads(io.artifact('association_gate', 'assessment-bundle/manifest.json').read_text())
    if (origin['origin_dependencies']['train_belief'] != io.request['dependencies']['train_belief']
            or origin['origin_dependencies']['execution_cache'] != io.request['dependencies']['execution_cache']):
        raise ValueError('repair belief/cache differs from the association gate lineage')
    index = json.loads(io.artifact('execution_cache', 'execution-index.json').read_text())
    domains = {}
    for domain, dataset in DOMAINS.items():
        entry = index['training_caches'][domain]['full']
        cache = io.artifact('execution_cache', entry['path'])
        if file_sha(cache) != entry['sha256']:
            raise ValueError('repair cache identity mismatch')
        payload = json.loads(cache.read_text())
        materials = {}
        for name in ('training-targets', 'public-context', 'evaluator-tests'):
            declared = index['repair_materials'][domain][name]
            path = io.artifact('execution_cache', declared['path'])
            if file_sha(path) != declared['sha256']:
                raise ValueError('repair material checksum mismatch')
            materials[name] = json.loads(path.read_text())
        private = io.outputs/f'{domain}-evaluator-only'
        private.mkdir()
        tasks = private/'tasks.jsonl'
        tasks.write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in materials['evaluator-tests']['records']))
        manifest = private/'manifest.json'
        manifest.write_text(json.dumps({'schema': 'apbpf-declared-repair-tests-v1',
            'evaluator_tasks_sha256': file_sha(tasks), 'execution_cache_completion_sha256': io.request['dependencies']['execution_cache']}, indent=2)+'\n')
        packets = io.outputs/f'{domain}-actor-packets'
        packets.mkdir()
        for seed in io.config['protocol']['seeds']:
            cell = training[domain, seed]
            if cell['entry']['cache_sha256'] != entry['sha256']:
                raise ValueError('repair checkpoint trained on another cache')
            checkpoint = cell['checkpoints']['belief']
            report_path = io.artifact('selection_gate', f'selection-reports/{domain}-seed{seed}.json')
            report = json.loads(report_path.read_text())
            rows = select_repair_rows(payload, materials['training-targets'], materials['public-context'], report,
                cache_sha256=entry['sha256'], checkpoint_sha256=file_sha(checkpoint), seed=seed)
            if sum(row['split'] == 'primary' for row in rows) != 500:
                raise ValueError('repair must retain all 500 selected primary sources')
            write_packet(rows, checkpoint, packets/f'actor-seed{seed}', seed=seed,
                binding={'cache_sha256': entry['sha256'], 'selection_report_sha256': file_sha(report_path),
                         'dependencies': io.request['dependencies'], 'task_bindings': {'evaluator': {
                             'manifest_sha256': file_sha(manifest), 'tasks_sha256': file_sha(tasks)}}})
        python = str(root/'.venv/bin/python')
        generation, evaluation = io.outputs/f'{domain}-generation', io.outputs/f'{domain}-execution'
        io.execute(f'{domain}-train-and-generate', [python, '-u', 'scripts/run_local_packet_repair.py',
            '--packet-root', str(packets), '--output', str(generation), '--domain', domain, '--gpu', args.gpu])
        io.execute(f'{domain}-fresh-execution', [python, '-u', 'scripts/run_local_packet_repair_evaluation.py',
            '--generation-root', str(generation), '--packet-root', str(packets), '--evaluator-root', str(private),
            '--output', str(evaluation)])
        status = json.loads((evaluation/'status.json').read_text())
        if status['status'] != 'complete' or status['smoke_only']:
            raise ValueError('complete non-smoke repair execution required')
        domains[dataset] = {str(seed): json.loads((evaluation/f'seed{seed}/summary.json').read_text())
                            for seed in io.config['protocol']['seeds']}
    evidence = {'schema': 'apbpf-stage-repair-v1', 'domains': domains,
                'scope': 'supportive exploratory repair; failed association/selection gates remain binding',
                'primary_sources_per_domain_seed': 500, 'arms': io.config['ablations']['repair'],
                'new_programs_executed_per_domain_seed': 3000,
                'generator_visibility': 'four public observations and train/development supervised targets only'}
    (io.outputs/'repair.json').write_text(json.dumps(evidence, indent=2)+'\n')
    io.finish({'actual_projector_training': True, 'actual_generation': True, 'actual_fresh_execution': True,
               'scope': evidence['scope'], 'domains': list(domains)})


if __name__ == '__main__':
    main()
