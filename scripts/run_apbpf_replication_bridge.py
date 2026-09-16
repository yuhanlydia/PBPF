#!/usr/bin/env python3
"""Bind existing replication replay caches at the DAG root and forward them.

The underlying workers remain unchanged. Imported execution is always labelled
as reused; replication fitting downstream is fresh and belongs to this run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from build_apbpf_full_replay_cache import assemble
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.worker_io import WorkerIO

WORKERS = {
    'materialize': ('run_apbpf_materialize_worker.py', None),
    'execution_cache': ('run_apbpf_execution_cache_worker.py', 'materialize'),
    'association': ('run_apbpf_association_worker.py', 'execution_cache'),
    'association_gate': ('run_apbpf_prediction_gate_worker.py', 'association'),
}


def import_cache(entry, domain, outputs, config):
    """Rebuild from bound executions; a self-asserted cache/proof is insufficient."""
    cache_path, proof_path = Path(entry['cache']), Path(entry['proof'])
    if (file_sha(cache_path) != entry['cache_sha256']
            or file_sha(proof_path) != entry['proof_sha256']):
        raise ValueError('replication cache/proof differs from frozen root manifest')
    cache, rebuilt = assemble(Path(entry['evaluation_root']), Path(entry['public_root']),
                              Path(entry['evaluator_root']), domain=domain, family='deepseek')
    raw = (json.dumps(cache, sort_keys=True)+'\n').encode()
    if hashlib.sha256(raw).hexdigest() != entry['cache_sha256']:
        raise ValueError('replication cache does not reproduce the bound raw execution')
    proof = json.loads(proof_path.read_text())
    if any(proof.get(k) != v for k, v in rebuilt.items()):
        raise ValueError('replication proof differs from reconstructed provenance')
    if (proof['cache_sha256'] != entry['cache_sha256']
            or proof['builder_sha256'] != file_sha(Path(__file__).with_name('build_apbpf_full_replay_cache.py'))
            or proof['generator'] != next(m for m in config['models'].values() if m['role'] == 'replication')):
        raise ValueError('replication generator or builder identity differs')
    sizes = {'train': 321 if domain == 'rbr' else 212, 'development': 400, 'test': 500}
    if (cache['problem_counts'] != sizes or cache['counts'] != {k: v*8 for k, v in sizes.items()}
            or cache['evaluation_role'] != 'exploratory_locked_primary_assessment'):
        raise ValueError('replication requires the full original source partitions')
    for kind in ('public', 'evaluator'):
        if file_sha(outputs/f'{domain}-{kind}/tasks.jsonl') != proof['task_bindings'][kind]['tasks_sha256']:
            raise ValueError('replication materialized tasks differ from this DAG root')
    for binding in proof['source_bindings']:
        run = json.loads((Path(binding['bank'])/'run.json').read_text())
        if (run['temperature'] != .8 or run['top_p'] != .95
                or run['max_new_tokens'] != (1024 if domain == 'rbr' else 512)):
            raise ValueError('replication decoding budget differs from fixed protocol')
    target = outputs/'replication-inputs'/domain
    target.mkdir(parents=True)
    (target/'cache.json').write_bytes(raw)
    shutil.copyfile(proof_path, target/'proof.json')
    return {'cache': f'{domain}/cache.json', 'cache_sha256': file_sha(target/'cache.json'),
            'proof': f'{domain}/proof.json', 'proof_sha256': file_sha(target/'proof.json')}


def import_manifest(io, path, checksum):
    if file_sha(path) != checksum:
        raise ValueError('replication manifest digest mismatch')
    manifest = json.loads(path.read_text())
    if manifest['schema'] != 'apbpf-root-replication-cache-v1' or set(manifest['domains']) != {'rbr', 'codearc'}:
        raise ValueError('exactly both replication domains required')
    entries = {d: import_cache(e, d, io.outputs, io.config) for d, e in manifest['domains'].items()}
    receipt = {'schema': 'apbpf-bound-replication-inputs-v1', 'fingerprint': io.request['fingerprint'],
               'root_manifest_sha256': checksum, 'domains': entries,
               'fresh_candidate_generation': False, 'fresh_candidate_execution': False,
               'visibility': 'evaluator-only; never mount in generator sandbox',
               'scope': 'explicit root-declared reuse of full existing replay; no imported model or efficacy report'}
    target = io.outputs/'replication-inputs'
    shutil.copyfile(path, target/'root-manifest.json')
    (target/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def forward(io, dependency):
    source = io.directory(dependency, 'replication-inputs')
    manifest = json.loads((source/'manifest.json').read_text())
    if (manifest['schema'] != 'apbpf-bound-replication-inputs-v1'
            or manifest['fingerprint'] != io.request['fingerprint']):
        raise ValueError('replication inputs belong to another run')
    target = io.outputs/'replication-inputs'
    shutil.copytree(source, target)
    for path in source.rglob('*'):
        if path.is_file() and file_sha(path) != file_sha(target/path.relative_to(source)):
            raise ValueError('replication forwarding changed an artifact')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=WORKERS, required=True)
    parser.add_argument('--replication-cache-manifest', type=Path)
    parser.add_argument('--replication-cache-sha256')
    args, remaining = parser.parse_known_args()
    if args.stage != 'materialize' and (args.replication_cache_manifest or args.replication_cache_sha256 or remaining):
        raise ValueError('only root materialize accepts external inputs')
    root = Path(__file__).resolve().parents[1]
    worker, dependency = WORKERS[args.stage]
    sources = [str(p.relative_to(root)) for p in (root/'src/pbpf').rglob('*.py')]
    io = WorkerIO(args.stage, sources + ['scripts/run_apbpf_replication_bridge.py',
        'scripts/build_apbpf_full_replay_cache.py', 'scripts/'+worker])
    if args.stage == 'materialize':
        command = io.config['site']['commands']['materialize']
        for flag, value in (('--replication-cache-manifest', str(args.replication_cache_manifest)),
                            ('--replication-cache-sha256', args.replication_cache_sha256)):
            if not value or command.count(flag) != 1 or command[command.index(flag)+1] != value:
                raise ValueError('replication root inputs must be frozen in the request config')
    command = [str(root/'.venv/bin/python'), str(root/'scripts'/worker), *remaining]
    if args.stage == 'association_gate':
        command += ['--stage', args.stage]
    intermediate = io.request_path.parent/'bridge-base-result.json'
    env = dict(os.environ, APBPF_RESULT=str(intermediate), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
    (io.outputs/'bridge-base-command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (io.outputs/'bridge-base.log').open('x') as log:
        subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    base = json.loads(intermediate.read_text())
    if any(base[k] != io.request[k] for k in ('stage', 'fingerprint', 'dependencies')):
        raise ValueError('underlying worker result identity mismatch')
    if dependency is None:
        import_manifest(io, args.replication_cache_manifest, args.replication_cache_sha256)
    else:
        forward(io, dependency)
    if args.stage == 'association_gate':
        shutil.copyfile(io.artifact('association', 'association.json'), io.outputs/'primary-association.json')
    summary = dict(base['summary'], replication_inputs_forwarded=True, replication_candidate_execution_reused=True)
    io.finish(summary, gate=base.get('gate'))


if __name__ == '__main__':
    main()
