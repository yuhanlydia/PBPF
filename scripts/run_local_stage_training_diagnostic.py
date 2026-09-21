#!/usr/bin/env python3
"""Run full-cache stage trainers independently while the two-domain DAG waits.

This is explicitly an exploratory single-domain diagnostic, not a sealed stage.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.config import resolve_config
from pbpf.apbpf.stage_prediction import aggregate_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--cache-proof', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    proof = json.loads(args.cache_proof.read_text()); checksum = file_sha(args.cache)
    if proof['cache_sha256'] != checksum:
        raise ValueError('cache differs from real-data assembly proof')
    payload = json.loads(args.cache.read_text())
    if (payload['evaluation_role'] != 'exploratory_locked_primary_assessment'
            or payload['problem_counts']['test'] != 500):
        raise ValueError('requires the full existing primary population')
    source_files = [*sorted((root/'src/pbpf').rglob('*.py')),
        Path(__file__).resolve(), root/'scripts/run_apbpf_train_worker.py',
        root/'scripts/run_apbpf_association_worker.py', root/'scripts/run_rbr_prediction_gate.py']
    hashes = {str(p.relative_to(root)): file_sha(p) for p in source_files}
    config = resolve_config(root/'configs/experiments/apbpf_iclr2027.yaml', 'local_exploratory').config
    state = {'status': 'running', 'pid': os.getpid(), 'cache_sha256': checksum,
             'cache_proof_sha256': file_sha(args.cache_proof), 'source_sha256': hashes,
             'dataset': payload['dataset'], 'seeds': config['protocol']['seeds'], 'steps': 1000,
             'scope': 'standalone exploratory full-primary diagnostic; no sealed-stage completion', 'runs': {}}

    def save():
        tmp = output/'status.partial'; tmp.write_text(json.dumps(state, indent=2)+'\n'); tmp.replace(output/'status.json')

    def check():
        if any(file_sha(root/p) != h for p, h in hashes.items()) or file_sha(args.cache) != checksum:
            raise ValueError('declared source/cache changed; a new diagnostic is required')

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root/'scripts'/filename)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
        return module

    save()
    try:
        trainer = load('full_cache_diagnostic_trainer', 'run_apbpf_train_worker.py')
        helper = load('full_cache_diagnostic_helper', 'run_rbr_prediction_gate.py')
        replay = load('full_cache_diagnostic_replay', 'run_apbpf_association_worker.py')
        values, populations = {}, {}
        for seed in state['seeds']:
            cells = {}
            for kind in ('baselines', 'belief'):
                check(); name = f'{kind}-seed{seed}'; directory = output/name
                state.update(current=name); state['runs'][name] = {'status':'running'}; save()
                report = trainer.train_models(payload, directory, kind=kind, seed=seed, steps=1000,
                    batch_size=64, learning_rate=.0003, feature_dim=256, hidden_dim=192,
                    protocol=config['protocol'], helper=helper)
                with np.load(directory/'primary-predictions.npz', allow_pickle=False) as archive:
                    predictions = {k:archive[k].copy() for k in archive.files}
                cells[kind] = {'report':report, 'predictions':predictions,
                              'checkpoints':{'belief':directory/'belief.pt'} if kind=='belief' else {}}
                state['runs'][name].update(status='complete', training_report_sha256=file_sha(directory/'training.json')); save()
            check(); state['current'] = f'controls-seed{seed}'; save()
            predictions, population, masks = replay.replay(payload, cells['belief'], helper=helper)
            predictions.update({k:v for k,v in cells['baselines']['predictions'].items() if k!='labels'})
            values[seed] = predictions; populations[seed] = population
            np.savez_compressed(output/f'controls-seed{seed}.npz', **predictions)
            # Preserve strata; the aggregate gate diagnostics always use full population.
            (output/f'strata-seed{seed}.json').write_text(json.dumps({k:int(v.sum()) for k,v in masks.items()},indent=2)+'\n')
        check()
        report = aggregate_predictions(values, populations, bootstrap_seed=201701,
                                        replicates=config['protocol']['bootstrap_draws'])
        report.update(scope=state['scope'], dataset=state['dataset'], cache_sha256=checksum)
        (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        state.update(status='complete', comparison_sha256=file_sha(output/'comparison.json')); save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error)); save(); raise


if __name__ == '__main__':
    main()
