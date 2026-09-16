#!/usr/bin/env python3
"""Score the required fixed-alpha history-only control on bound predictions."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_prediction import aggregate_predictions
from pbpf.real_gate import OUTCOMES, validate_rbr_cache


def history_predictions(rows):
    """Laplace smoothing alpha=1, fixed in advance; four observed labels only."""
    probabilities = []
    for row in rows:
        observed = [OUTCOMES.index(label) for label in row['outcomes'][:4]]
        if len(observed) != 4:
            raise ValueError('exactly four public outcomes required')
        counts = np.bincount(observed, minlength=len(OUTCOMES)).astype(np.float64) + 1.
        probabilities.extend([counts/counts.sum()]*6)
    return np.asarray(probabilities)


def assess(payload, arrays, populations, *, checksum):
    validate_rbr_cache(payload)
    if payload['evaluation_role'] != 'exploratory_locked_primary_assessment':
        raise ValueError('requires original full primary assessment')
    rows = [row for row in payload['records'] if row['split'] == 'test']
    population = [{'candidate_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
    if (len(rows) != 4000 or len({r['source_component_id'] for r in rows}) != 500
            or any(r['original_split'] != 'primary' for r in rows)
            or any(populations[s] != population for s in (1701, 1702, 1703))):
        raise ValueError('full source/candidate population must match fitted predictions')
    labels = np.array([OUTCOMES.index(o) for r in rows for o in r['outcomes'][4:]])
    rate = history_predictions(rows)
    values = {}
    for seed in (1701, 1702, 1703):
        if not np.array_equal(arrays[seed]['labels'], labels):
            raise ValueError('prediction labels differ from bound primary cache')
        values[seed] = {'labels': labels, 'aligned': arrays[seed]['aligned'], 'history_rate': rate}
    report = aggregate_predictions(values, populations, bootstrap_seed=201701, replicates=10000)
    report.update(schema='apbpf-history-rate-ablation-v1', dataset=payload['dataset'], cache_sha256=checksum,
        alpha=1., fitted_parameters=0, public_observations=4,
        estimand='fixed-alpha categorical history-rate versus each fitted aligned predictor; full primary sources',
        scope='required supplementary prediction ablation; does not replace tuned-Dirichlet fairness or association gates')
    return report, rate


def evaluate_training(cache, training, output):
    checksum = file_sha(cache)
    state = json.loads((training/'status.json').read_text())
    if state['status'] != 'complete' or state['cache_sha256'] != checksum:
        raise ValueError('complete fitting on this exact cache required')
    arrays, populations, bindings = {}, {}, {}
    for seed in (1701, 1702, 1703):
        directory = training/f'belief-seed{seed}'
        path = directory/'training.json'; report = json.loads(path.read_text())
        if (report['config']['seed'] != seed
                or state['runs'][f'belief-seed{seed}']['training_report_sha256'] != file_sha(path)
                or file_sha(directory/'belief.pt') != report['arms']['belief']['checkpoint_sha256']):
            raise ValueError('training report or checkpoint identity mismatch')
        pred = directory/'primary-predictions.npz'
        with np.load(pred, allow_pickle=False) as archive:
            arrays[seed] = {k: archive[k].copy() for k in ('aligned', 'labels')}
        # The independently saved control replay must contain the same aligned values.
        with np.load(training/f'controls-seed{seed}.npz', allow_pickle=False) as archive:
            if not np.array_equal(archive['labels'], arrays[seed]['labels']) or not np.allclose(
                    archive['aligned'], arrays[seed]['aligned'], rtol=1e-6, atol=1e-7):
                raise ValueError('aligned fit and control replay predictions disagree')
        populations[seed] = report['primary_population']
        bindings[str(seed)] = {'training_report_sha256': file_sha(path), 'predictions_sha256': file_sha(pred),
                               'controls_sha256': file_sha(training/f'controls-seed{seed}.npz')}
    report, rate = assess(json.loads(cache.read_text()), arrays, populations, checksum=checksum)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output/'history-rate-probabilities.npy', rate, allow_pickle=False)
    report.update(bindings=bindings, training_status_sha256=file_sha(training/'status.json'),
                  probabilities_sha256=file_sha(output/'history-rate-probabilities.npy'))
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--training-root', type=Path)
    parser.add_argument('--wait-for-cell', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.wait_for_cell) == bool(args.cache or args.training_root) or (not args.wait_for_cell and not (args.cache and args.training_root)):
        raise ValueError('supply either a cell to wait for, or explicit cache and training root')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), *sorted((root/'src/pbpf').rglob('*.py'))]
    hashes = {str(p.relative_to(root)): file_sha(p) for p in sources}
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'waiting_for_fitting', 'pid': os.getpid(), 'source_sha256': hashes,
             'scope': 'standalone required ablation, not a sealed DAG stage', 'alpha': 1.}
    def save():
        p = args.output/'status.partial'; p.write_text(json.dumps(state, indent=2)+'\n'); p.replace(args.output/'status.json')
    save()
    try:
        if args.wait_for_cell:
            while True:
                parent = json.loads((args.wait_for_cell/'status.json').read_text())
                if parent['status'] == 'complete': break
                proc = Path(f"/proc/{parent['pid']}")
                if parent['status'] == 'needs_debug' or not proc.exists():
                    raise RuntimeError('replication cell exited incomplete')
                command = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode()
                if 'run_local_full_replication_cell.py' not in command or str(args.wait_for_cell.resolve()) not in command:
                    raise RuntimeError('replication cell PID no longer identifies the expected supervisor')
                time.sleep(20)
            args.cache = args.wait_for_cell/'cache/cache.json'
            args.training_root = args.wait_for_cell/'training'
        if any(file_sha(root/n) != h for n, h in hashes.items()):
            raise ValueError('ablation code changed while waiting')
        state.update(status='running'); save()
        report = evaluate_training(args.cache, args.training_root, args.output/'assessment')
        state.update(status='complete', results_sha256=file_sha(args.output/'assessment/results.json'),
                     gap=report['comparisons']['history_rate']); save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error)); save(); raise


if __name__ == '__main__':
    main()
