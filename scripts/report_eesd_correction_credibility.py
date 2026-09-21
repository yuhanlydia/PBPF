#!/usr/bin/env python3
"""Independently evaluate sealed corrections or summarize a sealed prior evaluation.

Only hidden-only correctness on the same original/correction programs is labelled.
Public score quantiles do not select, drop, or regenerate trajectories. This does
not measure full-suite regressions from originally correct programs: the existing
correction cohort deliberately contains visible-failing originals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pbpf.eesd.correction_credibility import (
    digest, evaluate_bundle, identity, load_public_bundle, snapshot, summarize_credibility,
)


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return digest(path.read_bytes())


def prior_evaluation(directory):
    complete, _ = snapshot(directory / 'complete.json')
    if complete.get('schema') != 'eesd-correction-credibility-complete-v1':
        raise ValueError('sealed independent correction evaluation required')
    evaluation, evalsha = snapshot(directory / 'evaluation.json')
    lock, locksha = snapshot(directory / 'input-lock.json')
    if (complete.get('files', {}).get('evaluation.json') != evalsha
            or complete.get('files', {}).get('input-lock.json') != locksha
            or lock.get('schema') != 'eesd-correction-input-lock-v1'
            or evaluation.get('input_lock_sha256') != locksha
            or evaluation.get('input_binding') != lock.get('input_binding')):
        raise ValueError('independent evaluation seal mismatch')
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-dir', type=Path, required=True)
    parser.add_argument('--scored-dir', type=Path, required=True)
    parser.add_argument('--public-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--domain', choices=['rbr','codearc'], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--evaluator-root', type=Path, help='run isolated original/correction hidden execution')
    source.add_argument('--evaluation-dir', type=Path, help='only postprocess a previously sealed independent evaluation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bins', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=6.0)
    args = parser.parse_args()
    if args.bins < 1:
        parser.error('positive bin count required')
    bundle = load_public_bundle(args.generation_dir, args.scored_dir, args.public_dir, args.config, domain=args.domain)
    # Freeze public inputs and complete trajectory population BEFORE private-root reads.
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root/'src/pbpf/eesd/correction_credibility.py',
             root/'src/pbpf/eesd/distillation.py', root/'src/pbpf/eesd/evidence.py',
             root/'src/pbpf/apbpf/rbr_execution.py', root/'src/pbpf/apbpf/codearc_execution.py']
    lock = {'schema':'eesd-correction-input-lock-v1', 'input_binding':bundle['input_binding'],
        'population':[{**identity(r), 'evaluator_task_id':r['evaluator_task_id'], 'original_sha256':r['original_sha256'],
                       'correction_sha256':r['correction_sha256']} for r in bundle['records']],
        'source_sha256':{str(p.relative_to(root)):digest(p.read_bytes()) for p in files},
        'bins':args.bins, 'timeout':args.timeout, 'private_root_opened':False}
    args.output.mkdir(parents=True, exist_ok=False)
    locksha = save(args.output/'input-lock.json', lock)
    try:
        if args.evaluation_dir:
            evaluation = prior_evaluation(args.evaluation_dir)
            evaluation['prior_input_lock_sha256'] = evaluation['input_lock_sha256']
        else:
            from pbpf.apbpf.rbr_execution import execute_stdin
            from pbpf.apbpf.codearc_execution import execute_call
            executor = execute_stdin if args.domain == 'rbr' else execute_call
            evaluation = evaluate_bundle(bundle, args.evaluator_root, execute=executor, timeout=args.timeout)
        evaluation['input_lock_sha256'] = locksha
        summary = summarize_credibility(bundle, evaluation, bins=args.bins)
        evalsha = save(args.output/'evaluation.json', evaluation)
        summary['evaluation_sha256'] = evalsha
        summarysha = save(args.output/'credibility.json', summary)
        save(args.output/'complete.json', {'schema':'eesd-correction-credibility-complete-v1',
             'files':{'input-lock.json':locksha,'evaluation.json':evalsha,'credibility.json':summarysha}})
    except BaseException as error:
        save(args.output/'failure.json', {'status':'failed', 'error_type':type(error).__name__,
             'note':'no completed evaluation; infrastructure failures are not labelled as program outcomes'})
        raise
    print(json.dumps({'output':str(args.output), 'label_scope':'hidden-only',
                      'sources_by_split':{s:v['sources'] for s,v in summary['by_split'].items()}},sort_keys=True))


if __name__ == '__main__':
    main()
