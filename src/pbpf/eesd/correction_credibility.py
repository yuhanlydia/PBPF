"""Independent hidden-only correction labels and source-weighted public-score strata."""
from __future__ import annotations

import bisect
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from pbpf.registry import OUTCOMES
from .distillation import score_trajectory


def digest(data):
    return hashlib.sha256(data).hexdigest()


def code_hash(code):
    return digest(code.encode())


def snapshot(path, *, lines=False):
    raw = Path(path).read_bytes()
    value = [json.loads(line) for line in raw.splitlines() if line.strip()] if lines else json.loads(raw)
    return value, digest(raw)


def unique(rows, key):
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f'missing or duplicate {key}')
        result[value] = row
    if not result:
        raise ValueError(f'empty {key} population')
    return result


def outcomes(values):
    if not isinstance(values, list) or not values:
        raise ValueError('nonempty outcome vector required')
    result = []
    for value in values:
        if isinstance(value, str) and value in OUTCOMES:
            value = OUTCOMES.index(value)
        if type(value) is not int or not 0 <= value < len(OUTCOMES):
            raise ValueError('invalid outcome')
        result.append(value)
    return result


def identity(row):
    keys = ('trajectory_id', 'source_component_id', 'task_id', 'split')
    if any(not isinstance(row.get(k), str) or not row[k] for k in keys):
        raise ValueError('missing trajectory/source/task/split identity')
    return {k: row[k] for k in keys}


def load_public_bundle(generation_dir, scored_dir, public_dir, config_path, *, domain):
    """Validate existing terminal report/summary/audit seals before private access.

    Correction generation has report.json, not the candidate-bank complete.json.
    Exact hashes bind its existing completion contract without inventing a seal.
    """
    if domain not in ('rbr', 'codearc'):
        raise ValueError('unsupported execution domain')
    generation_dir, scored_dir, public_dir = map(Path, (generation_dir, scored_dir, public_dir))
    generated, gen_sha = snapshot(generation_dir / 'corrections.jsonl', lines=True)
    gen_report, gen_report_sha = snapshot(generation_dir / 'report.json')
    scored, scored_sha = snapshot(scored_dir / 'scored-corrections.jsonl', lines=True)
    summary, summary_sha = snapshot(scored_dir / 'summary.json')
    public, public_sha = snapshot(public_dir / 'public-corrections.jsonl', lines=True)
    audit, audit_sha = snapshot(public_dir / 'audit.json')
    raw_config = Path(config_path).read_bytes()
    cfg = yaml.safe_load(raw_config)
    if (cfg.get('schema') != 'eesd-iclr2027-v1'
            or gen_report.get('schema') != 'eesd-correction-generation-v1'
            or summary.get('schema') != 'eesd-scored-corrections-v1'
            or gen_report.get('domain') != domain
            or gen_report.get('private_fields_available_to_generator') is not False):
        raise ValueError('incorrect public generation/scoring schema or scope')
    key = {'eesd-public-correction-bank-v1': 'public_bank_sha256',
           'eesd-recursive-public-bank-v1': 'public_corrections_sha256'}.get(audit.get('schema'))
    if (key is None or audit.get(key) != public_sha
            or gen_report.get('public_bank_sha256') != public_sha
            or gen_report.get('output_sha256') != gen_sha
            or summary.get('input_sha256') != gen_sha or summary.get('scored_sha256') != scored_sha
            or summary.get('config_sha256') != digest(raw_config)):
        raise ValueError('public bank/generation/scoring checksum chain mismatch')
    gen_by_id, score_by_id = unique(generated, 'trajectory_id'), unique(scored, 'trajectory_id')
    public_by_task = unique(public, 'task_id')
    if (set(gen_by_id) != set(score_by_id)
            or set(public_by_task) != {r.get('task_id') for r in generated}
            or gen_report.get('trajectories') != len(generated)
            or summary.get('trajectories') != len(scored) or audit.get('rows') != len(public)):
        raise ValueError('sealed population missing or inconsistent')
    alpha, penalty = summary['alpha'], summary['uncertainty_penalty']
    utility = cfg['distillation']['transition_utility']
    records = []
    for tid, row in gen_by_id.items():
        identity(row)
        score = score_by_id[tid]
        pub = public_by_task[row['task_id']]
        if (row.get('schema') != 'eesd-correction-trajectory-v1'
                or pub.get('schema') != 'eesd-public-correction-row-v1'
                or row['split'] not in ('train', 'development')
                or any(pub.get(k) != row[k] for k in ('source_component_id', 'split'))
                or pub.get('problem_id') != row.get('problem_id')
                or row['original'] != pub['candidate']
                or pub.get('candidate_code_sha256') != code_hash(row['original'])
                or row.get('correction_code_sha256') != code_hash(row['correction'])):
            raise ValueError('public trajectory/source/code identity mismatch')
        before, after = outcomes(row['before_outcomes']), outcomes(row['after_outcomes'])
        if (len(before) != 4 or len(after) != 4 or all(v == 0 for v in before)
                or before != outcomes(pub['outcomes'])
                or [t['id'] for t in pub['tests']] != ['0', '1', '2', '3']):
            raise ValueError('requires exactly four public tests and visible-failing originals')
        for k, value in row.items():
            expected = before if k == 'before_outcomes' else after if k == 'after_outcomes' else value
            if score.get(k) != expected:
                raise ValueError(f'scored trajectory differs from sealed generation: {k}')
        diagnostics = score_trajectory(before, after, row['relevance'], alpha=alpha,
                                       utility=utility, uncertainty_penalty=penalty)
        weights = diagnostics.pop('weights')
        if score.get('eesd_diagnostics') != diagnostics or score.get('training_weights') != weights:
            raise ValueError('public scoring diagnostics do not reproduce')
        records.append({**identity(row), 'evaluator_task_id': row.get('problem_id') or row['task_id'], 'original': row['original'], 'correction': row['correction'],
            'original_sha256': code_hash(row['original']), 'correction_sha256': code_hash(row['correction']),
            'public_tests': pub['tests'],
            'public_conservative_utility': diagnostics['effective_conservative_utility']['conservative'],
            'public_effective_evidence_mass': sum(diagnostics['effective_posterior']) - 4 * alpha})
    return {'records': records, 'input_binding': {
        'domain': domain, 'generation_sha256': gen_sha, 'generation_report_sha256': gen_report_sha,
        'scored_sha256': scored_sha, 'scoring_summary_sha256': summary_sha,
        'config_sha256': digest(raw_config), 'public_bank_sha256': public_sha, 'public_audit_sha256': audit_sha}}


def evaluate_bundle(bundle, evaluator_root, *, execute, timeout=6.0):
    """Execute both sealed programs on the same private tests; infrastructure errors propagate."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('positive finite timeout required')
    unique(bundle['records'], 'trajectory_id')
    root = Path(evaluator_root)
    manifest, manifest_sha = snapshot(root / 'manifest.json')
    tasks, tasks_sha = snapshot(root / 'tasks.jsonl', lines=True)
    expected = 'apbpf-rbr-generated-materialization-v1' if bundle['input_binding']['domain'] == 'rbr' else 'apbpf-codearc-replay-materialization-v1'
    if manifest.get('schema') != expected or manifest.get('evaluator_tasks_sha256') != tasks_sha:
        raise ValueError('evaluator manifest/schema checksum mismatch')
    tasks = unique(tasks, 'task_id')
    jobs = []
    # Finish ALL alignment checks before starting any candidate execution.
    for row in bundle['records']:
        task = tasks.get(row['evaluator_task_id'])
        if task is None or any(task.get(k) != row[k] for k in ('source_component_id', 'split')):
            raise ValueError('evaluator task/source/split mismatch')
        tests = task.get('tests', [])
        if ([t.get('id') for t in tests] != [str(i) for i in range(10)]
                or any(t.get('hidden') is not (i >= 4) for i, t in enumerate(tests))):
            raise ValueError('evaluator requires ordered four-public/six-hidden tests')
        for shown, actual in zip(row['public_tests'], tests[:4]):
            for key in ('id', 'input', 'expected', 'expected_error'):
                if shown.get(key) != actual.get(key):
                    raise ValueError('public/evaluator test binding mismatch')
        if code_hash(row['original']) != row['original_sha256'] or code_hash(row['correction']) != row['correction_sha256']:
            raise ValueError('trajectory code hash mismatch')
        jobs.append((row, tests[4:]))
    records = []
    for row, tests in jobs:
        before = [execute(row['original'], t, timeout=timeout)['outcome'] for t in tests]
        after = [execute(row['correction'], t, timeout=timeout)['outcome'] for t in tests]
        outcomes(before); outcomes(after)
        records.append({**identity(row), 'evaluator_task_id': row['evaluator_task_id'], 'original_sha256': row['original_sha256'],
            'correction_sha256': row['correction_sha256'], 'hidden_test_ids': [t['id'] for t in tests],
            'hidden_test_set_sha256': digest(json.dumps(tests, sort_keys=True).encode()),
            'before_hidden': before, 'after_hidden': after})
    return {'schema': 'eesd-independent-correction-eval-v1', 'label_scope': 'hidden-only',
        'cohort': 'visible-failing-originals', 'input_binding': dict(bundle['input_binding']),
        'evaluator_manifest_sha256': manifest_sha, 'evaluator_tasks_sha256': tasks_sha,
        'timeout': timeout, 'records': records}


def source_statistics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source_component_id']].append(row)
    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None
    def event(r, kind):
        b, a = r['before_correct'], r['after_correct']
        return {'fix': a and not b, 'regression': b and not a,
                'preserved': b and a, 'unresolved': not b and not a, 'after': a}[kind]
    def rate(kind, condition=None):
        rates = []
        for source_rows in grouped.values():
            eligible = [r for r in source_rows if condition is None or r['before_correct'] is condition]
            if eligible:
                rates.append(mean(event(r, kind) for r in eligible))
        return mean(rates)
    fix_rate, regression_rate = rate('fix'), rate('regression')
    return {'sources': len(grouped), 'trajectories': len(rows),
        'fixes': sum(event(r, 'fix') for r in rows), 'regressions': sum(event(r, 'regression') for r in rows),
        'preserved': sum(event(r, 'preserved') for r in rows),
        'unresolved': sum(event(r, 'unresolved') for r in rows),
        'net_gain_count': sum(event(r, 'fix') - event(r, 'regression') for r in rows),
        'hidden_previously_correct_trajectories': sum(r['before_correct'] for r in rows),
        'hidden_previously_wrong_trajectories': sum(not r['before_correct'] for r in rows),
        'fix_rate': fix_rate, 'regression_rate': regression_rate, 'final_correctness': rate('after'),
        'preserved_rate': rate('preserved'), 'unresolved_rate': rate('unresolved'),
        'net_gain': fix_rate - regression_rate if grouped else None,
        'retained_correct_rate': rate('preserved', True),
        'fix_rate_on_hidden_wrong': rate('fix', False),
        'regression_rate_on_hidden_correct': rate('regression', True)}


def summarize_credibility(bundle, evaluation, *, bins=4):
    if type(bins) is not int or bins < 1:
        raise ValueError('bins must be a positive integer')
    if (evaluation.get('schema') != 'eesd-independent-correction-eval-v1'
            or evaluation.get('label_scope') != 'hidden-only'
            or evaluation.get('cohort') != 'visible-failing-originals'
            or evaluation.get('input_binding') != bundle['input_binding']):
        raise ValueError('independent evaluator binding/scope mismatch')
    scored = unique(bundle['records'], 'trajectory_id')
    evaluated = unique(evaluation.get('records', []), 'trajectory_id')
    if set(scored) != set(evaluated):
        raise ValueError('trajectory population missing or mismatched')
    splits = defaultdict(list)
    for tid, row in scored.items():
        record = evaluated[tid]
        if (identity(record) != identity(row)
                or any(record.get(k) != row[k] for k in ('original_sha256', 'correction_sha256', 'evaluator_task_id'))
                or record.get('hidden_test_ids') != [str(i) for i in range(4, 10)]
                or not isinstance(record.get('hidden_test_set_sha256'), str)
                or len(record['hidden_test_set_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in record['hidden_test_set_sha256'])):
            raise ValueError('independent trajectory/source/code/test identity mismatch')
        before, after = outcomes(record.get('before_hidden')), outcomes(record.get('after_hidden'))
        if len(before) != 6 or len(after) != 6:
            raise ValueError('hidden test coverage missing')
        if any(not math.isfinite(row[key]) for key in ('public_conservative_utility', 'public_effective_evidence_mass')):
            raise ValueError('nonfinite public scoring value')
        splits[row['split']].append({**row, 'before_correct': all(v == 0 for v in before),
                                    'after_correct': all(v == 0 for v in after)})
    by_split = {}
    for split, rows in splits.items():
        groupings = {}
        for key in ('public_conservative_utility', 'public_effective_evidence_mass'):
            cutpoints = np.quantile([r[key] for r in rows], np.arange(1, bins) / bins).tolist()
            partitions = [[] for _ in range(bins)]
            for row in rows:
                partitions[bisect.bisect_left(cutpoints, row[key])].append(row)
            groupings[key] = {'cutpoints': cutpoints, 'bins': [
                {'bin': i, **source_statistics(values)} for i, values in enumerate(partitions)]}
        by_split[split] = {**source_statistics(rows), 'groupings': groupings}
    return {'schema': 'eesd-correction-credibility-v1', 'label_scope': 'hidden-only',
        'cohort': 'visible-failing-originals', 'input_binding': dict(bundle['input_binding']),
        'evaluator_manifest_sha256': evaluation['evaluator_manifest_sha256'],
        'evaluator_tasks_sha256': evaluation['evaluator_tasks_sha256'], 'by_split': by_split,
        'aggregation': 'rates: within-source trajectory means, then equal-weight source means; event counts are trajectories; conditional rates average sources with eligible denominators',
        'stratification': 'public-score empirical trajectory quantiles per split; equal scores stay together; empty bins retained',
        'uncertainty': 'no independence claim or CI; a source may appear in multiple bins',
        'limitation': 'hidden-only transitions among visible-failing originals; not full-suite regression among originally correct programs'}
