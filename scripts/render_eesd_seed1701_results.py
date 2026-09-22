#!/usr/bin/env python3
"""Render two final seed-1701 tables from sealed paired primary evaluations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pbpf.eesd.distillation import TRAIN_RULES
from pbpf.eesd.training_outcomes import NON_ESTIMABLE


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/eesd-direct-20260921'
BASE = RUN / 'fresh-baselines'
DOMAINS = ('runbugrun', 'codearc', 'apps_replay', 'codecontests_replay')
FAMILIES = ('qwen25_7b', 'deepseek_6p7b', 'gemma3_4b')
LABELS = {'runbugrun': 'RunBugRun', 'codearc': 'CodeARC',
          'apps_replay': 'APPS Replay', 'codecontests_replay': 'CodeContests Replay',
          'qwen25_7b': 'Qwen2.5 Coder 7B', 'deepseek_6p7b': 'DeepSeek Coder 6.7B',
          'gemma3_4b': 'Gemma 3 4B'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def domain(value):
    return 'runbugrun' if value == 'rbr' else value


def load_report(path, *, expected_domain, expected_model=None):
    report = json.loads(path.read_text())
    records = report.get('records', [])
    if (report.get('schema') != 'eesd-fresh-policy-eval-v1'
            or domain(report.get('domain')) != expected_domain
            or report.get('execution_profile') != 'direct-no-sandbox'
            or report.get('sources') != 500 or len(records) != 500
            or report.get('all_tests_passes') != sum(
                row.get('all_tests_pass') is True for row in records)
            or any(type(row.get('all_tests_pass')) is not bool for row in records)
            or len({row['source_component_id'] for row in records}) != 500
            or (expected_model is not None and report.get('model') != expected_model)):
        raise ValueError('invalid paired primary evaluation: ' + str(path))
    return report, {row['source_component_id']: row['all_tests_pass'] for row in records}


def paired(base, method):
    if set(base) != set(method):
        raise ValueError('paired primary source populations differ')
    keys = sorted(base)
    b = np.asarray([base[key] for key in keys], dtype=np.int8)
    m = np.asarray([method[key] for key in keys], dtype=np.int8)
    difference = m - b
    rng = np.random.default_rng(314159)
    draws = np.empty(10000, dtype=float)
    for index in range(len(draws)):
        draws[index] = rng.choice(difference, size=len(difference), replace=True).mean()
    lo, hi = np.quantile(draws, [.025, .975])
    return {'sources': 500, 'baseline_passes': int(b.sum()),
            'method_passes': int(m.sum()), 'absolute_gain': float(difference.mean()),
            'gain_ci95': [float(lo), float(hi)],
            'fixes': int(((b == 0) & (m == 1)).sum()),
            'regressions': int(((b == 1) & (m == 0)).sum())}


def percent(passes):
    return f'{passes / 5:.1f}%'


def cell_value(value):
    if value['status'] == 'non_estimable':
        return '不可估计'
    return f"{value['comparison']['method_passes']}/500 ({percent(value['comparison']['method_passes'])})"


def inspect(*, preflight_only=False):
    manifest_path = BASE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'eesd-seed1701-baseline-matrix-v1'
            or manifest.get('cell_count') != 12
            or {(cell['domain'], cell['family']) for cell in manifest['cells']}
               != {(d, f) for d in DOMAINS for f in FAMILIES}):
        raise ValueError('twelve-cell baseline matrix missing or mismatched')
    baseline = {}
    for cell in manifest['cells']:
        path = Path(cell['report'])
        if sha(path) != cell['report_sha256']:
            raise ValueError('baseline report checksum differs')
        report, rows = load_report(path, expected_domain=cell['domain'])
        if report['all_tests_passes'] != cell['all_ten_pass']:
            raise ValueError('baseline manifest/pass count differs')
        baseline[(cell['domain'], cell['family'])] = (path, report, rows)

    missing = []
    for d in DOMAINS:
        for f in FAMILIES:
            for rule in TRAIN_RULES:
                if rule == 'no_update':
                    continue
                training = RUN / 'training' / d / f / 'round1' / rule / 'seed1701'
                binding = training / 'training-binding.json'
                if not binding.is_file():
                    missing.append(str(binding))
                    continue
                b = json.loads(binding.read_text())
                if b.get('schema') != 'eesd-training-binding-v1':
                    raise ValueError('invalid training binding: ' + str(binding))
                if b['eligibility']['status'] == NON_ESTIMABLE:
                    if (training / 'adapter').exists():
                        raise ValueError('non-estimable arm has an adapter')
                    continue
                evaluation = RUN / 'fresh-evaluations' / d / f / rule / 'seed1701/report.json'
                if not evaluation.is_file():
                    missing.append(str(evaluation))
    if preflight_only or missing:
        return {'status': 'waiting' if missing else 'ready',
                'baseline_cells': 12, 'update_arms': 84,
                'missing_count': len(missing), 'missing': missing}

    comparisons = {}
    for d in DOMAINS:
        for f in FAMILIES:
            _, base_report, base_rows = baseline[(d, f)]
            for rule in TRAIN_RULES:
                if rule == 'no_update':
                    continue
                training = RUN / 'training' / d / f / 'round1' / rule / 'seed1701'
                binding_path = training / 'training-binding.json'
                binding = json.loads(binding_path.read_text())
                key = f'{d}/{f}/{rule}'
                if binding['eligibility']['status'] == NON_ESTIMABLE:
                    comparisons[key] = {'status': 'non_estimable',
                                        'training_binding_sha256': sha(binding_path)}
                    continue
                path = RUN / 'fresh-evaluations' / d / f / rule / 'seed1701/report.json'
                report, rows = load_report(path, expected_domain=d,
                                           expected_model=base_report['model'])
                for field in ('revision', 'task_manifest_sha256',
                              'execution_profile', 'execution_lock_sha256'):
                    if report.get(field) != base_report.get(field):
                        raise ValueError('paired evaluation differs in ' + field + ': ' + key)
                if report.get('adapter_sha256') != binding.get('adapter_tree_sha256'):
                    raise ValueError('training/evaluation adapter digest differs: ' + key)
                comparisons[key] = {'status': 'evaluated',
                                    'comparison': paired(base_rows, rows),
                                    'training_binding_sha256': sha(binding_path),
                                    'evaluation_report_sha256': sha(path)}

    main_lines = ['| Benchmark | 模型 | Seed | no_update | eesd_full | Δ 百分点 | 95% CI 百分点 |',
                  '|---|---|---:|---:|---:|---:|---:|']
    rules = [rule for rule in TRAIN_RULES if rule != 'no_update']
    ablation_lines = ['| Benchmark | 模型 | Seed | no_update | ' +
                      ' | '.join(rules) + ' |',
                      '|---|---|---:|---:|' + ':---:|' * len(rules)]
    for d in DOMAINS:
        for f in FAMILIES:
            base_report = baseline[(d, f)][1]
            base_passes = base_report['all_tests_passes']
            full = comparisons[f'{d}/{f}/eesd_full']
            if full['status'] == 'non_estimable':
                full_text = '不可估计'
                gain_text = ci_text = '—'
            else:
                result = full['comparison']
                full_text = cell_value(full)
                gain_text = f"{result['absolute_gain'] * 100:+.1f}"
                lo, hi = result['gain_ci95']
                ci_text = f'[{lo * 100:+.1f}, {hi * 100:+.1f}]'
            prefix = f'| {LABELS[d]} | {LABELS[f]} | 1701 | {base_passes}/500 ({percent(base_passes)}) | '
            main_lines.append(prefix + f'{full_text} | {gain_text} | {ci_text} |')
            ablation_lines.append(prefix + ' | '.join(
                cell_value(comparisons[f'{d}/{f}/{rule}']) for rule in rules) + ' |')
    markdown = ('# PBPF seed 1701 主实验结果\n\n'
                '每组使用 500 个相同的 primary 来源；同组比较固定 benchmark、模型、seed、评估任务和 direct 执行锁。'
                '数值是每来源一个新候选的十项测试全通过率；Δ 与 95% CI 来自逐来源配对 bootstrap。\n\n'
                '## 主方法\n\n' + '\n'.join(main_lines) + '\n\n'
                '## 方法消融与对照\n\n' + '\n'.join(ablation_lines) + '\n\n'
                '“不可估计”表示该方法没有正权重训练轨迹，按预先规定不生成 adapter 或方法分数。\n')
    return {'status': 'complete', 'schema': 'eesd-seed1701-final-results-v1',
            'baseline_manifest_sha256': sha(manifest_path),
            'comparisons': comparisons, 'markdown': markdown}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    result = inspect(preflight_only=args.preflight_only)
    if args.preflight_only:
        print(json.dumps({k: v for k, v in result.items() if k != 'missing'}, sort_keys=True))
        return
    if result['status'] != 'complete':
        raise ValueError(f"final results are incomplete: {result['missing_count']} missing artifacts")
    operations = RUN / 'operations'
    markdown = operations / 'seed1701-final-results.md'
    metadata = operations / 'seed1701-final-results.json'
    if markdown.exists() or metadata.exists():
        raise FileExistsError('final results are create-once')
    markdown.write_text(result.pop('markdown'))
    metadata.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'cells': 12, 'method_comparisons': 84,
                      'markdown': str(markdown), 'metadata': str(metadata)}))


if __name__ == '__main__':
    main()
