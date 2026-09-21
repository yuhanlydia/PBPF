#!/usr/bin/env python3
"""Summarize actual recursive chains or shared-teacher matched controls offline."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

from pbpf.eesd.recursive_trajectories import summarize_recursive_trajectories, summarize_matched_controls


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute_path(value):
    require(isinstance(value, str) and bool(value) and Path(value).is_absolute(),
            'receipt requires an absolute path')
    return Path(value).resolve()


def read_reports(study, rule):
    reports, inputs = {}, []
    for i in range(4):
        arm = 'base' if i == 0 else rule
        path = study / f'round{i}' / arm / 'fresh-eval' / 'report.json'
        raw = path.read_bytes()
        reports[i] = json.loads(raw)
        inputs.append({'round': i, 'path': str(path.resolve()),
                       'sha256': hashlib.sha256(raw).hexdigest()})
    return reports, inputs


def verify_shared_receipts(study, run, run_sha, teacher, teacher_inputs):
    require(type(run.get('seed')) is int and type(run.get('response_token_budget')) is int
            and run['response_token_budget'] > 0, 'shared run seed/budget required')
    inputs = []
    for r in (1, 2, 3):
        path = study / f'round{r}' / 'update-inputs.json'
        raw = path.read_bytes(); receipt = json.loads(raw)
        expected = {'schema': 'eesd-recursive-update-inputs-v1', 'round': r,
                    'experience_policy': 'shared_eesd_teacher', 'seed': run['seed'],
                    'response_token_budget': run['response_token_budget'], 'shared_experience': True,
                    'teacher_rule': 'base' if r == 1 else 'eesd_full', 'teacher_round': r - 1,
                    'recursive_run_sha256': run_sha}
        require(all(receipt.get(key) == value and type(receipt.get(key)) is type(value)
                    for key, value in expected.items()), 'shared update receipt mode/round/seed/budget/run binding mismatch')
        expected_parent = Path(teacher_inputs[r - 1]['path'])
        expected_parent_sha = teacher_inputs[r - 1]['sha256']
        teacher_sha = teacher[r - 1].get('adapter_sha256')
        adapter = receipt.get('teacher_adapter')
        if r == 1:
            require(adapter is None and teacher_sha is None, 'round1 teacher must be base without adapter')
        else:
            absolute_path(adapter)
            require(isinstance(teacher_sha, str) and len(teacher_sha) == 64
                    and all(c in '0123456789abcdef' for c in teacher_sha), 'teacher adapter SHA required')
        require(receipt.get('teacher_adapter_sha256') == teacher_sha, 'teacher adapter/report SHA mismatch')
        arms = receipt.get('arms', {})
        require(set(arms) == {'equal_weight', 'eesd_full'}, 'both matched update arms required')
        scored_paths = []
        for rule, arm in arms.items():
            require(absolute_path(arm.get('parent_evaluation')) == expected_parent
                    and arm.get('parent_evaluation_sha256') == expected_parent_sha,
                    'update parent must be actual EESD teacher evaluation')
            require(arm.get('previous_adapter') == adapter
                    and arm.get('previous_adapter_sha256') == teacher_sha,
                    'update previous adapter must match teacher/report binding')
            scored = absolute_path(arm.get('scored'))
            scored_sha = digest(scored)
            require(arm.get('scored_sha256') == scored_sha
                    and receipt.get('shared_scored_sha256') == scored_sha,
                    'shared scored bank checksum mismatch')
            scored_paths.append(scored)
        require(scored_paths[0] == scored_paths[1], 'both arms must use one shared scored bank path')
        inputs.append({'round': r, 'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                       'teacher_rule': expected['teacher_rule'], 'teacher_round': r - 1,
                       'parent_evaluation': str(expected_parent), 'parent_evaluation_sha256': expected_parent_sha,
                       'shared_scored': str(scored_paths[0]), 'shared_scored_sha256': receipt['shared_scored_sha256']})
    return inputs


def incomplete_coverage(study, run, run_sha, rule):
    """Validate terminal rows and dependencies; never invent absent evaluations."""
    report_path = study / 'recursive-report.json'
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    if 'coverage' not in report:
        return None
    require(report.get('recursive_run_sha256') == run_sha and report.get('experience_policy') == run['experience_policy'],
            'recursive study/run binding mismatch')
    summaries = report.get('summary', [])
    require([r.get('round') for r in summaries] == [1, 2, 3], 'complete planned round inventory required')
    root = Path(__file__).resolve().parents[1]
    training = runpy.run_path(str(root / 'scripts/run_eesd_downstream.py'))
    model_config = absolute_path(report.get('model_config'))
    require(digest(model_config) == run.get('model_config_sha256'), 'recursive model config binding mismatch')
    states = {}; sealed_inputs = []
    shared = run['experience_policy'] == 'shared_eesd_teacher'
    zero = 'non_estimable_zero_positive_training_weight'
    for summary in summaries:
        r = summary['round']; path = study / f'round{r}' / 'outcomes.json'
        require(absolute_path(summary.get('outcomes_path')) == path and digest(path) == summary.get('outcomes_sha256'),
                'recursive round outcome checksum mismatch')
        outcome = json.loads(path.read_text()); update_path = study / f'round{r}' / 'update-inputs.json'
        require(outcome.get('schema') == 'eesd-recursive-round-outcomes-v1' and outcome.get('round') == r
                and outcome.get('recursive_run_sha256') == run_sha
                and outcome.get('update_inputs_sha256') == digest(update_path) == summary.get('update_inputs_sha256'),
                'round outcome/update/run binding mismatch')
        update = json.loads(update_path.read_text())
        require(update.get('round') == r and update.get('experience_policy') == run['experience_policy']
                and update.get('seed') == run['seed'] and update.get('response_token_budget') == run['response_token_budget']
                and update.get('recursive_run_sha256') == run_sha, 'update budget/run binding mismatch')
        arms = outcome.get('arms', {})
        require(set(arms) == {'equal_weight', 'eesd_full'} and arms == summary.get('arms'), 'planned outcome arms differ')
        blocked = {arm: row.get('dependency') for arm, row in arms.items() if row.get('status') == 'blocked_dependency'}
        executable = set(arms) - set(blocked)
        require(update.get('blocked_dependencies') == blocked and set(update.get('arms', {})) == executable,
                'blocked dependencies/executable update plans differ')
        if shared:
            require(update.get('shared_experience') is True and update.get('teacher_round') == r - 1
                    and update.get('teacher_rule') == ('base' if r == 1 else 'eesd_full')
                    and len(executable) in (0, 2), 'shared teacher round/arm contract mismatch')
            if executable:
                first, second = (update['arms'][name] for name in ('equal_weight', 'eesd_full'))
                require(absolute_path(first.get('scored')) == absolute_path(second.get('scored'))
                        and first.get('scored_sha256') == second.get('scored_sha256') == update.get('shared_scored_sha256'),
                        'shared arms require the same scored path and SHA')
                teacher_report = study / ('round0/base/fresh-eval/report.json' if r == 1 else f'round{r-1}/eesd_full/fresh-eval/report.json')
                teacher_sha = json.loads(teacher_report.read_text()).get('adapter_sha256')
                teacher_adapter = None if r == 1 else states[(r - 1, 'eesd_full')].get('adapter')
                require(update.get('teacher_adapter') == teacher_adapter and update.get('teacher_adapter_sha256') == teacher_sha
                        and all(p.get('previous_adapter') == teacher_adapter and p.get('previous_adapter_sha256') == teacher_sha
                                for p in (first, second)), 'shared teacher adapter/report SHA mismatch')
            else:
                require(update.get('shared_scored_sha256') is None and update.get('teacher_adapter') is None
                        and update.get('teacher_adapter_sha256') is None, 'blocked shared round cannot provide a bank/teacher')
        for arm, row in arms.items():
            status = row.get('status'); parent_rule = 'eesd_full' if shared else arm
            previous = states.get((r - 1, parent_rule))
            needs_block = previous is not None and previous['status'] != 'trained'
            require(status in {'trained', zero, 'blocked_dependency'} and (status == 'blocked_dependency') == needs_block,
                    'recursive outcome violates teacher dependency')
            require(row.get('requested_response_tokens') == run['response_token_budget'], 'outcome requested budget mismatch')
            if status == 'blocked_dependency':
                dep = row.get('dependency', {})
                origin = states.get((dep.get('round'), dep.get('rule')))
                require(origin is not None and origin['status'] == zero and dep.get('status') == zero
                        and (dep.get('rule') == 'eesd_full' if shared else dep.get('rule') == arm)
                        and dep.get('receipt') == origin.get('receipt') and dep.get('receipt_sha256') == origin.get('receipt_sha256'),
                        'blocked outcome must name its zero-weight ancestor receipt')
            else:
                receipt = absolute_path(row.get('receipt'))
                require(digest(receipt) == row.get('receipt_sha256'), 'training outcome receipt checksum mismatch')
                plan = update.get('arms', {}).get(arm, {})
                require(digest(absolute_path(plan.get('scored'))) == plan.get('scored_sha256'), 'outcome scored input checksum mismatch')
                if r == 1:
                    parent_eval = study / 'round0/base/fresh-eval/report.json'; parent_adapter = None
                else:
                    parent_eval = absolute_path(previous.get('evaluation')); parent_adapter = previous.get('adapter')
                require(absolute_path(plan.get('parent_evaluation')) == parent_eval
                        and plan.get('parent_evaluation_sha256') == digest(parent_eval)
                        and plan.get('previous_adapter') == parent_adapter, 'outcome update parent binding mismatch')
                directory = study / f'round{r}' / arm / 'training'
                require(receipt == directory / 'training-binding.json', 'terminal receipt must belong to actual round/arm training')
                task = training['make_training_task'](root=root, scored=absolute_path(plan['scored']),
                    model_config=model_config, rule=arm,
                    previous_adapter=absolute_path(parent_adapter) if parent_adapter is not None else None,
                    output=directory, seed=run['seed'], budget=run['response_token_budget'],
                    max_steps=run['max_steps'], anchor_beta=run['anchor_beta'])
                require(plan.get('previous_adapter_sha256') == task['previous_adapter_tree_sha256'],
                        'update parent adapter SHA differs from verified training request')
                require(task.get('status') in {'verified_complete', 'verified_non_estimable'},
                        'terminal training receipt is missing')
                verified = training['check_training_binding'](task)
                require(all(row.get(key) == verified[key] for key in ('status','adapter','receipt','receipt_sha256')),
                        'recursive outcome does not match verified training terminal status')
            if status != 'trained':
                require(row.get('pass_at_1') is None and row.get('evaluation') is None and row.get('adapter') is None
                        and row.get('actual_response_tokens') == 0, 'non-estimable outcome cannot contain fallback metrics/adapter')
            else:
                evaluation = study / f'round{r}' / arm / 'fresh-eval/report.json'
                require(absolute_path(row.get('evaluation')) == evaluation and digest(evaluation) == row.get('evaluation_sha256')
                        and json.loads(evaluation.read_text()).get('fresh_all_tests_pass_at_1') == row.get('pass_at_1')
                        and row.get('actual_response_tokens') == run['response_token_budget'], 'trained evaluation/budget binding mismatch')
            states[(r, arm)] = row
        sealed_inputs.append({'round': r, 'path': str(path), 'sha256': digest(path)})
    coverage = {'complete': all(v['status'] == 'trained' for v in states.values()), 'planned_updates': 6,
        'trained_updates': sum(v['status'] == 'trained' for v in states.values()),
        'non_estimable_updates': sum(v['status'] == zero for v in states.values()),
        'blocked_updates': sum(v['status'] == 'blocked_dependency' for v in states.values())}
    require(coverage == report['coverage'], 'recursive coverage counts mismatch')
    if coverage['complete']:
        return None
    base = study / 'round0/base/fresh-eval/report.json'
    return {'schema': 'eesd-recursive-incomplete-coverage-v1', 'status': 'non_estimable_incomplete_recursive_study',
        'coverage': coverage, 'rule': rule, 'experience_policy': run['experience_policy'],
        'main_effect': None, 'confidence_interval': None, 'p_value': None,
        'reason': 'planned update or teacher checkpoint unavailable; no fallback or complete-chain claim',
        'by_round': [{'round': 0, 'status': 'base', 'pass_at_1': json.loads(base.read_text())['fresh_all_tests_pass_at_1']}] +
                    [{'round': r, **states[(r, rule)]} for r in (1, 2, 3)],
        'planned_arm_outcomes': [{'round': r, 'rule': arm, **row} for (r, arm), row in states.items()],
        'round_outcome_inputs': sealed_inputs, 'study_report_sha256': digest(report_path),
        'recursive_run': {'path': str(study / 'recursive-run.json'), 'sha256': run_sha}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True,
                        help='one dataset/model/seed recursive study directory')
    parser.add_argument('--rule', choices=['equal_weight', 'eesd_full'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study = args.study_root.resolve()
    lock = study / 'recursive-run.json'
    raw = lock.read_bytes(); run = json.loads(raw)
    run_sha = hashlib.sha256(raw).hexdigest()
    require(run.get('schema') == 'eesd-recursive-run-v1' and type(run.get('rounds')) is int
            and run['rounds'] == 3 and run.get('experience_policy') in {'arm-specific', 'shared_eesd_teacher'},
            'explicit three-round recursive run policy lock required')
    partial = incomplete_coverage(study, run, run_sha, args.rule)
    if partial is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(partial, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'output': str(args.output), 'status': partial['status'], 'coverage': partial['coverage']}))
        return
    reports, inputs = read_reports(study, args.rule)
    mode = run['experience_policy']
    if mode == 'shared_eesd_teacher':
        teacher, teacher_inputs = read_reports(study, 'eesd_full')
        receipts = verify_shared_receipts(study, run, run_sha, teacher, teacher_inputs)
        matched = summarize_matched_controls(reports, teacher)
        if args.rule == 'equal_weight':
            result = matched
        else:
            result = summarize_recursive_trajectories(reports)
            result['teacher_relative_updates'] = matched
            result['independent_closed_loop'] = True
        result.update(update_inputs=receipts, teacher_input_reports=teacher_inputs,
            binding_scope='run/update receipts, actual evaluation reports and scored bytes; adapter trees not reverified')
    else:
        result = summarize_recursive_trajectories(reports)
        result.update(independent_closed_loop=True, study_role='supplemental_arm_specific')
    result.update(rule=args.rule, experience_policy=mode, input_reports=inputs,
                  recursive_run={'path': str(lock), 'sha256': run_sha})
    # Do not seal a summary if an input changed while it was being processed.
    for item in inputs + result.get('teacher_input_reports', []) + result.get('update_inputs', []):
        require(digest(item['path']) == item['sha256'], 'summary input changed during processing')
        if 'shared_scored' in item:
            require(digest(item['shared_scored']) == item['shared_scored_sha256'],
                    'shared scored bytes changed during processing')
    require(digest(lock) == run_sha, 'recursive run lock changed during processing')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'sources': result['sources'],
                      'final': result['by_round'][-1]}, sort_keys=True))


if __name__ == '__main__':
    main()
