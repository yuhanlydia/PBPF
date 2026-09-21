"""Postprocess sealed fresh evaluations; never generate or execute candidates."""
from __future__ import annotations


def _index_reports(reports: dict[int, dict]):
    """Validate the same source/task population and evaluation protocol."""
    if set(reports) != {0, 1, 2, 3} or any(type(k) is not int for k in reports):
        raise ValueError('require contiguous rounds 0..3')
    indexed = {}
    identity = None
    task_ids = None
    for round_index in range(4):
        report = reports[round_index]
        if report.get('schema') != 'eesd-fresh-policy-eval-v1':
            raise ValueError('fresh evaluation schema required')
        current_identity = {k: report.get(k) for k in
                            ('domain', 'model', 'revision', 'task_manifest_sha256')}
        if any(not isinstance(v, str) or not v for v in current_identity.values()):
            raise ValueError('missing evaluation identity')
        if identity is None:
            identity = current_identity
        if identity != current_identity:
            raise ValueError('evaluation identity differs across rounds')
        if report.get('selection') != 'none; exactly one fresh candidate per primary source':
            raise ValueError('require exactly one fresh candidate per primary source')
        rows = report.get('records')
        if not isinstance(rows, list) or not rows or report.get('sources') != len(rows):
            raise ValueError('invalid source population count')
        by_source = {}
        candidate_ids = set()
        for row in rows:
            for key in ('source_component_id', 'task_id', 'candidate_id'):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f'missing record identity: {key}')
            source = row['source_component_id']
            if source in by_source or row['candidate_id'] in candidate_ids:
                raise ValueError('duplicate source/candidate population')
            if type(row.get('all_tests_pass')) is not bool:
                raise ValueError('all_tests_pass must be a boolean')
            candidate_ids.add(row['candidate_id'])
            by_source[source] = row
        current_tasks = {s: row['task_id'] for s, row in by_source.items()}
        if task_ids is None:
            task_ids = current_tasks
        if set(current_tasks) != set(task_ids):
            raise ValueError('source population differs across rounds')
        if current_tasks != task_ids:
            raise ValueError('representative task differs across rounds')
        indexed[round_index] = by_source
    return indexed, identity, task_ids


def summarize_recursive_trajectories(reports: dict[int, dict]) -> dict:
    """Summarize a genuine rounds 0..3 policy chain.

    Recovery follows an earlier regression. Continuous retention uses round-0
    correct sources as denominator. Do not use for shared-teacher equal controls.
    """
    indexed, identity, task_ids = _index_reports(reports)

    trajectories = []
    for source in sorted(task_ids):
        values = [indexed[i][source]['all_tests_pass'] for i in range(4)]
        regressions, recoveries, repeats = [], [], []
        for i in range(1, 4):
            if values[i - 1] and not values[i]:
                if recoveries:
                    repeats.append(i)
                regressions.append(i)
            elif not values[i - 1] and values[i] and regressions:
                recoveries.append(i)
        trajectories.append({
            'source_component_id': source, 'task_id': task_ids[source],
            'candidate_ids': [indexed[i][source]['candidate_id'] for i in range(4)],
            'correctness': values, 'regression_rounds': regressions,
            'recovery_rounds': recoveries, 'regression_after_recovery_rounds': repeats,
            'ever_regressed': bool(regressions), 'regression_events': len(regressions),
            'regressions_after_recovery': len(repeats),
            'always_retained_base_correct': all(values),
        })
    n = len(trajectories)
    baseline_correct = sum(t['correctness'][0] for t in trajectories)
    by_round = []
    for i in range(4):
        retained = sum(all(t['correctness'][:i + 1]) for t in trajectories)
        by_round.append({
            'round': i, 'pass_at_1': sum(t['correctness'][i] for t in trajectories) / n,
            'regression_events_this_round': sum(i in t['regression_rounds'] for t in trajectories),
            'ever_regressed_sources': sum(any(j <= i for j in t['regression_rounds']) for t in trajectories),
            'cumulative_regression_events': sum(sum(j <= i for j in t['regression_rounds']) for t in trajectories),
            'regressions_after_recovery': sum(sum(j <= i for j in t['regression_after_recovery_rounds']) for t in trajectories),
            'always_retained_base_correct_count': retained,
            'always_retained_base_correct_fraction': retained / baseline_correct if baseline_correct else None,
        })
    return {
        'schema': 'eesd-recursive-trajectories-v1', **identity,
        'rounds': [0, 1, 2, 3], 'sources': n, 'base_correct_sources': baseline_correct,
        'definitions': {
            'ever_regressed_sources': 'unique sources with any PASS->FAIL up to this round',
            'cumulative_regression_events': 'all adjacent-round PASS->FAIL events, counting repeat sources',
            'regressions_after_recovery': 'PASS->FAIL after recovering from an earlier regression',
            'always_retained_base_correct_fraction': 'correct at every round so far / round-0 correct; null if denominator zero',
        },
        'by_round': by_round, 'trajectories': trajectories,
    }


def summarize_matched_controls(reports: dict[int, dict], teacher_reports: dict[int, dict]) -> dict:
    """Compare update r with EESD teacher r-1, never with control r-1."""
    arm, identity, task_ids = _index_reports(reports)
    teacher, teacher_identity, teacher_tasks = _index_reports(teacher_reports)
    if identity != teacher_identity or task_ids != teacher_tasks or reports[0] != teacher_reports[0]:
        raise ValueError('teacher identity/population/shared base differs')
    keys = sorted(task_ids)
    n = len(keys)
    base_correct = sum(arm[0][s]['all_tests_pass'] for s in keys)
    by_round = [{'round': 0, 'pass_at_1': base_correct / n, 'teacher_round': None,
                 'cumulative_teacher_relative_regression_events': 0}]
    total_regressions = 0
    events = {s: [] for s in keys}
    for r in (1, 2, 3):
        before = {s: teacher[r - 1][s]['all_tests_pass'] for s in keys}
        after = {s: arm[r][s]['all_tests_pass'] for s in keys}
        fixes = sum(not before[s] and after[s] for s in keys)
        regressions = sum(before[s] and not after[s] for s in keys)
        preserved = sum(before[s] and after[s] for s in keys)
        teacher_correct = sum(before.values())
        teacher_wrong = n - teacher_correct
        base_retained = sum(arm[0][s]['all_tests_pass'] and after[s] for s in keys)
        total_regressions += regressions
        by_round.append({
            'round': r, 'teacher_round': r - 1, 'teacher_rule': 'base' if r == 1 else 'eesd_full',
            'pass_at_1': sum(after.values()) / n, 'teacher_pass_at_1': teacher_correct / n,
            'fixes': fixes, 'regressions': regressions, 'preserved': preserved,
            'unresolved': n - fixes - regressions - preserved,
            'fix_rate': fixes / n, 'regression_rate': regressions / n,
            'net_gain': (fixes - regressions) / n,
            'teacher_correct_sources': teacher_correct,
            'regression_rate_on_teacher_correct': regressions / teacher_correct if teacher_correct else None,
            'fix_rate_on_teacher_wrong': fixes / teacher_wrong if teacher_wrong else None,
            'retained_teacher_correct_fraction': preserved / teacher_correct if teacher_correct else None,
            'base_correct_retained_fraction': base_retained / base_correct if base_correct else None,
            'cumulative_teacher_relative_regression_events': total_regressions,
        })
        for s in keys:
            events[s].append({'round': r, 'teacher_correct': before[s], 'control_correct': after[s],
                             'regression': before[s] and not after[s],
                             'teacher_candidate_id': teacher[r - 1][s]['candidate_id'],
                             'control_candidate_id': arm[r][s]['candidate_id']})
    return {
        'schema': 'eesd-recursive-matched-controls-v1', **identity,
        'sources': n, 'rounds': [0, 1, 2, 3], 'base_correct_sources': base_correct,
        'independent_closed_loop': False,
        'definitions': {
            'parent': 'round r compares to base at r=1, otherwise EESD at r-1',
            'retained_teacher_correct_fraction': 'teacher-correct remaining correct / teacher-correct; null if zero',
            'base_correct_retained_fraction': 'base-correct currently correct / base-correct; not uninterrupted retention',
            'cumulative_teacher_relative_regression_events': 'sum of per-update teacher-relative regressions; sources may recur, not an independent control policy chain',
        },
        'by_round': by_round,
        'source_events': [{'source_component_id': s, 'task_id': task_ids[s], 'updates': events[s]} for s in keys],
    }
