"""Build source-bound full-stage caches after primary population locking.

Unlike development-cache builders, this evaluator-side module accepts primary
records explicitly and puts them only in the legacy fitter's test partition.
"""
from collections import Counter
import hashlib

from pbpf.real_gate import CODEARC_CACHE_SCHEMA, RBR_CACHE_SCHEMA, validate_rbr_cache
from .codearc_prompt import bounded_task_text


def join_primary_executions(visible, hidden):
    """Join complementary phases without accepting filtered or reordered tests."""
    def indexed(rows, ids):
        result = {}
        for row in rows:
            key = row['task_id'], row['candidate_id']
            if key in result or [t['test_id'] for t in row['tests']] != ids:
                raise ValueError('duplicate candidate or invalid primary phase test inventory')
            result[key] = row
        return result

    left = indexed(visible, [str(i) for i in range(4)])
    right = indexed(hidden, [str(i) for i in range(4, 10)])
    if not left or left.keys() != right.keys():
        raise ValueError('primary visible/hidden candidate inventories differ')
    return [{'task_id': key[0], 'candidate_id': key[1],
             'tests': row['tests'] + right[key]['tests']} for key, row in left.items()]


def build_stage_cache(bank_rows, executions, public_tasks, private_tasks, *, domain):
    """Preserve all candidates and all three source splits; redact future answers.

    Caller must verify execution manifests and primary locks before private reads.
    This pure assembly function is not evidence that those locks were verified.
    """
    if domain not in {'rbr', 'codearc'}:
        raise ValueError('unknown stage-cache domain')
    evidence = {(r['task_id'], r['candidate_id']): r for r in executions}
    if len(evidence) != len(executions):
        raise ValueError('duplicate execution record')
    rows, used, sources, task_ids = [], set(), set(), set()
    for group in bank_rows:
        split = group['split']
        if split not in {'train', 'development', 'primary'}:
            raise ValueError('invalid original source split')
        task_id, source = group['task_id'], group['source_component_id']
        if source in sources or task_id in task_ids:
            raise ValueError('source/task repeated across generated banks')
        sources.add(source); task_ids.add(task_id)
        public, private = public_tasks[task_id], private_tasks[task_id]
        if any(t['split'] != split or t['source_component_id'] != source for t in (public, private)):
            raise ValueError('candidate/public/evaluator source binding mismatch')
        tests = private['tests']
        if ([t['id'] for t in tests] != [str(i) for i in range(10)]
                or any(t['hidden'] is not (i >= 4) for i, t in enumerate(tests))):
            raise ValueError('four public and six future tests required in protocol order')
        for shown, actual in zip(public['visible_tests'], tests[:4], strict=True):
            if any(shown[k] != actual[k] for k in ('id', 'input', 'expected')):
                raise ValueError('public examples differ from private evidence')
        if len(group['candidates']) != 8:
            raise ValueError('all eight declared candidates required')
        for candidate in group['candidates']:
            key = task_id, candidate['candidate_id']
            if key not in evidence or key in used:
                raise ValueError('candidate execution inventory mismatch')
            used.add(key)
            measured = evidence[key]['tests']
            if [t['test_id'] for t in measured] != [str(i) for i in range(10)]:
                raise ValueError('complete ten-test execution required')
            cases = []
            for i, (original, actual) in enumerate(zip(tests, measured, strict=True)):
                # No answer-derived hashes, truncation flags or exception type
                # are needed by predictors. Keep them in original evaluator logs.
                case = {'id': original['id'], 'input': original['input'],
                        'expected': original['expected'][:4096] if i < 4 else '',
                        'expected_redacted': i >= 4,
                        **{k: actual[k] for k in ('stdout', 'stderr', 'returncode', 'timed_out', 'outcome')}}
                case['actual'] = case.pop('stdout')[:4096]
                case['stderr'] = case['stderr'][:4096]
                cases.append(case)
            rows.append({'task_id': candidate['candidate_id'], 'problem_id': task_id,
                         'source_component_id': source, 'original_split': split,
                         'split': 'test' if split == 'primary' else split,
                         'task_text': bounded_task_text(public) if domain == 'codearc' and group.get('public_prompt_bounded') else public['task_text'],
                         'candidate': candidate['code'] or '\n', 'empty_candidate': not candidate['code'],
                         'candidate_code_sha256': hashlib.sha256(candidate['code'].encode()).hexdigest(),
                         'tests': cases, 'outcomes': [t['outcome'] for t in cases]})
    if used != set(evidence):
        raise ValueError('unused execution records; population filtering forbidden')
    if {r['split'] for r in rows} != {'train', 'development', 'test'}:
        raise ValueError('full stage cache requires all three disjoint source splits')
    payload = {'schema': RBR_CACHE_SCHEMA if domain == 'rbr' else CODEARC_CACHE_SCHEMA,
               'dataset': 'runbugrun' if domain == 'rbr' else 'codearc_replay',
               'seed': 1701, 'tests_per_candidate': 10, 'records': rows,
               'counts': dict(Counter(r['split'] for r in rows)),
               'problem_counts': {s: len({r['source_component_id'] for r in rows if r['split'] == s})
                                  for s in ('train', 'development', 'test')},
               'evaluation_role': 'exploratory_locked_primary_assessment',
               'population_policy': 'all generated candidates; primary mapped only to test; no correctness filtering',
               'execution_fields_visibility': 'evaluator-only except whitelisted public observations',
               'execution_protocol': {'stdin_policy': 'official-terminal-newline-if-missing' if domain == 'rbr' else 'function-call-replay',
                                      'expected_feature_visibility': 'observed-four-only' if domain == 'rbr' else 'query-input-only',
                                      'future_expected_values_redacted': True}}
    return validate_rbr_cache(payload)
