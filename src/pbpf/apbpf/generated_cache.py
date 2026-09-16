"""Development training caches for generated candidates with hidden answers redacted."""
from collections import Counter
import hashlib

from pbpf.real_gate import RBR_CACHE_SCHEMA, validate_rbr_cache
from .codearc_cache import build_cache as build_codearc_cache


def build_generated_cache(bank_rows, executions, public_tasks, private_tasks, *, domain):
    if domain not in {'rbr', 'codearc'}:
        raise ValueError('unknown generated task domain')
    for group in bank_rows:
        if group['split'] not in {'train', 'development'}:
            raise ValueError('primary candidates cannot enter development fitting')
        public, private = public_tasks[group['task_id']], private_tasks[group['task_id']]
        tests = private['tests']
        if ([t['id'] for t in tests] != [str(i) for i in range(10)]
                or any(t['hidden'] is not (i >= 4) for i, t in enumerate(tests))):
            raise ValueError('requires four public and six hidden tests in protocol order')
        for shown, actual in zip(public['visible_tests'], tests[:4], strict=True):
            if any(shown[k] != actual[k] for k in ('id', 'input', 'expected')):
                raise ValueError('public examples differ from evaluator test evidence')
    if domain == 'codearc':
        result = build_codearc_cache(bank_rows, executions, public_tasks, private_tasks)
    else:
        evidence = {(r['task_id'], r['candidate_id']): r for r in executions}
        if len(evidence) != len(executions):
            raise ValueError('duplicate candidate execution record')
        records, used = [], set()
        for group in bank_rows:
            public, private = public_tasks[group['task_id']], private_tasks[group['task_id']]
            if any(t['split'] != group['split'] or t['source_component_id'] != group['source_component_id']
                   for t in (public, private)):
                raise ValueError('candidate/public/evaluator source mismatch')
            for candidate in group['candidates']:
                key = group['task_id'], candidate['candidate_id']
                if key not in evidence or key in used:
                    raise ValueError('candidate execution inventory mismatch')
                used.add(key); measured = evidence[key]['tests']
                if [t['test_id'] for t in measured] != [str(i) for i in range(10)]:
                    raise ValueError('complete execution test order is required')
                cases = []
                for original, execution in zip(private['tests'], measured, strict=True):
                    cases.append({'id': original['id'], 'input': original['input'],
                        'expected': original['expected'], 'actual': execution['stdout'],
                        'stderr': execution['stderr'], 'returncode': execution['returncode'],
                        'timed_out': execution['timed_out'], 'outcome': execution['outcome']})
                records.append({'task_id': candidate['candidate_id'], 'problem_id': group['task_id'],
                    'source_component_id': group['source_component_id'], 'split': group['split'],
                    'task_text': public['task_text'], 'candidate': candidate['code'] or '\n',
                    'empty_candidate': not candidate['code'],
                    'candidate_code_sha256': hashlib.sha256(candidate['code'].encode()).hexdigest(),
                    'tests': cases, 'outcomes': [t['outcome'] for t in cases]})
        if used != set(evidence):
            raise ValueError('unused execution records; candidate filtering forbidden')
        result = {'schema': RBR_CACHE_SCHEMA, 'dataset': 'runbugrun', 'seed': 1701,
                  'tests_per_candidate': 10, 'records': records,
                  'counts': dict(Counter(r['split'] for r in records)),
                  'population_policy': 'all generated train/development candidates; no fixed-program screen',
                  'execution_fields_visibility': 'evaluator-only except whitelisted public observations'}
    for row in result['records']:
        for i, test in enumerate(row['tests']):
            if i >= 4:
                # Loss targets are outcomes. Future expected text is unnecessary for fitting.
                test['expected'] = ''
                test['expected_redacted'] = True
    result['execution_protocol'] = {
        'stdin_policy': 'official-terminal-newline-if-missing' if domain == 'rbr' else 'function-call-replay',
        'expected_feature_visibility': 'observed-four-only' if domain == 'rbr' else 'query-input-only',
        'future_expected_values_redacted': True,
    }
    return validate_rbr_cache(result)
