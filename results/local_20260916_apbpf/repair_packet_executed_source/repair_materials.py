"""Separate supervised repair targets from private execution test material."""
import hashlib
import json

from pbpf.real_gate import validate_rbr_cache


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_repair_materials(cache, public_tasks, private_tasks, *, domain):
    """Bind to the full cache; never export a primary reference implementation.

    Training targets include original training and development sources only.
    Development targets are for projector checkpoint selection, never gradients.
    Evaluator tests are a separate artifact, forbidden in the actor mount.
    """
    validate_rbr_cache(cache)
    expected_dataset = {'rbr': 'runbugrun', 'codearc': 'codearc_replay'}.get(domain)
    if expected_dataset is None or cache['dataset'] != expected_dataset:
        raise ValueError('repair domain and cache dataset differ')
    if cache.get('evaluation_role') != 'exploratory_locked_primary_assessment':
        raise ValueError('repair materials require original full source splits')
    groups = {}
    for row in cache['records']:
        groups.setdefault(row['problem_id'], []).append(row)
    targets, evaluator, contexts = [], [], []
    for task_id, rows in groups.items():
        public, private = public_tasks[task_id], private_tasks[task_id]
        split, source = rows[0]['original_split'], rows[0]['source_component_id']
        if len(rows) != 8 or len({r['task_id'] for r in rows}) != 8:
            raise ValueError('repair requires the original eight candidates per source')
        if any(r['original_split'] != split or r['source_component_id'] != source
               or r['split'] != ('test' if split == 'primary' else split) for r in rows):
            raise ValueError('repair source splits differ from cache membership')
        if any(t['task_id'] != task_id or t['source_component_id'] != source or t['split'] != split
               for t in (public, private)):
            raise ValueError('repair task/source/split identity mismatch')
        tests = private['tests']
        if ([t['id'] for t in tests] != [str(i) for i in range(10)]
                or any(t['hidden'] is not (i >= 4) for i, t in enumerate(tests))):
            raise ValueError('repair requires ordered four-public/six-hidden tests')
        for shown, actual in zip(public['visible_tests'], tests[:4], strict=True):
            keys = ('id', 'input', 'expected', 'expected_error') if domain == 'codearc' else ('id', 'input', 'expected')
            if any(shown[k] != actual[k] for k in keys):
                raise ValueError('repair public examples differ from evaluator tests')
        for row in rows:
            for i, (cached, test) in enumerate(zip(row['tests'], tests, strict=True)):
                if cached['id'] != test['id'] or cached['input'] != test['input']:
                    raise ValueError('repair invocation inventory differs from cache')
                if cached['expected'] != (test['expected'][:4096] if i < 4 else ''):
                    raise ValueError('repair cache answer redaction or visible answer mismatch')
        identity = {'task_id': task_id, 'source_component_id': source, 'split': split}
        context_keys = ('id', 'input', 'expected', 'expected_error') if domain == 'codearc' else ('id', 'input', 'expected')
        contexts.append({**identity, 'visible_tests': [{k: t[k] for k in context_keys}
                                                     for t in public['visible_tests']]})
        if split in {'train', 'development'}:
            code = private['reference_code']
            if not isinstance(code, str) or not code.strip():
                raise ValueError('nonempty training/development reference code required')
            targets.append({**identity, 'reference_code': code,
                            'reference_code_sha256': hashlib.sha256(code.encode()).hexdigest()})
        elif split != 'primary':
            raise ValueError('unknown repair source split')
        test_keys = ('id', 'input', 'expected', 'hidden', 'expected_error') if domain == 'codearc' else ('id', 'input', 'expected', 'hidden')
        evaluator.append({**identity, 'tests': [{k: t[k] for k in test_keys} for t in tests]})
    common = {'domain': domain, 'source_cache_payload_sha256': digest(cache)}
    training = {**common, 'schema': 'apbpf-repair-targets-v1', 'records': targets,
                'usage': 'train: gradients; development: checkpoint selection; primary references excluded'}
    private = {**common, 'schema': 'apbpf-repair-evaluator-tests-v1', 'records': evaluator,
               'usage': 'evaluator only; never mount in actor sandbox; freshly execute every repair'}
    public = {**common, 'schema': 'apbpf-repair-public-context-v1', 'records': contexts,
              'usage': 'four public examples and their declared expected-error semantics only'}
    return training, private, public


def actor_rows(cache, targets, context):
    """Whitelist exactly four public observations; attach nonprimary targets only."""
    if (targets['schema'] != 'apbpf-repair-targets-v1'
            or targets['source_cache_payload_sha256'] != digest(cache)):
        raise ValueError('repair targets are bound to another cache')
    if (context['schema'] != 'apbpf-repair-public-context-v1'
            or context['source_cache_payload_sha256'] != digest(cache)):
        raise ValueError('repair context is bound to another cache')
    contexts = {r['task_id']: r for r in context['records']}
    if len(contexts) != len(context['records']) or set(contexts) != {r['problem_id'] for r in cache['records']}:
        raise ValueError('repair public context inventory mismatch')
    refs = {r['task_id']: r for r in targets['records']}
    if len(refs) != len(targets['records']) or any(r['split'] not in {'train', 'development'} for r in refs.values()):
        raise ValueError('duplicate or primary repair reference')
    expected = {r['problem_id'] for r in cache['records'] if r['original_split'] != 'primary'}
    if set(refs) != expected:
        raise ValueError('repair training target inventory mismatch')
    output = []
    for row in cache['records']:
        split = row['original_split']
        if split != 'primary':
            ref = refs[row['problem_id']]
            if (ref['split'] != split or ref['source_component_id'] != row['source_component_id']
                    or hashlib.sha256(ref['reference_code'].encode()).hexdigest() != ref['reference_code_sha256']):
                raise ValueError('repair target source identity or content mismatch')
        value = {k: row[k] for k in ('task_id', 'problem_id', 'source_component_id', 'task_text', 'candidate')}
        value.update(split=split, tests=[{k: t[k] for k in ('id', 'input', 'expected', 'actual', 'stderr', 'outcome')}
                                       for t in row['tests'][:4]], outcomes=row['outcomes'][:4])
        public = contexts[row['problem_id']]
        if (public['split'] != split or public['source_component_id'] != row['source_component_id']
                or len(public['visible_tests']) != 4):
            raise ValueError('repair public context source or count mismatch')
        for shown, test in zip(public['visible_tests'], value['tests'], strict=True):
            if any(shown[k] != test[k] for k in ('id', 'input')) or shown['expected'][:4096] != test['expected']:
                raise ValueError('repair public context does not match observed tests')
            if targets['domain'] == 'codearc':
                if type(shown['expected_error']) is not bool:
                    raise ValueError('CodeARC public expected-error flag must be boolean')
                test['expected_error'] = shown['expected_error']
        if split != 'primary':
            value['reference_code'] = refs[row['problem_id']]['reference_code']
        output.append(value)
    return output
