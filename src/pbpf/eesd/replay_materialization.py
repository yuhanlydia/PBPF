"""Deterministic population selection for the separate Replay extension.

These pure functions check member/token identity and counts. Receipt, prompt,
graph and raw-data checksum verification belongs to the admission CLI; calling
the selector alone does not certify a scientific population.
"""
from __future__ import annotations

import hashlib


DOMAINS = ('apps_replay', 'codecontests_replay')
MODEL_KEYS = frozenset(('qwen25_7b', 'deepseek_6p7b', 'seed_coder_8b', 'starcoder2_15b'))
MAX_INPUT_TOKENS = 4096


def _rank(prefix: str, *parts: str) -> str:
    return hashlib.sha256((prefix + '|'.join(parts)).encode('utf-8')).hexdigest()


def _identity(row: dict) -> tuple[str, str, str]:
    if not isinstance(row, dict):
        raise ValueError('member/token row must be an object')
    values = tuple(row.get(key) for key in ('domain', 'task_id', 'source_id'))
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError('domain, task_id and source_id must be nonempty strings')
    if values[0] not in DOMAINS:
        raise ValueError(f'unsupported Replay domain: {values[0]}')
    return values


def choose_representatives(public_rows, token_rows, quarantined_sources):
    """Select CC-first, all-model-fitting hash representatives of a fixed graph.

    The caller supplies source IDs from the full graph, including connections
    through ineligible members. This function never rebuilds or cuts that graph.
    """
    members, tokens = {}, {}
    for row in public_rows:
        identity = _identity(row)
        if identity[1] in members:
            raise ValueError('duplicate public task identity')
        members[identity[1]] = row
    for row in token_rows:
        identity = _identity(row)
        if identity[1] in tokens:
            raise ValueError('duplicate token task identity')
        tokens[identity[1]] = row
    if set(members) != set(tokens):
        raise ValueError('public/token task population mismatch')

    if not isinstance(quarantined_sources, (set, frozenset, list, tuple)):
        raise ValueError('quarantined sources must be a collection of IDs, not a scalar')
    quarantined = set(quarantined_sources)
    if any(not isinstance(source, str) or not source for source in quarantined):
        raise ValueError('quarantined source IDs must be nonempty strings')
    fitting = {}
    for task_id, member in members.items():
        token_row = tokens[task_id]
        if _identity(member) != _identity(token_row):
            raise ValueError('public/token source or domain mismatch')
        models = token_row.get('models')
        if not isinstance(models, dict) or set(models) != MODEL_KEYS:
            raise ValueError('token audit must contain exactly the four locked models')
        lengths = []
        for values in models.values():
            count = values.get('input_tokens') if isinstance(values, dict) else None
            if type(count) is not int or count < 1:
                raise ValueError('input token count must be a positive integer')
            lengths.append(count)
        if member['source_id'] in quarantined or max(lengths) > MAX_INPUT_TOKENS:
            continue
        fitting.setdefault(member['source_id'], []).append(member)

    selected = []
    for source in sorted(fitting):
        candidates = fitting[source]
        preferred = ('codecontests_replay'
                     if any(row['domain'] == 'codecontests_replay' for row in candidates)
                     else 'apps_replay')
        selected.append(min(
            (row for row in candidates if row['domain'] == preferred),
            key=lambda row: (_rank('eesd-replay-representative-v1|', row['task_id']), row['task_id']),
        ))
    return selected


def select_population(public_rows, token_rows, quarantined_sources, *, development=200, primary=500):
    """Return both domains' exact development/primary populations or raise.

    Reduced positive counts support unit fixtures only; the production admission
    entry point uses the fixed defaults and exposes no size override.
    """
    if any(type(value) is not int or value < 1 for value in (development, primary)):
        raise ValueError('development and primary counts must be positive integers')
    representatives = choose_representatives(public_rows, token_rows, quarantined_sources)
    result = {}
    for domain in DOMAINS:
        rows = sorted(
            (row for row in representatives if row['domain'] == domain),
            key=lambda row: (
                _rank('eesd-replay-split-v1|1701|', domain, row['source_id']), row['source_id']),
        )
        if len(rows) < development + primary:
            raise ValueError(f'{domain} has {len(rows)} fitting sources; requires {development + primary}')
        result[domain] = [
            ('development' if index < development else 'primary', row)
            for index, row in enumerate(rows[:development + primary])
        ]
    return result


def _normalized_io(value: str) -> str:
    """Match inventory fingerprints; never use this value as execution input."""
    return '\n'.join(line.rstrip() for line in value.replace('\r\n', '\n').strip().splitlines())


def _valid_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def project_views(selected, private_rows):
    """Validate selected member bindings and return (public, evaluator) lists.

    Raw statement and IO strings are preserved. Fingerprints check normalized
    identity only. The caller must separately verify original raw bytes, full
    graph/provenance receipts, prompt/token bindings and production split counts;
    this projection cannot certify statement-parser completeness or judge validity.
    Unselected evaluator members may be present, but identities must be unique.
    """
    from copy import deepcopy

    if not isinstance(selected, dict) or set(selected) != set(DOMAINS):
        raise ValueError('selected population requires both Replay domains')
    indexed = {}
    for row in private_rows:
        identity = _identity(row)
        if identity[1] in indexed:
            raise ValueError('duplicate evaluator task identity')
        indexed[identity[1]] = row
    public, evaluator = [], []
    seen_tasks, seen_sources = set(), set()
    for domain in DOMAINS:
        values = selected[domain]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError('selected domain must contain a nonempty population')
        for entry in values:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError('selected entries must be (split, member) pairs')
            split, member = entry
            identity = _identity(member)
            if identity[0] != domain or split not in ('development', 'primary'):
                raise ValueError('selected domain/split mismatch')
            _, task_id, source_id = identity
            if task_id in seen_tasks or source_id in seen_sources:
                raise ValueError('selected task/source cannot be reused within or across domains')
            seen_tasks.add(task_id)
            seen_sources.add(source_id)
            private = indexed.get(task_id)
            if private is None or _identity(private) != identity:
                raise ValueError('selected public/evaluator identity mismatch or missing member')
            statement = member.get('statement')
            if not isinstance(statement, str) or not statement.strip() or len(statement) > 6000:
                raise ValueError('complete statement must be nonempty and at most 6000 raw characters')
            visible, tests = member.get('visible_tests'), private.get('ordered_tests')
            if not isinstance(visible, list) or len(visible) != 4:
                raise ValueError('exactly four public tests required')
            if not isinstance(tests, list) or len(tests) != 10:
                raise ValueError('exactly ten ordered evaluator tests required')
            known = private.get('known_exposed_input_sha256')
            if (not isinstance(known, list) or any(not _valid_sha256(value) for value in known)
                    or len(set(known)) != len(known)):
                raise ValueError('known exposed inputs require unique SHA256 identities')
            known = set(known)
            inputs = set()
            allowed_pools = ({'apps_input_output'} if domain == 'apps_replay'
                             else {'public_tests', 'private_tests', 'generated_tests'})
            for index, test in enumerate(tests):
                if not isinstance(test, dict) or test.get('id') != str(index):
                    raise ValueError('evaluator tests must retain string IDs in exact order 0..9')
                raw_input, raw_output = test.get('input'), test.get('output')
                if (not isinstance(raw_input, str) or not isinstance(raw_output, str)
                        or len(raw_input) > 8192 or len(raw_output) > 4096):
                    raise ValueError('raw test input/output must satisfy 8192/4096 character limits')
                normalized_input = _normalized_io(raw_input)
                if normalized_input in inputs:
                    raise ValueError('normalized test inputs must be unique')
                inputs.add(normalized_input)
                pair_sha = hashlib.sha256((normalized_input + '\0' + _normalized_io(raw_output)).encode()).hexdigest()
                if test.get('pair_sha256') != pair_sha:
                    raise ValueError('normalized pair SHA256 mismatch')
                pools = test.get('original_pools')
                if (not isinstance(pools, list) or not pools
                        or any(not isinstance(pool, str) or pool not in allowed_pools for pool in pools)
                        or len(set(pools)) != len(pools)):
                    raise ValueError('test provenance must contain valid unique original pools')
                exposed = hashlib.sha256(normalized_input.encode()).hexdigest() in known
                if type(test.get('known_exposed_input')) is not bool or test['known_exposed_input'] != exposed:
                    raise ValueError('test exposure flag differs from known exposed input identities')
                if index >= 4 and exposed:
                    raise ValueError('target input appears in known exposed inputs')
                if index < 4:
                    observed = visible[index]
                    if (not isinstance(observed, dict) or observed.get('input') != raw_input
                            or observed.get('output') != raw_output):
                        raise ValueError('public tests must exactly match first four raw evaluator tests')
            base = dict(task_id=task_id, source_component_id=source_id, split=split, domain=domain)
            public.append({**base, 'statement': statement,
                           'visible_tests': [{'input': test['input'], 'output': test['output']}
                                             for test in tests[:4]]})
            evaluation = {**base, 'statement': statement, 'tests': deepcopy(tests),
                          'known_exposed_input_sha256': list(private['known_exposed_input_sha256'])}
            for name in ('locator', 'prior_provenance'):
                if name in private:
                    evaluation[name] = deepcopy(private[name])
            evaluator.append(evaluation)
    return public, evaluator
