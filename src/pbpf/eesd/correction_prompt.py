"""Public-only, bounded correction model views; execution evidence stays intact."""
from __future__ import annotations

import hashlib
import json

POLICY = 'eesd-correction-public-head-tail-v1'
SYSTEM = 'You repair Python programs using public execution feedback.'
FIELDS = ('input', 'expected', 'actual', 'stderr')


def user_prompt(row, domain: str) -> str:
    mode = 'stdin/stdout Python program' if domain == 'rbr' else 'Python function solution'
    evidence = [
        {k: test[k] for k in ('id', 'input', 'expected', 'actual', 'stderr', 'outcome')}
        | ({'expected_error': test['expected_error']} if 'expected_error' in test else {})
        for test in row['tests']
    ]
    return (
        f'Repair the following {mode}. Use only the public execution evidence. '
        'Return only complete Python source code.\n' + json.dumps({
            'task': row['task_text'], 'program': row['candidate'],
            'public_executions': evidence,
        }, sort_keys=True, ensure_ascii=False)
    )


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def _view(row, cap):
    clipped = []
    def field(value, path):
        text = _text(value)
        if cap is None or len(text) <= cap:
            return value
        clipped.append({'field': path, 'original_characters': len(text), 'retained_characters': cap})
        head = (cap + 1) // 2
        tail = cap // 2
        marker = f'\n[PUBLIC TEXT TRUNCATED: original_characters={len(text)}; retained_characters={cap}]\n'
        return text[:head] + marker + (text[-tail:] if tail else '')
    view = dict(row, task_text=field(row['task_text'], 'task_text'))
    view['tests'] = [dict(test, **{key: field(test[key], f'tests[{i}].{key}') for key in FIELDS})
                     for i, test in enumerate(row['tests'])]
    return view, clipped


def prepare_correction_prompt(tokenizer, row, domain, max_input_tokens=4096, model_id=''):
    """Build the complete locked model view, failing instead of dropping a row.

    Non-string public values are JSON-rendered only when clipped, with a visible
    marker; untouched fields preserve their original JSON types. Full source code,
    slot IDs, outcomes and expected_error are never clipped.
    """
    if max_input_tokens < 1:
        raise ValueError('max_input_tokens must be positive')
    if len(row['tests']) != 4:
        raise ValueError('correction prompt requires exactly four public slots')
    if len({test['id'] for test in row['tests']}) != 4:
        raise ValueError('correction prompt requires four unique public slot IDs')
    kwargs = {'enable_thinking': False} if 'Qwen3' in model_id else {}
    # Avoid tokenizing megabytes before bounding. Small ordinary prompts retain
    # their exact original representation if they fit.
    public_size = len(_text(row['task_text'])) + sum(
        len(_text(test[key])) for test in row['tests'] for key in FIELDS)
    caps = [None] if public_size <= 4 * max_input_tokens else []
    cap = 2048
    while cap:
        caps.append(cap)
        cap = cap * 3 // 4
    caps.append(0)
    for cap in caps:
        view, clipped = _view(row, cap)
        prompt = user_prompt(view, domain)
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        ids = tokenizer(rendered, add_special_tokens=False, return_token_type_ids=False)['input_ids']
        if len(ids) <= max_input_tokens:
            return {'prompt': prompt, 'messages': messages, 'rendered': rendered, 'input_ids': ids,
                    'prompt_token_ids_sha256': hashlib.sha256(
                        json.dumps(ids, separators=(',', ':')).encode('utf-8')).hexdigest(),
                    'prompt_metadata': {'policy': POLICY, 'input_tokens': len(ids),
                                        'max_input_tokens': max_input_tokens,
                                        'field_character_cap': cap, 'truncated_fields': clipped}}
    raise ValueError(f"task {row.get('task_id', '<unknown>')} cannot fit correction input budget "
                     f'{max_input_tokens} with complete program and four public slots')
