"""Deterministic public-input bounds for unusually large replay examples."""
import json


def bounded_task_text(row, limit=2048):
    visible = row['visible_tests']
    if not any(len(t[field]) > limit for t in visible for field in ('input', 'expected')):
        return row['task_text']
    bounded = []
    for test in visible:
        value = dict(test)
        for field in ('input', 'expected'):
            text = test[field]
            if len(text) > limit:
                value[field] = text[:limit // 2] + f'\n[PUBLIC TEXT TRUNCATED: {len(text)} characters]\n' + text[-limit // 2:]
        bounded.append(value)
    return ('Infer a Python function named solution from the observed calls below. '
            'Return a complete Python implementation, including any imports. '
            'Do not print the examples as a substitute for implementing the function. '
            'Long public fields are explicitly truncated to respect the input budget.\n\n'
            + '\n\n'.join(json.dumps(t, ensure_ascii=False) for t in bounded))
