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


def prompt_for(tokenizer, row, *, family, max_input_tokens=4096, chat_template_kwargs=None):
    """Bound only public fields and retain all four replay example slots."""
    from pbpf.apbpf.rbr_prompt import chat_prompt

    visible = row['visible_tests']
    if [t['id'] for t in visible] != ['0', '1', '2', '3']:
        raise ValueError('exactly four public replay examples required')
    # Preserve normal prompts exactly; avoid tokenizing million-character outliers.
    if len(row['task_text']) <= max_input_tokens * 4:
        original = chat_prompt(tokenizer, family,
            'You write Python code that generalizes from observed input-output examples.',
            row['task_text'], chat_template_kwargs=chat_template_kwargs)
        tokens = tokenizer.encode(original, add_special_tokens=True)
        if len(tokens) <= max_input_tokens:
            return original, {'input_tokens': len(tokens), 'field_character_cap': None,
                              'clipped_fields': [], 'public_example_ids': ['0', '1', '2', '3']}
    limit = 2048
    while limit >= 16:
        content = bounded_task_text(row, limit=limit)
        prompt = chat_prompt(tokenizer, family,
            'You write Python code that generalizes from observed input-output examples.',
            content, chat_template_kwargs=chat_template_kwargs)
        # Match the generator's existing default add_special_tokens=True behavior.
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        if len(tokens) <= max_input_tokens:
            clipped = [field + t['id'] for t in visible for field in ('input', 'expected')
                       if len(t[field]) > limit]
            return prompt, {'input_tokens': len(tokens), 'field_character_cap': limit,
                            'clipped_fields': clipped, 'public_example_ids': ['0', '1', '2', '3']}
        limit = limit * 3 // 4
    raise ValueError('token budget cannot retain the public prompt structure')
