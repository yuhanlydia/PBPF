"""Public-only repair prompts preserving all four example slots under a cap."""
import ast
import json
import re


FIELDS = {'task_id', 'source_component_id', 'split', 'protocol', 'base_bug_id',
          'task_text', 'buggy_code', 'visible_tests'}


def prompt_for(tokenizer, row, *, max_input_tokens, chat_template_kwargs=None, family=None):
    if set(row) != FIELDS or row['protocol'] != 'RBR-generated-repair':
        raise ValueError('only whitelisted RBR public records accepted')
    if [t['id'] for t in row['visible_tests']] != ['0', '1', '2', '3'] or any(
            set(t) != {'id', 'input', 'expected'} for t in row['visible_tests']):
        raise ValueError('exactly four whitelisted public examples required')
    cap = 8192
    while cap >= 16:
        clipped = []
        def clip(value, field):
            if len(value) <= cap:
                return value
            clipped.append(field)
            return value[:cap]+'\n[remaining public text omitted]'
        content = ('Repair this Python program. Output one complete Python program that reads standard input '
                   'and writes standard output. Include all imports and the entry point.\n\nProblem:\n'
                   +clip(row['task_text'], 'task_text')+'\n\nBuggy program:\n```python\n'
                   +clip(row['buggy_code'], 'buggy_code')+'\n```\n\nFour public examples:\n')
        for test in row['visible_tests']:
            content += json.dumps({'id': test['id'], 'input': clip(test['input'], 'input'+test['id']),
                                   'expected': clip(test['expected'], 'expected'+test['id'])}, ensure_ascii=False)+'\n'
        template_kwargs = dict(chat_template_kwargs or {})
        prompt = chat_prompt(tokenizer, family,
            'You repair Python programs and return only complete source code.', content,
            chat_template_kwargs=template_kwargs)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) <= max_input_tokens:
            return prompt, {'input_tokens': len(tokens), 'field_character_cap': cap,
                            'clipped_fields': clipped, 'public_example_ids': ['0', '1', '2', '3']}
        cap = cap*3//4
    raise ValueError('token budget cannot retain the public prompt structure')


def extract_program(text):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, flags=re.S | re.I)
    for block in blocks:
        try:
            tree = ast.parse(block)
        except (SyntaxError, ValueError):
            continue
        if any(not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)) for node in tree.body):
            return block.strip()
    return (blocks[0] if blocks else text).strip()


def chat_prompt(tokenizer, family, system, content, *, chat_template_kwargs=None):
    """Use the pinned StarCoder2 single-user template, preserving task instructions."""
    if family == 'starcoder2_15b':
        messages = [{'role': 'user', 'content': system + '\n\n' + content}]
    else:
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          **dict(chat_template_kwargs or {}))


def generation_stop_kwargs(tokenizer, family):
    if family != 'starcoder2_15b':
        return {}
    # Official pinned model card requires both EOS and the next section marker.
    separator = tokenizer.convert_tokens_to_ids('###')
    if separator is None or separator == tokenizer.unk_token_id:
        raise ValueError('StarCoder2 tokenizer is missing the ### separator token')
    return {'eos_token_id': [tokenizer.eos_token_id, separator]}


def decode_completion(tokenizer, ids, family):
    stops = generation_stop_kwargs(tokenizer, family).get('eos_token_id', [tokenizer.eos_token_id])
    position = next((i for i, token in enumerate(ids) if token in stops), None)
    count = len(ids) if position is None else position + 1
    # Keep legacy decoding unchanged; ### is not special and must be stripped for StarCoder2.
    decoded_ids = ids[:position] if family == 'starcoder2_15b' and position is not None else ids
    text = tokenizer.decode(decoded_ids, skip_special_tokens=True)
    return text, count, position is None
