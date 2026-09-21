from pbpf.apbpf.codearc_prompt import bounded_task_text


def test_normal_public_prompts_unchanged_and_long_fields_bounded_without_dropping_calls():
    row = {'task_text': 'original prompt', 'visible_tests': [
        {'id': str(i), 'input': f'solution({i})', 'expected': 'ok', 'expected_error': False} for i in range(4)]}
    assert bounded_task_text(row) == row['task_text']
    row['visible_tests'][0]['expected'] = 'X' * 1_000_000
    text = bounded_task_text(row)
    assert len(text) < 4000
    assert 'PUBLIC TEXT TRUNCATED: 1000000 characters' in text
    assert all(f'solution({i})' in text for i in range(4))
    assert len(row['visible_tests'][0]['expected']) == 1_000_000


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages)
    def encode(self, text, **kwargs):
        return list(text)


def test_adaptive_prompt_preserves_four_slots_and_enforces_actual_token_cap():
    from pbpf.apbpf.codearc_prompt import prompt_for
    row = {'task_text': 'X' * 1_200_000, 'visible_tests': [
        {'id': str(i), 'input': f'solution({i})', 'expected': 'X' * 300_000,
         'expected_error': False} for i in range(4)]}
    prompt, metadata = prompt_for(CharacterTokenizer(), row, family='qwen25_7b', max_input_tokens=4096)
    assert len(prompt) <= 4096
    assert all(f'"id": "{i}"' in prompt and f'solution({i})' in prompt for i in range(4))
    assert metadata['public_example_ids'] == ['0', '1', '2', '3']
    assert metadata['input_tokens'] == len(prompt)
    assert metadata['field_character_cap'] < 2048
    assert metadata['clipped_fields'] == ['expected0', 'expected1', 'expected2', 'expected3']
    assert len(row['visible_tests'][0]['expected']) == 300_000


def test_normal_prompt_preserves_original_chat_content():
    from pbpf.apbpf.codearc_prompt import prompt_for
    row = {'task_text': 'original prompt', 'visible_tests': [
        {'id': str(i), 'input': f'solution({i})', 'expected': 'ok', 'expected_error': False} for i in range(4)]}
    prompt, metadata = prompt_for(CharacterTokenizer(), row, family='qwen25_7b', max_input_tokens=4096)
    assert prompt == 'You write Python code that generalizes from observed input-output examples.\noriginal prompt'
    assert metadata['clipped_fields'] == []


def test_prompt_with_2100_character_example_is_not_truncated_if_it_fits_tokens():
    from pbpf.apbpf.codearc_prompt import prompt_for
    row = {'task_text': 'original ' + 'X' * 2100, 'visible_tests': [
        {'id': str(i), 'input': f'solution({i})', 'expected': 'X' * 2100 if i == 0 else 'ok',
         'expected_error': False} for i in range(4)]}
    prompt, metadata = prompt_for(CharacterTokenizer(), row, family='qwen25_7b', max_input_tokens=4096)
    assert prompt.endswith(row['task_text'])
    assert metadata['clipped_fields'] == []
