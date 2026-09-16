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
