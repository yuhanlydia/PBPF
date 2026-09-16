import pytest
from pbpf.apbpf.rbr_prompt import prompt_for, extract_program


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages)
    def encode(self, text, **kwargs):
        return list(text)


def test_long_prompt_keeps_four_examples_and_rejects_hidden_fields():
    row = dict(task_id='RBR/p1', source_component_id='p1', split='train', protocol='RBR-generated-repair',
               base_bug_id='1', task_text='d'*2000, buggy_code='b'*2000,
               visible_tests=[{'id': str(i), 'input': 'x'*2000, 'expected': 'y'*2000} for i in range(4)])
    prompt, metadata = prompt_for(Tokenizer(), row, max_input_tokens=2000)
    assert len(prompt) <= 2000 and metadata['clipped_fields']
    assert all(f'"id": "{i}"' in prompt for i in range(4))
    with pytest.raises(ValueError):
        prompt_for(Tokenizer(), {**row, 'reference_code': 'SECRET'}, max_input_tokens=2000)


def test_program_extraction_skips_literal_example_blocks():
    text = '```\n42\n```\n```python\nprint(int(input()) + 1)\n```'
    assert extract_program(text) == 'print(int(input()) + 1)'
