from pbpf.apbpf.code_extraction import extract_solution


def test_short_implementation_wins_over_longer_usage_and_output_blocks():
    text = '```python\ndef solution(x): return x + 1\n```\n' + '```python\n' + "print(solution(1))\n" * 20 + '```\n```\n2\n```'
    assert extract_solution(text) == 'def solution(x): return x + 1'


def test_separate_imports_retained_but_usage_not_executed():
    text = '```python\nimport re\n```\n```python\ndef solution(x): return re.findall("a",x)\n```\n```python\nprint(solution("a"))\n```'
    code = extract_solution(text)
    namespace = {}; exec(code, namespace)
    assert namespace['solution']('abc') == ['a']
    assert 'print(' not in code


def test_truncated_definition_and_unfenced_completion_preserved():
    text = '```python\ndef solution(x):\n    return (\n```\n```python\n' + 'print(solution(1))\n' * 10 + '```'
    assert extract_solution(text).startswith('def solution')
    assert extract_solution('def solution(): return 1') == 'def solution(): return 1'
