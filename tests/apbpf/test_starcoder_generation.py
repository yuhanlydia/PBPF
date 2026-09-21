"""Exercise family-specific prompts and termination without loading model weights."""
import importlib.util
from pathlib import Path

import pytest
from pbpf.apbpf import rbr_prompt

ROOT = Path(__file__).resolve().parents[2]


class Tokenizer:
    eos_token_id = 0
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {'###': 610}[token]

    def apply_chat_template(self, messages, **kwargs):
        if self.single_user and [m['role'] for m in messages] != ['user']:
            raise ValueError('System messages are not allowed in this template.')
        return repr(messages)

    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, ids, **kwargs):
        return ''.join({1: 'print(1)', 610: '###', 0: ''}[i] for i in ids)


def test_starcoder_rbr_retains_instructions_and_all_examples_in_single_user():
    tokenizer = Tokenizer()
    tokenizer.single_user = True
    row = dict(task_id='RBR/p1', source_component_id='p1', split='train',
        protocol='RBR-generated-repair', base_bug_id='1', task_text='Increment', buggy_code='print(0)',
        visible_tests=[{'id': str(i), 'input': str(i), 'expected': str(i+1)} for i in range(4)])
    prompt, metadata = rbr_prompt.prompt_for(tokenizer, row, max_input_tokens=4096, family='starcoder2_15b')
    assert 'You repair Python programs' in prompt
    assert 'Increment' in prompt
    assert metadata['public_example_ids'] == ['0', '1', '2', '3']


def test_other_families_preserve_system_user_messages():
    tokenizer = Tokenizer()
    tokenizer.single_user = False
    assert rbr_prompt.chat_prompt(tokenizer, 'qwen25_7b', 'System', 'Task') == repr([
        {'role': 'system', 'content': 'System'}, {'role': 'user', 'content': 'Task'}])


@pytest.mark.parametrize('ids,want', [([1,610,0], ('print(1)',2,False)),
    ([1,0,0],('print(1)',2,False)), ([1],('print(1)',1,True))])
def test_starcoder_termination_counts_first_stop_and_strips_separator(ids,want):
    assert rbr_prompt.decode_completion(Tokenizer(), ids, 'starcoder2_15b') == want


def test_only_starcoder_overrides_generation_terminators():
    assert rbr_prompt.generation_stop_kwargs(Tokenizer(), 'starcoder2_15b') == {'eos_token_id': [0,610]}
    assert rbr_prompt.generation_stop_kwargs(Tokenizer(), 'qwen25_7b') == {}
    assert rbr_prompt.decode_completion(Tokenizer(), [1,610,0], 'qwen25_7b') == ('print(1)###',3,False)


@pytest.mark.parametrize('script', ['generate_apbpf_rbr_bank', 'generate_apbpf_codearc_bank'])
def test_generators_accept_starcoder_family(script):
    spec = importlib.util.spec_from_file_location(script, ROOT / 'scripts' / (script+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.MODELS['starcoder2_15b'] == (
        'bigcode/starcoder2-15b-instruct-v0.1', 'ffb8dd9776ba9a66d655ecd962e882f3013e9f7c')


def test_codearc_selects_one_deterministic_public_task_per_component(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    from types import SimpleNamespace
    script = 'generate_apbpf_codearc_bank'
    spec = importlib.util.spec_from_file_location(script, ROOT / 'scripts' / (script+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{'task_id': task, 'source_component_id': component, 'split': 'primary',
        'protocol': 'CodeARC-Replay', 'task_text': 'Task',
        'visible_tests': [{'id': str(i)} for i in range(4)]}
        for task, component in [('b','two'), ('z','one'), ('a','one')]]
    public = tmp_path / 'public'
    public.mkdir()
    tasks = public / 'tasks.jsonl'
    tasks.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (public / 'manifest.json').write_text(json.dumps({'public_tasks_sha256': hashlib.sha256(tasks.read_bytes()).hexdigest()}))
    class StopBeforeWeights:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError('stop before weights')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=StopBeforeWeights,
        AutoModelForCausalLM=None, BitsAndBytesConfig=None))
    monkeypatch.setattr(sys, 'argv', ['generator', '--public-root', str(public), '--output', str(tmp_path/'out'),
        '--split', 'primary', '--family', 'qwen25_7b'])
    with pytest.raises(RuntimeError, match='stop before weights'):
        module.main()
    run = json.loads((tmp_path/'out/run.json').read_text())
    assert run['task_ids'] == ['b','a']
    assert run['components'] == 2


@pytest.mark.parametrize('domain', ['rbr', 'codearc'])
def test_starcoder_generators_produce_clean_code_and_honor_dual_stops(tmp_path, monkeypatch, domain):
    import contextlib
    import hashlib
    import json
    import sys
    from types import SimpleNamespace
    import numpy as np
    script = 'generate_apbpf_' + domain + '_bank'
    spec = importlib.util.spec_from_file_location(script, ROOT / 'scripts' / (script+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Tensor:
        def __init__(self, values): self.values = np.asarray(values)
        @property
        def shape(self): return self.values.shape
        def __getitem__(self, key): return Tensor(self.values[key])
        def __iter__(self): return (Tensor(row) for row in self.values)
        def cpu(self): return self
        def tolist(self): return self.values.tolist()

    class Inputs(dict):
        def __init__(self, ids):
            self.input_ids = Tensor([ids])
            super().__init__(input_ids=self.input_ids)
        def to(self, device): return self

    class ActorTokenizer(Tokenizer):
        single_user = True
        eos_token = '<|endoftext|>'
        pad_token_id = 0
        def encode(self, text, **kwargs): return [ord(c) for c in text]
        def __call__(self, text, **kwargs): return Inputs(self.encode(text))

    class Actor:
        device = 'cpu'
        def eval(self): pass
        def generate(self, input_ids, **kwargs):
            # The fake models actual termination only if callers configure both stops.
            continuation = [1,610,0] if kwargs.get('eos_token_id') == [0,610] else [1,610,1]
            return Tensor([input_ids.values[0].tolist() + continuation])

    tokenizer = ActorTokenizer()
    actor = Actor()
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(manual_seed=lambda seed: None,
        bfloat16='bf16', inference_mode=contextlib.nullcontext))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: actor),
        BitsAndBytesConfig=lambda **k: k))
    row = dict(task_id='task1', source_component_id='source1', split='primary',
        protocol='RBR-generated-repair' if domain == 'rbr' else 'CodeARC-Replay', task_text='Task',
        visible_tests=[{'id': str(i), 'input': str(i), 'expected': str(i)} for i in range(4)])
    if domain == 'rbr': row.update(base_bug_id='1', buggy_code='print(0)')
    public = tmp_path/'public'
    public.mkdir()
    tasks = public/'tasks.jsonl'
    tasks.write_text(json.dumps(row)+'\n')
    (public/'manifest.json').write_text(json.dumps({'schema':'apbpf-rbr-generated-materialization-v1',
        'public_tasks_sha256': hashlib.sha256(tasks.read_bytes()).hexdigest()}))
    output = tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['generator','--public-root',str(public),'--output',str(output),
        '--split','primary','--family','starcoder2_15b','--candidates','1'])
    module.main()
    candidate = json.loads((output/'task1.json').read_text())['candidates'][0]
    assert candidate['code'] == 'print(1)'
    assert candidate['generated_tokens'] == 2
    assert candidate['hit_token_cap'] is False
    assert (output/'complete.json').exists()
    if domain == 'codearc':
        run = json.loads((output/'run.json').read_text())
        assert run['max_input_tokens'] == 4096
        assert len(run['codearc_prompt_source_sha256']) == 64
        record = json.loads((output/'task1.json').read_text())
        assert record['prompt_metadata']['input_tokens'] <= 4096
        assert record['prompt_metadata']['public_example_ids'] == ['0','1','2','3']
