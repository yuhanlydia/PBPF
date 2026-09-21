import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from pbpf.apbpf.codearc_bank import file_sha, load_bank


def test_serial_replication_keeps_eight_candidates_and_verifiable_resume(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2]/'scripts/generate_apbpf_rbr_replication_bank.py'
    spec = importlib.util.spec_from_file_location('replication_generation_contract', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    public = tmp_path/'public'; public.mkdir()
    row = {'task_id':'RBR/p1', 'source_component_id':'p1', 'split':'development',
           'protocol':'RBR-generated-repair', 'base_bug_id':'bug1', 'task_text':'Echo a number',
           'buggy_code':'print(0)', 'visible_tests':[{'id':str(i),'input':str(i),'expected':str(i)} for i in range(4)]}
    (public/'tasks.jsonl').write_text(json.dumps(row)+'\n')
    (public/'manifest.json').write_text(json.dumps({'schema':'apbpf-rbr-generated-materialization-v1',
                                                 'public_tasks_sha256':file_sha(public/'tasks.jsonl')}))
    proof = tmp_path/'proof.json'
    proof.write_text(json.dumps({'model':module.MODELS['deepseek'][0], 'revision':module.MODELS['deepseek'][1],
                  'files':[{'file':'weights.safetensors','sha256':'a'*64,'publisher_lfs_sha256':'a'*64}]}))
    calls = []

    class Batch(dict):
        @property
        def input_ids(self): return self['input_ids']
        def to(self, device): return self

    class Tokenizer:
        eos_token='EOS'; eos_token_id=0; pad_token_id=0
        def apply_chat_template(self, messages, **kwargs): return json.dumps(messages)
        def encode(self, prompt, **kwargs): return [1,2]
        def __call__(self, prompt, **kwargs): return Batch(input_ids=torch.tensor([[1,2]]))
        def decode(self, tokens, **kwargs): return f'print({int(tokens[-1])})'

    class Actor:
        device='cpu'
        def eval(self): pass
        def generate(self, **kwargs):
            # A constrained actor cannot fit a batch of eight: exercise real orchestration.
            assert kwargs['num_return_sequences'] == 1
            assert kwargs['max_new_tokens'] == 1024
            calls.append(torch.initial_seed())
            return torch.cat([kwargs['input_ids'],torch.randint(10,1000,(1,3))],dim=1)

    loads=[]
    def load_actor(*args, **kwargs): loads.append(args); return Actor()
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_actor),
        BitsAndBytesConfig=lambda **kwargs:kwargs))
    output = tmp_path/'bank'
    argv=[str(path),'--public-root',str(public),'--output',str(output),'--model-proof',str(proof),'--split','development']
    monkeypatch.setattr(sys,'argv',argv); module.main()
    run, rows, _ = load_bank(output)
    assert run['decode_batch_size'] == 1 and run['candidates'] == 8
    assert len(calls) == len(set(calls)) == 8
    assert len(rows[0]['candidates']) == 8
    assert [c['decode_seed'] for c in rows[0]['candidates']] == calls
    module.main()
    assert len(loads) == 1 and len(calls) == 8
    monkeypatch.setattr(sys,'argv',argv+['--seed','1702'])
    with pytest.raises(ValueError,match='resume identity'):
        module.main()
