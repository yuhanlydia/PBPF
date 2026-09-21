"""Exact response-token budgeting, including the final partial accumulation."""
import importlib.util
from pathlib import Path
import pytest


def runner():
    path = Path(__file__).resolve().parents[2] / 'scripts/run_eesd_weighted_sft.py'
    spec = importlib.util.spec_from_file_location('eesd_sft', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_different_arm_populations_consume_exact_same_budget():
    sft = runner()
    for lengths in ([3, 7, 11], [19], [2, 4]):
        plan = list(sft.training_schedule(lengths, seed=1701, max_steps=2,
            gradient_accumulation=4, response_token_budget=53))
        assert sum(n for _, n in plan) == 53
        assert all(0 < n <= lengths[i] for i, n in plan)
        assert all(n == lengths[i] for i, n in plan[:-1])
        assert plan == list(sft.training_schedule(lengths, seed=1701, max_steps=2,
            gradient_accumulation=4, response_token_budget=53))


def test_legacy_schedule_preserves_step_and_example_budget():
    plan = list(runner().training_schedule([3, 7], seed=5, max_steps=3,
        gradient_accumulation=4, response_token_budget=None))
    assert len(plan) == 12
    assert all(n == [3, 7][i] for i, n in plan)


@pytest.mark.parametrize('lengths,budget', [([], 5), ([0], 5), ([3], 0), ([3], -1)])
def test_invalid_budget_or_empty_population_rejected(lengths, budget):
    with pytest.raises(ValueError):
        list(runner().training_schedule(lengths, seed=5, max_steps=3,
            gradient_accumulation=4, response_token_budget=budget))


def test_partial_accumulation_normalizes_actual_token_count():
    import torch
    sft = runner()
    p = torch.nn.Parameter(torch.tensor(1.0))
    # Two microbatches: 3 tokens with loss 2*p each and 1 token with 4*p.
    (3 * 2 * p).backward()
    (1 * 4 * p).backward()
    sft.normalize_gradients([p], denominator=4)
    assert p.grad.item() == pytest.approx(2.5)


class Tokenizer:
    eos_token = '!'
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']
    def __call__(self, text, **kwargs):
        return {'input_ids': list(range(len(text)))}


def test_encoding_preserves_prompt_and_reports_response_truncation():
    sft = runner()
    encoded = sft.encode_training_row(Tokenizer(), {'prompt': 'abcd', 'correction': '123456'},
        max_length=8, model_id='test')
    assert encoded['prompt_ids'] == [0, 1, 2, 3]
    assert encoded['completion_ids'] == [0, 1, 2, 3]
    assert encoded['response_tokens_before_truncation'] == 7
    assert encoded['response_truncated'] is True
    with pytest.raises(ValueError, match='prompt exceeds'):
        sft.encode_training_row(Tokenizer(), {'prompt': 'abcd', 'correction': '123'},
            max_length=4, model_id='test')


def test_real_training_loop_uses_exact_budget_and_flushes_partial_group(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    import torch
    sft = runner()

    class Model(torch.nn.Module):
        device = torch.device('cpu')
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.linear = torch.nn.Linear(4, 16)
        def gradient_checkpointing_enable(self):
            pass
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.linear(self.embedding(input_ids)))
        def save_pretrained(self, path):
            path.mkdir()
            (path / 'saved').write_text('yes')

    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(
        LoraConfig=lambda **kwargs: None, PeftModel=None,
        get_peft_model=lambda model, config: model,
        prepare_model_for_kbit_training=lambda model: model))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: Model()),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer()),
        BitsAndBytesConfig=lambda **kw: None))
    monkeypatch.setattr(Tokenizer, 'pad_token_id', 0, raising=False)
    data = tmp_path / 'rows.jsonl'
    from pbpf.eesd.distillation import TRAIN_RULES
    data.write_text('\n'.join(json.dumps({'split': split, 'source_component_id': str(i), 'trajectory_id':str(i),
        'before_outcomes':[0]*4,'after_outcomes':[0]*4,'relevance':[1.]*4,
        'prompt': 'abcd', 'correction': text, 'training_weights': {rule:(0. if rule=='no_update' else 1.) for rule in TRAIN_RULES}})
        for i, (split, text) in enumerate([('train', '123456'), ('train', '12'), ('development', '123')])))
    cfg = tmp_path / 'model.yaml'
    cfg.write_text('model_id: test\nrevision: ' + 'a' * 40 + '\n')
    output = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['sft', '--input', str(data), '--model-config', str(cfg),
        '--rule', 'equal_weight', '--output', str(output), '--response-token-budget', '23',
        '--gradient-accumulation', '4', '--max-steps', '1', '--max-length', '64'])
    sft.main()
    report = json.loads((output / 'training-report.json').read_text())
    assert report['response_tokens'] == report['response_token_budget'] == 23
    assert report['optimizer_steps'] == 2  # one full group + a final partial group
    assert report['micro_steps'] > 4
    assert report['loss_normalization'] == 'response_tokens'
    assert (output / 'adapter' / 'saved').exists()


def test_training_uses_exact_generation_messages_and_checks_token_identity():
    import hashlib,json
    class MessageTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return ''.join(m['content'] for m in messages)
    row={'prompt':'abcd','correction':'x',
         'messages':[{'role':'system','content':'sys'},{'role':'user','content':'abcd'}],
         'prompt_metadata':{'input_tokens':7},
         'prompt_token_ids_sha256':hashlib.sha256(json.dumps(list(range(7)),separators=(',',':')).encode()).hexdigest()}
    encoded=runner().encode_training_row(MessageTokenizer(),row,max_length=10,model_id='test')
    assert encoded['prompt_ids']==list(range(7))
    bad={**row,'prompt_token_ids_sha256':'0'*64}
    with pytest.raises(ValueError,match='token identity'):
        runner().encode_training_row(MessageTokenizer(),bad,max_length=10,model_id='test')
    with pytest.raises(ValueError,match='messages'):
        runner().encode_training_row(MessageTokenizer(),{**row,'prompt':'different'},max_length=20,model_id='test')
