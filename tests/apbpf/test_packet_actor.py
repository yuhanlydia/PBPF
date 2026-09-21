import hashlib
import json

import numpy as np
import pytest

from pbpf.apbpf.repair_materials import actor_rows, build_repair_materials
from test_repair_materials import fixture
from test_stage_training import module


def packet(tmp_path):
    cache, public, private = fixture('codearc')
    targets, _, context = build_repair_materials(cache, public, private, domain='codearc')
    rows = actor_rows(cache, targets, context)
    rows = [r for r in rows if r['split'] != 'primary'] + [next(r for r in rows if r['split'] == 'primary')]
    (tmp_path/'actor-rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    np.savez(tmp_path/'posteriors.npz', diagnosis=np.zeros((len(rows), 8, 24), dtype=np.float32),
             log_weights=np.full((len(rows), 8), -np.log(8), dtype=np.float32))
    value = {'schema': 'apbpf-repair-actor-packet-v1', 'seed': 1701, 'primary_sources': 1,
             'counts': {s: sum(r['split'] == s for r in rows) for s in ('train', 'development', 'primary')},
             'files': {n: hashlib.sha256((tmp_path/n).read_bytes()).hexdigest() for n in ('actor-rows.jsonl', 'posteriors.npz')}}
    (tmp_path/'manifest.json').write_text(json.dumps(value))
    return rows


class Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return list(range(16))

    def __call__(self, text, **kwargs):
        return {'input_ids': [1, 2, 3]}


def test_packet_actor_verifies_inventory_and_public_prompt(tmp_path):
    rows = packet(tmp_path)
    worker = module('packet_actor_test', 'run_apbpf_packet_actor.py')
    manifest, actual, z, weights = worker.load_packet(tmp_path)
    assert rows == actual
    assert z.shape == (25, 8, 24)
    tokenizer = Tokenizer()
    ids, length = worker.prompt_ids(tokenizer, rows[-1], 'codearc', cap=10)
    assert ids == list(range(6, 16)) and length == 16
    prompt = json.dumps(tokenizer.messages)
    assert 'function solution' in prompt and 'stdin/stdout' not in prompt
    assert 'expected_error' in prompt
    assert 'reference_code' not in prompt and 'answer4' not in prompt
    with pytest.raises(ValueError, match='primary row'):
        worker.training_item(tokenizer, rows[-1], 'codearc', 100)
    item, reason = worker.training_item(tokenizer, rows[1], 'codearc', 100)
    assert reason is None and (item[1][:16] == -100).all() and item[1][16:].tolist() == [1, 2, 3, 0]
    item, reason = worker.training_item(tokenizer, rows[1], 'codearc', 20)
    assert item is None and reason['target_tokens'] == 4
    (tmp_path/'actor-rows.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        worker.load_packet(tmp_path)


def test_packet_actor_rejects_reference_injection_even_with_updated_checksum(tmp_path):
    rows = packet(tmp_path)
    rows[-1]['reference_code'] = 'PRIVATE_PRIMARY_REFERENCE'
    path = tmp_path/'actor-rows.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    p = tmp_path/'manifest.json'
    value = json.loads(p.read_text())
    value['files'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    p.write_text(json.dumps(value))
    worker = module('packet_actor_injection_test', 'run_apbpf_packet_actor.py')
    with pytest.raises(ValueError, match='public/target boundary'):
        worker.load_packet(tmp_path)
