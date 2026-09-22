"""A training checkpoint binds the exact request, adapter and optimizer state."""
import json
import hashlib
from pathlib import Path
import runpy

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/run_eesd_weighted_sft_checkpointed.py'
QUEUE = Path(__file__).resolve().parents[2] / (
    'runs/eesd-direct-20260921/operations/queue-seed1701-training-arms.py')


class TinyStudent:
    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir()
        (directory / 'adapter_config.json').write_text('{}')
        (directory / 'adapter_model.safetensors').write_bytes(b'tiny adapter')


def test_checkpoint_round_trip_and_integrity(tmp_path):
    module = runpy.run_path(str(SCRIPT))
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter])
    (parameter.square()).backward()
    optimizer.step()
    identity = {'rule': 'eesd_full', 'seed': 1701}
    progress = {'optimizer': optimizer.state_dict(), 'optimizer_steps': 10,
                'micro_steps': 160, 'response_tokens': 1000,
                'group_tokens': 0, 'group_examples': 0,
                'totals': {'loss': 1.0}, 'token_totals': {'loss': 1000.0}}
    module['save_checkpoint'](tmp_path, identity, TinyStudent(), optimizer, progress)
    directory = module['load_checkpoint'](tmp_path, identity)
    saved = torch.load(directory / 'state.pt', weights_only=False)
    assert saved['optimizer_steps'] == 10 and saved['response_tokens'] == 1000
    assert saved['optimizer']['state']
    with pytest.raises(ValueError, match='request or source differs'):
        module['load_checkpoint'](tmp_path, {'seed': 1702})
    (directory / 'adapter/adapter_config.json').write_text(json.dumps({'changed': True}))
    with pytest.raises(ValueError, match='adapter checksum differs'):
        module['load_checkpoint'](tmp_path, identity)


def test_training_queue_only_retries_matching_checkpoint(tmp_path):
    module = runpy.run_path(str(SCRIPT))
    queue = runpy.run_path(str(QUEUE))
    trainer_sha = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    identity = {'trainer_source_sha256': trainer_sha, 'input_sha256': 'input',
                'model_config_sha256': 'config', 'rule': 'eesd_full',
                'seed': 1701, 'response_token_budget': 2000}
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter])
    module['save_checkpoint'](tmp_path, identity, TinyStudent(), optimizer,
        {'optimizer': optimizer.state_dict(), 'optimizer_steps': 10,
         'micro_steps': 160, 'response_tokens': 1000,
         'group_tokens': 0, 'group_examples': 0, 'totals': {}, 'token_totals': {}})
    entry = {'task': {'command': [str(Path('/usr/bin/python3')), str(SCRIPT)],
                      'output': str(tmp_path),
                      'expected_report': {'trainer_source_sha256': trainer_sha,
                                          'input_sha256': 'input',
                                          'model_config_sha256': 'config',
                                          'rule': 'eesd_full', 'seed': 1701,
                                          'response_token_budget': 2000}}}
    assert queue['checkpoint_available'](entry)
    entry['task']['expected_report']['seed'] = 1702
    assert not queue['checkpoint_available'](entry)
