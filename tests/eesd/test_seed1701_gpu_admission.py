"""Both GPU queues reserve launches before nvidia-smi exposes a new process."""
import json
from pathlib import Path
import runpy
import time


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / 'runs/eesd-direct-20260921/operations/queue-seed1701-training-arms.py'
FRESH = ROOT / 'runs/eesd-direct-20260921/operations/queue-seed1701-fresh-evaluations.py'


def test_training_queue_counts_invisible_fresh_shard(tmp_path):
    module = runpy.run_path(str(TRAIN))
    reserve = module['include_fresh_reservations']
    state = {'cells': {'codearc/qwen25_7b/eesd_full': {
        'shards': [{'status': 'running', 'gpu': 1, 'pid': 1234}]}}}
    receipt = tmp_path / 'fresh.json'
    receipt.write_text(json.dumps(state))
    reserve.__globals__['fresh_receipt'] = receipt
    reserve.__globals__['active'] = lambda pid: True
    counts, free = {x: 0 for x in range(4)}, {x: 40000 for x in range(4)}
    reserve(counts, free, set())
    assert counts[1] == 1 and free[1] == 32500
    single = module['reserve_invisible']
    single.__globals__['active'] = lambda pid: True
    counts, free = {x: 1 for x in range(4)}, {x: 40000 for x in range(4)}
    single(counts, free, {1234}, gpu=1, pid=1234,
           memory_mib=7500, started_at=time.time())
    assert counts[1] == 1 and free[1] == 32500


def test_fresh_queue_counts_invisible_training_and_own_shard(tmp_path, monkeypatch):
    module = runpy.run_path(str(FRESH))
    gpu_state = module['gpu_state']
    receipt = tmp_path / 'training.json'
    receipt.write_text(json.dumps({'tasks': {'task': {
        'status': 'running', 'gpu': 0, 'pid': 1234,
        'family': 'qwen25_7b', 'started_at': time.time()}}}))
    gpu_state.__globals__['training_receipt'] = receipt
    gpu_state.__globals__['active'] = lambda pid: True
    monkeypatch.setattr(gpu_state.__globals__['subprocess'], 'check_output',
        lambda command, text: (
        '0, uuid0, 48000\n1, uuid1, 48000\n2, uuid2, 48000\n3, uuid3, 48000\n'
        if '--query-gpu=index,uuid,memory.free' in command else 'uuid0, 1234\n'))
    state = {'cells': {'codearc/gemma3_4b/eesd_full': {
        'adapter': '/unused',
        'shards': [{'status': 'running', 'gpu': 1, 'pid': 2345}]}}}
    counts, free = gpu_state(state)
    assert counts[0] == counts[1] == 1
    assert free[0] == 26000 and free[1] == 42500
