import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def coordinator():
    path = Path(__file__).resolve().parents[2]/'scripts/coordinate_rbr_replication_gpus.py'
    spec = importlib.util.spec_from_file_location('rbr_gpu_coordinator_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def until(predicate, timeout=5):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        result = predicate()
        if result:
            return result
        time.sleep(.02)
    raise AssertionError('test process did not reach expected state')


def test_parked_scheduler_keeps_child_running_and_cannot_retire_live_work(tmp_path):
    module = coordinator()
    child = tmp_path/'child.py'
    child.write_text('import pathlib, time\n'
                     'r=pathlib.Path(__file__).parent\n'
                     '(r/"started").write_text("yes")\n'
                     'while not (r/"release").exists(): time.sleep(.02)\n'
                     '(r/"finished").write_text("yes")\n')
    parent = tmp_path/'parent.py'
    parent.write_text('import pathlib, subprocess, sys\n'
                      'subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name("child.py"))], check=True)\n')
    process = subprocess.Popen([sys.executable, str(parent)], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        until(lambda: (tmp_path/'started').exists())
        expected = module.process_info(process.pid)
        assert len(expected['children']) == 1
        child_pid = expected['children'][0]
        parked = module.park_parent(expected)
        assert parked['state'] in ('T', 't')
        with pytest.raises(ValueError, match='live work'):
            module.retire_parked_parent(expected)
        (tmp_path/'release').write_text('go')
        until(lambda: (tmp_path/'finished').exists())
        until(lambda: (module.process_info(child_pid) or {}).get('state') == 'Z')
        assert module.process_info(process.pid)['state'] in ('T', 't')
        receipt = module.retire_parked_parent(expected)
        assert receipt['terminal_children'][0]['state'] == 'Z'
        assert process.wait(timeout=5) == -signal.SIGKILL
    finally:
        (tmp_path/'release').write_text('cleanup')
        if process.poll() is None:
            os.kill(process.pid, signal.SIGCONT)
            process.wait(timeout=5)


def test_wrong_process_identity_or_running_scheduler_is_never_signaled(monkeypatch):
    module = coordinator()
    actual = {'pid': 42, 'start_ticks': '100', 'command': 'known parent', 'state': 'R', 'children': []}
    monkeypatch.setattr(module, 'process_info', lambda pid: actual)
    def forbidden(*args):
        raise AssertionError('must not send a signal')
    monkeypatch.setattr(module.os, 'kill', forbidden)
    with pytest.raises(ValueError, match='identity'):
        module.park_parent({**actual, 'start_ticks': '99'})
    with pytest.raises(ValueError, match='running parent'):
        module.retire_parked_parent(actual)


def test_gpu0_check_does_not_treat_occupied_device_as_available(monkeypatch):
    module = coordinator()
    def query(command, **kwargs):
        assert command[:3] == ['nvidia-smi', '-i', '0']
        return '1234\n'
    monkeypatch.setattr(module.subprocess, 'check_output', query)
    assert not module.gpu0_idle()
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *args, **kwargs: '')
    assert module.gpu0_idle()


def test_handoff_validation_rejects_changed_generation_protocol(monkeypatch, tmp_path):
    module = coordinator()
    config = {'family': 'deepseek', 'split': 'train', 'offset': 0, 'components': 321,
              'seed': 1701, 'decode_batch_size': 1, 'max_new_tokens': 1024, 'max_input_tokens': 4096}
    monkeypatch.setattr(module, 'load_bank', lambda path: (config, [{}]*321, 'complete-hash'))
    assert module.validate_bank(tmp_path, 'rbr-deepseek-train321') == 'complete-hash'
    with pytest.raises(ValueError, match='identity differs'):
        module.validate_bank(tmp_path, 'rbr-deepseek-train321', {'model': 'different-model'})
    config['decode_batch_size'] = 8
    with pytest.raises(ValueError, match='identity differs'):
        module.validate_bank(tmp_path, 'rbr-deepseek-train321')


def test_largest_remaining_public_inventory_goes_to_first_free_gpu():
    module = coordinator()
    assert module.remaining_banks(['rbr-deepseek-development-pilot16'], 'rbr-deepseek-train321') == [
        'rbr-deepseek-primary500', 'rbr-deepseek-development384']


def test_complete_coordinator_flow_adopts_current_bank_and_assigns_disjoint_gpus(tmp_path, monkeypatch):
    """Simulated bank execution; real process parking is covered separately above."""
    import json
    module = coordinator()
    root = Path(module.__file__).resolve().parents[1]
    run, output, public = tmp_path/'run', tmp_path/'handoff', tmp_path/'public'
    run.mkdir(); public.mkdir()
    (public/'tasks.jsonl').write_text('fixture tasks')
    (public/'manifest.json').write_text('{}')
    proof = run/'deepseek_verified_manifest.json'
    proof.write_text('fixture model proof')
    names = ['scripts/run_local_rbr_replication_generation.py', 'scripts/generate_apbpf_rbr_replication_bank.py',
             'scripts/apbpf_generator_sandbox.sh', 'src/pbpf/apbpf/rbr_prompt.py', 'src/pbpf/apbpf/codearc_bank.py']
    parent = {'pid': os.getpid(), 'state': 'S', 'ppid': 1, 'start_ticks': 'fixture-parent',
              'command': 'python scripts/run_local_rbr_replication_generation.py', 'children': [42]}
    child = {'pid': 42, 'state': 'R', 'ppid': os.getpid(), 'start_ticks': 'fixture-child', 'children': [],
             'command': f'bwrap {run}/rbr-deepseek-train321 scripts/generate_apbpf_rbr_replication_bank.py'}
    original = {'status': 'running', 'pid': parent['pid'], 'public_root': str(public),
                'source_sha256': {n: module.file_sha(root/n) for n in names},
                'public_tasks_sha256': module.file_sha(public/'tasks.jsonl'),
                'model_proof_sha256': module.file_sha(proof), 'current': 'rbr-deepseek-train321',
                'runs': {'rbr-deepseek-development-pilot16': {'status': 'complete'},
                         'rbr-deepseek-train321': {'status': 'running'}}}
    (run/'rbr_deepseek_generation_status.json').write_text(json.dumps(original))
    (run/'rbr_qwen_generation_status.json').write_text(json.dumps({'status': 'complete', 'pid': 43}))
    for name in ['rbr-deepseek-development-pilot16', 'rbr-deepseek-train321', 'rbr-qwen-primary500']:
        (run/name).mkdir()
    (run/'rbr-deepseek-development-pilot16/complete.json').write_text('{}')
    launched, validated = [], []
    retired = False
    def info(pid):
        if pid == parent['pid']:
            return None if retired else dict(parent)
        if pid == child['pid']:
            if launched:
                child['state'] = 'Z'
                (run/'rbr-deepseek-train321/complete.json').write_text('{}')
            return dict(child)
        return None
    def park(expected):
        assert expected['pid'] == parent['pid']
        parent['state'] = 'T'
        return dict(parent)
    def retire(expected):
        nonlocal retired
        assert child['state'] == 'Z' and parent['state'] == 'T'
        assert 'rbr-deepseek-train321' in validated
        retired = True
        return {'retired_parent': expected, 'terminal_children': [dict(child)]}
    def bank(path):
        path = Path(path)
        if path.name == 'rbr-qwen-primary500':
            return {'family': 'qwen'}, [{}]*500, 'fixture-qwen-complete'
        assert (path/'complete.json').exists()
        plan = json.loads((output/'plan.json').read_text())
        split, count, offset = module.BANKS[path.name]
        config = {**plan['expected_bank_identity'], 'family': 'deepseek', 'split': split,
                  'components': count, 'offset': offset, 'seed': 1701, 'decode_batch_size': 1,
                  'max_new_tokens': 1024, 'max_input_tokens': 4096}
        validated.append(path.name)
        return config, [{}]*count, 'fixture-complete-'+path.name
    class Job:
        def __init__(self, command, **kwargs):
            self.bank = Path(command[3])
            gpu = int(kwargs['env']['CUDA_VISIBLE_DEVICES'])
            launched.append((gpu, self.bank.name))
            self.pid = 1000+len(launched)
            self.calls = 0
        def poll(self):
            self.calls += 1
            if self.calls < 3:
                return None
            (self.bank/'complete.json').write_text('{}')
            return 0
        def wait(self):
            return 0
    monkeypatch.chdir(root)
    monkeypatch.setattr(module, 'process_info', info)
    monkeypatch.setattr(module, 'park_parent', park)
    monkeypatch.setattr(module, 'retire_parked_parent', retire)
    monkeypatch.setattr(module, 'gpu0_idle', lambda: True)
    monkeypatch.setattr(module, 'load_bank', bank)
    monkeypatch.setattr(module.subprocess, 'Popen', Job)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(sys, 'argv', ['coordinator', '--run-root', str(run), '--output', str(output)])
    module.main()
    assert retired
    assert launched == [(0, 'rbr-deepseek-primary500'), (1, 'rbr-deepseek-development384')]
    assert json.loads((output/'status.json').read_text())['status'] == 'complete'
    final = json.loads((run/'rbr_deepseek_generation_status.json').read_text())
    assert final['status'] == 'complete' and final['gpu_assignments'] == {}
    assert all(final['runs'][name]['status'] == 'complete' for name in module.BANKS)
    assert (output/'handoff.json').exists() and (output/'retired-original-parent.json').exists()
