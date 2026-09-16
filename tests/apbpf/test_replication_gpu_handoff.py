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
