"""The downstream entry point must preflight science settings before launching."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from test_mechanism_runner import ROOT, load_script


@pytest.fixture
def inputs(tmp_path):
    config = yaml.safe_load((ROOT / 'configs/experiments/eesd_iclr2027.yaml').read_text())
    config['distillation']['transition_utility'] = [1, -1, 0, 0]  # Test-only science fixture.
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(yaml.safe_dump({'schema': 'eesd-cache-manifest-v1'}))
    return config_path, manifest, tmp_path / 'out'


@pytest.mark.parametrize('utility', [None, [1, -1, 0], [1, -1, 0, float('nan')], [1, -1, 0, True]])
def test_cli_rejects_missing_or_invalid_utility_before_job_launch(inputs, utility):
    config_path, manifest, output = inputs
    config = yaml.safe_load(config_path.read_text())
    if utility is None:
        del config['distillation']['transition_utility']
    else:
        config['distillation']['transition_utility'] = utility
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_eesd_downstream.py'),
        '--config', str(config_path), '--manifest', str(manifest), '--output', str(output), '--stage', 'fresh'],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'transition_utility' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('stage,expected', [
    ('fresh', ['no_update', 'equal_weight', 'final_correctness', 'scalar_confidence',
        'fixed_mass_dirichlet', 'eed_mean_no_uncertainty', 'eed_no_anchor', 'eesd_full']),
])
def test_downstream_routes_complete_rule_matrix_and_records_frozen_inputs(inputs, monkeypatch, stage, expected):
    runner = load_script('run_eesd_downstream')
    config, manifest, output = inputs
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, 'run', launch)
    monkeypatch.setattr(sys, 'argv', ['runner', '--config', str(config), '--manifest', str(manifest),
        '--output', str(output), '--stage', stage, '--seed', '1702', '--max-steps', '25'])
    runner.main()
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index('--rules') + 1:] == expected
    assert command[command.index('--seed') + 1] == '1702'
    assert command[command.index('--max-steps') + 1] == '25'
    snapshot_config = Path(command[command.index('--config') + 1])
    snapshot_manifest = Path(command[command.index('--manifest') + 1])
    assert snapshot_config.read_bytes() == config.read_bytes()
    assert snapshot_manifest.read_bytes() == manifest.read_bytes()
    record = json.loads(next(output.glob('downstream-launches/*/launch.json')).read_text())
    assert record['status'] == 'completed'
    assert record['parameters']['rules'] == expected
    assert record['parameters']['transition_utility'] == [1, -1, 0, 0]
    assert len(record['input_sha256']['config']) == 64
    assert 'scripts/run_eesd_matrix.py' in record['source_sha256']
    assert record['command'] == command


def test_downstream_records_failed_child_without_claiming_completion(inputs, monkeypatch):
    runner = load_script('run_eesd_downstream')
    config, manifest, output = inputs
    monkeypatch.setattr(runner.subprocess, 'run', lambda command, **kwargs: subprocess.CompletedProcess(command, 7))
    monkeypatch.setattr(sys, 'argv', ['runner', '--config', str(config), '--manifest', str(manifest),
        '--output', str(output), '--stage', 'fresh'])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 7
    record = json.loads(next(output.glob('downstream-launches/*/launch.json')).read_text())
    assert record['status'] == 'failed'
    assert record['returncode'] == 7


@pytest.mark.parametrize('stage,budget,message', [('train', None, 'explicit'), ('train', 0, 'positive'),
                                                ('recursive', 100, 'experience-policy')])
def test_budget_requires_explicit_train_value_and_rejects_recursive(inputs, stage, budget, message):
    config, manifest, output = inputs
    command = [sys.executable, str(ROOT / 'scripts/run_eesd_downstream.py'), '--config', str(config),
               '--manifest', str(manifest), '--output', str(output), '--stage', stage]
    if budget is not None:
        command += ['--response-token-budget', str(budget)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


@pytest.fixture
def training_inputs(inputs, tmp_path):
    config, manifest, output = inputs
    model = tmp_path / 'model.yaml'
    model.write_text(yaml.safe_dump({'model_id': 'test/model', 'revision': 'a' * 40}))
    manifest.write_text(yaml.safe_dump({'schema': 'eesd-cache-manifest-v1', 'correction_cells': [
        {'dataset': 'runbugrun', 'model': 'qwen25_7b', 'round': 1, 'model_config': str(model)}]}))
    scored = output / 'corrections/runbugrun/qwen25_7b/round1/scored-corrections.jsonl'
    scored.parent.mkdir(parents=True)
    from test_zero_training_outcomes import scored_rows
    scored.write_text(''.join(json.dumps(row)+'\n' for row in scored_rows(train_weight=1.)))
    return config, manifest, output


def training_argv(inputs):
    config, manifest, output = inputs
    return ['runner', '--config', str(config), '--manifest', str(manifest), '--output', str(output),
            '--stage', 'train', '--seed', '1702', '--response-token-budget', '53']


def fake_training_job(command, **kwargs):
    import hashlib
    def value(flag):
        return command[command.index(flag) + 1]
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    output = Path(value('--output'))
    (output / 'adapter').mkdir(parents=True)
    (output / 'adapter/adapter_model.safetensors').write_bytes(b'adapter fixture')
    budget = int(value('--response-token-budget'))
    rule = value('--rule')
    anchor = float(value('--anchor-beta')) if rule in {'fixed_mass_dirichlet', 'eed_mean_no_uncertainty', 'eesd_full'} else 0.
    report = {'input_sha256': sha(value('--input')), 'model_config_sha256': sha(value('--model-config')),
        'seed': int(value('--seed')), 'rule': rule, 'response_token_budget': budget, 'response_tokens': budget,
        'trainer_source_sha256': sha(ROOT / 'scripts/run_eesd_weighted_sft.py'), 'anchor_beta': anchor,
        'previous_adapter': value('--previous-adapter') if '--previous-adapter' in command else None}
    from pbpf.eesd.training_outcomes import protocol_binding
    report.update(protocol_binding(ROOT))
    (output / 'training-report.json').write_text(json.dumps(report))
    return subprocess.CompletedProcess(command, 0)


def test_train_routes_seven_explicit_token_budgets_and_checks_sealed_resume(training_inputs, monkeypatch):
    runner = load_script('run_eesd_downstream')
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        return fake_training_job(command, **kwargs)
    monkeypatch.setattr(runner.subprocess, 'run', launch)
    monkeypatch.setattr(sys, 'argv', training_argv(training_inputs))
    runner.main()
    assert [c[c.index('--rule') + 1] for c in commands] == [
        'equal_weight', 'final_correctness', 'scalar_confidence', 'fixed_mass_dirichlet',
        'eed_mean_no_uncertainty', 'eed_no_anchor', 'eesd_full']
    assert all(Path(c[1]).name == 'run_eesd_weighted_sft.py' for c in commands)
    assert all(c[c.index('--response-token-budget') + 1] == '53' for c in commands)
    output = training_inputs[2]
    bindings = list(output.glob('training/runbugrun/qwen25_7b/round1/*/seed1702/training-binding.json'))
    assert len(bindings) == 7
    assert all(len(json.loads(p.read_text())['adapter_tree_sha256']) == 64 for p in bindings)
    commands.clear()
    runner.main()
    assert commands == []
    adapter = bindings[0].parent / 'adapter/adapter_model.safetensors'
    adapter.write_bytes(b'changed adapter')
    with pytest.raises(ValueError, match='binding|checksum'):
        runner.main()
    assert commands == []


def test_unbound_existing_training_report_is_rejected_before_any_job(training_inputs, monkeypatch):
    runner = load_script('run_eesd_downstream')
    output = training_inputs[2] / 'training/runbugrun/qwen25_7b/round1/eesd_full/seed1702'
    output.mkdir(parents=True)
    (output / 'training-report.json').write_text('{}')
    commands = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda *args, **kwargs: commands.append(args))
    monkeypatch.setattr(sys, 'argv', training_argv(training_inputs))
    with pytest.raises(ValueError, match='unbound|binding'):
        runner.main()
    assert commands == []


@pytest.mark.parametrize('change', ['budget', 'input', 'model_config', 'trainer_source', 'seed', 'rule'])
def test_training_resume_rejects_changed_request_or_report_identity(training_inputs, monkeypatch, change):
    runner = load_script('run_eesd_downstream')
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        return fake_training_job(command, **kwargs)
    monkeypatch.setattr(runner.subprocess, 'run', launch)
    argv = training_argv(training_inputs)
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    commands.clear()
    config, manifest, output = training_inputs
    if change == 'budget':
        monkeypatch.setattr(sys, 'argv', [*argv[:-1], '54'])
    elif change == 'input':
        scored = output / 'corrections/runbugrun/qwen25_7b/round1/scored-corrections.jsonl'
        rows = [json.loads(line) for line in scored.read_text().splitlines()]
        rows[0]['prompt'] += ' changed'
        scored.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif change == 'model_config':
        model = Path(yaml.safe_load(manifest.read_text())['correction_cells'][0]['model_config'])
        model.write_text(model.read_text() + '\n# changed pinned configuration\n')
    elif change == 'trainer_source':
        original_digest = runner.file_digest
        monkeypatch.setattr(runner, 'file_digest', lambda path: '0' * 64
            if Path(path).name == 'run_eesd_weighted_sft.py' else original_digest(path))
    else:
        report_path = output / 'training/runbugrun/qwen25_7b/round1/equal_weight/seed1702/training-report.json'
        report = json.loads(report_path.read_text())
        report[change] = 1703 if change == 'seed' else 'eesd_full'
        report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='binding'):
        runner.main()
    assert commands == []


@pytest.mark.parametrize('policy',['arm-specific','shared_eesd_teacher'])
def test_downstream_recursive_forwards_explicit_budget_and_policy(inputs, tmp_path, monkeypatch,policy):
    runner = load_script('run_eesd_downstream')
    config, manifest, output = inputs
    model = tmp_path / 'model.yaml'
    model.write_text('model_id: test/model\nrevision: ' + 'a' * 40 + '\n')
    manifest.write_text(yaml.safe_dump({'schema': 'eesd-cache-manifest-v1', 'recursive_cells': [
        {'dataset': 'runbugrun', 'model': 'qwen25_7b', 'domain': 'rbr', 'family': 'qwen25_7b',
         'model_config': str(model), 'public_root': str(tmp_path), 'evaluator_root': str(tmp_path), 'rounds': 3}]}))
    commands = []
    def launch(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, 'run', launch)
    monkeypatch.setattr(sys, 'argv', ['runner', '--config', str(config), '--manifest', str(manifest),
        '--output', str(output), '--stage', 'recursive', '--seed', '1702',
        '--response-token-budget', '53', '--experience-policy', policy])
    runner.main()
    assert len(commands) == 1
    command = commands[0]
    assert Path(command[1]).name == 'run_eesd_recursive.py'
    assert command[command.index('--response-token-budget') + 1] == '53'
    assert command[command.index('--experience-policy') + 1] == policy
    assert command[command.index('--output') + 1] == str(output / 'recursive/runbugrun/qwen25_7b/seed1702')
