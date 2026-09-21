"""Regression coverage for the mechanism CLI and its generation boundary."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('seed,expected', [(None, [1701, 1702, 1703]), (1702, [1702])])
def test_mechanism_generation_honors_seeds_and_separates_outputs(tmp_path, monkeypatch, seed, expected):
    runner = load_script('run_eesd_matrix')
    manifest = tmp_path / 'manifest.yaml'
    manifest.write_text(yaml.safe_dump({'schema': 'eesd-cache-manifest-v1', 'mechanism_cells': [{
        'dataset': 'RunBugRun', 'model': 'qwen', 'family': 'qwen', 'domain': 'rbr',
        'public_root': str(tmp_path), 'evaluator_root': str(tmp_path),
    }]}))
    commands = []
    # GPU generation/execution is the external boundary; planning remains real.
    monkeypatch.setattr(runner, 'run', lambda command, **kwargs: commands.append(command))
    # Integrity is exercised with real sealed artifacts in test_mechanism_integrity.
    monkeypatch.setattr(runner, 'verify_mechanism_bank', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, 'verify_mechanism_cache', lambda *args, **kwargs: None)
    argv = ['runner', '--config', str(ROOT / 'configs/experiments/eesd_iclr2027.yaml'),
            '--manifest', str(manifest), '--output', str(tmp_path / 'out'), '--stage', 'mechanism']
    if seed is not None:
        argv += ['--seed', str(seed)]
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    generation = [c for c in commands if 'scripts/generate_apbpf_rbr_bank.py' in c]
    assert [int(c[c.index('--seed') + 1]) for c in generation] == [s for s in expected for _ in range(2)]
    assert len({c[c.index('--output') + 1] for c in generation}) == 2 * len(expected)
    reports = [c for c in commands if 'scripts/run_eesd_evidence_matrix.py' in c]
    assert [int(c[c.index('--seed') + 1]) for c in reports] == expected
    assert all('--greedy' not in c for c in generation)


def test_shell_wrapper_fails_for_missing_manifest(tmp_path):
    import os
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_eesd_iclr.sh')],
        env={**os.environ, 'EESD_MANIFEST': str(tmp_path / 'absent.yaml')}, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Missing' in result.stderr


@pytest.mark.parametrize("fail_first", [False, True])
def test_evidence_runs_canonical_config_and_reports_missing_history(tmp_path, monkeypatch, fail_first):
    runner = load_script('run_eesd_evidence_matrix')
    config = yaml.safe_load((ROOT / 'configs/experiments/eesd_iclr2027.yaml').read_text())
    config['evidence'].update(global_mass_grid=[1, 4], ece_bins=10,
        permutation_seeds=[1701, 1702, 1703], concentration_bins_for_n4=[1, 2, 3, 4], bootstrap_draws=5)
    config['bootstrap_seed'] = 314159
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    records = [{'split': split, 'problem_id': split,
        'tests': [{'input': str(i)} for i in range(6)],
        'outcomes': ['PASS', 'PASS', 'PASS', 'PASS', 'PASS', 'PASS']}
        for split in ('development', 'primary')]
    cache = tmp_path / 'cache.json'
    cache.write_text(json.dumps({'records': records}))
    output = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['runner', '--config', str(config_path), '--cache', str(cache),
        '--dataset', 'RunBugRun', '--model', 'qwen', '--seed', '1702', '--output', str(output)])
    if fail_first:
        original_write = runner.write_json
        def fail_write(*args, **kwargs):
            raise RuntimeError("simulated write failure")
        monkeypatch.setattr(runner, 'write_json', fail_write)
        with pytest.raises(RuntimeError, match="simulated write failure"):
            runner.main()
        assert not output.exists()
        failures = list(tmp_path.glob('out.partial-*/failure.json'))
        assert len(failures) == 1
        assert 'simulated write failure' in failures[0].read_text()
        monkeypatch.setattr(runner, 'write_json', original_write)
    runner.main()
    report = json.loads((output / 'report.json').read_text())
    assert report['seed'] == 1702
    assert len(report['factorial']) == 32
    assert [r['history_size'] for r in report['history_sweep']] == [1, 2, 4, 8]
    assert report['history_sweep'][-1]['status'] == 'unavailable'
    assert report['history_sweep'][-1]['reason']
    assert set(report['same_alpha']) == {'0.10', '0.01'}
    assert all('paired_bootstrap' in row for rows in report['same_alpha'].values() for row in rows)
    assert (output / 'complete.json').exists()
    assert sum(b['count'] for b in report['concentration_bins']) == report['counts']['assessment_examples']
    assert report['selected']['constant_mean_effective_mass'] == pytest.approx(4.0)


def test_output_transaction_preserves_prior_attempt_and_completed_result(tmp_path):
    runner = load_script('run_eesd_evidence_matrix')
    destination = tmp_path / 'report'
    destination.mkdir()
    (destination / 'partial.json').write_text('previous attempt')
    with runner.output_transaction(destination) as staging:
        (staging / 'complete.json').write_text('{}')
    archived = list(tmp_path.glob('report.incomplete-*/previous-output/partial.json'))
    assert len(archived) == 1
    assert archived[0].read_text() == 'previous attempt'
    with pytest.raises(FileExistsError):
        with runner.output_transaction(destination):
            pytest.fail('completed output must never be entered')
    assert (destination / 'complete.json').read_text() == '{}'


def test_source_split_overlap_rejected(tmp_path, monkeypatch):
    runner = load_script('run_eesd_evidence_matrix')
    rows = [{'split': split, 'source_component_id': 'shared',
             'tests': [{'input': 'x'}] * 5, 'outcomes': ['PASS'] * 5}
            for split in ('development', 'primary')]
    cache = tmp_path / 'cache.json'
    cache.write_text(json.dumps({'records': rows}))
    monkeypatch.setattr(sys, 'argv', ['runner', '--config', str(ROOT / 'configs/experiments/eesd_iclr2027.yaml'),
        '--cache', str(cache), '--dataset', 'd', '--model', 'm', '--output', str(tmp_path / 'out')])
    with pytest.raises(ValueError, match='source.*overlap'):
        runner.main()
    assert not (tmp_path / 'out').exists()


def test_seed_has_explicit_generator_profile_without_changing_other_families():
    module = load_script('run_eesd_matrix')
    for domain in ('rbr', 'codearc'):
        assert module.mechanism_generator_script(domain, 'seed_coder_8b') == f'scripts/generate_apbpf_seed_{domain}_bank.py'
        for family in ('qwen25_7b', 'deepseek_6p7b', 'starcoder2_15b'):
            assert module.mechanism_generator_script(domain, family) == f'scripts/generate_apbpf_{domain}_bank.py'
    with pytest.raises(ValueError):
        module.mechanism_generator_script('typo', 'seed_coder_8b')
