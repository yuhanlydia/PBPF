"""Seed-level reporting must not inflate replication or hide missing runs."""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('render_eesd_tables', ROOT / 'scripts/render_eesd_tables.py')
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def write_report(root, seed, fixed, *, model='qwen25_7b'):
    path = root / 'mechanism' / 'runbugrun' / model / f'seed{seed}' / 'report.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {name: {'nll': fixed + delta} for name, delta in {
        'ordinary': 1, 'fixed': 0, 'effective': -0.5, 'global_mass': 0.25,
        'effective_at_fixed_params': -0.25, 'fixed_at_effective_params': 0,
        'constant_mean_mass': 0, 'binary_fixed_at_effective_params': 0,
        'binary_effective': -0.5,
    }.items()}
    metrics['permuted_mass'] = {'mean_nll': fixed, 'std_nll': 0, 'runs': []}
    path.write_text(json.dumps({'schema': 'eesd-evidence-matrix-v1', 'metrics': metrics,
        'counts': {'assessment_examples': 1 if seed == 1701 else 1000},
        'bootstraps': {key: {'gain': value} for key, value in {
            'effective_vs_fixed_at_effective_params': 0.5,
            'effective_vs_fixed_at_fixed_params': 0.25,
            'effective_vs_tuned_global_mass': 0.75,
        }.items()}}))


def test_mechanism_aggregates_seeds_without_weighting_by_query_count(tmp_path):
    for seed, fixed in [(1701, 1), (1702, 2), (1703, 3)]:
        write_report(tmp_path, seed, fixed)
    coverage = {'mechanism': {}}
    text = '\n'.join(renderer.mechanism_table(tmp_path, coverage))
    cell = coverage['mechanism']['runbugrun/qwen25_7b']
    assert cell['complete'] is True
    assert cell['missing_seeds'] == []
    assert cell['summary']['fixed_nll'] == {'n': 3, 'mean': 2.0, 'std': 1.0}
    assert len(cell['runs']) == 3
    assert '2.0000 $\\pm$ 1.0000' in text
    assert 'Ordinary NLL' in text
    assert len(coverage['mechanism']) == 12
    assert sum(len(c['runs']) for c in coverage['mechanism'].values()) == 36
    absent = coverage['mechanism']['codearc/qwen3_coder_30b']
    assert absent['missing_seeds'] == [1701, 1702, 1703]
    assert absent['complete'] is False


def test_single_seed_is_incomplete_and_has_no_estimated_standard_deviation(tmp_path):
    write_report(tmp_path, 1701, 1)
    coverage = {'mechanism': {}}
    text = '\n'.join(renderer.mechanism_table(tmp_path, coverage))
    cell = coverage['mechanism']['runbugrun/qwen25_7b']
    assert cell['complete'] is False
    assert cell['missing_seeds'] == [1702, 1703]
    assert cell['summary']['fixed_nll'] == {'n': 1, 'mean': 1.0, 'std': None}
    assert '1.0000 (n=1)' in text
    assert cell['runs']['1702']['present'] is False


def test_unseeded_legacy_report_does_not_satisfy_seed_coverage(tmp_path):
    write_report(tmp_path, 1701, 1)
    path = tmp_path / 'mechanism/runbugrun/qwen25_7b'
    (path / 'seed1701/report.json').rename(path / 'report.json')
    coverage = {'mechanism': {}}
    renderer.mechanism_table(tmp_path, coverage)
    assert coverage['mechanism']['runbugrun/qwen25_7b']['missing_seeds'] == [1701, 1702, 1703]


def test_causal_ablation_reads_all_seed_reports(tmp_path):
    for seed, fixed in [(1701, 1), (1702, 2), (1703, 3)]:
        write_report(tmp_path, seed, fixed)
    coverage = {'tables': {}}
    text = '\n'.join(renderer.ablation_table(tmp_path, coverage))
    assert coverage['tables']['causal_ablation']['complete'] is True
    assert '2.0000 $\\pm$ 1.0000' in text
    assert '0.5000 $\\pm$ 0.0000' in text


def test_config_selects_four_model_families_without_obsolete_missing_cells(tmp_path, monkeypatch):
    import sys
    config = tmp_path / 'config.yaml'
    config.write_text('''schema: eesd-iclr2027-v1
seeds: [1701, 1702, 1703]
models:
  primary: configs/models/qwen2.5-coder-7b.yaml
  cross_family: configs/models/deepseek-coder-6.7b.yaml
  second_cross_family: configs/models/seed-coder-8b.yaml
  third_cross_family: configs/models/starcoder2-15b.yaml
''')
    output = tmp_path / 'table.tex'
    monkeypatch.setattr(sys, 'argv', ['renderer', '--results', str(tmp_path), '--config', str(config), '--output', str(output)])
    renderer.main()
    coverage = json.loads(output.with_suffix('.coverage.json').read_text())['mechanism']
    assert len(coverage) == 8
    assert 'codearc/starcoder2_15b' in coverage
    assert 'runbugrun/qwen3_8b' not in coverage
    assert 'StarCoder2-15B' in output.read_text()
