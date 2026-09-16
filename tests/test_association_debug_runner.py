"""Keep synthetic checks distinct from real-data evidence and avoid test tuning."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]


def driver():
    spec = importlib.util.spec_from_file_location('association_debug_runner', ROOT / 'scripts/run_association_debug.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controlled_histograms_cannot_reveal_diagnosis_but_pairs_can():
    batches, metadata = driver().controlled_batches(32, 16, 16, 17, torch.device('cpu'))
    assert set(batches) == {'train', 'development'}
    for b in batches.values():
        assert (b.outcomes[:, :4] == 0).sum(1).eq(2).all()
        assert (b.outcomes[:, :4] == 1).sum(1).eq(2).all()
        # One signed test/outcome pair reveals the hidden trigger direction.
        direction = b.tests[:, 0, 0].sign() * (2 * b.outcomes[:, 0] - 1)
        expected = (b.tests[:, 4:, 0] * direction[:, None] > 0).long()
        torch.testing.assert_close(expected, b.outcomes[:, 4:])
    assert metadata['scientific_claim'] == 'controlled_learnability_only'


def test_driver_writes_auditable_failed_or_passed_development_result(tmp_path):
    output = tmp_path / 'run'
    command = [sys.executable, str(ROOT / 'scripts/run_association_debug.py'),
               '--source', 'controlled', '--arm', 'history_is', '--output', str(output),
               '--steps', '2', '--train-size', '16', '--dev-size', '8',
               '--feature-dim', '16', '--hidden-dim', '12', '--latent-dim', '4',
               '--difficulty-dim', '2', '--particles', '4', '--device', 'cpu',
               '--bootstrap-replicates', '20', '--eval-particles', '8', '--shuffle-train-tests']
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / 'result.json').read_text())
    assert report['scope'] == 'development_only_exploratory'
    assert report['data']['scientific_claim'] == 'controlled_learnability_only'
    assert report['test_evaluated'] is False
    assert report['device'] == 'cpu'
    assert report['evaluation_particles'] == 8
    assert report['training_views'] == 'uniform_pair_permutation'
    assert report['selection']['development_gate_passed'] in (True, False)
    assert report['training_history'][-1]['step'] == 2
    assert (output / 'model.pt').is_file()
    assert (output / 'checksums.json').is_file()
    assert (output / 'baselines/pair_aware.pt').is_file()
    assert report['baselines']['pair_aware']['selected_step'] in (1, 2)
    assert 'association_gap' in report['baselines']['deterministic_interaction']['development']
    assert 'development_bootstrap' in report['baselines']['deterministic_interaction']
    assert (output / 'source/src/pbpf/belief/diagnostic.py').is_file()
    # Re-running cannot overwrite a previous result or silently resume it.
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert second.returncode != 0


def test_legacy_nondefault_coefficients_are_not_silently_mislabeled(tmp_path):
    args = type('Args', (), dict(arm='legacy', association_weight=2., evidence_weight=1.,
        invariance_weight=.1, steps=1, particles=2, batch_size=2, bootstrap_replicates=20,
        output=tmp_path / 'bad'))()
    with pytest.raises(ValueError, match='legacy'):
        driver().run(args)


def test_deterministic_controls_detect_pair_signal_without_order_signal():
    module = driver()
    batches, _ = module.controlled_batches(32, 64, 16, 17, torch.device('cpu'))

    class SignOracle(torch.nn.Module):
        def forward(self, batch):
            evidence = (batch.tests[:, :4, 0].sign() * (2 * batch.outcomes[:, :4] - 1)).mean(1)
            score = 5 * evidence[:, None] * batch.tests[:, 4:, 0].sign()
            logits = torch.full((*score.shape, 5), -20.)
            logits[:, :, 0], logits[:, :, 1] = -score, score
            return logits

    metrics, _ = module.evaluate_deterministic(SignOracle(), batches['development'], 13)
    assert metrics['aligned_nll'] < .001
    assert metrics['association_gap'] > 1.
    assert metrics['pair_order_effect'] < 1e-6
