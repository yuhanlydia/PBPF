"""Seed compatibility generators cannot silently launch a different model."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('script', ['generate_apbpf_seed_rbr_bank.py', 'generate_apbpf_seed_codearc_bank.py'])
def test_seed_generator_rejects_other_families_before_loading(tmp_path, script):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--public-root', str(tmp_path),
        '--output', str(tmp_path / 'out'), '--split', 'development', '--family', 'qwen25_7b'],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr
    assert 'seed_coder_8b' in result.stderr
    assert not (tmp_path / 'out').exists()
