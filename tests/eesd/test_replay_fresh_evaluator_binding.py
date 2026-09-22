"""Replay fresh evaluation reads the public population binding from each bank schema."""
from pathlib import Path
import runpy


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/evaluate_eesd_replay_fresh_bank.py'


def test_public_tasks_digest_reads_nested_generation_binding():
    digest = runpy.run_path(str(SCRIPT))['public_tasks_digest']
    assert digest({'bindings': {'public_tasks_sha256': 'nested'},
                   'public_tasks_sha256': 'stale'}) == 'nested'


def test_public_tasks_digest_accepts_legacy_gemma_bank():
    digest = runpy.run_path(str(SCRIPT))['public_tasks_digest']
    assert digest({'public_tasks_sha256': 'legacy'}) == 'legacy'
