from types import SimpleNamespace

import pytest
import torch

from test_stage_training import module


def test_uniform_resampled_weights_do_not_hide_complete_ancestry_collapse():
    worker = module('particle_diversity_test', 'run_apbpf_particle_diagnostic.py')
    trace = SimpleNamespace(log_weights=torch.zeros(1, 2, 4),
        resampling_indices=torch.tensor([[[2, 2, 2, 2], [0, 1, 2, 3]]]),
        resampled=torch.tensor([[True, False]]))
    rows = worker.diversity(trace)
    assert all(r['mean_ess'] == pytest.approx(4.) for r in rows)
    assert all(r['mean_unique_initial_particles'] == 1. for r in rows)
    assert all(r['single_ancestor_fraction'] == 1. for r in rows)
    assert rows[0]['resampled_fraction'] == 1. and rows[1]['resampled_fraction'] == 0.


def test_identity_resampling_retains_every_initial_particle():
    worker = module('particle_identity_test', 'run_apbpf_particle_diagnostic.py')
    trace = SimpleNamespace(log_weights=torch.zeros(2, 4, 8),
        resampling_indices=torch.arange(8).expand(2, 4, -1), resampled=torch.zeros(2, 4, dtype=torch.bool))
    assert all(row['mean_unique_fraction'] == 1. for row in worker.diversity(trace))
