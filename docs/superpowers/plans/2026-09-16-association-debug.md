# Association Debug Implementation Plan

**Goal:** Implement and measure an honest diagnostic reference path.
**Architecture:** Add a causal Deep Sets prefix-IS model, normalized one-sided
association training with latent MMD, and a development-only ablation runner.
**Tech Stack:** Python 3.11, PyTorch, NumPy, existing PBPF feature/cache utilities.
**Spec:** `docs/superpowers/specs/2026-09-16-association-debug-design.md`

## Constraints

Keep legacy APIs unchanged. Label new inference as prefix importance sampling,
not FIVO. Preserve original pilot artifacts. Do not evaluate test during tuning.
Use real computation and immutable experiment outputs; report failed gates.

## Tasks

- [x] Add `tests/belief/test_diagnostic_reference.py` with causal/permutation,
  hand-computed importance, MMD, gradient, units, and checkpoint-selection tests.
  Run `PYTHONPATH=src /root/PBPF/.venv/bin/python -m pytest
  tests/belief/test_diagnostic_reference.py -q` and record expected failures.
- [x] Implement `src/pbpf/belief/diagnostic.py`: `HistoryISBeliefModel`,
  `weighted_latent_mmd`, `diagnostic_objective`, `train_diagnostic_step`, and
  `select_diagnostic_checkpoint`. Re-run the new tests, then legacy belief/APBPF
  tests. Proposal uses visible pair mean plus histogram/count, and weights
  include prior/proposal density correction.
- [x] Add `scripts/run_association_debug.py` and behavioral runner tests.
  Controlled data use balanced binary triggers with code-independent diagnosis;
  real mode validates the rich cache and tensorizes train/development only.
  Run development-only ablations `legacy`, `objective`, `history_is`, keeping
  feature dimension, training seed, steps and particle counts recorded.
- [x] Execute the controlled positive control and real development ablations.
  Save configuration, input/source hashes, selected checkpoints, histories,
  baseline comparisons, and selection status. Do not promote synthetic results.
- [x] Review code and scientific claims, run required tests, write a final
  evidence-backed report with limitations and reproducible commands.

## Execution outcome

Implemented and reviewed the reference path, then added an explicitly ablated bilinear interaction after the controlled concatenation model failed to learn. Six runs completed; controlled interaction replicated across two seeds, real development association remained negative. See `results/ASSOCIATION_DEBUG_2026-09-16.md`. Full verification: 723 passed, 1 optional paper-build test skipped. Branch remains isolated for review.
