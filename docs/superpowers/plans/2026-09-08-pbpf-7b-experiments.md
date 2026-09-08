# PBPF 7B Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone PBPF package for exact finite posterior tests, frozen 7B outcome prediction and fixed-order multi-round repair.

**Architecture:** A lightweight offline core defines manifests, immutable trajectory banks and particle mathematics; optional torch/Transformers/PEFT modules provide learned belief, coherent particle-conditioned repair and QLoRA entry points. Every baseline consumes the same frozen candidates, test order and budget.

**Tech Stack:** Python 3.11/3.12, NumPy, SciPy, PyYAML; optional PyTorch, Transformers, PEFT, bitsandbytes, datasets.

**Spec:** `docs/superpowers/specs/2026-09-08-pbpf-7b-experiments-design.md`

## Global Constraints

- Package/CLI names are `pbpf` and `pbpf-run`; there is no JAG/GOAV code.
- Sequence generation samples one particle once; token-level re-mixture exists only as a named fault ablation.
- Tests are fixed-order and equal-budget across arms; gold/future outcomes are evaluator-side only.
- Formal configs pin model/data/container revisions and cannot run on the 16/24GB pilot profiles.
- No empirical gain is claimed by smoke or configuration validation.

---

### Task 1: Package, registries and protocol validation

**Files:** create `pyproject.toml`, `README.md`, `src/pbpf/{__init__,__main__,cli,config,registry,schema}.py`, `tests/test_registry.py`.

**Interfaces:** produce `load_experiment`, `validate_experiment`, frozen model/dataset/baseline specs, and `pbpf-run doctor|plan`.

- [ ] Write a failing test for all exact models, benchmark roles, outcomes, baseline provenance modes and formal-profile restrictions.
- [ ] Run the test and confirm missing package failure.
- [ ] Implement canonical config hashing and lazy dependency checks with no import-time torch/download side effects.
- [ ] Run the test and commit `feat: scaffold strict PBPF experiments`.

### Task 2: Leakage-safe manifests and trajectory banks

**Files:** create `src/pbpf/{manifest,bank,ledger,sandbox}.py`, `tests/test_data_protocol.py`.

**Interfaces:** produce `TaskRecord`, `CandidateVersion`, `Observation`, `TrajectoryBank`, `build_grouped_split`, `trainer_view`, `evaluator_view` and three cost ledgers.

- [ ] Write failing tests for RunBugRun lineage grouping, hidden future/gold fields, immutable test order, fresh versions after patches, candidate/test budget equality and tamper detection.
- [ ] Implement canonical hashes, evaluator sidecars, JSONL/NPZ serialization, categorical outcomes and deterministic fake sandbox.
- [ ] Run tests and commit `feat: add PBPF trajectory data protocol`.

### Task 3: Particle filter and exact finite audit

**Files:** create `src/pbpf/{particles,posterior,finite,metrics}.py`, `tests/test_particles.py`, `tests/test_finite_audit.py`.

**Interfaces:** produce `normalize_log_weights`, `effective_sample_size`, `systematic_resample`, `particle_update`, `future_nll`, `sequence_mixture_logprob`, and `run_finite_audit`.

- [ ] Write failing numerical tests for proposal correction, log-sum-exp stability, deterministic resampling, exact posterior convergence and product-of-mixtures counterexample.
- [ ] Implement the equations with explicit RNG objects and shape/finite-value guards.
- [ ] Add P=1/4/8/16/32, MAP/posterior-mean/uniform/shuffled controls and gate computation.
- [ ] Run tests and commit `feat: implement predictive belief particles`.

### Task 4: History/state baselines and model adapters

**Files:** create `src/pbpf/{history,state,models,benchmarks,kv}.py`, `tests/test_state_controls.py`.

**Interfaces:** produce `history_view(mode,k)`, `StateEncoder`, `ParticleConditioner`, `TransformersRepairBackend`, and benchmark JSONL adapters.

- [ ] Write failing tests for last/window/full/orderless/shuffled/wrong-task/masked views, matched-capacity controls, KV delta layer/rank/norm masks and one-particle-per-sequence sampling.
- [ ] Implement CPU representations and lazy optional torch hooks for text, soft-prompt, K-only, V-only and K+V conditioning.
- [ ] Add loaders for RunBugRun, CodeARC, EvalPlus/LiveCodeBench and an isolated SWE-bench adapter that rejects gold fields.
- [ ] Run tests and commit `feat: add PBPF history and model interfaces`.

### Task 5: Frozen prediction, repair and QLoRA launchers

**Files:** create `src/pbpf/{experiment,repair,loss,trainer,statistics,artifacts}.py`, `tests/test_experiment.py`, `configs/{models,benchmarks,experiments}/`, `scripts/{run_smoke,run_frozen_7b,run_repair}.sh`.

**Interfaces:** produce `run_prediction`, `run_repair`, `belief_loss`, `actor_loss`, `select_active_slot`, paired bootstrap and immutable reports.

- [ ] Write failing fake-backend tests proving four rounds, no early stop/reorder, shared candidate bank, one final program, generation-logprob tie break and stop-gradient separation.
- [ ] Implement frozen predictor and repair runners, optional QLoRA update, three ledgers, gates and create-once artifacts.
- [ ] Add exact smoke/16GB/24GB/H200 configs for every primary baseline/ablation and dry-run validation.
- [ ] Run all config plans and fake-backend smoke; commit `feat: add staged PBPF 7B experiments`.

### Task 6: Documentation and CI

**Files:** modify `README.md`; create `docs/{EXPERIMENTS,BASELINES,DATA,ARTIFACTS}.md`, `.github/workflows/ci.yml`, `tests/test_repository_contract.py`.

**Interfaces:** provide exact pull/install/doctor/prepare/run/verify commands and evidence/status boundaries.

- [ ] Write a failing repository-contract test for required matrices, configs, provenance labels and no-claim text.
- [ ] Complete docs, including Rollout Roulette/REx/RLEF distinctions and 16/24GB limitations.
- [ ] Run `pytest -q`, compileall, `git diff --check`, shell syntax and every dry-run.
- [ ] Commit `docs: document reproducible PBPF experiments`.

