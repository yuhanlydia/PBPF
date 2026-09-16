# A-PBPF Information-Theoretic Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable A-PBPF experiment path for association-aware particle
inference, diagnostic-value test acquisition, hard candidate selection, coherent
repair, ablations, and resumable ICLR evaluation.

**Architecture:** Preserve the existing PBPF runner and add an independent
`pbpf.apbpf` package with pure mathematical primitives, a factored neural belief
integration, and a versioned stage runner. Real RunBugRun scripts consume richer
execution records and the new objectives. One shell command executes or resumes
the locked experiment DAG.

**Tech Stack:** Python 3.11+, NumPy, PyYAML, optional PyTorch/Transformers,
pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-09-16-apbpf-information-inference-design.md`

**Implementation status (2026-09-16):** Tasks 1--6 have landed on
`codex/apbpf-information-inference`. Per the project owner's explicit request,
runtime validation in Task 7 was stopped; Torch/GPU paths and the new DAG remain
runtime-unverified. The real backend also requires operator-provided stage-worker
adapters described in `docs/APBPF_WORKERS.md`.

## Global Constraints

- Do not run GPU/model experiments in this implementation session.
- Do not alter the legacy `pbpf-iclr` canonical contract or historical results.
- New behavior is test-first; optional Torch tests may skip without Torch.
- Never use particle-weight entropy as diagnostic entropy.
- Constant histories never contribute to association auxiliary loss.
- All real-result placeholders remain prospective and clearly labeled.
- Formal defaults are seeds `1701,1702,1703`, four visible tests, six future
  tests, eight particles, and source-problem cluster bootstrap with 10,000 draws.
- Particle novelty is gated against pair-aware deterministic, exchangeable,
  tuned Dirichlet-rate, and no-particle bottleneck baselines.
- VoI evaluation runs only after an oracle subset establishes reachable
  headroom over fixed/random test policies.

---

### Task 1: Counterfactual association primitives

**Files:**
- Create: `src/pbpf/apbpf/__init__.py`
- Create: `src/pbpf/apbpf/counterfactual.py`
- Create: `src/pbpf/apbpf/association.py`
- Test: `tests/apbpf/test_counterfactual.py`
- Test: `tests/apbpf/test_association.py`

**Interfaces:**
- `association_eligibility(outcomes, visible_steps, future_outcomes=None)`
- `outcome_derangement(outcomes, visible_steps, seed)`
- `joint_permutation(tests, outcomes, visible_steps, seed)`
- `association_margin_loss(aligned_nll, shuffled_nll, eligible, margin)`
- `association_gap(aligned_nll, shuffled_nll, clusters, ...)`

- [ ] Write tests proving histogram preservation, nonidentity with duplicates,
  constant-history ineligibility, joint-pair invariance, deterministic replay,
  all-ineligible zero loss, and clustered gap sign.
- [ ] Run the focused tests and observe failures caused by missing interfaces.
- [ ] Implement the minimal validated NumPy primitives and optional Torch loss.
- [ ] Re-run focused tests and commit.

### Task 2: Diagnostic mutual information and hard-bank contracts

**Files:**
- Create: `src/pbpf/apbpf/acquisition.py`
- Create: `src/pbpf/apbpf/hard_bank.py`
- Test: `tests/apbpf/test_acquisition.py`
- Test: `tests/apbpf/test_hard_bank.py`

**Interfaces:**
- `expected_information_gain(component_probs, component_outcomes)`
- `ActiveTestPolicy.choose(state, remaining, budget)`
- `audit_hard_bank(visible_outcomes, hidden_labels)`
- `select_development_eligible_groups(records, minimum_groups)`

- [ ] Write hand-computed MI tests, zero-information controls, nonrepeat adaptive
  selection tests, matched-budget checks, and saturated/non-saturated bank tests.
- [ ] Run focused tests and observe missing-feature failures.
- [ ] Implement stable entropy/MI, immutable decision records, and headroom audit.
- [ ] Re-run focused tests and commit.

### Task 3: Factored neural belief and A-PBPF training objective

**Files:**
- Modify: `src/pbpf/belief/model.py`
- Modify: `src/pbpf/belief/features.py`
- Modify: `src/pbpf/belief/losses.py`
- Modify: `src/pbpf/train_belief.py`
- Test: `tests/belief/test_apbpf_neural.py`

**Interfaces:**
- `NeuralBeliefModel(..., difficulty_dim=8)` and `split_latent(z)`
- additive difficulty/diagnosis likelihood heads
- `association_aware_belief_loss(...)`
- `train_apbpf_step(...)` using common random numbers

- [ ] Write optional-Torch tests for dimensional validation, additive logits,
  zero eligible loss, positive margin gradients, factor invariance, and metrics.
- [ ] Run focused tests with Torch and observe intended failures; where Torch is
  absent, validate syntax/API through import-contract tests.
- [ ] Implement the factored heads and training step without changing legacy
  defaults or checkpoint loading.
- [ ] Re-run focused and legacy belief tests and commit.

### Task 4: Rich RunBugRun association gate

**Files:**
- Modify: `scripts/run_rbr_prediction_gate.py`
- Modify: `src/pbpf/real_gate.py`
- Create: `scripts/build_apbpf_hard_bank.py`
- Test: `tests/test_apbpf_real_gate.py`

**Interfaces:**
- versioned cache retains expected/actual/stderr/return metadata
- arms: aligned, outcome-shuffled, joint-reversed, presentation-permuted,
  orderless, semantics-masked, wrong-candidate, random-latent, history-rate
- full/eligible/future-variable strata and clustered CIs

- [ ] Write cache, firewall, transform, metric, and hard-bank CLI tests.
- [ ] Observe failures against the input-only cache and fixed flip control.
- [ ] Implement richer bounded records, canonical transformations, A-PBPF
  training flags, gates, and immutable report fields.
- [ ] Re-run focused tests and commit.

### Task 5: Versioned one-command A-PBPF DAG

**Files:**
- Create: `src/pbpf/apbpf/config.py`
- Create: `src/pbpf/apbpf/stages.py`
- Create: `src/pbpf/apbpf/cli.py`
- Create: `configs/experiments/apbpf_iclr2027.yaml`
- Create: `configs/apbpf/{models,datasets,protocol,gates,ablations,profiles}.yaml`
- Create: `scripts/run_apbpf_iclr.sh`
- Modify: `pyproject.toml`
- Test: `tests/apbpf/test_config.py`
- Test: `tests/apbpf/test_pipeline.py`

**Interfaces:**
- `pbpf-apbpf {doctor,run,verify,report,rerun-stage}`
- resumable stages from materialization through paper tables
- fake backend runs offline; real backend emits exact stage commands and failure
  metadata instead of silently substituting synthetic results

- [ ] Write config/fingerprint, gate propagation, resume, failure, fake-E2E, and
  shell-syntax tests.
- [ ] Run them and observe missing-command failures.
- [ ] Implement the independent schema, DAG, immutable artifacts, CLI, configs,
  and shell launcher.
- [ ] Run focused tests and commit.

### Task 6: Coherent repair, ablations, and operator documentation

**Files:**
- Modify: `scripts/run_rbr_repair_gate.py`
- Modify: `src/pbpf/arms/repair.py`
- Modify: `README.md`
- Create: `docs/APBPF_EXPERIMENTS.md`
- Create: `docs/APBPF_ARTIFACTS.md`
- Test: `tests/apbpf/test_coherent_repair.py`
- Modify: `tests/test_repository_contract.py`

**Interfaces:**
- one immutable component/fingerprint per full continuation
- mean, MAP, random, sample-once, token-remix-fault controls
- gate-aware repair launch and exact operator commands

- [ ] Write tests that fail if conditioning is resampled during generation, if
  downstream claims survive failed upstream gates, or if documentation/config
  inventories diverge.
- [ ] Implement trace fingerprints, named ablations, checkpoint compatibility,
  and complete run/debug documentation.
- [ ] Run focused tests and commit.

### Task 7: Whole-branch verification and publication

**Files:**
- Modify only defects found by review.

- [ ] Run all non-GPU/non-network tests, compileall, shell syntax, config doctor,
  fake-backend complete/resume/verify/report, `git diff --check`, and secret scan.
- [ ] Review the complete branch for claim leakage, hidden-test leakage, unmatched
  budgets, and silent fallback.
- [ ] Fix important findings and re-run covering tests.
- [ ] Push `codex/apbpf-information-inference` and report the commit and exact
  one-command launcher.
