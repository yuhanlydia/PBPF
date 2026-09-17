# Dirichlet Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare four user-approved statistical-feedback hypotheses without inventing unavailable rollout outcomes.

**Architecture:** Isolate pure NumPy probability/update/allocation primitives from recorded-data evaluation. Run the original and similarity-weighted predictor on the same development partition. Keep discounted inheritance and posterior-sampling allocation scientifically pending until actual repair trajectories are available.

**Tech Stack:** Python3.11, NumPy, existing PBPF rich-cache validation and lexical features, pytest.

**Spec:** User-approved four-arm table in this conversation: Dirichlet; test-related weighted Dirichlet; discounted inheritance after repair; Dirichlet-guided rollout allocation. Primary endpoint is equal-budget final success, not predictor NLL alone.

## Global Constraints

- Original held-out test rows must not enter parameter selection or assessment.
- Current cache uses literal stdin and is legacy exploratory evidence, not corrected-protocol or confirmatory data.
- No namespace bypass or execution of untrusted candidate code.
- Same source-group partitions and visible histories for competing predictors.
- Weight bandwidth and smoothing selected only on inner validation.
- Real rollout claims require actual child-code executions and budgets; unit tests or fabricated rewards cannot establish efficacy.
- No baseline deletion or threshold changes. Preserve negative results.

### Task 1: Pure statistical primitives and regression tests

Files: `src/pbpf/apbpf/dirichlet.py`, `tests/apbpf/test_dirichlet.py`.
Interfaces: `predict_rate(outcomes, alpha, weights=None, classes=5)` returns normalized class probabilities; `similarity_weights(query, history, strength)` returns nonnegative weights summing to history length; `inherit_counts(parent_counts, child_outcomes, retention)` discounts evidence once, then adds child observations; `allocate_rollouts(counts, budget, observe, seed)` returns exactly budget choices and updated local counts.

- [ ] Write failing tests: `predict_rate([0,0],1,classes=2)==[.75,.25]`; zero similarity strength returns ones; weighted identical histories reproduce base; retention0 forgets parent and retention1 retains it without mutating input; allocation callback called exactly budget times and each returned category increments only selected candidate.
- [ ] Run `/root/miniforge3/envs/pbpf-py311/bin/python -m pytest -q tests/apbpf/test_dirichlet.py`, confirm missing API failure.
- [ ] Implement counts using `np.bincount`, weights using max-subtracted exponential cosine scores normalized to fixed total evidence, posterior sampling using `rng.dirichlet(counts[i]+1)` and PASS index0. Label allocation score a heuristic for repair potential, not an identified child success probability.
- [ ] Rerun tests and input-rejection cases; review independently before publishing.

### Task 2: Legacy development prediction screen

Files: `scripts/run_dirichlet_variants.py`, output under a fresh `runs/dirichlet-variants-20260917/`.
Consumes Task1 prediction interfaces and existing `partition_development`.

- [ ] Validate rich cache; partition original training into fitting/validation; original development is assessment.
- [ ] Predeclare alpha grid [.01,.1,.25,.5,1,2,5,10], strength grid [0,1,4,16]. Similarity uses public input lexical features, no evaluator outcomes or stderr. Fix total evidence at four so bandwidth cannot simply change smoothing strength.
- [ ] Fit alpha and strength on validation only, record selection before assessment. Preserve source/code/input hashes and label old stdin protocol explicitly.
- [ ] Report NLL, accuracy, Brier, full development source-cluster bootstrap10000 draws, complete probability arrays. Keep baseline when selected strength0; do not force a nonzero variant.
- [ ] Verify artifact hashes and probability metrics independently. No inference that prediction improvement implies rollout success.

### Task 3: Real discounted-inheritance comparison

- [ ] Require parent-child code hashes, ordered public observations, actual child execution outcomes, source-separated fitting/assessment, and fixed generation/token/execution budgets.
- [ ] Compare retention grid [0,.25,.5,.75,1], choosing only on validation. Do not transfer hidden child outcomes into scheduling.
- [ ] Present missing trajectory/environment requirements if unavailable. No real run currently authorized on a new host; summaries alone are insufficient replay data.

### Task 4: Real allocation comparison

- [ ] Require complete potential-action repair banks or an available isolated online evaluator.
- [ ] Compare uniform, visible-pass greedy, and Dirichlet posterior-sampling allocation at same eight roots/four repair continuations and token cap. Log every executed action and evaluate one final selected candidate.
- [ ] If unavailable, leave scientific result pending and report unit validation separately. Do not select a winner from simulator-only results.

## Initial audit

Local authoritative checks: `/root/pbpf-runs`, `/root/PBPF/local`, `/data/cwj/PBPF/local` absent. Only legacy rich-cache is present; namespace probe fails Operation not permitted. Tasks1/2 can proceed as explicitly exploratory work. Tasks3/4 need external-state change. This limitation does not redefine the four-arm goal as completed.
