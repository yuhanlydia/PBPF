# Dirichlet Generalization Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate fixed-mass and effective-mass weighted Dirichlet across code generators and datasets, including frozen-parameter bidirectional transfer.

**Architecture:** First acquire and audit scored candidate caches; do not generate labels from aggregate reports. Reuse the existing NumPy predictors, with a separate cross-cache selection/evaluation runner only after schema and source identities are verified. Cache-based prediction needs no Docker.

**Tech Stack:** Existing pbpf-py311 Conda environment, NumPy, pytest, JSON/NPZ artifacts.

**Spec:** User approved cross-model prediction and Qwen-to-DeepSeek / DeepSeek-to-Qwen frozen-parameter transfer in this conversation, then explicitly requested a long goal and execution. Repair and rollout experiments are excluded.

## Global constraints

- Keep ordinary Dirichlet and current fixed-mass weighting as controls; effective-mass weighting is an experimental variant, not a declared winner.
- Alpha grid: .01, .1, .25, .5, 1, 2, 5, 10. Strength grid: 0, 1, 4, 16. Use the existing 256-dimensional lexical input encoder and four visible outcomes.
- Match dataset, task-source partitions, execution protocol, and visible history budget across models before attributing differences to the generator.
- Tune only on source-model validation for transfer. Freeze parameters before looking at target assessment labels. Reverse direction is a separate prespecified experiment.
- Verify source disjointness globally across both models, not just within each cache. Do not allow a source used for tuning under one generator into assessment under another.
- Audit original population roles: some existing development-only reports label an assessment subset `test`. A split name alone does not prove an untouched primary test set.
- Never recategorize old aggregate APBPF reports as new Dirichlet results. Existing examined primary populations are not prospectively untouched for all research decisions.
- Report NLL, Brier, accuracy, PASS precision/recall, and 10,000 whole-source bootstrap draws; state repeated-development and multiplicity limitations.
- No paid resources without user authorization. No unisolated execution of untrusted code. No requirement for Docker when scored caches exist.

## Task 1: Acquire and audit the data

Files: `docs/dirichlet-generalization-data-audit-20260917.md`; cache files under `runs/generalization-inputs/` once transferred.

- [x] Inventory local scored caches including ignored paths. Only `runs/apbpf-real-diagnostic/rbr-cache.json` is present; it is legacy RunBugRun corpus code, not two model-generated populations.
- [x] Inspect remote artifact reports at `origin/longgoal/apbpf-20260916` / `f3ec566`.
- [ ] Obtain full scored Qwen and DeepSeek CodeARC caches plus provenance, task/source mapping, and population manifests. Reports indicate both exist elsewhere; complete local inputs are absent.
- [ ] Obtain corrected RunBugRun Qwen and DeepSeek scored caches. The inspected DeepSeek report records incomplete generation, not verified current completion.
- [ ] Verify cache SHA256 against receipts; bind generator id/revision, execution protocol, source identities, public-input visibility, and actual outcomes. Reject missing information rather than guess.

## Task 2: Implement and verify matched-cache transfer

Files: create `scripts/run_dirichlet_transfer.py` and `tests/apbpf/test_dirichlet_transfer.py` after Task 1 establishes the actual cache schemas. Reuse `predict_rows` from the existing prediction runner; preserve old runner outputs.

- [ ] Write failing behavioral tests: changing target labels cannot change source-selected parameters; overlapping source groups are rejected; generator mismatch is rejected; shuffled input record order does not change source membership; source/target direction is bound in the manifest.
- [ ] Run `python -m pytest -q tests/apbpf/test_dirichlet_transfer.py` and observe the missing implementation failure.
- [ ] Implement cache normalization preserving original source ids, explicit role mapping, and generator/protocol metadata. Select each arm on source validation using the fixed grids above; write a hashed selection receipt before target assessment.
- [ ] Run the tests and existing `tests/apbpf` suite; obtain code review before primary assessment.

Task 2 is intentionally data-gated: no schema adapter or executable command is asserted to exist before complete cache files are audited. This is a research execution plan, not authorization to invent missing data contracts.

## Task 3: Execute and report

- [ ] First run within-model prediction controls on development portions for both generators, documenting prior exposure.
- [ ] Freeze source-selected configurations and execute both cross-model transfer directions on globally source-disjoint assessment populations.
- [ ] Repeat on a second dataset when the matched model pair exists. A Qwen-RunBugRun versus DeepSeek-CodeARC comparison alone is a joint domain/model shift, not an isolated model-transfer test.
- [ ] If a genuinely untouched primary population is available, lock all analysis decisions and assess it once; otherwise label the entire study exploratory and reserve new sources for confirmation.
- [ ] Independently recompute metrics from predictions and verify hashes. Save wins and losses. Do not infer repair success from outcome prediction.
- [ ] Commit reviewed runner/tests and a compact results report to the research branch; do not upload model weights or private data.

## Product long-goal state

The previous APBPF goal remains `blocked`. A `create_goal` call for this new
objective failed with `cannot create a new goal because this thread has an
unfinished goal; complete the existing goal first`. The old scientific goal
is not complete, so it was not falsely marked complete. A user/product-side
goal replacement or a fresh thread is needed to activate the new long goal.
