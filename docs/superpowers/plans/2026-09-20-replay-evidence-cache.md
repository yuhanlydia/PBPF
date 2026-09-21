# Replay Evidence Cache Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` and TDD when implementing this plan.

**Status:** implementation design only, 2026-09-20. No candidate execution, cache publication, configuration change, or admission of statistical results occurred during this review.

**Goal:** produce 24 independently sealed execution caches for the admitted two Replay domains × four models × three seeds, each covering exactly 200 development and 500 primary candidates and ten executions per candidate.

**Architecture:** a new trusted evaluator validates the published Replay admission and both sealed generation banks, maps raw evaluator IO to the immutable stdin executor, and publishes a redacted cache with a separate binding receipt. New Replay verifiers consume this schema. Frozen RBR/CodeARC builders, dispatchers, inference loaders and existing statistical families remain unchanged.

**Tech stack:** Python 3.11.16 builder, immutable Python 3.10.12/bubblewrap stdin executor, JSON seals, pytest synthetic fixtures. No model runtime is needed by the cache builder.

**Binding science:** [Replay extension spec](../../EESD_REPLAY_EXTENSION_SPEC_20260920.md). The estimand is calibration of the locked execution tool's five outcomes. It is not official APPS/CodeContests accuracy, semantic program correctness, or proof of previously unseen test answers.

## Existing contracts and concrete incompatibilities

- Published input: `runs/eesd-data/replay-extension-locked-20260920`, schema `eesd-replay-admission-v1`; admission SHA `73d96f7b0054726aef27b007e8a789ea22a739cfd338e9d096360261dbfbd40f`. Each domain has separate public/evaluator manifests with `tasks: {sha256, bytes}`. Do not look for the old `evaluator_tasks_sha256` field.
- `replay_generation.load_public_split(bundle, admission_sha256, domain, split, family)` returns ordered public rows, audits, model and bindings. `build_expected(...)` and `verify_replay_bank(..., preflight=...)` verify a bank against that external public identity. Merely checking `complete.json` or using generic `codearc_bank.load_bank` is insufficient.
- New banks have schema `eesd-replay-generation-v1`, `domain` equal to `apps_replay`/`codecontests_replay`, one synthesis candidate, `adapter=None`, model revision, seed, public identities, source hashes and admission bindings. Two banks are needed for each cache; split directories live below `replay-mechanism-banks/{domain}/{family}/seed{seed}`.
- Evaluator rows have `task_id/source_component_id/domain/split/statement/tests/known_exposed_input_sha256` plus provenance. Each test has string ID `0`–`9`, raw `input` and **`output`**, normalized `pair_sha256`, original pools and exposure flag. `execute_stdin` requires **`expected`** instead of `output`: supply the narrow in-memory mapping `{'input': test['input'], 'expected': test['output']}`. Do not normalize stored raw IO or pass all evaluator fields to the subprocess.
- `build_eesd_mechanism_cache.py`, `run_eesd_matrix.verify_mechanism_cache`, `mechanism_artifacts.load_mechanism_artifacts`, and the original inference reporter hardcode old domains/schemas/sources. They cannot certify Replay by renaming a dataset field. Do not edit or route through those verification paths.

## Minimal new files and interfaces

1. `src/pbpf/eesd/replay_execution_cache.py`: trusted evaluator input verification, pure job/result projection, strict cache verification and atomic publication helpers.
2. `scripts/build_eesd_replay_mechanism_cache.py`: CLI orchestration and isolated worker pool; no generation/model loading.
3. `tests/eesd/test_replay_execution_cache.py` and `test_replay_cache_cli.py`: synthetic sealed fixtures and mocked execution only during implementation.
4. A separate extension execution lock JSON and cache matrix JSON under `runs/eesd-setup/`, generated after source review. Do not mutate original matrix/config/lock files.

CLI inputs: `--bundle --admission-sha256 --generation-manifest --generation-manifest-sha256 --domain --family --seed --development-bank --primary-bank --execution-lock --execution-lock-sha256 --output`. Worker concurrency is an operational setting recorded in the execution receipt and bounded by the execution lock; scientific deadline stays six seconds, with no per-row override. `--preflight-only` validates data/bank identities and reports runtime readiness separately, creates no cache and executes no dataset candidate. Namespace readiness checks use only a fixed known program, only when explicitly requested by an execution-readiness mode; a file-validation-only preflight must not invoke subprocess execution.

Suggested library interfaces:

- `load_replay_execution_inputs(...) -> {ordered_jobs, bindings, runtime_profile}`. Verify all bindings before constructing the first executable job.
- `measure_job(job) -> strict_cache_record`. Execute ten tests in their sealed order; no early success/failure stop.
- `verify_replay_cache(cache_dir, *, expected_inputs, execution_lock) -> receipt`. Compare to external input bindings; never trust cache metadata as the source of expected population.

## Verification sequence and freeze boundary

1. Verify expected admission SHA, status/schema, public and evaluator manifest/task file hashes and sizes through the admission output inventory. Verify selected audit/model/template/spec bindings through the public loader. Open evaluator content only in this trusted evaluator path, after both generation banks have passed completion verification.
2. Verify generation manifest SHA/current source hashes and the exact two planned output paths for this domain/family/seed. Rebuild bank expectations from each public preflight and current frozen generator/library/prompt/helper; require complete banks, exact record/checksum inventory, matching candidate code/IDs and immutable seed provenance. Missing or changed seals fail; do not label them missing model outcomes.
3. Require exactly 200/500 ordered unique source/task identities, disjoint splits, one shared model/revision/seed, and equality to the admitted public population—not merely equal counts. No source/task/candidate duplicate, subset or alternate representative. Match evaluator task identities and splits one-to-one. Preserve canonical order: development in admitted file order, then primary in admitted file order.
4. Recheck ten ordered IDs, raw IO bounds and normalized pair identity/provenance; first four raw pairs must exactly equal public observations. Targets 4–9 must have disjoint normalized inputs from observations and the sealed identified-exposed-input set. Preserve the documented incomplete native-example extraction limitation; do not claim that this audit proves every target answer absent from the original statement. Bind complete evaluator bytes, not only normalized fingerprints.
5. Verify source hashes and runtime identity before any candidate execution. After execution, recheck all input/source identities before sealing. A changed input/runtime/source invalidates the run. Freeze new CLI/library and the immutable imported executor plus `codearc_execution.py` (`_limits` dependency), generation verification sources, published admission, generation manifest, public/evaluator bytes, spec, runtime lock and exact bank run/complete files.

Existing queue locks are not a substitute for this new execution-profile lock. Do not modify the current generation scheduler to build caches as part of this task.

## Execution profile and infrastructure handling

Reuse `pbpf.apbpf.rbr_execution.execute_stdin` without changing it: candidate `/usr/bin/python3` 3.10.12, bubblewrap 0.6.1, six-second wall deadline, terminal-newline insertion, case-sensitive line/token comparator with numeric absolute tolerance `1e-4`, relative tolerance zero. Bind builder Python 3.11.16 too: the immutable executor's initial `compile` runs in the builder interpreter, before candidate execution under 3.10.12. Preserve and disclose this existing two-interpreter behavior; do not silently move compilation or relabel runtime syntax failures.

Record executable resolved paths/SHA, versions, platform and execution source SHA. `_limits` currently binds CPU `ceil(timeout)`/`+1`, address space 1 GiB, file size 8 MiB, core zero and 64 descriptors; preserve the implementation and environment variables. Capture the observed runtime profile in a new readiness receipt and bind it into the execution lock. Do not infer readiness from installed binaries or a historical receipt alone: each actual cache run needs a fresh fixed-program namespace/isolated-stdin probe before any row. This must happen even when all candidates would return a compile error before attempting bubblewrap.

Current namespace failure is a real external blocker. Probe failure, `bwrap:` infrastructure error, missing binary, worker death, process-pool exception, resource/setup errors or malformed executor result abort publication as infrastructure failures. Never convert them to `RUNTIME_EXCEPTION`, `TIMEOUT`, `WRONG_OUTPUT`, or a skipped row. Ordinary candidate outcomes remain exactly PASS, WRONG_OUTPUT, COMPILE_ERROR, RUNTIME_EXCEPTION, TIMEOUT. A real candidate timeout is retained; never retry it to find a preferred outcome. Keep the measured empty/invalid generated programs—do not filter on quality.

The executor mounts only code and supplies stdin; expected outputs remain host-side for comparison. Never mount the bundle, evaluator files, model artifacts, solution fields or expected answers into the candidate sandbox. Do not substitute subprocess-without-bwrap when namespace isolation fails. No candidate or readiness program is executed by this design task.

## New cache schema and publication

Use a create-once **directory**, atomically published with no-replace semantics after all 700 records complete:

- `cache.json`, schema `eesd-replay-public-query-cache-v1`:
  - `task_kind: stdin_synthesis`, `domain`, `family`, `seed`, model/revision/adapter identity;
  - `tests_per_candidate: 10`, `observation_indices: [0,1,2,3]`, `target_indices: [4,5,6,7,8,9]`;
  - counts and source counts `{development:200,primary:500}`;
  - `records`, ordered as above. Each record has exactly `task_id` (candidate ID), `problem_id` (bank task ID), `source_component_id`, `split`, `candidate_code_sha256`, `tests: [{id,input}]`, `outcomes: [five-class string ×10]`;
  - explicit visibility/claim scope and bank/input/runtime binding references.
- `binding.json`, schema `eesd-replay-cache-binding-v1`: cache SHA/bytes; expected admission and generation-manifest SHAs; full public/evaluator manifests/task hashes; selected audit/token/template/spec digests; ordered task/source/test-input inventory digest; each bank's absolute path, split, run SHA, complete SHA and count; runtime readiness/execution-lock hashes; exact new/imported source SHA map.
- `complete.json`, schema `eesd-replay-cache-complete-v1`: hashes/sizes of cache and binding. No successful completion file on infrastructure failure.

All ten outcome strings belong to the trusted offline evaluation cache. Predictor-facing examples use only the first four outcome observations and query input features; targets' outcomes are labels, never predictor features, tuning feedback or generation input. Keep expected outputs, actual stdout/stderr, task statement, reference code and candidate code out of cache records. The complete cache is not a generator-readable public artifact merely because query inputs are exposed there.

No per-record execution resume in the minimum implementation: an interrupted unpublished staging directory is explicit incomplete infrastructure state. Preserve diagnostic receipts separately; do not accept a partial population. A restart policy must be reviewed before adding one. Complete-cache reuse requires all external bindings and every inventory/label-domain check to pass, not file existence alone.

## TDD tasks and acceptance gates

### Task 1 — trusted sealed-input adapter

- [ ] Write red tests using synthetic two-split public/bank/evaluator fixtures: raw CRLF/output mapping, mismatched first-four pairs, duplicate/missing source or test IDs, crosssplit bank swap, wrong admission/manifest/source SHA, incomplete bank, and mutated candidate code.
- [ ] Implement verified input loading and pure projections only; reuse Replay bank/public verification, never old-domain verifiers.
- [ ] Test that no executor is called when any input/bank verification fails. Verify missing evaluator labels cannot be filled from public examples.

### Task 2 — fail-closed measurement and publication

- [ ] Write red fake-executor tests covering all five categories, ten calls even after failure, target order, code SHA and output/diagnostic redaction.
- [ ] Add namespace-failure test with a compile-invalid candidate: readiness failure must occur first and produce no outcome/cache. Add worker death/malformed result/infrastructure exception, source drift after measurement, and partial publication tests.
- [ ] Implement fixed execution mapping and atomic no-replace publication. Add external-binding verifier and tests for post-seal label/input/bank/runtime tampering, existing output and incomplete inventory.
- [ ] Run focused CPU tests. A real readiness probe is a later integration gate; dataset execution remains blocked until it succeeds under the locked profile.

### Task 3 — independent execution matrix and inference handoff

- [ ] Prepare 24 cache jobs under a new Replay output root, each requiring both 200/500 banks. Missing bank is a pending job; an existing invalid seal is an error. Do not dispatch execution here or alter generation jobs.
- [ ] Freeze the new cache schema/sources/runtime lock before the first measured candidate. Record cache matrix SHA and source population digests. Produce only a preflight inventory while namespace support is unavailable.
- [ ] Hand the sealed-cache API and fixtures to a separate Replay evidence adapter; require its protocol/support lock before assessment inference. Tests below define compatibility requirements, not implementation of inference in this cache task.

## Offline EED/statistics compatibility, without extending old families

The record payload deliberately matches the pure `run_eesd_evidence_matrix.py` example builder's ordered inputs/outcome strings, so its hash-locked mathematical functions and `pbpf.eesd.evidence` can be reused through a new extension entrypoint. That runner's CLI alone does not validate Replay provenance and must not serve as the new seal verifier. Do not invoke old `load_mechanism_artifacts`/`reconstruct_contrasts` with forged `rbr`/`codearc` domains. New thin artifact/contrast wrappers must validate Replay seals and reproduce the same 49-slot definitions, selection grids/ties, predictions and public-support timing.

Primary NLL retains `core/effective_params`: eight slots (two new domains × four models), combining three fixed seeds by the existing source-paired inference rule. Secondary is `49 × 4 metrics × 8 cells − 8 = 1560`, with A8/n4 aliasing core/tuned, comparator metrics averaged rather than probabilities, and all A1–A10 slots preserved. Ordinary n=4 analyses predict six targets. A8 history counts 1/2/4/8 are explicitly offline sequential-observation diagnostics over the same ten executed tests (respectively 9/8/6/2 target labels); n=8 does not assert the generator saw eight examples or permit its six base targets into n=4 features.

Preserve development-only selection and A9 shared-source support sealing before assessment inferential reconstruction. Public-support computation may parse trusted cache bytes but must not access assessment outcomes as features; do not overclaim that no label bytes were loaded. Subsequent full reconstruction must match support/query IDs exactly. Use canonical string query IDs and Python integer seeds; all full populations retain the expected 500 sources. Unsupported or missing slots remain explicit and block complete-family decisions.

Reuse the generic `mechanism_inference` engine: source resampling paired across all seeds, query-within-source/seed equal weighting, nonlinear ECE recomputation, 10,000 PCG64 draws, centered two-sided p+1 and pointwise intervals; complete-family Holm at .05. Extension primary/secondary families are separate from the original families, not pooled or resized. The secondary minimum p `1/10001` exceeds `.05/1560`, so first-step Holm rejection is impossible; report that limitation. No claim of joint study-wide FWER .05. Bind the extension contrast/support/statistical lock before any assessment inference; this plan does not invent different tests or additional arms.
