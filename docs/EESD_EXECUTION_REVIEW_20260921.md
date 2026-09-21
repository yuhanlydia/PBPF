# Experiment execution review and direct-execution amendment — 2026-09-21

## User instruction and purpose

The user explicitly instructed execution without the sandbox after discussing the
namespace failure, and asked for a fresh review of the design. This amendment
changes the candidate execution environment. It does not change the scientific
question, locked populations, models, seeds, estimators, budgets, or metrics.
No new-study formal mechanism scores existed when this amendment was adopted.
The historical sandbox failure and all previous locks remain historical records.

## Findings

1. The GPU-only queue generated candidates without advancing them into scoring.
   At review, 15 of 48 generation cells were complete and no formal mechanism
   cache/report existed. Completed cells must enter CPU scoring immediately;
   waiting for every generation cell is unnecessary.
2. The scope is four independent model families, four mechanism datasets, and
   three generation seeds: 48 cells. The original RunBugRun/CodeARC matrix is
   24 cells; the separate APPS-Replay/CodeContests-Replay extension is 24 cells.
   Each cell contains 200 development and 500 primary candidates, ten executions
   each. Extension datasets add mechanism evidence, not new training domains.
3. Mechanism evidence means probability quality, especially NLL and paired
   controls, rather than merely code generation completing without error.
4. Full primary/secondary-family conclusions require the predeclared coverage.
   Individual completed-cell diagnostics may be produced first, but are not a
   substitute for missing cells or the three-seed source-cluster inference.
5. The 1,560-slot secondary Holm family has a declared resolution limitation:
   with 10,000 bootstrap draws the minimum p-value exceeds the first threshold.
   Keep this limitation visible; do not claim nonsignificance proves no effect.
6. HumanEval+ and MBPP+ remain required transfer evaluations. LiveCodeBench is
   optional under the main experiment specification; RUN_ALL's wording is stale.

## Revised execution order

1. Preserve the active generation queue and all sealed candidate banks.
2. Implement direct stdin and CodeARC-call execution with the original Python,
   input normalization, comparison rules, five outcome classes, and six-second
   deadline. Validate these with trusted small programs before dataset execution.
3. Run a completed Qwen/CodeARC cell through all 700 candidates / 7,000 tests,
   then the existing evidence runner. Seal actual execution provenance and output.
4. Feed every completed original-domain cell into the same scoring pipeline;
   integrate the same direct profile into the Replay extension's distinct inputs.
   Keep CPU execution bounded while GPU generation proceeds.
5. Reconstruct all locked ablations and collect separate original/extension
   statistical families when their required cells exist. Retain all signs.
6. Proceed to correction credibility, eight-arm single-round distillation,
   Qwen rounds 0–3 on both original domains, and required transfer evaluations.
   These phases still need execution-backend integration and actual runs.

## Direct execution profile

- No bubblewrap, Linux namespaces, container, or virtual-machine sandbox is used
  for candidate execution. This is not equivalent to the previous isolation.
- Run the candidate through `/usr/bin/python3 -I` as an unprivileged user, with
  supplementary groups removed and a small explicit environment. `/root` stays
  inaccessible by ordinary filesystem permissions. Do not pass credentials.
- Use per-invocation temporary working directories and original resource limits
  (six-second deadline, 1 GiB address space, 8 MiB output-file cap, 64 descriptors,
  no core files), plus a 64-process per-user limit and process-group cleanup.
  These limits are not a security boundary; direct execution retains host-network
  and otherwise permitted filesystem access.
- The process limit is an explicit environment change. Use the same profile for
  all compared models and disclose it, including any induced runtime failures.
- Record actual Python binary/version, environment, user identity, source hashes,
  validation probes, model-bank bindings, data hashes and output checksums.
- Create new execution and analysis receipts in a separate output tree. Do not
  edit the historical sandbox locks, reuse their readiness status, or overwrite
  the running GPU supervisor's frozen source dependencies.

## Completion criteria

The immediate milestone is a real sealed execution cache and mechanism report
for a complete cell. Tests, trusted probes, generated candidate counts and
configuration locks alone do not satisfy it. The whole experiment remains
incomplete until all required mechanism, decision, training, recursion and
transfer outcomes are reported, including null and negative findings.
