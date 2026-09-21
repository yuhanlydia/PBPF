# Replay extension statistical protocol

Adopted before any Replay execution cache or inference result exists. This
implements the separate families already specified in
`EESD_REPLAY_EXTENSION_SPEC_20260920.md`; it does not change the original study.

- Domains: `apps_replay`, `codecontests_replay`. Four families: Qwen2.5-Coder-7B,
  DeepSeek-Coder-6.7B, Seed-Coder-8B, StarCoder2-15B. Seeds 1701, 1702, 1703 are
  fixed repeated generations, not independently sampled source clusters.
- Each domain has the same admitted 200 development and 500 primary sources
  across models and seeds. No outcome-dependent selection, replacement or
  truncation. All 24 cache/report cells must be accounted for.
- Reuse the frozen evidence runner, grids, six prediction arms and all 49
  contrasts. Development labels alone select hyperparameters. A8/n4 is the
  reserved alias of core/tuned; unavailable slots are not deleted.
- Before full assessment reconstruction, derive A9 input-feature masks using
  development-selected settings, seal support for all three seeds in all eight
  cells, then require full reconstruction to match those identities and masks.
  Assessment JSON bytes may be hashed/decoded for provenance; its outcomes must
  not be accessed to derive public support.
- Primary family: eight `core/effective_params/nll` comparisons. Secondary family:
  49 contrasts × four metrics × eight cells minus those eight = 1,560 slots.
  Each family uses Holm at alpha .05 independently. This makes no combined
  study-wide FWER .05 claim across original and extension families.
- Reuse the paired source-cluster bootstrap: 10,000 PCG64 resamples, centered
  two-sided plus-one p-values, query-within-source/seed equal weighting, shared
  source sampling across all three seeds, and nonlinear ECE recomputed with ten
  bins. Multiple comparator realizations are averaged as metrics, not pooled
  probabilities. Base bootstrap seed remains 314159 with the frozen inference
  engine's domain/model/population derivation.
- The minimum p-value 1/10001 exceeds .05/1560, so the secondary family cannot
  reject at its first Holm step. Report this resolution limitation; absence of
  rejection is not evidence of no effect. Missing or unsupported comparisons
  prevent complete-family decisions rather than shrinking either family.

The implementation lock binds this document, extension spec, unmodified
mechanism configuration, admission/cache matrix and execution lock, mathematical
sources and new adapters. The 49-contrast adapter uses an isolated function
namespace with only its provenance loader replaced; it must not modify the
shared original module or its mathematical code.

Evidence output remains an execution-tool calibration study. No result is
labelled official APPS/CodeContests accuracy. No candidate may run without a
fresh successful trusted namespace probe under the locked execution profile.
