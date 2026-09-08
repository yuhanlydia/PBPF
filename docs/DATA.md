# Data and evaluator firewall

Benchmark adapters read JSONL with a public task identifier, prompt/buggy code,
and an ordered list of opaque test identifiers. Outcomes are evaluator-only
`(candidate_hash, test_id)` keys; task-global outcome sequences are not accepted
for scoring. Trainer input is rejected if it
contains gold/canonical solutions, gold patches, expected outputs, future
outcomes, hidden tests, or SWE-bench `test_patch`, `FAIL_TO_PASS`, or
`PASS_TO_PASS`. Evaluator sidecars are materialized separately and passed only
to metric computation after predictions are complete.

RunBugRun splitting must group connected source problem IDs, submission/user
lineage, translations, and shared test sources. `build_grouped_split` computes
transitive connected components before seeded assignment; no lineage component
can cross train/validation/test. Public target code and tests are
hidden-by-protocol, not described as unseen.

Prepare an already-authorized public JSONL without downloading weights:

```bash
python -c 'from pbpf.benchmarks import load_runbugrun; print(len(load_runbugrun("data/runbugrun-public.jsonl")))'
```

Build banks only from ordered observations. `CandidateVersion.with_patch` creates
a fresh immutable version with its parent hash; in-place patch mutation is not a
valid trajectory. `TrajectoryBank.write(trainer_root, evaluator_root,
candidate_outcomes=...)` requires two independent create-once roots suitable for
separate mounts. The trainer root contains public tasks, candidates,
observations, the hashed mutant registry, arrays, and a checksum manifest. The
evaluator root contains gold task sidecars plus candidate/test outcomes and its
own manifest. Trainers call `read_trainer`; evaluators explicitly call
`read_evaluator` with both roots.

The fake sandbox is deterministic and does not execute code. Real execution must
be supplied by an isolated sandbox adapter that maps task results into the five
categorical outcomes and reports infrastructure failure out-of-band. Test
selection remains fixed and cannot depend on GOAV or any arm-specific policy.
