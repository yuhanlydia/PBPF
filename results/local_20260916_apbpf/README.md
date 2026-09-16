# Local A-PBPF progress snapshot — 2026-09-16

This is an **incomplete, nonconfirmatory** experiment milestone. The long-running
local workflow remains active. No positive A-PBPF claim or complete real-stage DAG
is established by these files. Legacy H200 formal experiments are pending external
resources; the user confirmed that only local execution is available.

## Completed findings

- The original directory was aligned to upstream `7dd64de`, preserving local fixes.
- Three default RBR A-PBPF seeds (1701–1703, 1000 steps) all failed association and
  strong-baseline fairness gates. Full-population association gaps were about
  0.00196, −0.00097, and −0.00155 nats per future test (required: 0.03 plus positive CI).
- Expanded RBR sources yielded 6362 training and 1264 development candidates.
  Four predeclared development-only variants all failed the gates. The widest
  lexical-feature variant achieved a 0.00704 gap; it still trailed strong baselines.
  The original held-out population was excluded from this development sweep.
- Of 1264 development assessment candidates, 899 have constant visible outcomes.
  Their outcome-shuffle association gap is exactly zero. This difficulty is reported
  without changing the full-population estimand or lowering the threshold.
- The wider-feature 3000-step follow-up produced a 0.00740 gap (CI crosses zero)
  and worse aligned NLL, 0.47963. The default 3000-step variant also failed: gap -0.00308, aligned NLL 0.45618.
- CodeARC sources were pinned and split by exact-AST source components before
  generation: 212 training, 400 development, 500 primary groups. Candidate generation
  has access only to four public calls and outputs; no reference programs or future
  calls are mounted into its process.
- A 16-group development pilot generated 128 candidates. Replaying ten calls each
  produced 328/1280 successful calls and 19/128 candidates passing all ten. The small
  pilot has no selector headroom and does not meet the 300-mixed-group hard-bank gate.
- Development reference controls exposed executor defects (NumPy BLAS mounts,
  Python-version exception text, timeout classification), which were corrected.
  The corrected reference run matched 4002/4010 behaviors, including nine expected
  timeouts that remain TIMEOUT, not PASS. Eight residual discrepancies reflect
  dictionary printed ordering. Exact primary scoring is retained.

## Running and unresolved

Qwen generation is running on GPUs 0–2; GPU 3 belongs to another project. The
complete primary population will be evaluated visibly, sealed using the public
candidate inventory, and only then evaluated on hidden calls. A local supervisor
implements this sequence. It does **not** represent the sealed 21-stage DAG.

One development public prompt (`CodeARC/508`) contains 1,128,840 characters and
caused a 53.23 GiB allocation on the 24 GiB GPU. The failed 110-group partial run
is retained. A separate full development attempt bounds individual public fields
to 2048 characters and model input to 4096 tokens; all groups are retained. Normal
public prompts remain unchanged. Training/primary generation does not encounter
this oversized example and continues under its original identity.

The pinned DeepSeek replication model download recovered from two interrupted HTTP
streams and completed publisher LFS SHA-256 verification. No replication result is claimed yet. Active acquisition,
selection, repair, full cross-family replication, and genuine stage-worker adapters
remain unfinished.

## Validation and reproduction

The aligned baseline passed 710 tests. After additions, one full run had 717 passes
and four sandbox timeouts; all four passed in a focused 26-test rerun. A later full
run had 727 passes and one bounded-verifier timeout; that case and the input-bound
regression passed in a focused rerun. Disk I/O pressure was high during these runs;
we retain the failed logs and do not present either full invocation as wholly green.
Additional executor and cache checks passed (8 and 11 tests, respectively).

Entrypoints are `scripts/materialize_apbpf_codearc.py`,
`scripts/generate_apbpf_codearc_bank.py`,
`scripts/generate_apbpf_codearc_bounded_bank.py`,
`scripts/evaluate_apbpf_codearc_bank.py`,
`scripts/run_local_codearc_pipeline.py`, and
`scripts/build_codearc_development_cache.py`. The local supervisor requires the
already materialized/generating banks under its supplied run root. Evaluation
outputs are create-once, bind source/data checksums, and preserve failed gates.

`SHA256SUMS.json` covers this milestone's copied reports and logs. Full raw local
artifacts and checkpoints remain under `local/longgoal/`; subsequent results require
new snapshots. CodeARC replay is not the official interactive leaderboard protocol.

## Active-testing development diagnostic (2026-09-17)

A full 1264-candidate / 200-source development replay compared fixed, random,
diagnostic-MI, and predictive-entropy acquisition using the shipped mean-field
adapter. At budgets one and two, an evaluator-only oracle subset had NLL advantages
0.06762 and 0.05425 over fixed. Diagnostic MI achieved 0.00656 and −0.00644, both
with confidence intervals crossing zero. At budget four all four public tests are
consumed; canonical oracle and fixed predictions coincide, as checked in the run.
The four-test MI advantage was −0.00013, providing no active-testing gate success.
These are cached observation budgets, not fresh sandbox execution counts, and
remain exploratory because the upstream gates failed. Five additional regression
checks verified exact public budgets and rejection of a hidden query.

The next predeclared diagnostic compares the trained SMC posterior updates with
the adapter's fixed-particle mean-field updates using the same development cache
and checkpoint. The mismatch is a hypothesis, not a demonstrated cause.
