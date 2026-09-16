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

Qwen primary500 and training212 generation are complete; bounded development384
is complete and executing its 3072 candidates. DeepSeek primary500 generation is running on GPU 1; a
corrected repair pilot trains on GPU 0. The CodeARC prediction queue will use
GPU 2, followed by the remaining DeepSeek train/development generation. GPU 3 belongs to another project. The
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

## SMC comparison and extraction corrections (2026-09-17)

The SMC replay comparison completed on the same 1264 development candidates,
including a joint-particle-MI ablation. Diagnostic-MI advantages over fixed were
0.00568, −0.00659, and −0.00198 at budgets 1, 2, and 4; every CI crosses zero.
Changing inference alone did not establish active-selection gains. Seven checks
passed, including agreement with the existing diagnostic-MI adapter, selected-only
feedback, exact budgets, and first-choice independence from all test outcomes.
Exact executed source files are archived in `smc_executed_source/`.

DeepSeek's initial pilot had a parsing defect: choosing the longest fenced block
often extracted example calls instead of the shorter `solution` definition.
Reextracting all 128 unchanged raw completions by a deterministic, syntax-based
rule changed 75 candidates and increased all-ten passing candidates from 3 to 16
and successful calls from 83 to 254. Qwen remains 19/128 and 328/1280. These are
small development pilots; they do not establish a replication gate. All original
results are retained. Full banks are now uniformly reextracted before evaluation
and primary pre-hidden locking. Three extraction regression checks passed.

A separate repair integration defect used raw test-input features for newer
checkpoints trained with JSON public-test features. The repair path now follows
the checkpoint visibility setting, preserving raw features for the actual legacy
format verified in commit `64b6032`. A regression test checks equality with the
prediction encoder and exclusion of actual output/stderr.

A frozen-code-model feature experiment completed: 20629 distinct public texts
from the development-only RBR cache, Qwen last-hidden-state mean pooling, fixed
512-dimensional projection, no label fitting. The extractor runs with only this
public inventory mounted. Predictor training used the same features for all
strong baselines and validated the exact source-population digest. Three feature/
repair checks and a small training-integration check passed. Seed 1701 (1000 steps)
achieved aligned NLL 0.45434 and association gap 0.00633, CI [−0.00252, 0.01530].
Association and baseline fairness still failed. Deep Sets and pair-aware baseline
NLLs were 0.35331 and 0.35414. Seeds 1702–1703 also completed with the same recipe;
this result does not establish that changing the encoder fixes the mechanism.

The latest full regression invocation had 745 passes and three bounded-verifier
timeouts. All three passed in a focused rerun with no code or deadline changes
(two pytest-cache warnings reflect the read-only sandbox). Both logs are retained (copied log trailing whitespace is normalized; raw logs
remain in `local/longgoal/`).
Compileall, shell syntax checks and git diff checks also passed.
The first corrected repair-pilot launch failed because `HF_HOME` was unset inside
the offline sandbox (`HOME=/tmp`). It was restarted with the existing verified
model cache explicitly configured; no source or experiment budget was changed.


## Three-seed semantic features and threshold stopping (2026-09-17)

The semantic-feature association gaps were 0.00633, 0.00260, and 0.01581 for
seeds 1701–1703. Seed 1703 had a positive CI [0.00477, 0.02947], but all three
still failed the required 0.03 effect size and strong-baseline fairness. These
are complete development-only results; no seed was discarded.

The threshold-stopping diagnostic split the 200 development-assessment sources
into 100 calibration sources (602 candidates) and 100 assessment sources
(662 candidates) using a prespecified source hash order. The original held-out
population is absent. The rule stops when maximum remaining public-test
diagnostic MI falls below a calibration-locked threshold. All four prefixes use
the trained SMC path; future labels enter evaluator NLLs only.

Requiring calibration NLL no worse than the same diagnostic-MI policy at four
tests selected threshold 0. The assessment used 3.99849 tests on average: only
0.03776% savings, with unchanged NLL. Threshold 0.1 saved 38.82% but worsened
NLL by 0.04367 (paired source-bootstrap CI for the advantage
[−0.05488, −0.03271]); it was not selected. The entire prespecified curve is
retained. These are cached observation counts, not measured physical execution
savings, and do not pass or establish a confirmatory active-testing gate.

The CodeARC prediction queue exposed a concurrent status-file read failure before
any training began. A persistent supervisor now retries partial JSON snapshots,
publishes its own status atomically, and preserves source identities and failures.
The failed queue attempt is retained locally. Eleven focused checks passed for
stopping, unchanged default SMC acquisition, and supervisor recovery behavior.
`run_apbpf_stopping_replay.py` reproduces the stopping experiment;
`run_local_codearc_prediction.py` waits for the completed development-only
CodeARC executions and the GPU handoff, then builds the cache and runs all three
prediction seeds. It remains a standalone workflow, not a complete real DAG worker.


## CodeARC association and actual candidate selection (2026-09-17)

All 400 development source groups (3200 candidates) completed ten-call execution.
The bank has 70 mixed, 293 all-fail and 37 all-pass groups, below the unchanged
300-mixed-group gate. Hidden outcomes were predominantly WRONG_OUTPUT
(12383/19200), with 953 runtime exceptions, 84 compile errors and 34 timeouts.
Visible-pass selection scored 25.50%; oracle Pass@8 is 26.75%, leaving only
1.25 percentage points, insufficient for a 3-point gain over visible-pass.

All three CodeARC association seeds completed with aligned NLLs
0.53334, 0.54161 and 0.55381; association gaps were −0.00128, 0.00187 and 0.00014.
Every association CI crosses zero, and all seeds failed baseline fairness.
Original primary candidates are absent from fitting and development assessment.

A separate learned-utility extension now evaluates actual selection. It trains
one shared success head over posterior particles, plus pair-aware, Deep Sets,
no-particle-bottleneck and tuned-rate controls. Only task text, candidate code,
four public test inputs and their outcomes enter selection features; future
inputs, expected outputs and execution outcomes are excluded. Utility targets
are six-hidden-call success labels used only for training/validation or scoring.
All learned heads use independent fitting/inner-validation source partitions;
five source folds select the strongest deterministic comparator using the other
folds. Fixed candidate order breaks ties. Every development group is retained.

| Seed | Particle selector Pass@1 | Cross-fitted comparator | Difference | Paired source 95% CI |
| --- | ---: | ---: | ---: | --- |
| 1701 | 24.75% | 25.00% | −0.25 pp | [−1.75, 1.25] pp |
| 1702 | 24.50% | 25.75% | −1.25 pp | [−3.00, 0.25] pp |
| 1703 | 23.50% | 25.00% | −1.50 pp | [−3.25, 0.25] pp |

No selection gate passed. These standalone utility-head experiments remain
exploratory; they are not a claim that the full real-stage DAG is complete.
Two regression tests cover exclusion of all future content and comparator
choice invariance to its own assessment-fold labels. All three 1000-step
selection runs completed with checksummed source/data/checkpoint identities.
Use `scripts/run_apbpf_selection_replay.py` with the matching development cache
and prediction checkpoint to reproduce them.

The Qwen primary bank completed public-only execution and pre-hidden sealing;
its hidden execution is underway. DeepSeek primary and training banks continue
generating; its remaining development bank is queued. Corrected repair training
continues, including full inner-validation passes between training segments.


The primary500 audit subsequently completed: 107 mixed, 350 all-fail and
43 all-pass groups; visible-pass selection 28.0%, oracle ceiling 30.0%.
The bank gate remains failed because the development mixed-group requirement
failed. The raw audit's `pilot_primary_source_disjoint: false` denotes absence
of the required passing pilot artifact; it does not establish actual source
collision. The initial materialization assigned disjoint source components.
No primary measurements are used to retune this recipe.

The raw RBR eligibility inventory found 1221 official-training sources and only
164 test-only sources under the current size/test-count constraints. Therefore
a 500-source primary bank cannot be built from the official-exclusive test pool.
A new generated-bank partition needs an explicitly exploratory source split and
fresh belief/utility fitting; existing belief checkpoints must not be reused on
sources they previously saw. The official-exclusive pool remains reserved.
