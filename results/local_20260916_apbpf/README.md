# Local A-PBPF progress snapshot — 2026-09-16

This is an **incomplete, nonconfirmatory** experiment milestone. The long-running
local workflow remains active. No positive A-PBPF claim or complete real-stage DAG
is established by these files. Legacy H200 formal experiments are pending external
resources; the user confirmed that only local execution is available.

**RBR protocol correction (2026-09-17):** the RBR results below used literal stdin. The pinned official runner appends a missing terminal newline. A new development control confirms this local discrepancy; those historic results remain diagnostics and require corrected execution/retraining. See `rbr_stdin_protocol_audit.json`. CodeARC results are unaffected.

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

Qwen CodeARC generation, primary audit, three prediction seeds, and three utility
selection seeds are complete; their scientific gates remain failed. All three
corrected-stdin RBR prediction seeds and the legacy repair pilot are also complete.
DeepSeek primary generation runs on GPU 1 and development generation on GPU 2;
its train212 bank is complete. GPU 0 now generates the RBR Qwen bank after the
repair handoff. GPU 3 belongs to another project. CPU supervisors execute the
new RBR/DeepSeek banks; a source-frozen queue will run the first five real stages
once all RBR Qwen candidates are ready. Only materialization (1/21) has actually
completed as a real stage so far.

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


## First real stage adapter: both-domain materialization (2026-09-17)

The new `run_apbpf_materialize_worker.py` completed through the actual stage
runner, with request fingerprints, evidence inventories and runner-authored
lineage. It read pinned raw RBR/CodeNet and CodeARC files and created separate
public/evaluator materializations: RBR 321/400/500 and CodeARC 212/400/500
training/development/primary source components. RBR sources come from official
training files; official test-only sources remain reserved. No correctness
filter was used. Its new partition requires fresh belief/utility training and
is explicitly exploratory because earlier experiments saw these source problems.

The `local_exploratory` real profile marks stages nonconfirmatory from the
outset and cannot become main-table eligible. `--through-stage` enables explicit
partial provisioning/execution while full verification still requires all 21
stages. The actual run completed **1/21** stages, with 20 pending; this is data
preparation evidence, not generation or a passing scientific gate. The first
materialization run is retained; the final run uses the recovery-command fix.

Seven focused checks passed, covering public/private record separation, source
splits independent of reference correctness, real partial-run resume/rerun,
failed-gate stops and refusal to verify an incomplete DAG. An actual generator
sandbox probe read all 1221 RBR public tasks, each with exactly four tests,
while both raw-data and evaluator-artifact directories were unmounted.
The full regression completed: **757 passed**, with one existing NumPy/PyTorch
read-only-array warning. Compileall, shell syntax and diff checks also passed.
Full attempt-owned data remain local; the snapshot includes their checksums,
stage completion record and read-only run report.

## RBR stdin correction and generated-bank execution (2026-09-17)

The [pinned official executor](https://github.com/giganticode/run_bug_run/blob/374251a9d65410f37e1136049cb7ff5dcca3d0ae/lib/run_bug_run/test_runner.rb#L344-L350)
adds a terminal newline to raw inputs that lack one. Two development reference
programs use `sys.stdin.readline()[:-1]`; literal input dropped the final digit.
The first 400-source reference control passed 3995/4000 calls (398/400 complete
programs). With only stdin normalization changed, the same control passed
4000/4000 calls and all 400 programs. The five old failures remain in the audit.
This does not establish that the other prior RBR failures have the same cause.
The local output matcher still uses its declared global absolute tolerance of
1e-4, rather than the official per-problem tolerance table.

New generated-bank tools use four whitelisted public tests, fixed eight-candidate
inventories, bounded prompts, pinned Qwen/DeepSeek weights and decoding budgets,
raw-completion retention, and source-bound resume checks. RBR programs execute in
per-call bubblewrap sandboxes with CPU/memory/output limits. Primary hidden
execution requires a checksum-bound full population lock with the RBR provenance
prefix; a CodeARC lock is rejected before private files are opened.

The legacy predictor preparation also now normalizes stdin and records its
execution protocol. Its existing 7934-candidate pre-screen inventory was recovered
from pinned raw sources (6523 train, 1290 development, 121 held-out candidates).
A fresh cache rebuild re-executed all formerly rejected fixed-program candidates. It completed with 6392 train, 1267 development, and 117 held-out candidates; 158 fixed programs were rejected (previously 193).
The historic cache did not record its timeout; the new rebuild explicitly fixes
2 seconds, so it is not presented as a fully matched one-variable comparison.
Three fixed 1000-step seeds (1701–1703) are queued on CPU, with seed 1701 complete and seed 1702 running,
using original train/development only. The first corrected prediction result is recorded below; the other seeds are still running. The live repair run keeps its original source identity and will
be archived as a literal-input diagnostic before corrected follow-up work.

`scripts/run_local_bank_evaluation.py` now queues full RBR Qwen and CodeARC
DeepSeek execution. It retains all 400 development groups, records the unchanged
300-mixed-group threshold, then evaluates all 500 primary groups visibly, creates
the pre-hidden lock, executes hidden calls, and audits the fixed population.
Cross-split sources and model/decoding identities must agree with the declaration.
The workflows do not complete additional sealed stages by themselves.

Focused validation passed 13 generated-bank/executor checks, 1 held-out-exclusion
check for corrected replay supervision, and 2 complete-bank inventory checks.
The full run passed 747 tests and failed 21: shell `umask=002` made test-owned factory dependencies group-writable, which existing trust checks reject. With `umask 077`, all 21 failures and the 3 subsequently added supervisor tests passed (24/24). In total, 771 distinct tests passed across the invocations; the initially failed full-run log is retained. Compile checks and `git diff --check` passed.

The corrected development comparison recovered 32 training candidates and 3
development candidates, while 2 training fixed programs were newly rejected.
Among common candidates, 72 training test labels and 10 development test labels
changed. These differences do not by themselves establish better model quality.
`rbr_development_protocol_difference.json` excludes original held-out metrics.

Corrected seed 1701 still fails association, invariance and baseline fairness:
aligned NLL 0.44975 versus 0.36935 for the no-particle comparator; association gap
0.0000462 nats, 95% CI [-0.011454, 0.011776], below the unchanged 0.03 threshold.
The input correction has not established the claimed model improvement.

A two-repeat, fixed-2-second check of the two newly rejected training fixed
programs found timeouts under both literal and normalized stdin for p03298;
p03458 passed both modes in both repeats. Execution stability is another source
of fixed-screen variation. The new cache remains frozen; these replays do not
reintroduce selected successes or tune the scoring deadline.

## Completed corrected seeds and real bank-stage adapters (2026-09-17)

The normalized-stdin RBR prediction replay completed all three fixed seeds.
Association gaps are 0.0000462, -0.0015569 and 0.0022625 nats for 1701–1703;
all three confidence intervals cross zero and all three fail association,
pair-invariance and strong-baseline fairness. Fixing the executor did not
establish the method claim. All reports retain development-only status.

The 1500-step projector / eight-task repair diagnostic completed under its
original literal-input protocol. Every arm solved 4/8 programs. The coherent
sample-once arm's future pass fraction was 0.7083 versus 0.6667 without latent,
but the random-latent control also reached 0.7083. This does not demonstrate a
useful conditioning gain. Its exact executed source is archived locally.
The repair executor now uses the corrected newline protocol for future runs.

Replaying all 48 unchanged completions under the bounded normalized-input
executor at the same 2-second deadline changed three calls, all PASS→TIMEOUT
on source p03298/task432870 (mean, MAP, sample-once). The new replay therefore
has 3/8 all-ten successes for those arms and 4/8 for the other arms. The original
projector training still used the old cache. These are timing/resource-sensitive
diagnostics, not paired evidence of an input-normalization or model effect.
`repair_stdin_rescore_v2.json` counts solved over all ten tests, matching the
original report; it also separately records future-only all-pass. The local v1
summary had incorrectly used future-only success and was superseded without
rerunning or selecting execution outcomes.

Four additional real adapters now implement public locking, execution caching,
hard-bank auditing and its gate. Explicit candidate reuse is bound by a frozen
manifest in the root request; only complete public generation bytes are imported.
All public/hidden executions are fresh, and both primary locks are checked
before private task data is opened. The CodeARC import was exercised on all
1112 source components / 8896 candidates. A first import check exposed the two
extra duplicate-source task IDs and bounded-prompt metadata; it was corrected to
match the original generator's fixed source representative rule. Twelve focused
checks passed, including true and false audit-gate worker contract fixtures.
Those fixtures are not real experimental stages.

The queued `apbpf-real-bank-prefix-v1` will execute through the fifth stage once
the full RBR Qwen banks finish. At this snapshot only the prior materialize
stage has actually completed (1/21). Sixteen later stages still lack adapters;
no complete DAG or passing scientific gate is claimed. DeepSeek train212 has
finished generation and its development384 generation and public-bank execution
supervisor continue. RBR Qwen generation now owns GPU0 after repair completion.

The first RBR Qwen generated pilot completed all 16 fixed development source
groups (128 candidates): 702/1280 individual calls passed and 55 candidates
passed all ten calls. Nine groups mix successful and unsuccessful candidates;
five are all-fail and two all-pass. Of 128 candidates, 112 have constant four-test
visible histories. These descriptive counts do not meet the 300-mixed-group
pilot threshold or establish association/selection performance. Full generation
and evaluation continue without changing the locked source population.

Validation of this adapter milestone: the complete invocation had 779 passes and
3 failures. One new regression fixture failed to register its dynamically loaded
repair module in `sys.modules`; the fixture was corrected. Two existing legacy
local-DAG checks hit bounded-verifier deadlines under disk pressure. All three
passed on the focused rerun (3/3, unchanged runtime deadlines). Thus all 782
distinct collected tests passed across these invocations. Both full and rerun
logs are retained; the original full invocation is not described as all green.
Compilation and `git diff --check` also passed.


## Generated-bank replication and memory recovery (2026-09-17)

DeepSeek CodeARC primary generation completed all 500 fixed source groups. GPU1
then moved to RBR DeepSeek replication: 321 train, 400 development and 500 primary
sources, eight independently seeded serial decodes per source, at most 4096 input
and 1024 new tokens per candidate. This serial decoding policy is recorded in the
new run identity; it is not claimed to be byte-identical to batched Qwen sampling.
Model revision and publisher weight hashes are pinned, and generation sees only
the public task materialization. The RBR pilot has started; the full family run
has not yet completed.

DeepSeek CodeARC development generation exhausted GPU2 memory at source 111
(CodeARC/508), after completing 110/384 groups. Attempt1 and its downstream
failures are retained. Attempt2 resumes the same generation configuration with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the previously failing group
and subsequent groups now complete. The plan freezes the first 110 output hashes
and the original run identity. This demonstrates progress past the failure,
not completion of the remaining bank. Evaluation v2 will score the entire fixed
population again; failed v1 outputs remain available, with no outcome selection.

The new development-cache builder binds complete generation and execution hashes,
keeps every candidate (including all-fail programs), excludes primary sources,
and blanks expected-answer text for the six future tests before feature creation.
RBR retains the four public expected outputs; CodeARC features remain query-input
only. This is a new generated-bank protocol, distinct from historic RBR expanded
cache experiments that explicitly declared every expected output public.
Actual-data checks retained all 4896 Qwen CodeARC candidates and redacted 29376
future expected slots without changing per-candidate public features or outcomes.
The 128-candidate RBR pilot check retained every execution and redacted 768 future
slots. These are cache-contract checks, not new efficacy results.

Three CPU prediction queues (RBR Qwen, RBR DeepSeek, CodeARC DeepSeek) wait for all
train/development execution steps, then build fresh source-disjoint development
caches and fit the fixed 1701–1703 seeds for 1000 steps each, with unchanged
association and baseline gates. Original primary sources are excluded. The
CodeARC prediction v1 stopped on upstream OOM without fitting; v2 follows the
restarted evaluation. Five focused tests passed, covering serial seed/resume
identity and future-answer feature isolation; compilation checks passed.

Only 1/21 real DAG stages has completed. Five stage adapters exist, and sixteen
are still missing. New standalone generation/evaluation/training queues do not
increase the completed-stage count or establish a scientific improvement.
