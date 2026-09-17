# Local A-PBPF progress snapshot — 2026-09-16

This is an **incomplete, nonconfirmatory** experiment milestone. The long-running
local workflow remains active. No positive A-PBPF claim or complete real-stage DAG
is established by these files. Legacy H200 formal experiments are pending external
resources; the user confirmed that only local execution is available.

**Latest checkpoint:** all 21 local real-stage adapters are now implemented, but
only materialization (1/21) has completed as a sealed real stage. The full pipeline
is queued behind the remaining RBR banks. CodeARC full-primary association and
selection are complete for both Qwen and DeepSeek, with negative effects in both
families. All nine configured prediction controls are now complete for both
CodeARC families, including the previously missing fixed `history_rate`.
See `full_pipeline_progress_v2.json`, `ablation_coverage_audit.json` and the final
sections below. The latest checkpoint is
`association_weight_comparison_queue_progress.json`: the all-three-seed
weight100 comparison is queued and identity-validated. Qwen RBR generation and
execution are complete for all 500 primary sources / 4000 candidates. Its full
three-seed cell and fixed history-rate control are complete: association is
0.000493 and selection advantage 0.004667, both with CIs crossing zero.
The two-GPU handoff is now real: coordinator796796 runs distinct DeepSeek
primary/train banks on GPU0/1; the original child continues and its old parent
remains parked. All three repair fits are complete and six-arm generation runs
on GPU2. The 132-batch gradient audit found weak association gradients in all
three prior refits; one fixed association-weight100 intervention is running,
with three seeds and1000steps, no primary tuning or altered evaluation gates.
All three Qwen RBR development prediction seeds fail association/fairness gates;
their checkpoint replay reproduces all 12 seed/control NLLs and the pooled
association interval crosses zero. GPU generation and CodeARC repair continue. Earlier
sections are historical checkpoints, not the current running-process inventory.

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


## Full stage caches and real training adapters (2026-09-17)

The execution-cache adapter now publishes source-bound full caches and separate
development-only derivatives for both domains. It checks both population locks
before evaluator access and joins primary visible and hidden records by exact
candidate identity and complementary test IDs. All eight candidates per source
remain present, including empty/all-fail candidates. Primary sources map only to
the full cache's test partition; six future expected answers are blanked.

A real-data assembly check joined the existing complete Qwen CodeARC records:
1696 train, 3200 development and 4000 primary candidates (8896 total), covering
212/400/500 source components. The derived development cache matched the previous
cache's public features and outcomes while excluding all primary sources. This
uses existing execution evidence; it is not a newly completed execution stage.

`run_apbpf_train_worker.py` adds actual `train_baselines` and `train_belief`
adapters. Both consume the exact declared full cache and fit seeds 1701–1703
in both domains. Training uses fixed hash-text features, 1000 steps, batch64,
learning rate0.0003, feature256/hidden192; protocol particle count and loss
weights remain unchanged. Only original train data enters optimization and only
original development data selects checkpoints. Three neural baseline states and
the tuned Dirichlet parameter are saved; belief checkpoints and all primary
prediction matrices are also retained. Primary predictions follow completed
fitting. Scientific gates and cross-fitted selection remain downstream work.

Thirteen cache/worker checks passed. Four training checks passed, including
counterfactual changes to primary hidden labels that leave fitted weights,
checkpoint selection and predictions unchanged, plus complete two-domain,
three-seed worker contract calls. These short fixture fits do not count as
scientific training runs. Compilation and whitespace checks passed.

The unstarted prefix-v1 supervisor was verified waiting, with no stage directory
or child process, then archived before the execution-cache source changed.
Prefix-v2 is now queued through `train_belief` with explicit exploratory
continuation after failed gates. Its new source identity is frozen; GPU jobs and
independent development prediction queues continue. Seven real adapters now
exist, fourteen are still missing, and only one real stage has completed.


## Fairness and association adapters; DeepSeek RBR pilot (2026-09-17)

Four more adapters implement baseline fairness, full-population association
replay, the association gate and the pair-invariance gate. They validate the
actual training artifacts, all three fixed seeds, both domains, matched fitting
budgets and exact 500-source/eight-candidate primary inventories. The replay
restores actual belief checkpoint bytes and first reproduces aligned predictions
before computing outcome-shuffle and pair-preserving controls, plus masked,
wrong-candidate and random-latent diagnostics. Ambiguity strata are descriptive;
all constant histories and failed candidates remain in the gate population.

The predeclared local aggregation averages paired per-example losses across
1701–1703. It does not average probabilities or select a best seed. Whole-source
bootstrap draws keep observations from all seeds in the same source cluster;
10000 draws and all existing numeric thresholds remain unchanged. Per-seed
results and raw prediction matrices remain available. Fairness must pass in both
domains for every baseline; its scalar gate rows use domain-minimum gaps and CI
lower bounds. The association and invariance gate decisions are recomputed by
the existing runner.

Eight focused checks passed, including seed-loss aggregation, actual two-domain
fairness worker success/failure paths, restored-checkpoint controls, and gate
agreement with the locked runner. The A-PBPF regression suite passed 118 checks
with one existing read-only NumPy-to-Torch warning in acquisition replay.
Compilation and whitespace checks passed.

The unstarted seven-stage prefix-v2 was verified childless/waiting and archived.
Prefix-v3 now queues eleven stages through pair_invariance_gate, retaining failed
scientific gates under explicit exploratory continuation. Only materialization
has actually completed as a real stage (1/21); ten adapters remain absent.
Later acquisition workers will also need verified feature/checkpoint artifacts
forwarded through their declared gate dependencies; current scalar gate outputs
alone are not sufficient for those stages.

The new standalone CodeARC full-cache diagnostic is actually fitting all three
fixed seeds against the full existing train/development/primary split while RBR
generation proceeds. It saves real baseline and belief checkpoints and replays
all controls on 500 primary groups. At this snapshot completed fits are baselines-seed1701, belief-seed1701, baselines-seed1702;
the current task is belief-seed1702. This is permanently exploratory,
uses the previously assembled execution evidence, and is not a sealed stage or
a completed three-seed scientific result.

DeepSeek RBR generation completed its 16-source pilot and moved to train321.
Pilot evaluation retained 128 candidates/1280 calls: 558 calls and 44 complete
programs passed. Ten source groups are mixed, four all-fail and two all-pass;
111 candidates have constant four-test categorical histories. The prior Qwen
pilot on the same source inventory had 702 passing calls, 55 complete successes
and nine mixed groups. These small pilot counts do not establish a family-level
benefit or satisfy the full hard-bank gate.

The DeepSeek pilot has five syntax-invalid programs and no token-cap hits. Four
raw completions already contain malformed Python; one supplies only prose that
no repair is necessary. All 58 timeout calls occur on p03704, in candidates that
perform very large integer enumeration or while-loop search. Raw outputs, all
failures and the six-second scoring deadline remain unchanged. This evidence
does not support fixing these failures by changing extraction or extending the
output cap selectively.


## Completed full-primary diagnosis and query stages (2026-09-17)

The standalone CodeARC full-cache diagnostic completed all three fixed seeds,
with 212 original training, 400 development and 500 primary source groups.
The seed-mean full-primary association gap is -0.00261796 nats/test (95% source
CI [-0.00530136, 0.00006508]); every seed has a negative gap. All four strong
baselines outperform belief: paired NLL advantages for belief are -0.0756490
(pair-aware), -0.0591516 (Deep Sets), -0.0498716 (Dirichlet), and -0.0629411
(no-particle bottleneck). All four confidence intervals lie below zero.
Association, invariance and baseline-fairness claims remain unsupported.

Failure diagnosis retained the whole population. Of 4000 primary candidates,
2715 have constant visible histories and exactly zero shuffle gap. The other
1285 candidates have gap -0.00814929, with CI [-0.0165768, 0.00009206]. Thus
constant-history dilution alone does not explain the missing positive effect.
Best checkpoints were selected on development at steps400/450/350 for
1701/1702/1703, with NLL0.5281/0.5299/0.5329. Step1000 development NLL worsened
to0.6547/0.6484/0.6468. The evaluator already restored the better checkpoints;
no primary-based reselection or favorable-source filtering was performed.
The exact executed training/association source was archived before later
adapter edits. Raw checkpoints and prediction arrays remain locally bound by
checksums; repository summaries include the training curves and report digests.

Assessment bundles now forward byte-identical evaluator caches and belief
checkpoints through declared direct stage dependencies. Four real adapters add
oracle headroom, its gate, active testing and its gate. Oracle subsets use
canonical ordering for both fixed and uniform-random subset comparators, with
the main local oracle check fixed in advance at budget2. Budgets1/2/4 are all
reported. Active policies use only the four public slots; all six future labels
remain evaluator-side. Diagnostic MI, fixed, random, predictive entropy and
joint-particle MI all preserve exact fixed observation budgets. Threshold
stopping is calibrated on original development sources before primary replay,
with complete separate threshold--quality curves retained. Cached observation
counts do not establish physical runtime savings.

The actual CodeARC oracle replay completed three seeds over all 500 primary
sources/4000 candidates. At budget2, oracle NLL advantage is0.0762357 over fixed
(95% CI[0.0659394,0.0877039]) and0.0714757 over random canonical subsets
(CI[0.0643118,0.0791274]), above the unchanged0.03 threshold in this one-domain
standalone diagnostic. This is an evaluator upper bound using hidden labels,
not a realizable policy or a passing staged two-domain gate. At budget4, oracle,
fixed and random canonical subsets coincide exactly because the public pool
contains only four tests. The standalone progress counter stopped at its last
64-item update (3968); the preserved count audit verifies all4000 records for
each seed and the complete report. No records were missing.

The full CodeARC active-policy/stopping diagnostic is now running with the same
checkpoints and complete primary population. Prior association/fairness failures
remain binding. The waiting prefix-v3 was archived without interrupting a stage;
prefix-v4 queues fifteen real stages through active_testing_gate. Six later
adapters remain absent and only1/21 real stages has actually completed.

Seventeen focused query/bundle/prediction checks passed. Full A-PBPF regression
passed127 checks with the pre-existing read-only NumPy-to-Torch warning in the
legacy diagnostic adapter. Compilation and whitespace checks passed.


### Completed active testing and selection; development representation debug queued

The full CodeARC active replay completed all three seeds, each with 4000 primary
candidates from 500 sources. At four observations, diagnostic MI has NLL
0.5455255 versus fixed 0.5451005: advantage -0.0004250, 95% CI
[-0.0027857, 0.0020009]. Its advantage over random order is 0.0002127,
CI [-0.0016743, 0.0020902]. The original 0.03 threshold is not met.
Development-calibrated stopping uses 3.998833 observations on average, a
0.02917% reduction, and fails the hidden-quality matching check. The positive
oracle upper bound has not become an effective executable acquisition policy.

Full-bank utility selection also completed three seeds with 1000-step fitting
budgets. All primary sources and candidates were retained. The three particle
Pass@1 values are 0.268, 0.274 and 0.266; the cross-fitted deterministic
comparators are 0.282, 0.280 and 0.282. The aggregate paired advantage is
-0.012, CI [-0.022, -0.002], below the unchanged +0.03 requirement. Fresh and
pretrained deterministic heads, tuned Dirichlet and visible-pass-rate controls
are retained. The candidate-bank oracle is 0.30 and visible-pass-rate selection
is 0.28, leaving only two percentage points of oracle headroom relative to that
specific control. This ceiling is descriptive, not a new criterion or permission
to filter the primary bank. Original development alone chooses neural utility
checkpoints; source-disjoint folds choose the deterministic comparator.

The selection worker and gate bring implemented adapters to 17/21. The previous
waiting prefix-v4 was archived before executing any stage; prefix-v5 declares
all stages through selection_gate, with failed gates preserved in exploratory
lineage. Only 1/21 stages has actually completed as a real DAG stage. Repair,
replication, replication_gate and paper_tables still need adapters.

A new CodeARC development-only diagnostic compares lexical hashing against
frozen Qwen representations at the same 512 feature dimensions, three seeds,
1000 training steps and four strong baseline controls. It uses 170 fitting,
42 inner-validation and 400 original-development sources; all 500 original
primary sources are excluded. Public text extraction receives no execution
outputs, reference answers or labels. GPU2 extraction is queued after the live
DeepSeek generation finishes; lexical controls run on CPU meanwhile. This is
an exploratory representation hypothesis, with no primary re-evaluation.

RBR Qwen training generation completed all 321 sources and its execution is
running; development generation has begun. RBR DeepSeek training generation
and the resumed CodeARC DeepSeek development generation continue unchanged.
The full A-PBPF regression passed 131 checks; the additional public-text
isolation check passed separately. One existing read-only NumPy warning remains.


### DeepSeek CodeARC generation complete; actual repair inputs prepared

DeepSeek generation has completed all 1112 CodeARC sources: 212 training,
400 development and 500 primary, with eight candidates each (8896 total).
A completion audit verifies source disjointness and checksummed banks. All110
completed groups retained after the CUDA OOM are byte-identical, and the
original run identity is unchanged. Only the previously declared expandable
allocator setting changed. Development execution has started automatically;
this generation completion does not establish a replication gate result.

The execution-cache adapter now exports separated repair targets, evaluator
tests and public context. Training/development reference programs are retained
for supervision; all primary reference programs are excluded. Hidden tests
remain evaluator-only and CodeARC exception semantics are preserved. The
selection gate forwards the exact selected-candidate reports to its direct
repair dependency without exposing those label-bearing reports to the actor.

Actual CodeARC repair packets have been prepared for all three seeds. Each
contains1696 training candidates,3200 development candidates and the actual
particle-selected candidate for each of500 primary sources. Every actor row
has exactly four public observations. Eight24-dimensional diagnosis particles
and their weights were recomputed from each seed's actual belief checkpoint;
the8-dimensional difficulty component is excluded. No future invocation,
future execution evidence, primary reference target or primary success label
is included in these actor packets. Packet preparation is not repair training,
generation or scoring. Those remaining steps still need implementation.

Prefix-v5 was archived while waiting, without executing any stage. Prefix-v6
now declares the changed execution-cache and selection-gate contracts. The
adapter count remains17 and real completed-stage count remains1/21. Full A-PBPF
regression passed145 checks;13 repair checks passed again after strengthening
the public-context binding checks. Exact executed packet-preparation source is
retained for the earlier completed run.


### Real repair training and six-arm generation connected

A public-only packet actor now trains the actual diagnosis prefix against
nonprimary reference programs, generates all six declared repair controls,
and records fixed-component traces, raw token IDs and generated code. CodeARC
prompts explicitly request the callable solution interface and preserve public
expected-exception semantics. Primary targets and future tests are absent from
the actor packet. A separate evaluator seals the entire generated source/arm
inventory before opening private tests, then freshly executes each program.

A real-data engineering smoke used four short training candidates, two short
development candidates and one selected primary candidate. Development source
selection reduced its two candidates from the same source to one validation
item. One projector step completed with finite loss and gradients; all six
32-token-budget generations completed and their60 newly executed tests passed.
This shortest-text engineering fixture establishes executable plumbing only;
it is not evidence of repair efficacy and does not pass any scientific gate.

The full CodeARC run has started on GPU2 with all three seeds,1500 projector
steps each and512-token continuations. Each seed retains all500 selected
primary sources and all six arms. Training uses1672 of1696 candidates and
validation uses394 of400 source representatives under the fixed1536-token
context cap;24 training candidates and6 validation representatives exceed that
cap and are explicitly audited. Primary generation retains all500 sources.
Normalization uses training posteriors only; checkpoint selection uses
original development only. The evaluation queue waits for sealed generation
and does not infer repair scores from the original candidate cache.

A real `repair` stage adapter connects both domains and all three seeds through
declared direct dependencies. Prefix-v6 was archived while still waiting;
prefix-v7 provisions18 stages through repair. Replication, replication_gate and
paper_tables still need adapters, and real completed-stage count remains1/21.
147 A-PBPF checks passed, in addition to the actual isolated GPU/CPU smoke.

The dimension-matched CodeARC development feature comparison completed all six
fits. Semantic association gaps for1701–1703 are0.0092842,0.0065694 and0.0014091,
versus lexical0.0011261,-0.0015974 and-0.0032847. Only the first semantic
seed has a positive confidence lower bound, and all semantic gaps remain below
0.03. Every semantic seed fails baseline fairness and pair invariance. Mean
aligned NLL improves by0.0314739 across the three fixed seeds;
this is a descriptive comparison, not a pooled confidence interval or evidence
of particle necessity. Complete per-seed reports and the comparison are retained.


### DeepSeek full-bank outcomes and three remaining replication cells

The DeepSeek CodeARC replay completed all1112 sources. Development has99 mixed,
292 all-fail and9 all-pass groups, below the fixed300-mixed-group requirement.
Primary has133 mixed,344 all-fail and23 all-pass groups. Visible-pass-rate
selection achieves0.27 hidden selected Pass@1; the bank's hidden oracle is0.312,
a0.042 headroom. The audit remains audit-only because the development pilot
criterion failed. Its absent-pilot disjointness flag is not evidence of actual
source overlap; full cache assembly independently checks source disjointness.

The independent development-only prediction seeds1701–1703 also completed.
Association gaps are-0.0014895,-0.0071972 and-0.0011281; all three confidence
intervals include zero and all association/fairness/invariance gates fail.
These development results are not substituted for full-primary replication.

A generic complete-replay cache assembler now verifies generator identity,
all original splits, evaluator/public manifests and exact primary locks.
Two independent CodeARC DeepSeek assembly invocations produced byte-identical
full caches (1696train/3200development/4000primary candidates). A queued
replication-cell runner fits the original three1000-step seeds, all matched
baselines, complete association controls and utility selection. DeepSeek CodeARC
is actively training; RBR Qwen and DeepSeek wait for their complete replays.
The already-completed Qwen CodeARC cell has association-0.002618 and selection
advantage-0.012, preserving both negative findings and original source bindings.

Replication-gate and exploratory table-export adapters are implemented. The
replication gate requires the full2-domain by2-family matrix and unchanged
positive-direction agreement. Tables retain every failed gate and cannot
become main-table eligible.20/21 adapters now exist; the actual replication
adapter and its root-declared input chain remain incomplete. The waiting real
prefix still covers18 stages through repair, and only1/21 real stages has
actually completed.156 A-PBPF checks passed.

Full CodeARC repair is still running. At the first250-step development check,
teacher-forced reference NLL was0.770587 with the prefix versus1.260503 without
it on394 source representatives/34047 target tokens. This is supervised
sequence likelihood, not new-program test success or evidence of diagnostic
particle necessity; the six-arm generation/execution comparison remains pending.

## Complete local adapter coverage and full DeepSeek cell (2026-09-17)

The final real `replication` adapter is implemented. A separate bridge wraps the
unchanged materialize, execution-cache, association and association-gate workers.
Both DeepSeek replay caches must be declared by immutable root manifest. The
bridge reconstructs each cache from its bound raw candidate executions, checks the
pinned generator and decoding budget, and compares the materialized public and
evaluator task inventories. It forwards these inputs only through declared direct
DAG dependencies. Existing candidate execution is explicitly reused; replication
belief/baseline/selection fitting is fresh. Qwen cells come from the same run's
association and selection stages. No standalone trained result is silently imported.

The full 21-stage supervisor is queued as `apbpf-real-full-pipeline-v1`, waiting for
complete Qwen banks and DeepSeek RBR replay. The old 18-stage prefix was verified
childless and without any stage attempt, then archived and superseded. GPU0/GPU1
RBR generation and GPU2 packet repair continue untouched. This is implementation
coverage and a queued run, **not 21 completed experiments**. The old materialize
run is still the only sealed real-stage completion. All scientific gates retain
their original thresholds and all execution remains exploratory.

The actual CodeARC DeepSeek full-primary cell completed all three seeds on 500
sources and 4000 candidates. Association is **−0.00242874 nats**, source-bootstrap
95% CI [−0.00486621, −0.00004403]. Selection against the strongest cross-fitted
deterministic comparator is **−0.018**, CI [−0.0293333, −0.0073333]. Both effects
are negative, consistent in direction with the earlier negative Qwen CodeARC
cell. The four-cell replication matrix remains incomplete until RBR finishes;
these results already fail the required positive direction in every cell.
All seed reports, cache provenance and aggregate negative results are retained.

Validation: 167 A-PBPF tests passed (one existing read-only NumPy warning), followed
by 12 final adapter checks after adding bounded retry for partially written legacy
status files. Root-import validation on the real 8896-candidate DeepSeek CodeARC
cache reproduced its digest exactly. This engineering check is not a completed
DAG stage. Repair seed1701 has reached step1000/1500 and is validating; neither its
teacher-forced loss nor an unfinished generation run is evidence of repair efficacy.

## Required prediction-control audit and finite-contract verification (2026-09-17)

Auditing `configs/apbpf/ablations.yaml` found that full-stage reports contained
tuned Dirichlet but omitted the distinct fixed-alpha `history_rate` control.
The missing ablation is now implemented without fitting or changing the existing
models, association results, primary population or numerical gates. It predicts
six future categorical outcomes from the four observed categories using fixed
Laplace alpha=1. Tests verify independence from hidden labels and semantic fields,
visible-order invariance and rejection of incomplete public histories.

Actual supplementary evaluation on all 500 CodeARC primary sources and all three
seeds completed for both families. Aligned prediction beats this fixed weak
control by 0.278225 NLL for Qwen (95% CI [0.245953, 0.307247]) and 0.297694 for
DeepSeek ([0.268244, 0.324061]). These positive weak-control comparisons do not
alter the negative association results or failures against stronger matched
baselines. All nine configured prediction controls now have complete CodeARC
results; RBR remains queued behind bank generation and fitting.

The stage bridge writes supplementary history-rate evidence while preserving the
original association report byte-for-byte. Replication evaluates the same control
on freshly fitted DeepSeek models, and paper tables export all four cells. A real
full-CodeARC engineering check reproduced standalone metrics exactly. Full queue
v1 was confirmed childless with no stage execution, archived, then replaced by
v2 (all 21 stages with the additional ablation). Existing generation and repair
jobs were not interrupted. The A-PBPF suite passed 170 tests, with one pre-existing
read-only NumPy warning. This does not establish completed real-stage execution.

The existing standalone legacy finite-contract run contains all 60000 cases:
50000 train, 5000 development, 5000 test, across 120 disjoint families. Every
archive SHA-256 and the specification SHA-256 were verified. Every reported
per-arm aggregate metric and all finite gate decisions were independently
recomputed from the archived metrics. All gates remain failed; on test the
32-particle median posterior KL is 2.10842, relative future-NLL gap is 0.0336261,
and HPD coverage is 0.7994. This is a verified negative local diagnostic, not a
sealed formal S1 result. Historical execution source hashes are absent, and this
audit does not retroactively claim immutable source provenance.

## Frozen-model particle sensitivity and repair training (2026-09-17)

A predeclared development-only numerical diagnostic evaluated every combination
of two feature encoders, three frozen model seeds and 8/32/128 inference particles
(18 cells). It retained all 3200 candidates from the 400 original development
sources; no original primary source entered the cache. Model, cache, feature and
population receipts were verified. Each original eight-particle result was
numerically reproduced before interpreting the larger budgets. Raw predictions,
resampling ancestry and ESS are retained locally with complete checksums.

| Frozen encoder | Particles | Aligned NLL | Association gap | Source-bootstrap 95% CI |
|---|---:|---:|---:|---|
| Lexical512 | 8 | 0.538300 | −0.001252 | [−0.004252, 0.001622] |
| Lexical512 | 32 | 0.506779 | 0.000106 | [−0.001088, 0.001293] |
| Lexical512 | 128 | 0.498018 | 0.000330 | [−0.000158, 0.000818] |
| Semantic512 | 8 | 0.506826 | 0.005754 | [−0.001798, 0.013506] |
| Semantic512 | 32 | 0.479641 | 0.001043 | [−0.002506, 0.004843] |
| Semantic512 | 128 | 0.471963 | 0.000803 | [−0.001094, 0.002984] |

The paired NLL improvements from8 to128 particles are 0.040282 for lexical
features (CI [0.032732, 0.049663]) and 0.034863 for semantic features
([0.028227, 0.042142]). These are inference-budget effects, not active-test
acquisition gains. Every association interval still crosses zero and the
0.03-nat criterion remains unmet. Average surviving initial particles rise from
6.32 to91.23 (lexical) and from5.93 to83.65 (semantic). Uniform post-resampling
weights are not counted as restored ancestry diversity; two focused regression
checks verify that distinction. Full details and all budgets are in
`codearc_particle_sensitivity_summary.json` and `codearc_particle_sensitivity_results.json`.

A bounded follow-up is now fitting all three semantic model seeds with32
particles, keeping the same 1000-step optimization budget, frozen features,
170 fitting/42 inner-validation/400 development-assessment source partition,
and four strong baselines. It tests whether training approximation contributes
to weak association. Its plan is recorded before fitting; no primary evaluation
or original protocol replacement occurs. Compute increases with particle count,
so this is not a compute-matched improvement claim. All three results will be
retained regardless of direction.

CodeARC repair seed1701 completed1500 steps. The saved projector matches the
best development checkpoint at step1250, with teacher-forced NLL0.745874 versus
1.260503 without a prefix on394 validation items/34047 target tokens. The full
validation history and checkpoint checksum are retained. Seed1702 is running;
repair success still requires all remaining fits, six-arm code generation and
fresh hidden execution. Qwen RBR development384 generation also completed and
its execution queue started; GPU0 has moved to primary500 generation.

A reproducible audit of the existing60000-case finite outputs found that the
32-particle final approximation retains about6.4 states above the metric's
numerical floor. On test, those floor states carry15.95% of exact posterior mass,
and the true latent lies there in16.06% of cases. Their mean KL contribution is
3.99042 versus mean total KL3.92179 (other states can contribute negatively).
This describes support loss in the archived discrete problem. It cannot separate
initial proposal omission from later resampling loss, and does not establish a
cause for the neural continuous-latent model. No finite gate was changed.

## Generated RBR development completion and matched-budget comparison

Qwen RBR development execution is complete for all 400 sources, 3200 candidates
and 32000 tests, using the corrected official terminal-newline convention.
There are 231 mixed-outcome source groups, 82 all-fail and 87 all-pass groups.
The mixed count is below the required 300; this is a development diagnostic,
not a completed sealed hard-bank gate. The predictor cache contains 257 fitting,
64 inner-validation and 400 assessment sources, excludes every original primary
source, and redacts future expected answers from features.

All three fixed 1000-step prediction fits have completed. Their reports and
checkpoint/population receipts were verified before this snapshot.

| Seed | Aligned NLL | Association gap | Source-bootstrap 95% CI |
|---|---:|---:|---|
| 1701 | 0.419151 | 0.014146 | [−0.002615, 0.031804] |
| 1702 | 0.407933 | 0.007906 | [−0.002287, 0.018890] |
| 1703 | 0.432493 | 0.000940 | [−0.008041, 0.009549] |

All three association and strong-baseline fairness gates fail. These are
individual seed intervals, not a pooled three-seed interval. Complete receipts
and gate decisions are in `rbr_qwen_generated_development_complete.json`.

The first semantic 32-particle retraining seed also completed: association
0.003730, CI [−0.005455, 0.012829], with failed association/fairness and passed
pair invariance. Its aligned NLL is 0.483049. At the same 32-particle inference
budget, the old eight-particle-trained seed1701 model has NLL 0.481475, so this
single-seed comparison does not show a training benefit. The remaining refits
continue. `codearc_semantic_p32_comparison_plan.json` declares an automatic
comparison of all three old/new models at 32 inference particles, with source
bootstrap intervals and prediction receipts. It waits for all three refits and
uses only development assessment sources; it is not additional training or a
primary evaluation. Training compute is not matched across these models.

The completed RBR development checkpoint replay restores all three models and
reproduces every aligned, outcome-shuffled, joint-reversed and
presentation-permuted NLL, with a maximum absolute error of 2.25e-8. Pooling
paired losses across seeds while keeping each of the 400 sources as one
bootstrap cluster yields aligned NLL 0.419859 and association gain 0.007664,
95% CI [−0.001760, 0.017550]. Pair-preserving reversal and permutation gaps are
0.004180 and 0.002970 respectively; their intervals also cross zero. This is
still negative association evidence under the unchanged criterion. The shared
aggregation helper's historical `primary_candidates` field denotes 3200
development candidates in this report; no primary examples were evaluated.
See `rbr_qwen_generated_development_pooled.json` for aggregate and per-seed
contrasts, raw prediction checksums, and the original gate decisions. Bootstrap
draws use fixed seed 201701 for the summary, so individual intervals can differ
slightly from the original seed-specific reports.

Semantic 32-particle refit seed1702 has also completed, with association
0.005944, CI [−0.003116, 0.015245] and aligned NLL 0.494202. Association,
strong-baseline fairness and pair invariance fail. Seed1703 and the queued
same-inference-budget comparison remain in progress at this checkpoint.

## Completed particle retraining and order diagnosis

All three semantic 32-particle development refits have now completed. The third
seed has association gain 0.010543, CI [0.001907, 0.020179], below the unchanged
0.03 criterion; fairness and pair invariance also fail. Across all three seeds,
association is 0.006739, CI [−0.000534, 0.014309]. At the same 32-particle
inference budget, the new models have NLL 0.485320 versus 0.479641 for the old
eight-particle-trained models. The paired improvement is −0.005679, CI
[−0.011639, 0.000053]. Increased training particle count therefore has not
established a benefit. All checkpoints, raw replay predictions and original
gate decisions remain bound to `codearc_semantic_p32_comparison_results.json`.

Code inspection identified two possible numerical sources of order sensitivity:
the proposal depends on the first observation, and intermediate resampling
discards particle support. A fixed 2x2 inference diagnostic used all three
original eight-particle-trained semantic models, all 400 development sources,
and 32 inference particles. It crossed learned-first-observation versus
root-prior proposals with ESS-0.5 resampling versus no resampling. All 12 cells
and their four controls completed; the standard cell reproduced the existing
prediction arrays before any alternative was interpreted.

| Proposal | Resampling | Aligned NLL | Association gap | Association 95% CI |
|---|---|---:|---:|---|
| Learned first observation | ESS 0.5 | 0.479641 | 0.001043 | [−0.002506, 0.004843] |
| Learned first observation | None | 0.476244 | 0.000935 | [−0.002108, 0.004004] |
| Root prior | ESS 0.5 | 0.508474 | −0.000114 | [−0.000426, 0.000179] |
| Root prior | None | 0.507851 | −0.000005 | [−0.000018, 0.000008] |

Removing resampling with the learned proposal improves NLL by 0.003397,
CI [0.002032, 0.004985], but does not establish association. The largest
pair-order probability difference across all examples/seeds is 0.6700 for the
standard implementation, 0.6298 without resampling, 0.1799 for the prior with
resampling, and 4.77e-7 for the prior without resampling. Thus removing
resampling alone does not remove observed order sensitivity. The prior/no-
resampling control is invariant to numerical precision but worsens NLL by
0.028210, with improvement CI [−0.039227, −0.018579]. Numerical invariance has
not produced a successful replacement method. These are exploratory diagnostic
intervals, with no primary evaluation, checkpoint refitting, or gate changes.

`codearc_order_decomposition_results.json` retains every cell and raw prediction
checksum. A focused regression test independently checks static-Bayes weighting,
pair-order invariance and changed-outcome sensitivity of the prior/no-resampling
control; it passes. The training and inference diagnostic scripts do not modify
the source files frozen by the live packet actor and full pipeline.

## Exchangeable proposal diagnostic and second repair fit

A fixed follow-up pools the four visible per-observation proposal Gaussians by
matching their mean and marginal variance with one diagonal Gaussian. This is
a single Gaussian proposal with its exact prior/proposal density correction,
not an evaluation of a mixture density. It retains all three frozen semantic
models, 32 particles and all 400 development sources, and disables resampling.
Every original model/feature receipt and reference prediction checksum was
verified. No checkpoint was refit and no primary source was evaluated.

The pooled aligned NLL is 0.475059. The paired improvements are:

| Reference inference | NLL improvement | Source-bootstrap 95% CI |
|---|---:|---|
| Learned first-observation proposal, ESS 0.5 | 0.004581 | [0.001477, 0.007741] |
| Learned first-observation proposal, no resampling | 0.001185 | [−0.001609, 0.003818] |
| Root-prior proposal, no resampling | 0.032792 | [0.022278, 0.045004] |

All pair-preserving order differences are at most 4.77e-7 in probability.
However, association is −0.00000466, CI [−0.00002968, 0.00002092], far below
the unchanged gate. Resolving numerical order sensitivity therefore has not
recovered the required association signal. These exploratory comparisons use
four proposal evaluations rather than one and are not compute-matched claims.
All outputs and references are in `codearc_pooled_proposal_results.json`.
Two focused tests pass: they verify exact importance weighting, order invariance,
dependence on every visible outcome, and exclusion of future tests/outcomes.

Repair seed1702 completed all 1500 steps and restored its best development
projector from step1250, with teacher-forced NLL 0.745385 versus 1.260503 without
a prefix on the same 394 items/34047 tokens. The saved projector tensors were
verified against the best-state tensors and the supervisor's checkpoint hash.
The final-step development NLL was 0.746973. The full validation trajectory is in
`packet_repair_seed1702_training_complete.json`. Seed1703 has started; six-arm
generation and fresh hidden execution remain required before repair efficacy
can be assessed.

## Likelihood assignment audit and bounded training follow-up

The next diagnostic fixes the same 32 prior particles for aligned and shuffled
visible outcomes, so the proposal and sampling support cannot change between
them. Direct likelihood weighting reproduces both archived predictions for all
three frozen semantic models within 4.77e-7. All 3200 development candidates
remain included: 1025 have mixed visible outcomes and 2175 have constant visible
outcomes. Constant histories have exactly zero assignment contrast.

| Frozen seed | Mean centered log-ratio RMS, mixed histories | Mean posterior total variation, mixed histories | Mean maximum future-probability difference, mixed histories |
|---|---:|---:|---:|
| 1701 | 0.004025 | 0.000975 | 0.000226 |
| 1702 | 0.006353 | 0.001710 | 0.000418 |
| 1703 | 0.004054 | 0.001270 | 0.000160 |

The log ratio is centered across particles because a constant likelihood ratio
does not change normalized posterior weights. These descriptive results show
weak dependence on outcome assignment on the sampled support. They do not
establish a property of the exact continuous posterior. Raw per-candidate arrays,
all-population summaries and quantiles are retained by
`codearc_likelihood_assignment_results.json`. A focused test distinguishes a
known separable likelihood, whose assignment ratio is constant over particles,
from an interacting likelihood with a large posterior change.

One fixed follow-up now refits all three seeds with a prior proposal and no
resampling, using the same 32-particle budget, 1000 steps, semantic features,
170/42/400 source partition, original losses, optimizer and four strong
baselines. It tests whether removing the first-observation proposal route during
training helps the likelihood learn assignment dependence. No primary sources
are used and every seed will be retained. Its immutable plan is
`codearc_prior_training_plan.json`. A focused test verifies prefix visibility
and an actual likelihood update while the original proposal-head parameters
remain unused. Checkpoint loading must use the declared `PriorTrainingBelief`
class and per-seed variant sidecar; the default model has matching tensor shapes
but different filtering behavior. No efficacy conclusion is available yet.

The post-training comparison is now declared and queued in
`codearc_prior_comparison_plan.json`. It waits for all three fits and restores
the declared model variant, rather than inferring behavior from tensor shapes.
Both the old and new models were trained with 32 particles. Comparing them
under the same prior/no-resampling inference separates the training change
from the inference change; a second contrast retains the old standard inference
to measure the combined procedure change. The comparator must reproduce the
new per-seed reports and the old standard prediction arrays before pooling
source-level evidence. `prior_comparison_queue_progress.json` records the
verified live processes and current training/generation counts. No comparison
result is available at this checkpoint.

The first prior/no-resampling refit (seed1701) has now completed, with its
variant sidecar and model/report/population receipts verified. Aligned NLL is
0.456188; association is −0.00003226, CI [−0.00008408, 0.00002093]. Association,
strong-baseline fairness and the original pair-invariance gate all fail. The
pair-aware deterministic baseline has NLL 0.439310 and Deep Sets has 0.441837.
This is one seed only; the all-seed same-inference comparison is still pending.
The second seed started automatically. Its report is
`codearc_prior_training_seed1701.json`, and its mandatory variant binding is
`codearc_prior_training_seed1701_variant.json`. Observed generation throughput
and verified live process identities are retained in the progress record;
throughput measurements do not guarantee future completion times.

## Queued parallel replication scheduling

Observed RBR generation throughput is substantially lower for the required
serial DeepSeek decoding than for Qwen. A coordinator now waits for the complete
Qwen primary bank and an idle GPU0 before changing the DeepSeek schedule. It
will park only the original scheduler, keep the existing GPU1 generator alive,
and assign remaining whole banks to distinct output directories on GPU0/GPU1.
The largest declared source inventory is assigned first, without inspecting
outcomes. The old scheduler can be retired only after the adopted bank passes
complete checksum/identity validation and its child is terminal.

All generation settings remain fixed: pinned model and weights proof, generator
source, public tasks, per-candidate seed rule, eight serial candidates, sampling
parameters and 4096/1024 input/output token limits. The completed pilot and
active training bank match the coordinator's expected identity. Five focused
tests pass, including a real parked-parent/live-child test and rejection of
retirement while a child is still running. GPU2 and GPU3 are not reassigned.

`rbr_deepseek_parallel_handoff_plan.json` is the active declaration. A childless
initial waiting coordinator was superseded before any handoff to improve the
fixed bank ordering; its audit remains available and no generator was stopped.
At this checkpoint the replacement coordinator is still waiting, the original
scheduler is unparked and remains the generation owner, and no parallel
generation result or speedup is claimed. This changes scheduling only; it does
not complete an additional sealed scientific stage or change failed gates.

The second prior/no-resampling refit is complete and its model/report/population
receipt and variant sidecar have been verified. Seed1702 has aligned NLL
0.450032 and association −0.00007146, CI [−0.00017435, −0.00000241]. Original
association, strong-baseline fairness and pair-invariance gates remain failed.
The third seed is still running; no pooled training-intervention result is
available yet. See `codearc_prior_training_seed1702.json` and its variant sidecar.

The coordinator's additional full-flow fixture exercises current-bank adoption,
largest-bank-first GPU0/GPU1 assignment into distinct directories, old-parent
retirement only after validation, and final complete-status publication. All six
targeted tests pass; `replication_gpu_handoff_flow_tests.log` records this check.
It supplements the real process-parking test but is not a real GPU handoff or
an empirical generation result. The actual coordinator remains queued.

## Prior refits and repair training complete (2026-09-17)

All three development-only prior-proposal/no-resampling fits and the matched
32-particle replay completed. `codearc_prior_comparison_results.json` contains
all comparisons; `codearc_prior_comparison_audit.json` verifies every prediction
archive, old/new model receipt, variant binding and independent mean loss.
Maximum report-replay NLL error is 3.67e-8. Joint reversal and presentation
permutation change probabilities by at most 4.77e-7, while the original relative
invariance gate remains recorded exactly as evaluated (two failures when the
association denominator is near zero).

The mean NLL is 0.452309. Holding prior/no-resampling inference fixed, retraining
improves NLL by 0.041393 (95% source-bootstrap CI 0.029057–0.054459). Against
old standard inference the improvement is 0.033011 (0.021907–0.044391), a contrast
that changes both training and inference. Both use 32 particles; compute is not
matched. Association is −0.00003386 (−0.00007306–0.00000465), and all seeds fail
original association and strong-baseline fairness gates. These are development
results on 400 sources / 3200 candidates; the shared aggregator's historical
`primary_candidates` field does not denote primary evaluation here.

`packet_repair_seed1703_training_complete.json` verifies the final 1500-step
checkpoint restores the best development projector, selected at step 1000
(NLL 0.749572; no-latent validation NLL 1.260503). All three fits now have verified
best-checkpoint restoration and exclude primary data from training/selection.
Generation proceeds across 500 sources × six arms × three seeds, followed by
fresh hidden-test execution. Teacher-forced NLL is not a repair-success result.

## Refit likelihood-assignment and history-use audit (2026-09-17)

`scripts/audit_apbpf_refit_assignment.py` evaluates old/new 32-trained models
for all three seeds on the same 400 development sources. Within each model,
aligned and shuffled histories use identical 32 prior draws. Across models the
standard-normal noise matches, but the learned prior transforms differ.
Every direct-likelihood prediction reproduces the completed comparison archive
within 5.82e-7; all six diagnostic archives and their descriptive means were
independently verified. No fitting or primary assessment was added.

| Three-seed descriptive mean | Old | Prior refit |
| --- | ---: | ---: |
| Prior-only future NLL, all candidates | 0.738804 | 0.759330 |
| History-conditioned future NLL, all candidates | 0.493701 | 0.452309 |
| History NLL gain, all candidates | 0.245102 | 0.307021 |
| Assignment log-likelihood ratio RMS, mixed histories | 0.005453 | 0.018701 |
| Aligned/shuffled posterior total variation, mixed histories | 0.001573 | 0.003480 |
| Per-candidate max future-probability change, mixed histories | 0.000270 | 0.000783 |

The static-only improvement hypothesis is not supported: both models use history
substantially, and prior-only prediction is worse after refitting. Assignment
sensitivity increased but remains small on this sampled support; the original
association gates still fail. These descriptive values have no new significance
claim, and do not prove count-only dependence or characterize the exact continuous
posterior. There are 1025 mixed and 2175 constant visible histories; the latter
have zero assignment effect. Evidence is in `codearc_refit_assignment_results.json`
and `codearc_refit_assignment_verified.json`. Next debugging should inspect
interactions between test features and diagnosis latents, or gradient allocation,
before increasing training duration. GPU generation continues independently.

## Training-gradient audit and fixed-weight follow-up (2026-09-17)

The audit ran the unchanged training backward over all 1360 fitting candidates
(170 sources) in 22 fixed batches for each of six old/new seed cells. It measured
weighted loss gradients without updating any parameter. The component sum
reproduces actual backward within 4.18e-7; all132 batch records and summary means
are verified. One targeted observer test passed. This measures checkpoint-local
raw gradients, not Adam-preconditioned updates or a causal explanation of fitting.

| Seed | Old association/base gradient norm ratio | Prior-refit ratio |
| --- | ---: | ---: |
| 1701 | 0.137820 | 0.001661 |
| 1702 | 0.102788 | 0.002045 |
| 1703 | 0.095520 | 0.003119 |

Ratios are unweighted means over fixed batches. Old and new models use their
respective training inference paths. Prior-refit base/association mean gradient
cosines are −0.0514, −0.0147 and −0.0509. `codearc_training_gradients_verified.json`
and the per-cell records preserve the evidence.

A single fixed follow-up raises association weight from1 to100, leaving prior
inference, other loss weights, optimizer,1000steps,32particles,three seeds and
all evaluation gates unchanged. The coefficient was chosen from fitting-gradient
scale before any follow-up fit. All results are retained; a larger coefficient
may harm NLL or merely worsen shuffled predictions. The plan is
`codearc_association_weight100_plan.json`; it is development-only and running.

## Qwen RBR full cell and actual GPU handoff (2026-09-17)

All500 primary source receipts, the full cache, three-seed training reports,
model receipts and selection input bindings were verified. Association was
independently recomputed from all three raw prediction archives. The full-cell
association gap is0.0004925 (95% CI −0.0055622–0.0065747); selection advantage
is0.0046667 (−0.0093333–0.0200000). Both fail the original0.03-plus-positive-CI
criterion. Fixed history-rate improvement is0.308833 (0.283904–0.332216), a weak
control comparison that does not repair failed core gates. The primary bank has
274 mixed,123 all-fail and103 all-pass sources; development has231 mixed sources,
below the required300. These are standalone exploratory results, not sealed
DAG stages. See `rbr_qwen_full_cell_verified.json` and the associated reports.

`rbr_deepseek_actual_handoff.json` records the successful scheduler handoff.
The original GPU1 generator child was preserved; its original parent is parked,
while the coordinator owns future scheduling and a distinct primary bank onGPU0.
The old parent must not be manually resumed into directories now owned by the
coordinator. GPU2 continues repair generation, and unrelatedGPU3 is untouched.

## Weight100 comparison queued (2026-09-17)

`scripts/summarize_apbpf_association_weight.py` waits for every fixed seed and
binds both coefficient declarations, mandatory model-variant sidecars, model
receipts and original weight1 prediction archives. Both sides use the same
prior/no-resampling class and32-particle inference. It will reproduce all four
control NLLs for every old/new model and report paired loss changes separately
for aligned and shuffled histories, plus both order controls. Positive changes
mean weight1 NLL minus weight100 NLL. A larger association gap caused solely by
worse shuffled predictions must not be described as improved aligned prediction.

The comparison is running as a waiter, with source/plan identities checked;
no weight100 outcome is available yet. See the comparison plan and
`association_weight_comparison_queue_progress.json`. Actual checkpoint replay
and numerical verification remain pending until all three fits finish.
