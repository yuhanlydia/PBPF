# Zero-positive-weight training arms: policy review

Status: **PROPOSAL — not adopted and not implemented**.

Date: 2026-09-20. This review is prospective, before examining real scored
correction outcomes. It does not change the experiment configuration, trainer,
execution queue, model policy, or scientific results.

## 1. Existing behavior and failure boundary

The seven update arms plus `no_update` must remain in the single-round inventory.
An empty selection is a possible outcome of a filtering/weighting method, not
permission to delete the arm or substitute another method.

- `scripts/run_eesd_weighted_sft.py`, `load_rows` (lines 33–53): checks positive
  weights across train **and development**. All-zero weights fail before creating
  the output directory. Positive development weights with all-zero training
  weights pass this check, create an output directory (line 168), tokenize all
  rows (lines 187–188), then fail at the training-only check (lines 190–196).
  Neither path writes a scientific empty-selection receipt.
- `scripts/run_eesd_downstream.py`, `make_training_task` (lines 85–115): preflight
  checks paths/hashes but does not classify empty training selections. Its
  expected report requires actual response tokens to equal the requested budget.
- `check_training_binding` (lines 54–81) requires a training report and a nonempty
  adapter tree. A zero-token report alone cannot pass this contract.
- The downstream execution loop (lines 354–366) stops at the first failed
  subprocess, leaving unrelated pending arms unexecuted.
- `scripts/run_eesd_recursive.py`, `train_next` (lines 200–212), always returns a
  new output adapter path. Later rounds depend on that checkpoint. Under the
  shared-teacher protocol, both arms in round r start from EESD at round r−1;
  equal-weight is a matched fork, not its own continuing teacher.
- Fresh evaluation in `scripts/run_eesd_matrix.py` (lines 442–449) and transfer
  planning in `scripts/run_eesd_downstream.py` (lines 208–211) require every
  non-`no_update` arm to have an adapter directory.

Line numbers describe the reviewed source state; function names identify the
interfaces if subsequent edits shift them.

## 2. Proposed classification before model allocation

Validate the common scored population and split identity, required train and
held-out development records, all finite weights in [0,1], model/input hashes,
rule identity, and parent policy identity first. Count positive-weight **training**
trajectories separately from development trajectories.

Classify a valid, nonempty training population with zero positive-weight training
trajectories as `non_estimable_zero_positive_training_weight`. Development
positivity cannot rescue this condition: development records never supply
training gradients.

An empty/missing dataset, invalid weight, unsupported split, corrupt seal,
invalid checkpoint, or tokenizer failure remains a data/infrastructure error.
It must not be relabeled as a method's empty selection. `no_update` is the
predeclared baseline and does not enter this classification as a failed update.
No epsilon weights, alternate rule, threshold changes, extra trajectories,
held-out data, or anchor-only updates are substituted.

## 3. Main analysis: preserve the row, mark non-estimable

The main comparison requires the declared equal **actual** response-token budget
per update. An empty selection cannot consume that budget according to the
positive-weight training protocol. Preserve its dataset/model/seed/round/arm row
with:

- `status: non_estimable_zero_positive_training_weight`;
- requested response-token budget recorded unchanged;
- actual response tokens = 0, optimizer steps = 0, micro steps = 0;
- positive training trajectories = 0 and audited total train/dev counts;
- main matched-budget effect, confidence interval, and p value = `null`, with an
  explicit reason rather than an imputed zero effect;
- no new adapter, no successful-training marker, and no equal-budget-complete
  claim.

Retain the planned denominator and make coverage visible. Do not silently average
only successful arms/seeds or describe a reduced subset as the complete planned
comparison. A failure in one seed leaves that seed's row present and any required
complete-seed main comparison non-estimable. Results for independent valid cells
may still be reported under their own original scope.

This is a scientific feasibility outcome, distinct from infrastructure failure.
It is not evidence of zero treatment effect.

## 4. Optional descriptive fallback, recorded separately

If this proposal is explicitly adopted, an additional descriptive evaluation may
reuse the unchanged **input parent policy**. It has a separate result namespace
and `analysis_role: descriptive_zero_token_fallback`; its update token/step counts
remain zero. It cannot populate the main matched-budget result fields.

Represent the policy as a reference to the pinned base model or an existing,
verified parent adapter with its tree SHA. Do not create empty, identity, copied,
or symlinked adapters merely to satisfy the existing adapter-directory contract.
The fallback retains the arm identity and reason, even if its checkpoint equals
a baseline or another arm. Metrics may be evaluated under the locked evaluation
protocol; reuse of an existing evaluation requires identical checkpoint, data,
decoding, and sample identity, with an explicit provenance reference.

Undefined training means/losses are `null`, never divisions by zero or fabricated
zero losses. The fallback receipt must state that it is not a completed budgeted
update and cannot be used as a continuing teacher for the main recursive study.

## 5. Recursive dependency handling

### Shared EESD continuing teacher

If EESD's update at round r has zero positive training trajectories:

1. Retain the EESD round-r main row as non-estimable. Its optional zero-token
   descriptive evaluation may reference the round-r input teacher.
2. Execute the equal-weight round-r fork if its independently validated inputs
   are usable: it depends on the already-existing round-(r−1) teacher, not on
   the failed EESD round-r update.
3. Mark **both main arms at rounds r+1 onward** `blocked_dependency`, naming the
   missing EESD round-r trained teacher and receipt. Do not generate purported
   main-study experience from the unchanged teacher or count a zero update as
   a successful recursive improvement.
4. Any continuation from that unchanged teacher requires a separate, explicitly
   adopted protocol and separate results. It must never occur implicitly.

If only the equal-weight round-r update is empty, retain that row as
non-estimable; a successful EESD round-r teacher allows both main forks in the
next round to proceed under the existing shared-teacher design.

### Arm-specific supplemental recursion

An empty update blocks later main/supplemental descendants of that same arm's
lineage. An independently valid other lineage may proceed. The provenance must
name the applicable experience policy; shared and arm-specific dependency graphs
must not be conflated.

### Unrelated work

Continue independent models, datasets, seeds, and update arms. Main failures and
blocked descendants remain explicit planned rows. Only jobs requiring the absent
checkpoint are blocked. Do not stop the entire matrix merely because one valid
arm selected no training trajectories.

## 6. Minimal implementation interfaces required after adoption

1. **Trainer preflight / `load_rows`:** separate structural validation from
   training-selection eligibility. Return a structured classification and split
   counts before creating partial output or allocating model weights. Preserve
   clear distinctions between invalid inputs and valid empty selections.
2. **Trainer result contract:** add an immutable empty-selection outcome receipt,
   bound to input, model config, rule, seed, requested budget, trainer/protocol
   source, parent policy, and positive-count audit. No adapter field may imply
   a newly trained model. Adopt explicit protocol-version hashing so previous
   successful or failed artifacts cannot be silently reinterpreted.
3. **`make_training_task` / `check_training_binding`:** handle a tagged union of
   successful budgeted training and validated non-estimable outcome. Successful
   training retains exact token-budget and adapter-hash requirements; the new
   receipt is a terminal scientific status, not `verified_complete` training.
4. **Downstream scheduling:** record all seven update-arm statuses and the
   baseline; continue independent tasks after a scientific empty-selection
   outcome. Track infrastructure failures separately. Resume only from matching
   immutable outcome receipts and preserve failed/incomplete artifacts.
5. **Policy resolution for fresh/transfer:** accept an explicit base/adapter
   policy reference for a separately requested descriptive fallback. This needs
   a nonfrozen routing interface; do not patch a frozen matrix merely to accept
   missing adapters. Keep the eight-arm main ledger intact even where transfer
   has its own smaller predeclared control inventory.
6. **Recursive `train_next` and dependency routing:** return structured outcomes
   rather than always `output/adapter`. Record an absent main checkpoint and
   materialize `blocked_dependency` descendant rows. Never promote a fallback
   checkpoint to main continuing-teacher status.
7. **Reports / coverage:** distinguish trained, non-estimable, blocked dependency,
   and infrastructure-error states; separate descriptive fallback metrics from
   main equal-actual-budget metrics. Do not coerce missing estimates to zero.

## 7. Focused tests required before use

- All weights zero; development-only positive weights; and nonempty positive
  training weights are correctly distinguished. Empty/invalid inputs remain
  errors. Classification occurs before GPU/model allocation and creates no
  misleading partial adapter.
- Empty-selection receipts bind input, parent checkpoint, rule, seed, requested
  budget, and source/protocol hashes. Resume rejects tampering and successful
  training receipts cannot accept actual tokens = 0 for a positive budget.
- All eight main arms remain in coverage; an empty arm does not prevent an
  independent arm/cell/seed from executing. No implicit renormalization over
  successful seeds or substitution of main metrics occurs.
- Descriptive fallback resolves to the exact parent identity, creates no adapter,
  records zero actual tokens, and cannot masquerade as successful training.
- Shared teacher: failed EESD round r permits the valid same-round equal fork,
  then blocks both next-round descendants without generation/training calls.
  Empty equal-weight with successful EESD does not block the next teacher round.
- Arm-specific mode blocks only its affected lineage. Fallback results never
  enter main teacher selection or recursive-improvement summaries.
- Loss/mean fields are null when undefined; downstream tables and receipts retain
  every planned non-estimable/blocked row with a concrete reason.

No actual empty-selection outcomes have been inferred by this review. Adoption
and these implementation/test gates remain pending.
