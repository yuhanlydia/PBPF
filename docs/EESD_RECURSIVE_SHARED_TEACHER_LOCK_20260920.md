# Prospective recursive shared-teacher amendment (2026-09-20)

This is a new, prospective choice made before recursive scientific runs. Section
7 of the experiment contract requires shared experience and the same starting
checkpoint/token budget at each update. The previous implementation shared only
round-0 experience, then continued two independent policies. Choosing EESD as the
continuing teacher is introduced here; it was not specified by the original
contract and must not be described as original preregistration. No assessment
result was used to choose this teacher or this protocol.

## Main mode: `shared_eesd_teacher`

Let T0 be the pinned base policy. At each r in {1,2,3}:

1. Freeze T(r−1). Generate one shared train/development original-and-correction
   experience bank from that teacher using the existing public execution rules.
2. Compute the scored trajectories and both arms' weights on that same bank.
3. Fork exactly two updates from the identical T(r−1) checkpoint: equal_weight
   and eesd_full. Both receive the same seed and explicit positive response-token
   budget, using existing trainer receipt checks. This adds no update arms.
4. Evaluate both updated policies with one fresh greedy primary candidate per
   source. Compare each with the same teacher entering the update and with base.
5. Set Tr to EESD_r by the fixed rule, irrespective of evaluation or effect sign.
   Do not select a winner based on assessment results.

The scalar response-token budget is explicit at launch and is held common across
both arms and rounds. Its derivation/acceptance remains governed by the separate
budget plan; this amendment does not invent a numeric budget. Same input bank
and token budget do not imply identical optimization batches when weighting
rules filter examples. The policy-anchor distinction remains part of each arm.

The experience directory is `round{r-1}/shared-experience`. Output policy names
remain `round{r}/equal_weight` and `round{r}/eesd_full` for compatibility. In this
mode equal_weight_r is a matched one-step control from the EESD teacher; it is
NOT the continuation of equal_weight_(r−1). Therefore its adjacent displayed
rounds must not be used to claim independent-policy cumulative regressions.
EESD_0→EESD_1→EESD_2→EESD_3 is the actual continuing trajectory.

Both `vs-previous.json` reports use base at round 1 and EESD_(r−1) thereafter.
`vs-base.json` and same-round `eesd-vs-equal.json` remain. Reports must label
teacher-relative FIX/regression/retention separately from cumulative EESD-chain
statistics. Conditional retention uses the entering teacher's correct sources.

## Supplemental mode: `arm-specific`

The legacy mode remains available only by explicit CLI choice and is labeled
`supplemental_arm_specific`. It retains shared base experience at round 1, then
each policy's own experience/checkpoint at later rounds. It does not fulfill the
main shared-bank matched-control claim and is not pooled with the main mode.
Its per-arm `vs-previous` retains the genuine same-arm previous policy.

## Seals and resume contract

The run identity includes experience_policy, config/model/data/source hashes,
round count, seed, and response-token budget. Switching modes or changing identity
under an existing output directory fails. Existing trainer report/adapter binding
checks and unbound-output rejection remain in force; this amendment does not
certify or bypass other generation/evaluation provenance gates.

Before either update, each `roundN/update-inputs.json` binds:

- schema `eesd-recursive-update-inputs-v1`, round, seed, experience_policy,
  shared_experience and common response_token_budget;
- teacher_rule (`base` at round 1, otherwise `eesd_full` in main mode),
  teacher_round, teacher_adapter path and teacher_adapter_sha256;
- shared_scored_sha256: SHA256 of the shared scored-corrections JSONL, not an
  ambiguous directory checksum;
- arms[rule]: scored path/SHA, previous_adapter path/tree SHA,
  parent_evaluation path/SHA; and recursive_run_sha256.

Base adapter fields are null. Paths are absolute strings. Each round summary
contains update_inputs_sha256 plus teacher_rule/teacher_round. Resumption checks
the complete receipt against the current teacher, scored bank and budget before
training; it never replaces a conflicting receipt. Unbound existing training
under a round without update-inputs is rejected.

Only the independent prospective downstream config is amended. The frozen
mechanism config, candidate generators, execution semantics and mechanism runs
are unchanged. CPU synthetic orchestration tests establish routing and binding;
they do not establish GPU feasibility or count as formal recursive results.
