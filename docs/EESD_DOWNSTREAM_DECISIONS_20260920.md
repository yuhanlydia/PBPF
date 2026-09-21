# Prospective downstream decisions

No candidate outcomes have been scored and no training has begun.

The separate `configs/experiments/eesd_downstream_prospective_20260920.yaml`
records the following choices without changing the active mechanism config.

- Transition utility is `[1, -1, 0, 0]` in FIX, REGRESSION, PRESERVED,
  UNRESOLVED order, aligning expected utility with net repair gain.
- Before training any arm in a cell/seed, use
  `scripts/plan_eesd_training_budget.py` on the sealed public correction bank.
  The reference is 200 optimizer steps with 16 microbatches per step under
  equal weighting. All update arms receive exactly that response-token budget.
  It includes response EOS and excludes development tokens. This matches token
  exposure, not mathematical equivalence to the old trajectory-normalized loss.
- Freeze the resulting plan, input hashes and numeric budget before launching
  any arm. Do not adjust the budget using assessment outcomes.

Still required: real public correction banks, sandbox execution availability,
and an anchored-training memory check. The zero-positive-weight main-analysis
policy is adopted in `EESD_ZERO_WEIGHT_ADOPTION_20260920.md` and implemented:
such arms retain null matched-budget estimates, zero actual training tokens,
and no new adapter; unrelated arms proceed and missing recursive teachers block
their descendants. The optional unchanged-policy fallback is not adopted.
Zero selections receive CPU structural validation and parent-checkpoint checks,
but do not load a tokenizer or claim a fresh tokenization audit. Positive
training arms retain their generation/training token-identity validation.
The shared-bank/same-start recursive design is now specified in
`EESD_RECURSIVE_SHARED_TEACHER_LOCK_20260920.md`. EESD is the prospectively
fixed continuing teacher; each equal-weight update is a matched control from
that teacher, not an independent recursive chain. The runner implementation
has focused tests; no recursive training or scientific evaluation has run.

## Statistical reporting implementation and remaining execution gate

The original config requires `holm_secondary: true`. The present mechanism
runner only computes unadjusted single-seed NLL bootstrap intervals and the
renderer summarizes seeds descriptively. Neither constitutes Holm correction.

The separate prospectively adopted plan is
`EESD_MECHANISM_MULTIPLICITY_DRAFT_20260920.md`, sealed by
`runs/eesd-setup/statistical-amendment-lock-20260920.json`. Its new postprocessor,
`scripts/report_eesd_mechanism_inference.py`, implements source-paired,
three-seed aggregation, recomputed ECE, and complete Holm families of 8 primary
and 1,560 secondary comparisons. Unsupported or missing cells remain visible
and block family-wide rejection claims. Public A9 support is sealed before
assessment outcomes are reconstructed.

The prescribed 10,000 draws impose a minimum p value of 1/10,001; no secondary
comparison can pass the first Holm threshold at this family size. This is a
power limitation, not evidence of no effect. The original runner's unadjusted
reports remain unchanged. Focused tests and a missing-input preflight have run;
formal inference still awaits all sealed execution caches and reports.
