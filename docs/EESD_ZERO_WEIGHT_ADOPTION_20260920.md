# Prospective zero-positive-training-weight policy

Adopted 2026-09-20, before any real correction outcomes are scored or any
training arm is run. This is an analysis/operational amendment, not an original
preregistration. Implementation and verification are in progress.

Protocol identifier: `eesd-zero-positive-training-weight-v1`.

The detailed rationale and reviewed interfaces are retained in
`EESD_ZERO_WEIGHT_POLICY_REVIEW_20260920.md`. This adoption covers its main
analysis and dependency policy, with the following explicit scope:

- A structurally valid scored population with train and held-out development
  records, finite legal weights, and no positive-weight **train** record is
  `non_estimable_zero_positive_training_weight`. Development weights cannot
  qualify it for training. Invalid or missing inputs remain errors.
- Preserve the planned arm/seed/cell row, requested budget and input/parent
  identity. Record actual tokens and optimization steps as zero, scientific
  estimates as null, and no new adapter. The matched-budget comparison is not
  estimable; this is not a zero effect or a successful budgeted update.
- Continue independent arms/cells/seeds. A missing EESD update in shared-teacher
  recursion blocks subsequent rounds of both arms, while its valid same-round
  equal-weight fork may still run. A missing equal-weight update does not block
  a successful EESD teacher. Supplemental arm-specific recursion blocks only
  descendants of the affected arm.
- Fresh and transfer coverage retain absent trained policies explicitly. They
  must not resolve an absent adapter to the base model or an old teacher.
- The optional descriptive unchanged-parent fallback in the review is **not
  adopted or implemented** in this amendment. It supplies no main-study result
  and is not needed to continue independent valid work.
- Resume requires immutable, input/source/budget/parent-bound outcome receipts.
  Existing unbound failed outputs are not retroactively reclassified.

No existing frozen mechanism generator, evidence protocol, or active mechanism
configuration is changed. Tests and receipts for synthetic empty selections
demonstrate software handling only; they are not experiment outcomes.
