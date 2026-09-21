# Supplementary evaluation dimensions — 2026-09-21

Requested by the user after the first direct-execution mechanism cell was seen.
These additions are supplementary/exploratory, not retroactively preregistered
primary endpoints. They do not change generation budgets, model selection,
training weights, stopping rules, or the original statistical families.

## Existing required dimensions

- Correctness: task-level all-tests Pass@1, per-test success where identified.
  Five-class outcome prediction accuracy is a separate quantity, not code Pass@1.
- Probability quality: NLL, Brier, ECE, including the locked paired baselines.
- Repair value: FIX, REGRESSION, PRESERVED, UNRESOLVED and net gain.
- Iterative stability: per-round correctness retention, regressions and policy KL.
- Cost: input/output tokens, execution/optimization time, measured peak GPU memory.

## Added diversity summaries

The existing bank has one candidate per source per seed (1701/1702/1703). Align
exact source identities across seeds and evaluate only complete three-seed groups.
Do not create an extra generation budget implicitly or drop failed candidates.

1. Exact-code unique fraction: number of distinct complete candidate code texts
   divided by three. Report empty/invalid code rates separately. Whitespace alone
   can inflate this count; do not call it semantic diversity.
2. Syntax diversity: distinct `ast.dump(..., include_attributes=False)` values
   among syntactically valid candidates. Report its valid-candidate denominator
   and compilation failures. This removes comments/formatting but does not prove
   functional equivalence or algorithm diversity.
3. Observed behavior diversity: distinct ordered ten-test outcome vectors per
   source, plus pairwise outcome disagreement. This measures diversity of observed
   success/failure categories, not diversity of exact stdout or unseen behavior.
4. Useful diversity: fraction of sources where at least one of the three fixed
   seed candidates passes every evaluator test, compared with mean per-seed
   all-tests success. Label it 'three-seed union success / oracle coverage'; it is
   not a deployable selection score or an independently sampled Pass@3 estimate.
5. Error overlap: pairwise shared task failure and per-test failure disagreement;
   show jointly with union success. Different wrong answers are not a benefit.
6. Correct-solution syntax diversity: unique valid ASTs among candidates passing
   every evaluator test, with the number of correct candidates/sources reported.
   This conditional diagnostic cannot replace unconditional correctness.

## Reporting constraints

- Three samples per source give coarse estimates; show distributions and counts,
  not unsupported claims of broad algorithmic diversity.
- Aggregate by source, with paired source-cluster uncertainty when comparing
  models or rounds. Generation seeds are repeated measurements, not independent
  source samples. Keep incomplete coverage explicit.
- Use sealed bank code and verified execution caches; outputs do not flow into
  training, candidate filtering, hyperparameter selection, or stopping decisions.
- APPS/CodeContests results retain their Replay protocol labels, not official
  benchmark accuracy labels. Do not pool their mechanisms with EvalPlus transfer.
- Until code and receipt integration is implemented, this is an analysis plan,
  not a completed diversity result.
