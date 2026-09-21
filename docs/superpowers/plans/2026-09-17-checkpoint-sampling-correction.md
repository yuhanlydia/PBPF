# Checkpoint sampling correction and third-seed replication

The previous fixed-checkpoint audits found materially different NLL across
particle draws. Selecting over 21 checkpoints using one draw is unnecessarily
sensitive to that draw. Correct the estimator before comparing new trials.

1. Average predictive probabilities over four independent draws, with fixed
   history corruption and matched noise for aligned/shuffled controls. Use this
   same estimator for checkpoint selection, final scoring and bootstrap inputs.
   Keep the diagnostic screen and absolute-NLL guard unchanged. Retain an
   explicit single-draw compatibility option.
2. Regression-test probability averaging (not NLL averaging), fixed corruption,
   recorded seeds and disjoint selection/verification ensembles. Review before
   launching the experiment.
3. Run seed 1703 for original interaction, interaction-only, high-gain K8, and
   high-gain K32. All use 1000 steps, fixed history, the same immutable semantic
   cache, four K64 selection draws, and identical development visibility.
4. Freeze selected checkpoints; separately vary sampling, corruption and
   evaluation particle budget. Compare averaged predictions using common
   independent draws. Preserve negative results and all source/artifact hashes.
5. Report whether the initialization result repeats and whether K32 adds value.
   The new optimization seed is not fresh data. The previously exposed test
   split remains untouched. This round does not test active selection or repair.

Pre-run machine-readable plan and raw artifacts:
`/root/pbpf-runs/selection-fix-20260917/plan.json`.

Four K64 self-normalized filters averaged together are not one pooled K256
importance estimate. Reducing sampling sensitivity is not evidence that latent
bug identity is identifiable, nor that development selection bias is eliminated.
