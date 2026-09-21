# Acquisition objective and update-strength debugging

This bounded round is complete: 30 unique one-step replay conditions and 20
two-step conditions, using five frozen checkpoints, two new sampling seeds,
two pool/target partitions and evaluation K64/K256. No models were retrained.
A target-predictive information score is implemented and tested. Its one-step
point estimates improve over diagnosis MI for all 12 high-gain model/panel
aggregates, but sustained superiority over random selection is not established.

## What was changed and checked

`pbpf.apbpf.acquisition.predictive_information_gain` computes the mean across
targets of `I(query outcome; target outcome)` under the finite joint-particle
mixture. For conditionally independent outcomes given the latent and ordinary
Bayesian updating, this is model-expected reduction in mean marginal target
log loss. It is not information about a named bug, joint target-vector entropy,
or a guarantee about realized data. Inputs are predicted distributions only.

The test with two independent latent bits demonstrates the distinction: a
query can reveal a full bit about the latent while revealing nothing about a
target that depends on the other bit. The new score ignores that irrelevant
bit. Noisy-query tests verify the expected entropy-reduction identity.

During implementation, a numerical bug was reproduced for rare correlated
events: multiplying two small marginals underflowed before taking their ratio,
returning infinity. Computing the ratio through separate log marginals fixes
this; a regression covers probability 1e-200. This new-function bug was not
the cause of the earlier empirical acquisition failures. The one-step probe
archived the earlier ratio implementation; its float32-derived probabilities
do not reach that float64 underflow regime. The two-step probe uses the fixed
implementation, and all corresponding first-step losses agree exactly.

The existing diagnosis-MI policy remains available. The new score is a tested
experimental alternative, not a default replacement presented as proven best.
Both target MI and the cosine control see the fixed target **features**;
diagnosis MI, joint MI and entropy do not use that additional target inventory.
None of the deployable scores reads target outcomes.

## One-step results

All values below are random-policy NLL minus selected-policy NLL: positive is
better than random. Each cell averages realized losses over two particle draws,
not probabilities. The primary partition observes tests 0–3, queries one of
4–6 and scores untouched targets 7–9. The swapped partition exchanges pool and
targets. Initial histories and checkpoints remain fixed.

| Model | Primary K64: diagnosis MI | Target MI | Target-feature cosine |
|---|---:|---:|---:|
| Original reference, seed 1703 | .009945 | .005681 | .005845 |
| High-gain K8, seed 1703 | -.010775 | -.001091 | -.001499 |
| High-gain K32, seed 1703 | -.009686 | .008743 | -.001131 |
| High-gain K8, seed 1704 | -.003039 | -.000136 | -.000975 |
| High-gain K32, seed 1704 | -.008439 | .003559 | -.004849 |

Across all three panels, target MI has a better mean loss than diagnosis MI in
12/12 high-gain model/panel aggregates; gains range .001139–.018429 NLL. These
are dependent, adaptively selected development comparisons, not 12 independent
replications. The corresponding source-bootstrap difference intervals all
include zero. For example, K32/1703/primary/K64 improves by .018429 with a
descriptive 95% interval [-.000085,.048298]. The target-MI result still depends
on the partition and particle draw. The simpler cosine control sometimes
matches it; target-feature access is a material part of the comparison.

## Update strength and two-step results

For each checkpoint, beta in `{0,.25,.5,1}` scales only the new query's log
likelihood in the posterior update. Beta was selected using **training data
only**, averaging losses over all three possible queries and two particle
draws. Every model selected beta=1, and that value was frozen across development
partitions and K. Consequently the `train_fitted` and `bayes` records are
identical conditions, not additional replications. This grid provides no
train-based justification to weaken updating; it does not rule out other
calibration methods or policy-specific effects.

Two-step replay recomputes MI after the first selected outcome, without
repeating a query. All policies use the same initial finite joint support and
ordinary likelihood reweighting. Random is the exact average over the three
possible unordered query pairs; update order commutes for this updater.

The target-MI advantage is not consistent across two-step panels. In the
primary partition, its gains over random are negative for all four high-gain
models, ranging -.009260 to -.002133. In the swapped partition they are
positive (.001005–.012893). Thus changing the scoring objective helps some
comparisons, but does not solve active testing. K32 training and K256 evaluation
also do not provide a uniform remedy. No post-hoc threshold is used to discard
these negative results.

The outcome audit finds similar middle/late class marginals but different
within-candidate variability. Nonconstant four-test histories occur in 27.7%
of training candidates versus 44.0% of development candidates. This is a
distributional difference worth investigating, not proof that it causes the
selection failure. Realized query-gain rankings remain only weakly aligned
with model information scores in many panels.

## Reporting corrections, verification and limits

The initial probe's random `harm_fraction` counted whether each candidate's
expected random-query loss increased. That is different from the probability
that a random query harms. The reviewed result corrects it using all cached
query losses, with the same 1e-7 tolerance for every policy. Raw records are
preserved. The two-step probe uses the corrected definition directly.

The oracle is target-label-clairvoyant and only chooses the best available
query or pair under this frozen updater; it is not a deployable policy or a
general acquisition ceiling. Training-fit data were already used in model
training. Development is the same 91 candidates/31 sources as before, with
adaptive reuse. No held-out test split was evaluated. K64 and K256 draws with
the same seed are not nested samples. No fresh test execution or repair was
performed in these cached-outcome replays.

All 29 tests under `tests/apbpf` pass. Independent review verified no repeated
queries in all 20 two-step records, exact matching one-step prefix losses,
correct random expectations and no label leakage in policy selection. All 528
files in the run/audit checksum manifests were verified.

Raw plans, source snapshots, train-only fits, all positive/negative scores,
per-candidate losses, failure examples and source-bootstrap inputs are in:

- `/root/pbpf-runs/acquisition-debug-20260917/`
- `/root/pbpf-runs/acquisition-two-step-20260917/`

`reviewed-result.json` records the derived harm correction and cosine control;
`summary.json` records the aggregates. The checked-in companion
`acquisition_objective_debug_2026-09-17.json` includes every aggregate and
verification metadata. The .03 association screen remains advisory. This
round completes the announced debug experiments; the broader claim of reliable
real-data diagnostic action remains open.
