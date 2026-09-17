# Advisory association and exploratory active-testing debug

At the user's request, the debug runner now selects minimum development NLL
and treats association as a continuous diagnostic. A positive gap is a reason
to continue debugging, not a claim of identifiable bugs. The .03 screen remains
in historical metadata with `gate_enforced=false`; it cannot veto selection.
`--checkpoint-policy historical_screen` restores the old behavior. The training
association margin is unchanged. All six most recent frozen checkpoints already
minimized NLL, so this change requires no retroactive reselection or retraining.

## Is one source responsible for association?

In the two recent high-gain K32 models, excluding any single development source
leaves the association gap positive: minimum .021159 for seed 1703 and .030517
for seed 1704. Positive/negative source counts are 15/7 and 19/3, with nine
near-zero groups in each case. Thus no single source accounts for all the
positive signal, but this does not establish broad generalization or rule out
concentration in several sources. The audit uses the existing four-draw averaged
predictions; it adds no fresh data.

## What to debug next: useful test selection

Eighteen frozen-checkpoint replay configurations were completed: three models,
two sampling seeds, and three history/particle panels. No new model was trained.
Every case starts with observed tests, chooses one query, reveals its cached
outcome, and scores a fixed set of untouched target outcomes. Targets are never
used by deployable selection policies.

- Two-history panel: initial tests 0–1, query pool 2–5, targets 6–9, K64.
- Four-history panels: initial tests 0–3, pool 4–6, targets 7–9, K64 and K256.
- Diagnosis MI crosses the posterior difficulty and diagnosis marginals, an
  explicit mean-field/interventional acquisition approximation.
- Joint MI and predictive entropy use the original joint particle posterior.
- All policies update the same finite joint particle support with the observed
  likelihood. There is no proposal refresh and only one acquisition step.
- Fixed selects the first pool entry. Random is the exact uniform expectation
  of realized NLL over the pool, not NLL of averaged predictions.
- The reported oracle is **target-label-clairvoyant**: it chooses the lowest
  realized target NLL among these queries under this fixed updater. It is not
  an executable policy or a general acquisition upper bound.

The table reports random NLL minus policy NLL, averaged over two sampling
seeds. Positive means better than random. Sampling seeds are noise probes, not
independent data replications. Different history panels also use different
pools/targets and cannot isolate history length as a causal factor.

| Initial history | K | Frozen model | Diagnosis MI | Joint MI | Entropy | Fixed |
|---:|---:|---|---:|---:|---:|---:|
| 2 | 64 | reference-s1703 | +0.033739 | +0.033333 | +0.037313 | +0.023307 |
| 2 | 64 | high-gain-k32-s1703 | +0.000749 | +0.012977 | +0.017887 | +0.035499 |
| 2 | 64 | high-gain-k32-s1704 | +0.033602 | +0.038964 | +0.039558 | +0.029295 |
| 4 | 64 | reference-s1703 | +0.011532 | +0.008737 | +0.007346 | -0.007015 |
| 4 | 64 | high-gain-k32-s1703 | -0.022275 | -0.012091 | -0.013604 | +0.000221 |
| 4 | 64 | high-gain-k32-s1704 | -0.002950 | -0.003030 | +0.000828 | +0.004134 |
| 4 | 256 | reference-s1703 | +0.013653 | +0.009540 | +0.009039 | -0.007726 |
| 4 | 256 | high-gain-k32-s1703 | -0.012709 | -0.003148 | -0.001254 | -0.001109 |
| 4 | 256 | high-gain-k32-s1704 | -0.008275 | -0.000928 | -0.004315 | +0.002808 |

The high-gain K32 seed-1703 model has little diagnosis-MI benefit with two
initial observations and negative benefit in both four-history panels. Seed
1704 benefits in the two-history panel but not the four-history panels.
Increasing K to 256 does not consistently remove the weakness. The original
reference has positive point estimates despite its weak association gap.
Therefore a larger aligned/shuffled gap does not by itself establish improved
acquisition, and finite particle count alone does not explain these observations.
Several individual descriptive intervals include zero; all are adaptive
development analyses, not confirmatory hypothesis tests.

## Next bounded experiments

1. Compare model-predicted information against the realized target-loss change
   for every available query. Inspect cases where high MI selects a damaging
   observation; distinguish likelihood calibration from query ranking.
2. Compare diagnosis MI to expected information about future test outcomes,
   using only future test features, not labels. This tests whether information
   about latent particles aligns with the actual prediction objective. Keep
   joint MI, entropy, fixed and random controls.
3. Replay K8-trained and K32-trained high-gain models with identical pools and
   evaluate additional fixed partitions, then a two-step acquisition budget.
   This separates training-budget effects from partition and updater effects.

These are debugging priorities, not new mandatory gates. Small candidate
selection/repair probes can follow, but no evidence here establishes repair
control. Fresh source-disjoint data will be needed for a generalization claim.

## Evidence and verification

Raw scripts, pre-run plans, source snapshots, cached-outcome replay choices,
per-candidate losses and 2000-replicate source-bootstrap intervals are under:

- `/root/pbpf-runs/active-replay-20260917/`
- `/root/pbpf-runs/active-replay-four-history-20260917/`
- `/root/pbpf-runs/active-replay-four-history-k256-20260917/`

Each probe is an exploratory artifact, not a production acquisition runner.
Reproduction requires a fresh output directory and the saved feature/cache and
checkpoint paths. The companion JSON retains all 18 configurations, including
negative results. These are the same 91 development candidates and 31 sources;
the held-out test split was not evaluated. Equal seeds at different K do not
imply nested particle sets. Independent review found no label-leakage or math
blockers in the initial probe; its limits on oracle interpretation are included
above. The checkpoint-policy regression suite passes 26 tests.
