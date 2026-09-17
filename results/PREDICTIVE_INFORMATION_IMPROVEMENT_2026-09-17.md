# Predictive-information changes and matched reruns

Two implementation changes and 80 replay conditions are complete. Exact
two-step planning did not improve realized loss and remains an ablation.
Pooling four independent normalized posterior draws before predictive-MI
scoring produced small favorable development point estimates relative to a single
256-particle estimate. Predictive entropy remained a matched control in every
condition; the pooled method has not established superiority over it.

## 1. Exact two-step planning: correct algorithm, negative empirical result

`predictive_information_scores(..., budget=2)` adds immediate target MI to the
expected best remaining target MI after each possible first-query outcome.
After the actual first observation, the final query uses budget=1. It reads
predicted outcome distributions, never actual unexecuted outcomes. All query
and target outcomes must be conditionally independent given the component.
Optimality is only under that finite-mixture model.

Tests cover complementary XOR queries that greedy scoring misses, an adaptive
selector-bit example requiring different second queries for different outcomes,
zero-probability branches, no repeated queries and budget-one compatibility.

Forty conditions cover five frozen checkpoints, two pool/target partitions,
K64 and four sampling seeds (62721/62722 plus 63721/63722). Old greedy target MI,
entropy, fixed and exact random-pair expectation share the same posterior and
updater. All 20 overlapping old-seed baselines reproduce the previous
per-candidate losses exactly.

Across the four high-gain models, planning changed the final query pair in only
2.82% of cases. Model-expected improvement over greedy was .0000194 NLL; actual
improvement was -.000428. Old-seed and new-seed means were -.000198 and -.000658.
Thus this pool offers little modeled benefit from lookahead, and the additional
planning did not help on realized outcomes. It is not the preferred change.

## 2. Pool posterior draws before computing information

`pool_predictive_particles` accepts normalized weights `[R,K]`, query predictions
`[R,K,Q,C]` and target predictions `[R,K,T,C]`. It assigns each draw mass `1/R`
and flattens the mixture, preserving particle/prediction alignment. Predictive
information is then computed on this mixture, rather than averaging separate
MI scores. After observing a query, all mixture weights update globally;
draw masses are not reset to uniform.

This is an equal mixture of self-normalized posterior estimates, not one
globally normalized importance estimate or an evidence-weighted combination of
draws. Disagreement between draws is part of this approximate mixture, not proof
of calibration. Tests verify between-draw predictive dependence, global updating
of draw mass, and malformed-input rejection.

The matched rerun has another 40 conditions: five checkpoints, two partitions,
two seed anchors, and `single256` versus `pooled4x64`. Pooling uses offsets
0/101/202/303 from each anchor. Both have 256 initial support particles and use
two greedy adaptive queries. Four proposal executions and one larger execution
need not have equal wall time. The comparison changes sampled support and
normalization strategy, not only a scalar MI formula.

## Results with entropy retained

The following are descriptive means over the four high-gain models, two
partitions and two seed anchors. Positive NLL gain means the target-MI policy
has lower loss. Reference-model results remain in the complete JSON.

| Quantity | Single 256 | Pooled 4x64 |
|---|---:|---:|
| Target-MI final NLL | .411754 | .410968 |
| Target-MI gain over random | .004725 | .005795 |
| Target-MI gain over matched entropy | -.004176 | -.000507 |

Pooling improves target-policy NLL by .000786, with a descriptive source-bootstrap
95% interval [-.003018,.004834]. Its change in gain over random is .001070
([-.002312,.004550]). The gap relative to entropy improves by .003669
([.000044,.007850]), but pooled target MI itself still has an interval spanning
zero relative to entropy ([-.008573,.007701]). This is not a confirmatory win.

| Frozen model | Single: gain over random | Pooled: gain over random | Pooled: gain over entropy |
|---|---:|---:|---:|
| Original reference, 1703 | .003375 | .003905 | -.001333 |
| High-gain K8, 1703 | .005784 | .006235 | -.000122 |
| High-gain K32, 1703 | .003426 | .003448 | -.002948 |
| High-gain K8, 1704 | .005498 | .008081 | .002872 |
| High-gain K32, 1704 | .004193 | .005418 | -.001827 |

Across the two seed anchors, high-gain target-MI first-query disagreement falls
from 33.24% to 32.01%; final-pair disagreement falls from 32.83% to 32.01%.
Sampling sensitivity remains substantial. The pooled variant is a useful
experimental option, not a stability guarantee. Entropy should remain the
primary comparison, and neither implementation silently replaces an existing
default policy.

## Reproducibility and verification

Raw pre-run plans, scripts, exact source snapshots, checkpoints' fingerprints,
per-candidate losses, chosen queries and bootstrap inputs are under:

- `/root/pbpf-runs/predictive-lookahead-20260917/`
- `/root/pbpf-runs/predictive-pooling-20260917/`

The companion `predictive_information_improvement_2026-09-17.json` retains all
80 conditions, including the reference and negative lookahead results. Its
summary averages realized losses across the fixed sampling-dependent policies,
then resamples the 31 source groups after collapsing each candidate across
settings; it does not average predictive probabilities or treat settings as
independent data replications.

All 39 `tests/apbpf` tests pass. All 880 manifest-listed artifact files were
verified. No replay repeats an executed query, and the target-label-clairvoyant
best-pair diagnostic is no worse than any executed policy up to rounding.
Independent reviews found no math, mixture-update or label-leakage blockers.

This remains adaptive analysis of the same 91 development candidates. The new
seed anchors are additional sampling checks, not new data; the pooling choice
was investigated after the lookahead result. Target policies use target
features, while entropy does not. Models were not retrained, held-out test data
were not evaluated, and cached outcomes were replayed rather than newly
executed. The .03 association screen remains advisory.
