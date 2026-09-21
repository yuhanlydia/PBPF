# Proposed mechanism inference and multiplicity amendment

**Status: prospectively adopted amendment, 2026-09-20; not an original preregistration.** This file proposes
scientific choices that the existing contract leaves unspecified. It implements
nothing and reports no experimental significance. Adoption requires a dated lock
before examining the new assessment effects; if effects have already been seen,
label this an analysis amendment rather than a prospective preregistration. No
hidden results were inspected to choose the rules below.

## 1. Existing commitments and new assumptions

`EESD_ICLR2027_EXPERIMENTS.md` sections 4–5 require A1–A10, three generation seeds,
10,000 paired source-cluster draws and all declared cells. The EESD YAML declares
NLL primary, Brier/ECE/accuracy secondary, and `holm_secondary: true`. It does not
define families, tests, sidedness, cross-seed inference or missing-test rules.
The older 2026-09-14 design describes tests→candidate→task→seed aggregation.

Recommendations introduced here are: the exact primary contrast; a separate
primary Holm family; the complete secondary family below; equal-source/equal-seed
estimands; centered cluster-bootstrap p values; fixed rather than resampled
seeds; the bin population convention; and a same-kernel global-mass control.
They must not be presented as previously registered decisions. The primary
contrast recommendation tests mass adaptation at validation-selected EED
parameters. It does not retrospectively privilege whichever existing contrast
has the better result.

## 2. Scope, population, and pairing

Eight cells: {RunBugRun corrected-stdin, CodeARC-Replay} × {qwen25_7b,
deepseek_6p7b, seed_coder_8b, starcoder2_15b}. Each cell requires generation seeds
1701/1702/1703. Validation selects parameters; assessment alone estimates effects.
The declared assessment source population must be identical across seeds within
a cell. Never intersect away a missing source, seed, candidate or target.

For each seed, aggregate query loss/correctness equally within a source; then
average the three seed-specific source statistics equally; finally average
sources equally. The independent sampling unit is source, not query, candidate,
or seed. Query means agree with hierarchical test/candidate means only when the
locked bank has the same target count per candidate; audit this invariant. If it
fails, stop for a separately locked weighting amendment, rather than silently
changing the estimand. Within a seed and contrast, both arms must refer to exactly
the same source/candidate/ordered target IDs. Across seeds, candidate identities
need not coincide: retain the entire source block containing all three seeds.

This estimates performance conditional on these three fixed generation seeds
and validation-selected settings, across the source population. It does not
estimate uncertainty over unseen generation seeds or validation selection. Seed
mean/sample-SD remains a separate descriptive output, not an independent-sample
significance calculation. Models and domains are not pooled into one p value.

## 3. Metrics and effect orientation

- NLL: mean negative log probability of the observed class, probability clipped
  below at 1e-12 to match the runner. **Remains the primary metric.**
- Brier: sum of squared class-probability errors, matching `score_predictions`.
- Accuracy: argmax correctness, matching the existing class ordering/tie rule.
- ECE: ten fixed equal-width confidence bins on [0,1], with the runner's exact
  endpoint convention. In each seed, give each source equal total weight and
  each included query within source equal weight; compute confidence/accuracy
  and absolute calibration gap from the full weighted predictions. Average the
  three ECE values. For every resample, recompute bin weights, accuracy and
  confidence; do not average per-query or per-source ECE values.

For NLL/Brier/ECE define gain = comparator minus proposed; for accuracy use
proposed minus comparator. Positive always favors the proposed arm. Record both
arm metrics and gain. Nonlinear ECE is not the mean of source-wise differences.

## 4. Complete hypothesis inventory

Use canonical IDs `(domain, model, contrast, metric)`; seeds are repeats within a
hypothesis, not extra hypotheses. Every listed contrast has all four metrics.
Reference history is four unless explicitly stated. All development tuning uses
only the locked grids and deterministic tie-breaking from the sealed runner.

| Contrast group | Count per cell | Exact comparator → proposed |
|---|---:|---|
| A1 relevance | 1 | independently validation-tuned ordinary → fixed relevance |
| A2/A3/A4 factorial | 32 | fixed → effective at each of 8 alpha × 4 strength values; alpha .10 and .01 rows are A2/A3 aliases, not duplicate tests |
| A5 global mass, existing | 1 | independently validation-tuned global mass arm → tuned EED, as currently emitted |
| A5 same-kernel control, proposed completion | 1 | global mass chosen on validation over the locked mass grid, with alpha and strength held at EED's selected values → EED at those same values |
| A6 average mass | 2 | assessment-public mean mass → EED; validation-public mean mass → EED, both at EED alpha/strength |
| A7 mass alignment | 1 | average metric of the three fixed mass permutations (1701/1702/1703) → aligned EED, at EED alpha/strength |
| A8 history | 4 | separately validation-tuned fixed → separately validation-tuned effective for each history n=1/2/4/8 |
| A9 concentration | 3 | fixed → effective at EED-selected parameters within public mass bins [1,2), [2,3), [3,4] |
| A10 taxonomy | 1 | binary fixed → binary effective at the multiclass EED-selected alpha/strength, using identical queries |
| Existing tuned comparison | 1 | independently tuned fixed → independently tuned EED |
| Existing fixed-selected-parameter control | 1 | fixed → effective, both at fixed-selected alpha/strength |
| Existing EED-selected-parameter control | 1 | fixed → effective, both at EED-selected alpha/strength |

Total: 49 reserved comparison slots per cell. A8 at n=4 and the existing
tuned comparison are a deterministic alias, not distinct estimands: share their
computed effect and p value while retaining both reserved slots conservatively.
This inventory must not be described as 49 unique comparisons. A5's extra same-kernel control is needed because the
current independently tuned global arm can choose a different kernel strength,
whereas the A5 prose requires the same relevance kernel. Report both, clearly
labeling the added control. Holding alpha as well as kernel strength fixed is
a new supplementary choice; it cannot replace the existing independently tuned
A5 control based on which result is favorable. A7 averages metrics, not probability vectors: mixing
predictions would test an ensemble instead of the declared permutation control.
Its permutation seeds are fixed nuisance replicates, not additional source units
or independent p values. Strength zero produces identical fixed/effective arms;
retain these hypotheses with gain zero and p=1.

A10 reports the binary contrast and the matched multiclass contrast already in
the inventory. Do not test raw binary versus five-class NLL differences: their
outcome spaces and entropies differ. Taxonomy robustness means reporting both
within-taxonomy effects, with no claim that a nonsignificant difference proves
invariance. A8 effects are within each history's target set; differences across
history lengths are descriptive because the future target set changes.

### Primary family P

One NLL contrast per cell: the EED-selected-parameter control (last table row).
H0: population gain = 0; H1: gain != 0. Eight hypotheses; two-sided alpha=.05,
Holm across these eight. The choice of a primary family correction is a new
conservative recommendation. Report all eight estimates regardless of sign.

### Secondary family S

All table entries × all four metrics × all eight cells, excluding only the eight
primary hypotheses already in P: **(49×4−1)×8 = 1560 hypotheses**. NLL remains the
primary *metric*, but NLL tests of additional comparisons are secondary
*hypotheses*. This avoids silently excluding A1–A10 NLL searches from multiplicity.
Use one global Holm family S, not separate favorable cell/metric/ablation subsets.
Do not count A2/A3 aliases twice. Genuine planned tests that happen to yield
identical predictions remain listed. No early rejection gate changes S's size.

Each secondary H0 is zero gain, with a two-sided alternative and alpha=.05.
Family P and family S each target nominal FWER .05 using approximate bootstrap p values; this does not claim
study-wide .05 across their union. If union-wide control is required, adopt a
new explicit allocation or single family before assessment; do not improvise.

## 5. A9 bin populations and global public statistics

Bins use public-feature-derived effective mass conditional on sealed
validation-outcome-selected settings. They do not use assessment outcomes.
Freeze support membership before reading assessment labels. For
a bin, use sources having at least one query in that bin in **every** seed;
compute within-bin query means per source/seed and average seeds/sources equally.
This public-only complete-bin support convention is a new conditional estimand,
not a general-population mechanism effect. Publish full source membership and
excluded counts for each seed/bin; do not exclude a source because of its loss.
A source may appear in multiple bins, so their tests are correlated. This is
compatible with Holm; never treat these bins as independent samples.

Two sources is a computational minimum, not a guarantee of reliable inference.
Report support size and degenerate bootstrap distributions. If the common bin support has fewer than two sources, the hypothesis is
unavailable, not zero-effect. Keep its reserved family entry. Use the same
whole-source resample indices across all metrics/arms within that bin; use full
cell indices for full-population contrasts. Cross-bin covariance estimates or
interaction claims are outside this proposal. Public global means, permutations,
and tuned parameters are fixed to their sealed full-bank values in bootstrap;
this is conditional inference, not a refit of the estimator on each resample.

## 6. Resampling, raw p, intervals and Holm

For each cell and population, use 10,000 paired draws with replacement of N whole
source IDs (N = locked source count). Use NumPy PCG64 with seed equal to the unsigned big-endian integer formed by
SHA256 of UTF-8 `314159|domain|model|population`, first eight digest bytes;
`population` is `all` or `mass_bin_1/2/3`. Source IDs are lexicographically sorted.
Record the NumPy version and derived seed; no Python randomized hash. Share a draw's indices across
all compared arms, metrics and all three seeds. Sources repeated in the draw
retain multiplicity. No independent seed or query resampling.

Let observed gain be d and recomputed gain on draw b be d_b. Recommended raw test:

    p = (1 + count_b[abs(d_b - d) >= abs(d)]) / 10001

This is a centered, two-sided cluster-bootstrap approximation to a zero-gain
null, not an exact randomization test. Its assumptions are independent source
clusters and adequate cluster count; bootstrap behavior for nonsmooth ECE or
boundary/degenerate distributions must be disclosed. Identical arm predictions
receive p=1. Do not substitute the uncentered probability that bootstrap gain is
negative. Preserve raw p without rounding before correction.

Report percentile 2.5%/97.5% intervals from d_b as **pointwise**, not simultaneous
or multiplicity-adjusted confidence intervals. Inference claims use adjusted p,
not whether a pointwise interval crosses zero. Primary/secondary classification
is fixed independently of the observed effects.

Sort each family's raw p increasingly with canonical-ID tie-breaks. At rank i
(1-based) use p_adj(i)=min(1,max_{j<=i}[(m-j+1)*p(j)]). Reject only if p_adj<=.05.
Persist the complete inventory, raw and adjusted p, effect sign, bootstrap seed,
N, source/seed support, and all unavailable reasons.

**Resolution constraint:** with 10,000 draws, the smallest raw p is 1/10001.
For S of size 1560, the first Holm threshold is .05/1560 (~0.0000321), below this
resolution. Therefore this finite-draw proposal cannot reject any member of S.
That is an explicit conservative limitation, not a software bug. It exposes a
real incompatibility between broad confirmatory testing and the inherited 10k
setting. Do not use p=0 to bypass it. Before adoption, choose explicitly between:
(a) retain this descriptive/conservative secondary plan; or (b) prospectively
increase raw-test simulation effort (at least 31,199 draws merely to make the
first threshold reachable, preferably materially more), keeping 10k CI draws;
or (c) justify a different prespecified test/family strategy. The adopted amendment below selects (a). Primary P is resolvable at 10k.

## 7. Missingness and claim gate

Missing cell, seed, source, identity linkage, seal, invalid probabilities or
unavailable contrast prevents a complete family conclusion. Preserve every
reserved hypothesis; do not shrink m. A conservative p=1 placeholder may be used
for bookkeeping only, explicitly marked untested, never as evidence of a null.
Do not publish rejection claims from an incomplete family. Family P can be
complete while S remains incomplete; this does not complete the ablation claim.
Expected structural unavailability, including n=8 when too few future outcomes
remain, must be resolved by a separately dated protocol amendment or left
unavailable. It must never be repaired by relabeling evaluator outcomes as public
or by silently dropping the offending rows. Directional/descriptive output is
allowed with a prominent incomplete status.

## 8. Implementability from sealed artifacts

Current `predictions.npz` stores labels, source clusters, ordinary/fixed/effective,
global_mass, both matched-parameter controls and effective mass. It supports the
main contrasts and A9 after identity validation, but contains no candidate/target
IDs. `complete.json` hashes predictions/report; report hashes cache/config and
runner/evidence sources. Validate this chain before any reconstruction.

Reconstruct query identity from the sealed cache's ordered records and ordered
tests, including source, split, candidate/task identity and target index/ID.
Compare reconstructed labels/clusters/main-arm predictions with NPZ exactly or
with exact labels/clusters and probability tolerance rtol=1e-12, atol=1e-14. Refuse ambiguous duplicate record IDs;
never align solely on equal labels. Across seeds align source IDs, not row index.

| Requirement | Current artifact / necessary independent reconstruction |
|---|---|
| A1, core controls, existing A5 | Six saved arm arrays; validate identity via cache |
| A2/A3/A4 | Reports contain scalar factorial metrics and some NLL CIs, insufficient for ECE/repeated-source inference; rebuild all 32 paired predictions from sealed cache/config |
| A5 same-kernel | Additional validation-only mass selection with fixed EED alpha/strength; not in current report/NPZ |
| A6 | Scalar means/settings saved; rebuild constant-mass predictions for both controls |
| A7 | Three permutation scalar metrics saved, not predictions; reproduce exact ordered multiset and fixed permutations |
| A8 | Selected settings/metrics or unavailable status saved; rebuild each viable history's target identities and predictions |
| A9 | effective_mass and matched arm arrays saved; derive public-only memberships and audit common support |
| A10 | Binary metrics saved, not binary predictions; reconstruct binary labels/arms with the locked mapping and parameters |

Do not infer p values from report-level means, standard deviations or rounded
intervals. An independent postprocessor may reconstruct statistics from sealed
cache without modifying frozen generation/mechanism code. Its own source/config,
input digests, prediction-reconstruction checks and final inventory must be
sealed. Raw evaluator outcomes remain confined to the evaluator/postprocessing
boundary and must never be sent back to a generator.

## 9. Adoption checklist

Before statistical implementation, resolve and sign: family/primary contrast;
conditional fixed-seed scope; A9 common-support convention; same-kernel A5;
raw-p method and its assumptions; 10k-versus-1560-family resolution conflict;
the proposed RNG mapping and reconstruction tolerance; structural A8 availability.
Until then, all existing mechanism tables remain descriptive and this file is
only a complete proposal, not evidence that Holm has been performed.

## 9. Adoption and review record

Adopted before scientific scoring on 2026-09-20, after independent supervisor
review. The filename retains DRAFT for provenance; this section and the status
above supersede the earlier draft status. Select option (a): retain the full
1,560 secondary slots and 10,000 draws, explicitly limiting secondary results
to descriptive effects and conservative non-rejection. The eight-hypothesis
primary family remains numerically resolvable. No p=0 substitution, adaptive
extra sampling, or selective family reduction is permitted.

Holm uses approximate bootstrap p values; this does not establish exact FWER
control. In particular, nonsmooth ECE and small-support bins can have poorly
calibrated bootstrap p values. Do not claim exact or study-wide error control.
Implementation and validation remain outstanding; adoption alone does not
produce any result or fulfill the statistical reporting gate.
