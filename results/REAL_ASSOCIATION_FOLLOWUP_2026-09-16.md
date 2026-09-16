# Real association follow-up: preserve paired evidence until the query

Real-data latent diagnosis remains unresolved. The completed diagnostic work
adds a useful distinction: predictive pairing signal exists in this development
population, but the neural particle model has not learned to exploit it.
A deterministic query-memory control is evidence for available association,
not interpretable bug identities, particle necessity, active-testing gains, or
repair success.

## Protocol

Existing source-disjoint cache: 440 training candidates / 158 source components,
91 development candidates / 31 components. Four visible and six future tests.
Held-out test records were not encoded, tensorized, scored, or used for selection.
All new results are exploratory development evidence on a repeatedly inspected
population. No new confirmatory claim is justified by development intervals.

The locked neural matrix compares lexical/contextual CodeBERT representations
and fixed/uniformly permuted training histories. Both tests and their outcomes
are permuted together; development histories remain fixed. All cells use seed
1701, 1000 steps, 8 training particles and 64 evaluation particles. Candidate
sampling is shared with deterministic baselines. Checkpoint selection retains
the 0.03 nats/test association threshold and the absolute-NLL guard. No threshold
was relaxed after observing failures.

CodeBERT revision: `3b0952feddeffad0063f274080e3c23d75e7eb39`.
Feature SHA256: `32de78763c722c9f1f214ddb37e30932cd139542f25731d25b04a155d8c1bfde`.
Only public task/code/input fields enter encoding; projection centering uses
training features only. The treatment includes context and pooling changes,
not pretrained weights alone. Of 5898 unique encoded records, 59 test inputs,
5050 contexts and 241 single texts were truncated; this encoder does not provide
full-program semantics for long candidates.

## Completed neural matrix

| Features | Training histories | Selected step | Development NLL | Shuffle gap | Gate |
|---|---|---:|---:|---:|---|
| Lexical | Fixed | 250 | .469081 | .000232 | Fail |
| Lexical | Permuted pairs | 350 | .452519 | -.000192 | Fail |
| Contextual CodeBERT | Fixed | 550 | .462123 | .000611 | Fail |
| Contextual CodeBERT | Permuted pairs | 700 | .453408 | .000525 | Fail |

All four runs completed 1000 steps and all 436 recorded artifact checksums
verified. No particle setting qualified for a second optimization-seed run.
The semantic fixed-prefix gap has a tiny positive descriptive interval, but is
roughly fifty times smaller than the unchanged .03 practical threshold.

The strongest ordinary predictors are the pair-aware deterministic controls
with permuted training histories: lexical NLL .421479 / gap -.001212 and semantic
NLL .421210 / gap -.001308. Their predictive improvement also does not establish
association. Fixed-history semantic pair-aware prediction reaches gap .020134
at NLL .449531, still below .03; semantic deterministic interaction reaches gap
.022145 at much worse NLL .596132. All controls and raw values are retained in
the companion JSON, including failures.

## Query-memory control

This exploratory extension was added after the locked matrix started. A fixed
similarity probe suggested available pairing signal. The reusable implementation
then selects a grid of smoothing alpha, temperature and mixture using training
NLL only. No future outcomes enter its prediction API.

For query test t, let a_i(t) be softmax cosine similarity to visible test i.
Set w_i(t) = (1-beta)/n + beta*a_i(t), then

`p(y | t,H) = (n * sum_i w_i(t) 1[y_i=y] + alpha) / (n + 5*alpha)`.

At beta=0 this is exactly the histogram control. Unlike a global compressed
state, this computation preserves individual pairs until the query is known.
It requires no association margin or learned particle inference.

| Control | Train-selected alpha / temperature / mixture | Development NLL | Shuffle gap | Descriptive source-bootstrap 95% interval |
|---|---|---:|---:|---|
| Lexical query memory | .03 / .03 / .75 | .429781 | .090209 | [.025602, .175368] |
| Semantic query memory | .03 / .1 / .75 | .436416 | .115098 | [.038737, .204092] |
| Train-selected histogram | .03 / irrelevant / 0 | .455412 | 0 | — |

Main shuffle seed 51701. Across four corruption seeds, lexical gaps range
.077081–.097777 and semantic gaps .070472–.115098; their individual descriptive
intervals exclude zero. These reuse the same population and are sensitivity
checks, **not independent replications**. Pair-preserving permutation changes
probabilities by at most 1.58e-7. The visible-label-only API and tests prevent
future-label access.

NLL gains over the histogram are .025630 (lexical; interval
[-.004223,.062981]) and .018995 (semantic; [-.019647,.059010]). Both intervals
include zero. Thus the result supports association but does not establish a
stable advantage over difficulty-only prediction. Lexical NLL is lower than
semantic NLL; these results do not establish CodeBERT superiority.

Relative to the strongest matched-feature learned predictor, the kernel has
slightly worse NLL: +.008303 lexical and +.015207 semantic. Paired descriptive
intervals for kernel gain are [-.083139,.072739] and [-.100833,.062285].
These data therefore do not establish kernel prediction superiority either.

## What the debugging actually changes

Do not equate a failed neural shuffle gap with absence of signal in these data.
The direct query-memory computation demonstrates a usable relation that can be
lost through global summarization, optimization or inference. These experiments
do not isolate which of those mechanisms causes neural failure.

The revised neural association term also deserves precise interpretation:
`relu(m + NLL_aligned - stop_gradient(NLL_shuffled))` has, on active examples,
only the aligned-NLL gradient. It reweights predictive training; it is not a
full contrastive representation objective. Removing stop-gradient without a
calibration constraint can instead reward making the corrupted predictor worse.
Neither version by itself proves operational bug diagnosis.

The implemented alternative keeps a histogram fallback and query-conditioned
paired memory. Any next latent architecture should first preserve this direct
path and beat its calibrated control before adding claims about diagnosis
entropy or repair. A genuinely new, locked source-disjoint evaluation is still
needed to establish generalization; the previously exposed test split must not
be recycled as a fresh confirmation.

## Verification and artifacts

28 focused tests passed, including field whitelisting, projection/cache binding,
pair permutation, deterministic counterfactual controls, exact histogram limit,
visible-only kernel API, and the existing diagnostic inference tests. Independent
review reproduced the kernel scores and histogram-bootstrap intervals. The prior
full suite had passed 723 tests with one optional-dependency skip; it was not
rerun for this isolated follow-up.

Exact feature cache, checkpoints, source snapshots, baseline states, raw logs
and checksums are under `/root/pbpf-runs/real-association-features-20260916/`.
`kernel-verified.json` contains the training grid and both comparator bootstraps.
Reproduction commands are in `docs/DIAGNOSTIC_DEBUG.md`.
