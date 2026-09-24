# EESD paper-order experiment roadmap — 2026-09-24

Branch: `research/eesd-shapley-relevance-20260924`

This document maps the final paper narrative to the exact experiment order. It
separates sealed legacy EESD evidence from the new axiomatic EESD implementation.

## Version boundary

- **Sealed 2026-09-22 results** (e.g. DeepSeek/CodeARC +5.4 pp) were produced by
  the previous EESD implementation. They are retained as empirical motivation /
  previous-EESD evidence and must not be relabeled as results of the new axiomatic
  Shapley method.
- **Final axiomatic EESD** is:
  exact edit-token evidence Shapley -> normalized relevance -> Renyi-2 effective
  support -> Jeffreys Dirichlet over improve/unchanged/regress -> positive posterior
  mean advantage -> KL-anchored weighted SFT.
- Axiomatic scientific claims require new Gate B/C/D results.

## Paper order

### 1. Introduction + Figure 1: discover the trust problem

Empirical observation:
- sealed seed-1701 12-cell previous-EESD matrix is heterogeneous;
- strongest sealed positive cell: DeepSeek/CodeARC +5.4 pp, CI [+2.8,+8.0];
- RunBugRun positive point estimates: DeepSeek +2.8 pp, Qwen +2.4 pp;
- complete 12-cell matrix stays in the appendix.

Mechanism observation:
- RunBugRun exploratory NLL: ordinary 0.458301, fixed 0.440499, effective 0.426433;
- fixed/effective have identical argmax on all 594 outcomes.

Paper claim at this point:
correctness alone is too coarse; self-generated corrections need an explicit
correction-to-update trust layer.

No new experiment is required to state this motivation, but the figure must label
the 12-cell matrix as sealed previous-EESD evidence, not final axiomatic-EESD results.

### 2. Problem formulation

For one task:
- initial program a0;
- four public executions;
- one self-generated correction a1;
- before/after outcomes;
- automatic execution delta d_i in {improved, unchanged, regressed}.

Question:
how strongly should this correction update the policy?

No experiment.

### 3. Method + Figure 2: axiomatic EESD

#### 3.1 Exact evidence Shapley
For execution subset S:
v(S) = mean log p(a1_edit | E_S) - mean log p(a0_edit | E_S).
Four executions -> 16 exact coalitions.
phi_i = Shapley_i(v); relevance s_i = |phi_i|.

#### 3.2 Effective evidence
rho_i = s_i / sum_j s_j.
m_eff = 1 / sum_i rho_i^2 = exp(H_2(rho)).

#### 3.3 Jeffreys posterior
alpha'_k = 1/2 + m_eff * sum_i rho_i 1[d_i=k].

#### 3.4 Bayes update weight
w = [ E(theta_improved - theta_regressed | E) ]_+
  = [ (alpha'_improved - alpha'_regressed) / sum_k alpha'_k ]_+.

#### 3.5 Actual implemented policy objective
The paper should match the trainer:
L = w * CE(correction) + beta * KL(pi_t || pi_theta)
on response-token positions.
Do not describe the implementation as a literal candidate-mixture q_t unless a
separate derivation explicitly establishes equivalence.

### 4. Theory

Main-text properties:
1. exact Shapley efficiency / symmetry / dummy / additivity as the attribution basis;
2. 1 <= m_eff <= n;
3. uniform relevance recovers ordinary counts;
4. relevance-scale invariance;
5. fixed support direction + lower m_eff implies stronger shrinkage to the symmetric prior;
6. 0 <= w <= 1 and w=0 when posterior expected improve-minus-regress is non-positive.

Legacy NLL derivative / tuned lexical mechanism theory may remain in the appendix
if it is used only to interpret the old exploratory mechanism experiment.

### 5. Experiments

The final paper should answer four RQs.

#### RQ1. Does execution evidence actually identify trustworthy corrections?

**Gate B — signal gate (before any new SFT).**
Run exactly three predeclared seed-1701 cells:
1. DeepSeek-Coder-6.7B / CodeARC (old strongest positive);
2. Qwen2.5-Coder-7B / RunBugRun (old moderate positive);
3. Gemma-3-4B / RunBugRun (old negative robustness cell).

For each, uniformly sample 64 trajectories without replacement, seed 1701:
```
python scripts/score_eesd_shapley_relevance.py \
  --input <corrections.jsonl> \
  --model-config <model.yaml> \
  --output <new-output> \
  --mode exact \
  --sample-size 64 \
  --sample-seed 1701
```

Report:
- mean/median game_span;
- mean attribution_l1;
- effective-mass distribution;
- zero-attribution fallback rate.

Decision:
- near-zero game_span -> stop Shapley branch;
- nonzero game_span but m_eff near 4 -> attribution exists but adaptive mass has little room;
- nonzero game_span + broad m_eff range -> proceed.

#### RQ2. Does Shapley + effective mass predict hidden correction benefit?

**Gate C — full attribution + independent hidden credibility.**
For the same three cells, rerun exact Shapley on the full correction population,
then evaluate with hidden outcomes that never enter training.

Comparators:
- Shapley + effective mass (main);
- Shapley + fixed mass (isolates EED);
- legacy lexical relevance + effective mass (isolates Shapley);
- scalar public correctness;
- final correctness.

Metrics:
- posterior benefit probability: hidden soft-NLL, Brier, Spearman;
- actual Bayes training weight: MSE/MAE and Spearman vs hidden positive net gain;
- 10,000 whole-source paired bootstrap draws.

Required causal comparisons:
A. Shapley-effective vs lexical-effective -> attribution contribution.
B. Shapley-effective vs Shapley-fixed -> adaptive mass contribution.

#### RQ3. Does final axiomatic EESD improve the next policy?

**Gate D — matched-budget one-round learning.**
Run the same three seed-1701 cells with:
`scripts/run_eesd_axiomatic_cell.py`.

Keep fixed:
- correction population;
- previous policy;
- response-token budget;
- primary 500-source evaluator;
- seed 1701;
- direct execution profile.

Headline metric:
all-tests Pass@1 vs sealed no_update.

Secondary:
fix rate, regression rate, net gain, retained-correct rate, policy KL.

Desired interpretation:
- DeepSeek/CodeARC: preserve or improve the old positive effect;
- Qwen/RunBugRun: retain positive transfer if real;
- Gemma/RunBugRun: reduce harmful updating, demonstrating trust-layer conservatism.

#### RQ4. Which component matters and does it generalize?

Only after Gates B/C/D are scientifically viable:

**Ablations**
1. exact Shapley vs legacy lexical relevance;
2. effective mass vs fixed mass under identical Shapley relevance;
3. edit-token Shapley vs full-response Shapley;
4. Bayes mean-advantage weight vs scalar confidence / final correctness / equal weight;
5. EESD full vs no-KL-anchor;
6. Jeffreys alpha=0.5 main rule, with prior sensitivity only in appendix.

**Generalization**
- expand final main method to all 12 benchmark/backbone cells for seed 1701;
- keep the complete matrix, not positive-only analysis;
- then repeat final method + key ablations for seeds 1702 and 1703.

**Recursive stability**
After one-round evidence is stable:
- Qwen on RunBugRun and CodeARC;
- rounds 0,1,2,3;
- shared EESD teacher protocol;
- equal-weight matched control from the same entering teacher;
- report Pass@1, fixes, regressions, retained correctness, net gain, policy KL.

### 6. Appendix order

A. Full 12-cell seed-1701 matrix.
B. Full Shapley signal/credibility diagnostics.
C. Full component ablations and prior sensitivity.
D. Per-seed tables (1701/1702/1703).
E. Recursive round-by-round results.
F. Historical lexical-EED mechanism study (594 outcomes / 32 sources).
G. Historical PBPF prediction-to-action diagnostics.
H. Cost, generation counts, execution counts, token budgets, and diversity diagnostics.
I. Optional transfer (HumanEval+/MBPP+, then LiveCodeBench/SWE-bench only if time permits).

## Immediate execution order on the machine

### Gate A0 — repair experiment-contract CI boundary
The latest branch-wide CI currently fails because the repository mixes the new
axiomatic experiment schema with frozen legacy contracts, missing large run-lock
files on GitHub CI, optional ML dependencies, and old absolute /root paths.
Before GPU experiments:
- ensure the two axiomatic unit-test files pass in the actual experiment environment;
- compile the touched axiomatic source/scripts;
- separate legacy repository-contract failures from failures in touched code;
- do not weaken frozen legacy scientific checks merely to make CI green.

### Gate A1 — minimal axiomatic test command
```
python -m pytest -q \
  tests/eesd/test_shapley_relevance.py \
  tests/eesd/test_axiomatic_trust.py
python -m compileall -q \
  src/pbpf/eesd/shapley_relevance.py \
  src/pbpf/eesd/evidence.py \
  src/pbpf/eesd/distillation.py \
  scripts/score_eesd_shapley_relevance.py \
  scripts/score_eesd_corrections.py \
  scripts/report_eesd_axiomatic_validation.py \
  scripts/run_eesd_axiomatic_cell.py
```

Then: Gate B -> Gate C -> Gate D -> 12-cell expansion -> seeds 1702/1703 ->
component ablations -> recursive rounds -> optional transfer.

## Files implementing this roadmap

- `src/pbpf/eesd/shapley_relevance.py`
- `src/pbpf/eesd/evidence.py`
- `src/pbpf/eesd/distillation.py`
- `scripts/score_eesd_shapley_relevance.py`
- `scripts/report_eesd_axiomatic_validation.py`
- `scripts/run_eesd_axiomatic_cell.py`
- `configs/experiments/eesd_axiomatic_validation_20260924.yaml`
- `configs/experiments/eesd_iclr2027.yaml`
- `docs/EESD_AXIOMATIC_VALIDATION_20260924.md`
