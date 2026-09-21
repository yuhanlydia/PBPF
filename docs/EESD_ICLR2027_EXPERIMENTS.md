# EESD ICLR 2027 experiment lock

Status: **prospective experiment specification plus inherited measured diagnostics**.

This document is the experiment contract for the EESD paper. It is intentionally
broader than the 32-source exploratory Effective-Evidence result. The goal is to
finish the paper with three kinds of evidence:

1. **mechanism** — does adaptive evidence mass improve probability quality?
2. **action value** — does a better correction belief change selection/repair decisions?
3. **recursive stability** — does evidence-aware self-distillation retain gains and avoid regressions across rounds?

No result may be dropped because its sign is unfavorable. Primary populations,
seeds, metrics, and ablations are fixed here before the new runs.

## 1. Main benchmark matrix

| Dataset / protocol | Mechanism | Decision | One-round EESD | Multi-round EESD | Paper role |
|---|---:|---:|---:|---:|---|
| RunBugRun corrected-stdin | yes | yes | yes | yes | primary repair domain |
| CodeARC-Replay | yes | yes | yes | yes | cross-domain structured program induction |
| HumanEval+ | no | official pass@1 | transfer only | no | official EvalPlus downstream transfer |
| MBPP+ | no | official pass@1 | transfer only | no | official EvalPlus downstream transfer |
| LiveCodeBench v6 temporal slice | no | pass@1 | optional transfer | no | contamination-mitigated temporal transfer |
| SWE-bench Lite | no | repair only | optional | no | repository-level external validity, appendix |

RunBugRun and CodeARC use four public executions and the remaining evaluator-only
executions. HumanEval+/MBPP+ are not converted into an artificial EED mechanism protocol. They are official downstream transfer evaluations: adapters learned from RunBugRun corrections are evaluated with EvalPlus Base+Extra pass@1. LiveCodeBench is a later temporal downstream replication. Do not pool these downstream correctness benchmarks with the public-input/private-outcome mechanism estimand.

## 2. Model matrix

The required main model set is deliberately limited to models already represented
by pinned configs in this repository so the expansion is runnable on 16--24 GB
hardware with 4-bit loading.

| Key | Model | Role |
|---|---|---|
| qwen25_7b | Qwen2.5-Coder-7B-Instruct | primary |
| deepseek_6p7b | DeepSeek-Coder-6.7B-Instruct | cross-family replication |
| seed_coder_8b | Seed-Coder-8B-Instruct | second cross-family replication |
| starcoder2_15b | StarCoder2-15B-Instruct-v0.1 | third cross-family replication |

Execution scope amendment (2026-09-20, before new generation): the user limited
the required matrix to four different model families above. Qwen2.5-Coder-1.5B,
Qwen3-8B and Qwen3-Coder-30B-A3B remain optional historical configs.
No observed mechanism result was used to reduce this scope.

Minimum paper breadth:
- all four model settings on RunBugRun and CodeARC mechanism experiments;
- qwen25_7b and deepseek_6p7b adapters trained on RunBugRun, then evaluated on HumanEval+ and MBPP+ with the official EvalPlus sanitizer/evaluator;
- qwen25_7b on LiveCodeBench temporal transfer if the cheaper gates pass;
- qwen25_7b plus one cross-family model for full one-round EESD;
- qwen25_7b for the full multi-round recursive study.

Do **not** call Qwen2.5 and Qwen3 independent model families.

## 3. Core estimator table

The already measured exploratory table must remain in the repository and paper:

| Estimator | Selected alpha | Strength | NLL down | Accuracy up |
|---|---:|---:|---:|---:|
| Ordinary Dirichlet | 0.10 | 0 | 0.458301 | 82.6599% |
| Fixed-mass relevance | 0.10 | 16 | 0.440499 | 84.0067% |
| Effective evidence | 0.01 | 16 | **0.426433** | 84.0067% |

This table is a mechanism screen, not the final causal ablation, because alpha differs.

## 4. Mandatory mechanism ablations

These are main-paper or first-page appendix ablations. They are chosen to answer
distinct reviewer questions, not to maximize the number of rows.

| ID | Comparison | Fixed quantities | Question answered |
|---|---|---|---|
| A1 | ordinary vs fixed relevance | alpha tuned per arm | does relevance matter? |
| A2 | fixed vs effective, alpha=0.10 | same kernel, strength, alpha | does adaptive mass itself matter at the fixed baseline prior? |
| A3 | fixed vs effective, alpha=0.01 | same kernel, strength, alpha | does adaptive mass itself matter at the EED prior? |
| A4 | full alpha x strength factorial | same alpha and strength within every pair | is the sign stable or a cherry-picked operating point? |
| A5 | tuned global mass vs EED | same relevance kernel | is query adaptivity useful beyond choosing a better single mass? |
| A6 | constant mean(EED mass) vs EED | same alpha, strength, mean mass | is per-query mass adaptation useful beyond average shrinkage? |
| A7 | permuted EED mass vs aligned EED mass | same mass multiset | is local alignment between concentration and uncertainty useful? |
| A8 | n in {1,2,4,8} | same split and kernel | how does the effect scale with history size? |
| A9 | effective-mass concentration bins | bins depend only on public features | does the gain occur where the mechanism predicts it should? |
| A10 | 5-class vs PASS/non-PASS | same examples | is the effect an artifact of the failure taxonomy? |

The full factorial is reported even if only one region favors EED. A global-mass
baseline is mandatory because query-dependent EED is not identifiable from prior
shrinkage at one query without such a control.

## 5. Robustness ablations

Appendix unless a result becomes central:

- alpha grid: {0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10};
- relevance strength: {0, 1, 4, 16};
- visible history size: {1, 2, 4, 8} when the cache contains enough executions;
- 3 fixed seeds: 1701, 1702, 1703 for generated candidate banks;
- lexical signed-hash kernel vs frozen-model semantic kernel where the latter is available;
- source-cluster bootstrap with 10,000 paired draws;
- ECE, NLL, Brier, accuracy, and argmax agreement;
- confidence / NLL difference by effective-mass bin.

A duplicate-test stress test belongs in limitations, not as a promised positive
ablation: inverse-squared relevance concentration is **not** a general dependence
estimator and can fail to recognize redundant tests with equal relevance.

## 6. Correction-level ablations for EESD

For each original -> correction pair record four transition counts:

- FIX: fail -> pass
- REGRESSION: pass -> fail
- PRESERVED: pass -> pass
- UNRESOLVED: fail -> fail

Required training-rule comparisons:

| ID | Training rule | Purpose |
|---|---|---|
| D0 | no update | base policy |
| D1 | all accepted corrections, equal weight | standard self-distillation control |
| D2 | final-correctness filter only | tests whether binary acceptance is enough |
| D3 | scalar confidence weighting | generic uncertainty-weighting control |
| D4 | fixed-mass Dirichlet utility | isolates EED mass |
| D5 | EED mean utility, no uncertainty penalty | isolates conservative posterior penalty |
| D6 | EED utility, no policy anchor | tests whether anchoring prevents drift |
| D7 | **full EESD** | proposed method |

Primary correction metrics:
- next-round pass@1 / all-tests success;
- FIX rate;
- regression rate;
- net correction gain = FIX - REGRESSION;
- retained-correct fraction;
- policy drift / KL proxy;
- fraction of trajectories receiving positive training weight.

## 7. Recursive experiment

Primary model: Qwen2.5-Coder-7B-Instruct.
Primary datasets: RunBugRun corrected-stdin and CodeARC-Replay.

Rounds: 0, 1, 2, 3.

At every round:
1. freeze the current policy;
2. generate a shared candidate/correction bank;
3. execute the locked public suite;
4. construct correction transitions;
5. compute training weights using the arm's fixed rule;
6. update from the same starting checkpoint and same token budget;
7. seal the next policy before evaluator-only scoring.

Report both absolute performance and retention:

| Round | base/equal SD pass@1 | EESD pass@1 | equal SD regressions | EESD regressions | net gain |
|---|---:|---:|---:|---:|---:|

A self-improvement claim requires that gains survive at least one subsequent
generation/update round. A one-round gain alone is not called recursive improvement.

## 8. Paper tables after completion

Main text:
1. **Table 1** — problem-space / related-work comparison.
2. **Table 2** — estimator mechanism across datasets and models.
3. **Table 3** — same-alpha/global-mass/constant-mass/permuted-mass ablations.
4. **Table 4** — one-round EESD downstream results.
5. **Figure 3** — performance and regression over recursive rounds.

Appendix:
- full alpha x strength factorial;
- all history-size sweeps;
- per-seed results;
- all cross-model cells;
- legacy PBPF prediction/repair diagnostics;
- CodeARC Qwen/DeepSeek negative selection diagnostics;
- active-testing/stopping diagnostics;
- cost and token ledgers.

## 9. Stop / submit gates

The paper is ready for an ICLR main-paper claim only if:

- at least two independent dataset domains show a mechanism-aligned EED advantage
  over fixed mass under same-alpha or global-mass controls;
- at least one downstream decision or self-distillation endpoint improves at matched
  candidate and token budget;
- no main claim depends solely on the repeatedly viewed 32-source legacy assessment;
- cross-model direction is reported for every predeclared cell, including negative ones;
- the recursive study reports regression accumulation, not only final pass@1.

If mechanism generalizes but downstream EESD is null, submit as an uncertainty /
evidence-aggregation paper and remove recursive-improvement claims rather than
selectively dropping the null endpoint.


## 10. Recovered legacy tables from the previous APBPF branch

These tables were present in the longgoal experiment artifacts but were omitted when the 9/17
paper blueprint narrowed to the Effective-Evidence mechanism. They remain scientifically useful
as **diagnostic motivation** and must stay visible in the new paper appendix.

### Legacy Table A. Full-primary replication cells

| Domain | Generator | Sources | Candidates | Association gap | 95% CI | Selection advantage | 95% CI |
|---|---|---:|---:|---:|---|---:|---|
| CodeARC-Replay | Qwen | 500 | 4000 | -0.002618 | [-0.005301, 0.000065] | -1.20 pp | [-2.20, -0.20] pp |
| CodeARC-Replay | DeepSeek | 500 | 4000 | -0.002429 | [-0.004866, -0.000044] | -1.80 pp | [-2.93, -0.73] pp |
| RunBugRun | Qwen | 500 | 4000 | +0.000493 | [-0.005562, 0.006575] | +0.47 pp | [-0.93, 2.00] pp |

### Legacy Table B. Proper scoring versus raw accuracy

| Dataset / generator | learned NLL | learned acc. | history-rate NLL | history-rate acc. |
|---|---:|---:|---:|---:|
| CodeARC / Qwen | 0.544627 | 80.09% | 0.822852 | 81.41% |
| CodeARC / DeepSeek | 0.489617 | 82.26% | 0.787312 | 84.14% |
| RunBugRun / Qwen | 0.401980 | 87.61% | 0.710813 | 91.30% |

This is directly relevant to EESD: higher categorical accuracy can coexist with substantially
worse probability quality, so binary correctness and calibrated learning strength cannot be
treated as interchangeable signals.

### Legacy Table C. CodeARC Qwen selection headroom

| Selector | seed 1701 | seed 1702 | seed 1703 |
|---|---:|---:|---:|
| particle | 26.8 | 27.4 | 26.6 |
| pair-aware | 28.2 | 28.4 | 28.4 |
| deep sets | 28.0 | 28.0 | 28.4 |
| no-particle bottleneck | 28.4 | 28.0 | 28.4 |
| tuned Dirichlet | 28.0 | 28.0 | 28.0 |
| visible pass rate | 28.0 | 28.0 | 28.0 |
| cross-fitted deterministic | 28.2 | 28.0 | 28.2 |
| random expectation | 18.6 | 18.6 | 18.6 |
| oracle | 30.0 | 30.0 | 30.0 |

Values are selected Pass@1 percentages. The gap between deployed selectors and the 30.0% oracle
shows decision headroom, while the negative particle advantage shows that a richer predictor alone
does not automatically exploit that headroom.

These recovered tables support the gap statement:
**predictive information is not automatically actionable learning information.**
