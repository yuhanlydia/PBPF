# Axiomatic EESD implementation and validation — 2026-09-24

## Main method now implemented

The main path no longer uses lexical similarity, a hand-authored transition utility,
or a manually chosen uncertainty penalty.

For one correction and its four public executions:

1. **Exact evidence attribution.** Every public execution is a Shapley player.
   The coalition value is the correction-vs-original teacher-forced log-probability
   contrast on edited tokens only. Four executions require exactly 16 coalitions.
2. **Relevance distribution.** Signed Shapley values are retained for diagnostics;
   EED relevance is their magnitude normalized over executions.
3. **Evidence amount.** The total evidence mass is the Renyi-2 effective support
   `1 / sum(p_i^2)`.
4. **Execution effect.** Before/after execution success automatically induces one
   of three states: improved, unchanged, regressed.
5. **Bayesian trust.** A symmetric Jeffreys Dirichlet prior (alpha=1/2 per state)
   plus effective evidence yields a posterior over the three states.
6. **No hand utility.** The training weight is the positive posterior-predictive mean execution advantage,
   `max(0, E[theta_improved-theta_regressed | E])`. Because the prior is symmetric,
   weak effective evidence is automatically shrunk toward zero. The posterior
   probability `P(theta_improved > theta_regressed)` is retained only as a
   calibration diagnostic, not as the training weight.
7. **Policy stability.** `eesd_full` retains the existing KL anchor to the previous
   policy; `eed_no_anchor` uses the identical trust weights without the anchor.

Legacy cosine relevance is preserved only inside the original trajectory artifact
for paired scientific comparison.

## Code paths

- `src/pbpf/eesd/shapley_relevance.py`: exact Shapley / LOO / edit-token masking.
- `scripts/score_eesd_shapley_relevance.py`: model-based 16-coalition attribution.
- `src/pbpf/eesd/evidence.py`: Renyi effective mass and three-state Bayesian trust.
- `src/pbpf/eesd/distillation.py`: training-arm weights.
- `scripts/report_eesd_axiomatic_validation.py`: hidden-only credibility report.
- `scripts/run_eesd_axiomatic_cell.py`: matched-budget one-cell downstream run.
- `configs/experiments/eesd_axiomatic_validation_20260924.yaml`: locked minimal matrix.

The sealed 2026-09-22 seed1701 results are not rewritten.

## Validation 0 — mathematical/software gate

Run:

```bash
python -m pytest -q \
  tests/eesd/test_shapley_relevance.py \
  tests/eesd/test_axiomatic_trust.py
python -m compileall -q src scripts tests
```

The full CI additionally runs the complete CPU test suite on Python 3.11 and 3.12.

The axiomatic tests cover:
- Shapley efficiency, additivity examples and interaction splitting;
- exact Shapley vs leave-one-out distinction;
- edit-only token targeting;
- coalition prompt filtering;
- uniform relevance recovering ordinary counts;
- relevance-scale invariance;
- effective mass changing confidence without changing relative support;
- posterior confidence increasing with evidence mass in a fixed positive direction;
- unchanged execution histories producing zero update weight;
- closed-form incomplete-beta sanity cases.

The pure-Python incomplete-beta implementation was also numerically checked against
`scipy.special.betainc` over a grid of alpha/beta values from 0.01 to 20 and
x from 1e-6 to 1-1e-6; the observed maximum absolute difference in that local check
was approximately 2.9e-14. This numerical cross-check is development verification,
not a paper result.

## Validation 1 — attribution signal gate

For an existing generated correction bank, compute exact Shapley relevance without
regenerating corrections:

```bash
python scripts/score_eesd_shapley_relevance.py \
  --input <cell>/generated-corrections/corrections.jsonl \
  --model-config <pinned-model-yaml> \
  --output <cell>/shapley-signal-gate \
  --mode exact \
  --sample-size 64 \
  --sample-seed 1701
```

For a correction bank generated from an adapter, pass that exact `--adapter`.
The 64-trajectory run is a compute gate only. If its attribution signal is non-degenerate,
rerun without `--sample-size` to produce the full population used by hidden
credibility and downstream scoring. Never choose trajectories by observed outcome.

Inspect `report.json`:
- `mean_game_span` / `median_game_span`: does evidence actually alter edit preference?
- `mean_attribution_l1`: is attribution signal nonzero?
- `mean/min/max_effective_mass`: does Shapley produce useful mass variation?

Interpretation:
- tiny game span: the model ignores individual execution evidence; changing the
  mass estimator cannot fix that;
- large game span but mass near 4: evidence matters but is distributed broadly;
- large game span and broad mass distribution: the EED mechanism has room to act.

Do not select examples or cells based on these diagnostics.

## Validation 2 — independent hidden credibility

Use an independently produced hidden correction evaluation:

```bash
python scripts/report_eesd_axiomatic_validation.py \
  --corrections-shapley <cell>/shapley-relevance/corrections-shapley.jsonl \
  --hidden-eval <cell>/independent-hidden-evaluation.json \
  --alpha 0.5 \
  --output <cell>/axiomatic-credibility.json
```

This report compares, on exactly the same trajectories:
- Shapley + effective mass (main mechanism);
- Shapley + fixed mass (isolates EED);
- legacy lexical relevance + effective mass (isolates Shapley relevance);
- scalar public correctness;
- final public correctness.

Hidden outcomes never feed training, hyperparameter selection, correction generation,
or stopping. The report uses hidden fix-vs-regression direction only for evaluation.

Mechanism evidence is reported in two aligned views:
- posterior benefit probability: soft-NLL/Brier for hidden beneficial-vs-harmful direction;
- actual Bayes training weight: MSE/MAE and Spearman against hidden positive net gain.

The two key paired probability comparisons are Shapley-effective vs lexical-effective
(isolating attribution) and Shapley-effective vs Shapley-fixed (isolating adaptive
effective mass). Use the 10,000 whole-source paired bootstrap interval; do not drop
negative cells.

## Validation 3 — matched-budget downstream gate

Run seed 1701 first on these three predeclared cells:

1. **DeepSeek-Coder-6.7B / CodeARC** — old EESD: +5.4 pp, strongest positive cell.
2. **Qwen2.5-Coder-7B / RunBugRun** — old EESD: +2.4 pp, moderate positive cell.
3. **Gemma-3-4B / RunBugRun** — old EESD: -2.0 pp, negative robustness cell.

Use the exact old cell's train/development correction population, previous policy,
response-token budget, primary 500-source evaluator and seed. Change only:
- relevance: lexical cosine -> exact evidence Shapley;
- trust rule: hand utility / uncertainty penalty -> posterior benefit confidence.

One cell:

```bash
python scripts/run_eesd_axiomatic_cell.py \
  --corrections <existing-train-dev-corrections.jsonl> \
  --model-config <pinned-model-yaml> \
  --config configs/experiments/eesd_iclr2027.yaml \
  --domain <rbr-or-codearc> \
  --public-root <locked-public-root> \
  --evaluator-root <locked-evaluator-root> \
  --family <same-family-as-old-cell> \
  --baseline-report <sealed-no_update-report.json> \
  --response-token-budget <exact-old-sealed-budget> \
  --seed 1701 \
  --alpha 0.5 \
  --anchor-beta 0.03 \
  --execution-lock <sealed-direct-execution-lock.json> \
  --execution-lock-sha256 <exact-lock-sha256> \
  --output <new-create-once-output>
```

The wrapper runs:
exact Shapley -> axiomatic scoring -> matched-budget EESD SFT -> fresh primary
generation -> fresh all-test evaluation -> paired comparison with no_update.

Do not overwrite old runs.

## Scientific interpretation

The method is supported only if the components separate cleanly.

- **Shapley relevance claim:** Shapley-effective should improve hidden credibility
  over legacy lexical-effective.
- **Effective-evidence claim:** Shapley-effective should improve hidden credibility
  over Shapley-fixed under identical Shapley relevance.
- **Learning claim:** downstream update should improve or preserve the positive
  cells without introducing stronger negative transfer in the robustness cell.
- **KL claim:** compare `eed_no_anchor` and `eesd_full` on any cell entering the
  full ablation matrix; identical trust weights isolate the anchor.

A negative outcome is informative:
- if Shapley beats lexical but EED does not beat fixed mass, the new contribution is
  evidence attribution rather than adaptive evidence mass;
- if hidden credibility improves but downstream performance does not, the remaining
  problem is the policy-update objective rather than the trust estimator;
- if game spans are near zero, the model is not conditionally using the evidence and
  no relevance estimator based on its conditional likelihood can recover causal signal.

Do not rewrite these distinctions after seeing the results.


## Minimal run order

Do not start with the full matrix.

### Gate A — software
Run the two axiomatic unit-test files and compile all Python sources.

### Gate B — 64-trajectory exact-Shapley signal
Run exactly the three predeclared seed-1701 cells:
DeepSeek/CodeARC, Qwen/RunBugRun, Gemma/RunBugRun. Use the same fixed sample seed
1701 and do not inspect outcome quality to choose trajectories. Record game-span,
attribution-L1 and effective-mass distributions.

### Gate C — full attribution + hidden credibility
If Gate B executes correctly, score the full correction population for all three cells
and run the independent hidden credibility report. This gate determines whether the
new attribution and adaptive mass are actually informative before any SFT cost.

### Gate D — matched-budget one-round learning
Only then run `run_eesd_axiomatic_cell.py` on the same three cells using the exact
sealed old response-token budgets and direct execution lock. Compare to the already
sealed no-update and old-EESD seed-1701 reports.

### Expansion
If the method survives all three roles (strong-positive, moderate-positive,
negative-robustness), expand to all 12 seed-1701 benchmark/model cells. Only after
that repeat the main and key ablations for seeds 1702/1703.
