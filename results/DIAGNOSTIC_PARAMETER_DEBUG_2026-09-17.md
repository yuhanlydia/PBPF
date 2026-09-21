# Real-data diagnostic debugging: weak interactions, not just a hard gate

Eleven new real-data runs locate a reproducible initialization sensitivity.
Removing the additive diagnosis MLP improves ordinary prediction. On that
simplified model, multiplying only the initialized test × g interaction weights
by ten creates a substantially larger association gap in both training seeds.
The result survives separate sampling/corruption perturbations and evaluation
with 256 particles. It does not establish bug identifiability or repair control.

## Matched comparisons

All rows below use fixed selected checkpoints and the same development
population: 91 candidates, 31 source components, 40 candidates with nonconstant
visible histories, four visible and six future tests. Probability distributions
are averaged over sampling seeds 51701/71701/81701/91701, with corruption seed
fixed at 51701, before NLL is calculated. This differs slightly from averaging
four separate NLLs. All feature preparation and training exclude the held-out
test split. Architecture development repeatedly used this development set.

| Model | Training seed | Averaged-prediction NLL | Association gap | Descriptive source-bootstrap 95% interval for gap |
|---|---:|---:|---:|---|
| Original interaction reference | 1701 | .458578 | .000428 | [.000034, .000949] |
| Interaction-only diagnosis | 1701 | .428329 | .000037 | [-.000044, .000138] |
| Interaction-only, 10× initialization | 1701 | .432941 | .020033 | [.005603, .037845] |
| Original interaction reference | 1702 | .456348 | .000297 | [-.000291, .000907] |
| Interaction-only diagnosis | 1702 | .432693 | .000088 | [-.000036, .000226] |
| Interaction-only, 10× initialization | 1702 | .432676 | .016982 | [.001187, .037190] |

Relative to the reference, high-gain NLL improves by .025637 (descriptive interval
[.004642,.048751]) and .023672 ([.007036,.041441]). Relative to the simplified
interaction-only predictor, high gain costs .004612 NLL in seed 1701 and changes
NLL by less than .00002 in seed 1702; neither corresponding interval excludes
zero. The improvement combines removal of an additive head and changed
initialization. We have **not** tested whether scaling the original additive
architecture alone would have the same effect.

These are adaptive development results. Confidence intervals are descriptive,
not fresh confirmatory inference after model/checkpoint selection. Two training
seeds are optimization replications on the same population, not two datasets.
The .03 full-population screen remains unchanged: larger positive signal does
not mean all evaluations now cross that threshold. The original selection
screen passes high-gain seed 1702 but not seed 1701; independent fixed-corruption
averages above are below .03. The debugging improvement does not depend on
relabeling either result.

## What was actually debugged

### 1. The concatenation encoder barely binds pairs

For an affine encoder, averaging `A*t_i + B*onehot(y_i) + b` retains only mean
test features and the outcome histogram, exactly eliminating correspondence.
The trained semantic pair encoder's Tanh preactivations stay near its linear
region (absolute maximum .249). Reassigning outcomes changes its pooled output
by a median .000518 relative, and changes proposal diagnosis means by only
.000031 standard deviations per coordinate (RMS).

Explicit one-hot-outcome × test feature blocks preserve pairing even through a
linear encoder. Both binding runs raise median standardized proposal changes to
.00594/.00888. But predictive gaps remain roughly .0004–.0006. The encoder is a
real weak point, **not the sole explanation of prediction failure**.

For the reference, using the exact same proposal particles and reweighting only
by shuffled-history likelihood reproduces almost all the tiny original gap
(.000614 versus .000611). A proposal is a sampling distribution, not the target
posterior; improving proposal sensitivity alone need not improve the learned
likelihood's diagnostic use.

### 2. The additive diagnosis path dominates the explicit interaction

On eligible development candidates, posterior-averaged, class-centered logits
have RMS 2.726 for the additive diagnosis MLP versus .0208 for the explicit
bilinear test × g path. The additive path's test-varying RMS is .184, indicating
that most of its magnitude is test-constant. The difficulty head RMS is 1.151.
These amplitude diagnostics motivate a hypothesis; they alone are not a causal
proof or a measure of predictive importance.

The `interaction_only` arm removes that diagnosis MLP while keeping the original
encoder, difficulty head, prior and inference. Its diagnosis contribution is
zero when g or test features are zero. This does not prove disentanglement:
test embeddings can themselves contain candidate-global components. Ordinary
NLL improves in both seeds, but association remains weak.

The `high_gain` arm changes only the initialized interaction-head weight scale
relative to `interaction_only`. Tests confirm all other initialized parameter
tensors match exactly at the same seed. Weights remain trainable; this is not
post-hoc temperature scaling. The matched experiments support initialization
scale as a cause of weak association **in this architecture and dataset**.

### 3. Several plausible parameter changes do not solve association

The following averages hold the corruption seed fixed and vary only sampling
noise. Original reference seed 1701 averages NLL .460846 and gap .000455.

| Single-factor change | Mean NLL | Mean full-population gap | Interpretation |
|---|---:|---:|---|
| Evidence weight 1 → .1 | .518049 | .001436 | Larger tiny gap, much worse prediction |
| Invariance weight .1 → 0 | .458557 | .000657 | Small change, not a solution |
| Association weight 1 → 10 | .578578 | .000957 | Damages prediction without substantial pairing use |
| Training particles 8 → 32 | .455473 | .000450 | Slightly better prediction, unchanged association |
| Bound encoder, seed 1701 | .457496 | .000643 | Stronger proposal response, weak predictive effect |
| Bound encoder, seed 1702 | .452103 | .000374 | Same qualitative result |

The association objective detaches shuffled NLL, so increasing its weight mostly
increases aligned-prediction gradients on active eligible examples. It does not
supply an explicit contrastive gradient through the shuffled branch. This helps
explain why weight tuning alone need not teach correspondence, but the runs do
not isolate that objective choice from all other optimization effects.

### 4. The old Monte Carlo label hid a seed confound

Previously `_predict` used one seed for particle draws/resampling uniforms and
for history corruption. The old `monte_carlo_repeats` therefore varied both.
Their original records, checkpoint selections and primary metrics are retained;
the repetitions must be interpreted as mixed sensitivity checks.

The optional `counterfactual_seed` now separates these sources. Default calls
preserve previous behavior. New runner repetitions fix corruption and vary only
sampling; per-evaluation seed metadata and the protocol are recorded. Regression
tests confirm fixed histories with changing noise, fixed noise with changing
histories, and default-call equivalence.

Frozen-checkpoint audits use independent sampling/corruption panels. Re-audits
also verify dataset and semantic-feature hashes against checkpoint metadata.
No checkpoint was reselected using these panels.

## Stability of the improved setting

Across four K64 sampling draws at a fixed corruption:

- Seed 1701: gap .015283–.026148, mean .020264; mean NLL .436423.
- Seed 1702: gap .014106–.020455, mean .017060; mean NLL .436177.

Changing corruption while fixing particle noise also preserves positive gaps.
At K256, gaps are .017177/.016777 for seed 1701 and .024240/.020969 for seed 1702.
Mean ESS/K is about .26–.31: population-average ESS remains moderate;
individual-candidate collapse is not ruled out. Equal RNG seeds at different K do not imply nested particle sets for
every candidate; these are budget-sensitivity checks, not paired extensions.

## Reproduction, verification and limitations

Use `--arm binding`, `--arm interaction_only`, or `--arm high_gain` in
`scripts/run_association_debug.py`, keeping the existing semantic cache, fixed
history, 1000 steps, training K8 and evaluation K64. Parameter-only runs use
`--evidence-weight .1`, `--invariance-weight 0`, `--association-weight 10`, or
`--particles 32` with the original interaction arm.

For this historical round, also pass `--selection-replicates 1` when using the
updated runner, plus `--checkpoint-policy historical_screen`; the later
correction defaults to four draws and advisory association reporting.

`inspect_diagnostic_state.py` instruments frozen checkpoints.
`audit_diagnostic_sampling.py` snapshots source and separates sampling, corruption
and evaluation-budget changes. Full configurations and pre-run stage plans are
under `/root/pbpf-runs/parameter-debug-20260917/` with exact checkpoints, source
snapshots, logs, checksums, predictions and bootstrap inputs. The companion JSON
summarizes every run and audit; no failed trial was dropped.

39 focused tests pass; a further six-test run verifies exact matched
initialization and data-fingerprint guards. Eleven real training runs completed.
The prior full-suite result (723 passed, one optional skip) is historical and was
not rerun for this isolated follow-up. Independent reviews found no blockers.

The stable result is stronger **predictive association on this development
population**. It is not proof that g denotes a specific bug type, that particles
beat the best deterministic/query-memory controls, or that diagnosis improves
active testing or repair. Fresh source-disjoint confirmation remains necessary
before broad generalization claims.
