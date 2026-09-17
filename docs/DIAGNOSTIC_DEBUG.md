# Diagnostic debugging reference

This optional path investigates the failed lexical-feature RunBugRun pilot.
It is **development-only and exploratory**. It never tensorizes or evaluates
the held-out test split, and does not replace the `apbpf-iclr-v1` worker DAG.

## What changed

`src/pbpf/belief/diagnostic.py` provides:

- `HistoryISBeliefModel`: an exchangeable Gaussian proposal using each visible
  prefix. Importance weights include the full prefix likelihood and prior/q
  correction. Prefixes are separately sampled; this is **not legacy FIVO**.
- `InteractionHistoryISBeliefModel`: the same estimator plus an explicit
  `W(g * W_t(test))` likelihood residual. This separately ablated interaction
  makes test-trigger relations easier to express.
- `train_diagnostic_step`: per-test evidence/future units, an aligned-only
  association gradient, and weighted latent MMD for difficulty invariance.
  MMD is not KL; none of these choices establishes latent identifiability.
- `DeterministicInteractionPredictor`: a non-particle control using
  outcome-conditioned test summaries and a bilinear decoder. It shares input
  visibility, but is not parameter/compute matched to the particle model.
- `select_diagnostic_checkpoint`: screen on development association and pair
  invariance within 0.02 NLL of the best absolute predictor. If no checkpoint
  qualifies, select minimum NLL and explicitly report screen failure.

Legacy constructors and training remain available. The one shared numerical
fix makes SMC CDF probabilities agree with native-precision ESS probabilities,
so an underflowed zero-mass particle is not selected at a zero-offset boundary.
The regression uses explicit draws instead of a version-sensitive RNG seed.

## Reproduction

Requires Python >=3.11 and the `neural` dependencies. From an installed checkout:

```bash
PYTHONPATH=src python scripts/run_association_debug.py \
  --source controlled --arm interaction --seed 1701 \
  --feature-dim 16 --hidden-dim 64 --latent-dim 8 --difficulty-dim 2 \
  --particles 16 --steps 1000 --output runs/debug/controlled-interaction-1701

PYTHONPATH=src python scripts/run_association_debug.py \
  --source runbugrun --cache /absolute/path/to/rich-cache-v3.json \
  --arm interaction --seed 1701 --steps 1000 \
  --output runs/debug/real-interaction-1701
```

Run separate outputs with `--arm legacy`, `--arm objective`, and
`--arm history_is` for the existing training, revised objective only, and
revised objective plus exchangeable prefix importance sampling. `interaction`
adds the explicit decoder interaction. All arms fit the same deterministic
baseline suite. `legacy` freezes coefficients and rejects nondefault overrides.

The controlled task has identical 2-PASS/2-FAIL visible histograms for every
candidate. A hidden random direction determines failure on signed tests;
task/code nuisance features do not reveal it. Only paired evidence reveals the
direction. Success means this mechanism is learnable, not that code semantics
or real bug diagnosis has been solved.

## Artifacts and interpretation

Every run creates a fresh directory and refuses overwrite. It saves:

- `manifest.json`, exact Python source snapshots, source/cache hashes, resolved
  device/environment metadata;
- scheduled model checkpoints and selected `model.pt`;
- selected deterministic baseline weights, steps and validation histories;
- `result.json`, including failed screens, raw NLLs, pair-order effects,
  gradient contribution norms at the first step and three Monte Carlo repeats;
- checksums of all output files, or `failure.json` if execution fails.

The development bootstrap is descriptive: those same data selected the
checkpoint. It is not a fresh confidence interval for a confirmatory claim.
Changing seeds on the same real split is not an independent population.
Do not use a large shuffle gap to excuse bad aligned NLL. Do not claim particles
are necessary without outperforming the deterministic interaction control.
Feature hashing remains a substantial limitation of these real-data ablations;
a frozen semantic encoder and a new locked test population are subsequent
experiments, not implicit properties of this run.

## Real-data representation and query-memory follow-up

`--eval-particles 64` separates evaluation accuracy from the training budget.
`--shuffle-train-tests` uniformly permutes complete training test/outcome pairs,
changing which four are visible while preserving pairing. Development prefixes
stay fixed. All deterministic baselines now use the same candidate minibatch
schedule as the particle model and report shuffle/pair-preserving controls.
This changes their sampling schedule relative to the earlier debug runner;
source snapshots identify the exact version of every old run.

`prepare_semantic_features.py` freezes pinned CodeBERT and caches only public
train/development fields. The semantic treatment includes contextual test
encoding, input-token pooling, training-only centering, a fixed Gaussian
projection and normalization. It does not isolate pretrained weights from
context representation. Long code/context truncation remains a limitation.
Requires `transformers==4.51.3` and pinned local Hugging Face model files, with
their download revision metadata. The `neural` dependencies alone do not include
this optional encoder.

```bash
PYTHONPATH=src python scripts/prepare_semantic_features.py \
  --cache /absolute/path/to/rich-cache-v3.json \
  --model-dir /absolute/path/to/codebert-base-3b0952f \
  --output runs/debug/codebert-features.npz

PYTHONPATH=src python scripts/run_association_debug.py \
  --source runbugrun --cache /absolute/path/to/rich-cache-v3.json \
  --arm interaction --feature-cache runs/debug/codebert-features.npz \
  --eval-particles 64 --shuffle-train-tests \
  --output runs/debug/semantic-views-1701

PYTHONPATH=src python scripts/run_kernel_association_probe.py \
  --cache /absolute/path/to/rich-cache-v3.json \
  --feature-cache runs/debug/codebert-features.npz \
  --output runs/debug/kernel-probe.json
```

The last command is a separate deterministic control: retain individual visible
pairs, then use each future test to weight their outcomes through cosine
similarity. Mix these weights with uniform history counts and add calibrated
Dirichlet pseudocounts. The grid is selected using training NLL only. Its API
accepts visible labels only; future labels are used solely by the scoring code.
It is neither a particle posterior nor an interpretable bug classifier and
cannot directly justify diagnosis mutual information or latent-conditioned
repair. Report both shuffle gaps and paired source-bootstrap comparisons to the
histogram control. Four corruption seeds are sensitivity checks, not four
independent data replications. See the follow-up results for the actual outcomes.

## Parameter and mechanism debugging (2026-09-17)

Three additional ablations preserve prefix-IS inference and visibility rules:

- `--arm binding`: bind each test feature vector to its one-hot outcome before
  learned averaging. This increases encoder capacity and is explicitly an
  architectural ablation.
- `--arm interaction_only`: keep the original encoder, remove the additive
  diagnosis MLP, and use only bilinear test × g diagnosis logits plus the
  difficulty head. This reduces parameter count.
- `--arm high_gain`: the interaction-only model with tenfold initialized
  interaction-head weights. All other initialized parameter tensors match at a
  shared seed. The weights remain trainable.

```bash
PYTHONPATH=src python scripts/run_association_debug.py \
  --source runbugrun --cache /absolute/path/to/rich-cache-v3.json \
  --feature-cache runs/debug/codebert-features.npz \
  --arm high_gain --seed 1701 --steps 1000 --particles 8 --eval-particles 64 \
  --output runs/debug/high-gain-1701

PYTHONPATH=src python scripts/inspect_diagnostic_state.py \
  --runs runs/debug/high-gain-1701 --particles 256 \
  --seed 71701 --counterfactual-seed 51701 \
  --output runs/debug/high-gain-state.json

PYTHONPATH=src python scripts/audit_diagnostic_sampling.py \
  --runs runs/debug/high-gain-1701 runs/debug/high-gain-1702 \
  --output runs/debug/high-gain-sensitivity
```

A seed-routing bug in the earlier evaluation mixed sampling noise and history
corruption inside what was called `monte_carlo_repeats`. Historical repetitions
are retained as mixed sensitivity checks; original primary metrics/checkpoint
choices are unchanged. New repetitions fix the counterfactual seed and vary
sampling only. The optional low-level `counterfactual_seed` defaults to the old
behavior for compatibility. Audits freeze selected checkpoints, validate both
input fingerprints, and separately vary sampling, corruption and particle
budget. Larger K does not imply nested random particles across every candidate.

See `results/DIAGNOSTIC_PARAMETER_DEBUG_2026-09-17.md` for all positive and negative
trials. The stronger association remains an exploratory development result.

## Checkpoint sampling correction (2026-09-17 follow-up)

The runner now defaults to `--selection-replicates 4`. Each scheduled checkpoint
is scored by averaging predictive probabilities from four independent particle
draws, then computing NLL. History corruption stays fixed across these draws;
each aligned/shuffled comparison uses matched sampling seeds. Selected-model
scoring and the descriptive bootstrap use the same averaged probabilities.
The three subsequent Monte Carlo repetitions each use a disjoint four-draw
ensemble, with history corruption still fixed. This reduces sensitivity to one
favorable particle draw; it does not correct adaptive reuse of development data.

Use `--selection-replicates 1` with the reproduction commands above to reproduce
the earlier selection protocol. Historical reports and checkpoints remain
unchanged. Four independent K=64 self-normalized estimates averaged together
are not equivalent to one K=256 importance estimate. Neither the 0.03 diagnostic
screen nor the 0.02 NLL guard has changed.

Six subsequent real-data runs and the two-seed K8/K32 comparison are reported
in `results/DIAGNOSTIC_SELECTION_FIX_2026-09-17.md`. The evaluator correction
does not itself establish a causal prediction improvement; all new model
comparisons share the corrected estimator.
