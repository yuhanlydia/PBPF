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
