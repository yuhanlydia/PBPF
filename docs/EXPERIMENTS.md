# Experiment execution

## Status boundary

All shipped YAML files have `claim_status: preregistered_configuration_only`.
Validating a plan, passing a fake-backend smoke, or completing a pilot does not
constitute evidence of a 7B prediction or repair gain. Only a frozen formal run
whose immutable artifacts pass the preregistered gates may be interpreted.

## Validation and smoke

From a clean checkout:

```bash
python -m pip install -e '.[test]'
pbpf-run doctor
for config in configs/experiments/*.yaml; do pbpf-run plan "$config"; done
PYTHONPATH=src python -m pytest -q
scripts/run_smoke.sh
```

The smoke runs sequential per-observation particle updates, candidate/test-keyed
prediction, four conditioned repairs, fail-closed stage gates, and an immutable
run bundle without a model download. The 7B launchers validate the exact config
they would hand to an authorized orchestration job:

```bash
scripts/run_frozen_7b.sh configs/experiments/frozen_7b_16gb.yaml
scripts/run_repair.sh configs/experiments/repair_7b_24gb.yaml
scripts/run_repair.sh configs/experiments/formal_h200.yaml
```

## Stages and gates

Stage A runs family-held-out finite programs at P=1/4/8/16/32 and reports future
NLL after prefix four, Brier, ECE, posterior KL, randomized 90% HPD coverage, ESS,
and unique ancestors. Stage B freezes the actor and fits belief/adapter modules on
grouped RunBugRun before transfer to CodeARC and EvalPlus. It advances only when
future NLL improves by at least 5% over the strongest matched deterministic arm,
the paired cluster-bootstrap confidence interval is above zero, Brier does not
regress, and shuffled/random controls remove at least 80% of the gain.

Stage C consumes the same immutable G=8 candidate bank in every arm. It always
executes four repair rounds, selects one active slot by predicted remaining-suite
success with generation log-probability as the fixed tie-break, and submits one
program. It advances only at +3 absolute Pass@1 points with a positive paired CI
and retained future-NLL advantage. Stage D freezes the mini-SWE-agent scaffold,
uses Lite for development, then runs Verified exactly once.

Formal online runs use three seeds and 10,000 paired hierarchical bootstrap
replicates at the source-task cluster level with one shared resampled seed vector
per replicate. Secondary comparisons use Holm
correction. Abort scaling above 90% peak memory, 1% infrastructure failures, or a
25% projected cost overrun.

## Hardware limits

The 16GB profile is a smoke profile: NF4/BF16 compute, 2,048 context, microbatch
1, accumulation 32, sequential P=8, last four layers, rank 4. The 24GB profile is
an online pilot: 3,072 context, microbatch 1, accumulation 16, P=8 in chunks of 2,
last eight layers, rank 8. Neither may be marked formal. `h200_formal` is the only
shipped formal configuration and requires full model/data SHAs and an image
digest. Model reports remain separate across Qwen 2.5, Qwen 3, and DeepSeek.
