# APBPF RunBugRun diagnostic — 2026-09-16

**Negative real-data diagnostic; not a completed confirmatory APBPF workflow.**
All three seeds finished 1000 training steps and strong-baseline evaluation on
real executed candidate data, with 10000 source-cluster bootstrap draws. The overall gate
failed for every seed. Do not substitute these reports for real stage-worker
evidence or interpret passing software tests as scientific success.

Run code: `f575be349b8f86db4c1eac5781b5d8c6df860237`; Python 3.11.16,
Torch 2.14.0+cpu. The later Unicode JSONL parsing fix does not change these saved
results. The dataset is the helper's checksum-pinned **legacy RunBugRun v0.0.1**,
not the newer v2 SQLite source referenced by the formal repository configuration.

The shared rich-cache-v3 SHA-256 is
`a512f2d740933380e134de0b8f8155b13edd5ddad250d108f19b937c90a2bda4`.
It contains train 447 candidates/159 problems, development 99/32, test 117/94.
Twelve fixed-program validity failures were excluded by the existing helper.
All seeds reuse the partition/cache prepared with seed 1701; later seeds change
training randomness, not the source split. Expected outputs were explicitly
declared public. Full-population evaluation retains non-ambiguous cases.

| Seed | Aligned NLL | Shuffled-minus-aligned NLL | 95% source-cluster CI | Overall gate |
| --- | ---: | ---: | --- | --- |
| 1701 | 0.5116702 | 0.0069987 | [-0.0441086, 0.0472550] | fail |
| 1702 | 0.5421220 | -0.0025546 | [-0.0205917, 0.0166556] | fail |
| 1703 | 0.5094656 | 0.0116072 | [-0.0500861, 0.0612395] | fail |

The association requirement is gap >=0.03 with positive lower confidence bound.
All three also fail baseline fairness; tuned Dirichlet NLL is 0.3731224 for all
seeds. Pair-invariance fails for seeds 1701/1702 and passes for 1703. Each report
contains all four strong-baseline scores, controls, strata, and gate decisions.
No failure was relabeled as success, threshold lowered, or held-out result used
to select a new checkpoint. Checkpoints were selected on development NLL.

## Artifacts and reproduction

- `seed-*.json`: original reports; `seed-*.population.json`: original inventories.
- `seed-*.checksums.json`: original hashes of reports, inventories and checkpoints.
- `checkpoints.tar.gz`: the three original CPU checkpoint files; extract beside
  the reports in a new empty directory to check the original manifests.
- `development-particle-diagnostic.json`: exploratory development-only comparison
  using seeds 1701/1702, particles 8/32/128 and three inference seeds. More particles
  reduced sampling error, but did not establish association. This is not a new
  confirmatory result or a change to the eight-particle protocol.
- `SHA256SUMS`: checksums for this published package (excluding itself).

Original invocation, from the repository root with the CPU environment installed:

```bash
for seed in 1701 1702 1703; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -u scripts/run_rbr_prediction_gate.py \
      --apbpf --expected-is-public --seed "$seed" --workers 8 \
      --cache runs/apbpf-real-diagnostic/rbr-cache.json \
      --output "runs/apbpf-real-diagnostic/seed-$seed.json"
done
```

Preserve seed order and common cache for source-partition reproducibility. Exact
execution outcomes may depend on host/runtime limits. Base model weights and raw
downloaded source data are not included here. The checkpoint archive is generated
locally, not a third-party pickle; load it only in a trusted environment.

The missing formal work includes source/candidate locks before hidden execution,
the prescribed hard-bank population, both confirmatory domains/model families,
and validated real stage adapters. The full runner remains unprovisioned.
