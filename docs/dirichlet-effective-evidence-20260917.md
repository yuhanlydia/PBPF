# Effective-evidence Dirichlet diagnostic

Status: completed exploratory prediction comparison, **not** corrected-protocol
validation or a real repair/rollout comparison.

## Method and selection

For normalized similarity weights p, use evidence weights
`p / sum(p**2)`. Total mass is the heuristic effective sample size
`1 / sum(p**2)`, between one and four for four observations. This discounts
concentration, not dependence between executions, and is not an exact Bayesian
posterior. Uniform weights recover the ordinary count model.

All arms use four visible outcomes and public test inputs. The existing alpha
grid and strength grid are unchanged; each arm is selected on inner validation
from original training sources. Original development is the assessment set;
original test is excluded. This development set has been examined repeatedly,
so results and intervals are exploratory, not confirmatory.

| Arm | Validation-selected alpha / strength | Assessment NLL | Accuracy |
| --- | --- | --- | --- |
| Ordinary Dirichlet | 0.1 / 0 | 0.458301 | 82.6599% |
| Fixed-mass weighted | 0.1 / 16 | 0.440499 | 84.0067% |
| Effective-mass weighted | 0.01 / 16 | 0.426433 | 84.0067% |

Effective versus fixed-mass NLL gain: 0.014066; descriptive 95% interval
[-0.028929, 0.049744], 10,000 source-cluster bootstrap draws, 32 sources.
The effective arm has worse validation NLL (0.391283 versus 0.390021).
It must not replace the fixed-mass arm solely because assessment NLL is better.
Both arms predict exactly the same classes on all 594 assessment examples.
With a symmetric prior, positive rescaling of counts does not change argmax
at a fixed strength. This experiment tests probabilities, not improved code
repair or candidate selection. Alpha also differs, so the aggregate comparison
does not isolate evidence scaling at a fixed prior.

Reproduction (output directory must not already exist):

```bash
python scripts/run_dirichlet_variants.py \
  --cache runs/apbpf-real-diagnostic/rbr-cache.json \
  --output runs/dirichlet-effective-evidence-20260917-v2 \
  --effective-evidence
```

Input cache SHA256:
`a512f2d740933380e134de0b8f8155b13edd5ddad250d108f19b937c90a2bda4`.
Local output directory contains plan, selection, predictions, report, and
completion hashes. Cache uses legacy literal stdin; it is not corrected by
this statistical comparison. Artifact/source hashes and metrics were checked
independently after execution. No cached labels were edited to simulate a rerun.
The v2 rerun clarifies per-arm normalization metadata; the method and metrics
are unchanged. The shuffled control in the report pertains to fixed-mass
weighting only, not an association test of the effective-mass arm.

## Remaining execution requirements

On this host, both a direct namespace probe and Docker image layer registration
fail with `unshare: operation not permitted`. Docker daemon responds, but that
does not establish that containers can run. No untrusted candidate was run
without isolation.

The remote branch `origin/longgoal/apbpf-20260916` at `f3ec566` contains
`results/local_20260916_apbpf/rbr_stdin_cache_rebuild_status.json`, reporting a
corrected cache at another host's
`/data/cwj/PBPF/local/longgoal/rbr_expanded_cache_stdin_v4.json`, SHA256
`ce9acd40ae2ee178532c777e318595b08a8965662c3355f9afc9666fe6e1f2a7`.
That file is absent here. The report alone cannot be used as prediction data.

To continue corrected prediction evaluation, transfer that cache with its
provenance or provide a host supporting isolated execution. Validate the cache
schema and source partitions before adapting the runner; its population differs
from the old cache, so absolute scores are not directly comparable. Evaluate
all baselines again under the same corrected protocol.

Real inheritance and allocation experiments additionally require actual
parent-child repair code, ordered visible execution results, generation/token
budgets, and either complete action banks or an isolated online executor.
Static candidate test records alone are insufficient to measure repair gains.
