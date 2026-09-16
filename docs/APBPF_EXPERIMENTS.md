# A-PBPF operator workflow

## Scope and status

The A-PBPF path tests whether execution feedback contains recoverable
test--outcome association information. It is prospective: no command or
artifact in this repository establishes a positive A-PBPF result. The frozen
legacy `pbpf-iclr` workflow remains separate.

The historical frozen-Qwen repair pilot remains negative evidence: it solved
`0/8` PBPF-conditioned candidates versus `1/8` with no latent, and lowered the
future-test pass fraction from `35.4%` to `22.9%`. Do not relabel, aggregate, or
replace that result with A-PBPF output.

## Exact launcher after worker provisioning

After provisioning all real stage adapters according to `APBPF_WORKERS.md`, the
complete resumable A-PBPF launcher from the repository root is exactly:

```bash
bash scripts/run_apbpf_iclr.sh --config configs/experiments/apbpf_iclr2027.yaml --profile local_24gb --site /path/to/apbpf-site.yaml --output-root runs/apbpf --resume
```

The canonical `local_24gb` profile is real and requires the audited site overlay;
replace only the example path with the operator-provisioned file. The formal
`slurm_h200x16` profile uses the same arguments with its audited overlay. The
explicit fake `local_smoke` profile is smoke-only and always produces a no-claim
label. Never substitute a synthetic backend, guessed snapshots, or local hidden
labels for a real run.

The underlying commands are available for diagnosis only; the launcher above is
the canonical run/resume command:

```bash
pbpf-apbpf doctor --config configs/experiments/apbpf_iclr2027.yaml --profile local_24gb --site /path/to/apbpf-site.yaml
pbpf-apbpf verify --config configs/experiments/apbpf_iclr2027.yaml --profile local_24gb --site /path/to/apbpf-site.yaml --output-root runs/apbpf
pbpf-apbpf report --config configs/experiments/apbpf_iclr2027.yaml --profile local_24gb --site /path/to/apbpf-site.yaml --output-root runs/apbpf
pbpf-apbpf rerun-stage --config configs/experiments/apbpf_iclr2027.yaml --profile local_24gb --site /path/to/apbpf-site.yaml --output-root runs/apbpf --stage NAME
```

## Gates and repair

Repair is downstream evidence, not the primary A-PBPF claim. The launcher must
stop confirmatory repair when association or selection fails. To preserve a
diagnostic run after an upstream failure, the operator may explicitly request:

```bash
bash scripts/run_apbpf_iclr.sh --config configs/experiments/apbpf_iclr2027.yaml --profile PROFILE --site /path/to/apbpf-site.yaml --output-root runs/apbpf --resume --continue-exploratory
```

This is not a gate override. The failed gate decision and SHA-256 are retained;
all descendants are marked `nonconfirmatory`, and verify/report must reject them
as confirmatory evidence. The focused RunBugRun repair helper follows the same
rule when given an immutable upstream decision:

```bash
python scripts/run_rbr_repair_gate.py \
  --belief-checkpoint RUN/stages/train_belief/attempt-NNNNNN/outputs/CHECKPOINT.pt \
  --upstream-gate RUN/stages/association_gate/attempt-NNNNNN/result.json \
  --upstream-gate RUN/stages/selection_gate/attempt-NNNNNN/result.json
```

A candidate confirmatory helper run requires intact runner attempts (including
their sibling `identity.json`, `request.json`, and `complete.json`). The
selection attempt's runner-authored ancestor checksum must identify the exact
association attempt supplied on the command line, so rerun attempts cannot be
mixed. Both attempts must come from an unfailed real-backend confirmatory run.
The `--belief-checkpoint` bytes must also match an evidence artifact of the exact
`train_belief` completion inherited by the selection gate. A failed decision
raises an error before projector training or actor loading. An
exploratory repair helper run must supply `--continue-exploratory`, which writes
the failed gate hash and nonconfirmatory claim status into its report.

For active testing, `NeuralDiagnosticAdapter` converts the factored belief into
explicit diagnosis components and interventionally crosses them with a
mean-field difficulty nuisance marginal. Both marginals receive a parallel
Bayes update after every public outcome. This is an explicit approximation, not
the original correlated SMC posterior; joint-particle mutual information is a
required ablation. `ActiveTestPolicy.acquire_trace` recomputes diagnostic mutual
information before each test and returns the final posterior for downstream
scoring. The adapter defaults to the RunBugRun `OUTCOMES` string registry;
integer-coded datasets must pass their own five-label registry. Raw latent or
particle-weight entropy is never the acquisition score.

## Conditioning controls

Each generated repair continuation writes a `conditioning_trace` bound to the
posterior particles, normalized weights, selected component(s), and the
continuation identity. The confirmatory conditioning implementation is
`sample_once`: it selects one posterior component before actor decoding and
holds the resulting prefix unchanged for the entire completion. For factored
A-PBPF checkpoints, the projector receives only that particle's diagnosis
slice; the test-invariant difficulty slice is excluded from actor conditioning.

| Report arm | Meaning | Confirmatory interpretation |
|---|---|---|
| `no_latent` | no soft-prefix tokens | baseline only |
| `mean` | posterior-mean prefix | collapse control |
| `map` | maximum-weight component | collapse control |
| `random` | matched-norm random latent | negative control |
| `sample_once` | one posterior component for one full continuation | required coherent method |
| `token_remix_fault` | reselects a component at each generated token | intentionally incorrect ablation; never a method claim |

`token_remix_fault` recomputes the next-token context under an explicit,
pre-recorded component schedule. Its trace has multiple component indices and
`immutable_component=false`; it cannot be silently confused with sample-once.

## Operator checks

Before treating any output as evidence, inspect the A-PBPF report for:

- the `apbpf-iclr-v1` fingerprint and the resolved configuration hash;
- source-problem split/bootstrap identities and the locked hard-bank inventory;
- association, pair-invariance, active-testing, and selection gate decisions;
- each repair trace's `conditioning_fingerprint`, component fingerprint, and
  immutable-component flag;
- both association- and selection-gate hashes for any candidate confirmatory repair;
- explicit `smoke-only-no-claim`, `prospective`, or `nonconfirmatory` labels
  where applicable.

Use `pbpf-apbpf report` to diagnose an incomplete or failed run, then
`pbpf-apbpf verify` after the complete DAG finishes. Both are read-only:
verification validates identity, stage checksums, gate lineage, and artifact
inventory; neither command repairs, regenerates, or reinterprets failed evidence.
