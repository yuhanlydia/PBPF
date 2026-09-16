# A-PBPF artifact contract

## Identity and immutability

Every A-PBPF run is bound to the independent `apbpf-iclr-v1` fingerprint. Its
resolved configuration, code/config identity, source-problem split, hard-bank
inventory, model/data revisions, seeds, and gate parameters are immutable.
`--resume` accepts only work with the same fingerprint; mismatched or partial
work is retained for diagnosis and never merged into a new run.

The A-PBPF contract deliberately does not amend frozen `pbpf-iclr` artifacts or
legacy RunBugRun/EvalPlus diagnostics. In particular, the historical negative
repair pilot remains a separate immutable report.

## Required run inventory

This table describes the access-controlled internal evaluator inventory. The
public paper bundle contains only aggregate metrics and cryptographic identities
for hidden evidence, never the hidden records themselves.

| Artifact | Required contents |
|---|---|
| resolved configuration | fully resolved YAML/JSON and SHA-256 config hash |
| experiment identity | `apbpf-iclr-v1` fingerprint, code/model/data identities, seed and profile label |
| hard-bank materialization | source-component IDs, ordered candidate IDs, visible/hidden inventories, saturation/headroom audit, checksums |
| association evaluation | aligned, outcome-shuffled, joint-reversal, presentation-permutation, masking, wrong-candidate and strata metrics with clustered intervals |
| acquisition evaluation | fixed/random/active matched budgets, unexecuted-test decisions, component MI/EIG values, no-repeat evidence |
| selection evaluation | candidate-bank identity, cross-fitted selector comparator, selected Pass@1 and clustered interval |
| repair evaluation | arm name, parent/candidate lineage, generated source hash, visible/future outcomes, token/request accounting, diagnosis-only conditioning contract, conditioning trace |
| gate decision | input artifact hashes, pass/fail result, threshold, failure reason, and decision SHA-256 |
| stage completion | immutable work records, stage summary, checksum, and actionable failure metadata |

The `paper_tables` worker must create a separate public export and verify that
private evaluator truth, credentials, and hidden outputs are absent. In
particular, credentials are forbidden in recorded worker argv and must be
injected through the operator's secret mechanism.

## Repair trace schema

The RunBugRun helper writes `pbpf-rbr-repair-gate-v2` reports. Each task/arm row
contains a `conditioning_trace` with the following fields:

For a factored A-PBPF belief, the associated
`apbpf-rbr-diagnosis-prefix-v4` projector records `conditioning_source` as
`diagnosis_component`, its exact conditioning dimension, and the excluded
difficulty dimension. Older joint-latent projector checkpoints are rejected for
factored repair rather than silently reintroducing the difficulty shortcut.
Partial v4 projector checkpoints carry optimizer, NumPy RNG, component-sampling
RNG, target-step, full data/fixed-code/config/implementation identity, and
`complete=false` state. They are atomically replaced and resume only under the
same identity; evaluation accepts only a `complete=true` v4 checkpoint at the
requested step count and matching data payload.

| Field | Contract |
|---|---|
| `posterior_fingerprint` | SHA-256 over the exact particle values and log weights used for the continuation |
| `component_indices` | one selected component for `sample_once`; the pre-recorded per-token schedule only for `token_remix_fault` |
| `component_fingerprint` | hash binding posterior, component index/schedule, latent value, and conditioning mode |
| `conditioning_fingerprint` | hash binding task, arm, continuation, posterior, component fingerprint, and immutability flag |
| `immutable_component` | `true` only when one conditioned latent is held for the complete continuation |
| `component_mode` | named control semantics, never an inferred label |

Formal repair arms record the same invariant as a
`pbpf-repair-continuation-v1` trace: parent version, particle index, latent/soft
prefix, and a `continuation_fingerprint` binding decoding controls. A trace does
not prove a repair succeeded; it only makes the conditioning claim auditable.

## Gate lineage

When both upstream association and selection decisions pass, repair output can remain
prospective until all later gates pass. When it fails, the normal launch blocks
repair. `--continue-exploratory` preserves the failed gate SHA-256 under
`upstream_gate`, sets the descendant claim status to exploratory/nonconfirmatory,
and permanently disqualifies that lineage from confirmatory aggregation.

`verify` must reject a claim-bearing downstream artifact if its upstream gate is
missing, failed, or fingerprint-mismatched. It may retain those artifacts for
debugging and historical audit.
