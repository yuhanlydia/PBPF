# Cross-model data audit

Inspected remote: `origin/longgoal/apbpf-20260916`, commit `f3ec566`.
Inspected local locations: `/root/PBPF`, `/root/pbpf_external`, `/dev/shm`.
No new model-transfer evaluation has run. Local predictor code and model
weights are not sufficient without scored candidate execution records.

| Dataset | Generator | Evidence elsewhere | Local scored cache |
| --- | --- | --- | --- |
| CodeARC | Qwen | Completed replication report; cache SHA256 `ceac963cacb3e8f6cb38992233ea8c99503ad339f190d22499e5117afb87533b` | Absent |
| CodeARC | DeepSeek | Full cache proof; SHA256 `fbbe930dfa26b7463210d9f21939b6f4f1c5a2b6fea01349f07e53f5bbffce99` | Absent |
| RunBugRun | Qwen | Full cache proof; SHA256 `4af78c8a7856ae4996c3643bc803b849d9c976f06b518b14cf8998045ae3dc39` | Absent |
| RunBugRun | DeepSeek | Generation handoff report with pending banks; does not attest current completion | Absent |

Reports under `results/local_20260916_apbpf/` on that remote branch:

- `codearc_qwen_completed_replication_cell.json`
- `codearc_deepseek_full_cache_proof.json`
- `rbr_qwen_full_cache_proof.json`
- `rbr_deepseek_actual_handoff.json`
- `generated_prediction_launches.json`

The CodeARC DeepSeek proof refers to source banks under
`/data/cwj/PBPF/local/longgoal/codearc-deepseek-full-evaluation-v2/`.
Those paths do not exist on the current host. The full caches and their
provenance/source manifests must be transferred, not only aggregate reports.

Start with the CodeARC matched pair if both caches can be transferred. Check
public input representation, outcome taxonomy, original population roles,
and source overlap before sharing the RunBugRun predictor adapter. The reports
include previously evaluated primary populations, so do not call them newly
untouched test sets without a provenance audit.

The old local cache has no Qwen/DeepSeek generator binding. Relabeling it as
two models, or splitting it arbitrarily into two model populations, would not
test cross-model generalization.

## GitHub distribution check

The repository Releases API returned no releases, and the Actions artifacts
API returned `total_count: 0` when checked in this session. The newly fetched
`origin/debug/association-inference` branch at `7a534a0` contains additional
prediction/planning code but no matching full execution cache found in its
file inventory. These checks did not locate a downloadable scored cache;
they do not establish that the files are absent from the other machine.
