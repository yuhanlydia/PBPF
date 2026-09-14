# PBPF real-model diagnostics (2026-09-14)

These runs exercise the legacy diagnostic interface with the pinned
`Qwen/Qwen2.5-Coder-7B-Instruct` revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242` in NF4 on one RTX 4090. They are
explicitly **diagnostic-no-PBPF-claim** results. The current runner uses a
heuristic predictor/scorer, random one-token soft prompts, and flat particle
likelihoods. It therefore does not learn a PBPF belief and has no matched arm
capable of attributing repair success to PBPF.

| Run | Tasks with any passing initial candidate | Final pass@1 | Rescued from no passing initial candidate |
|---|---:|---:|---:|
| frozen 16GB diagnostic | 45/164 | 83/164 (50.6%) | 38 |
| repair 24GB diagnostic | 46/164 | 85/164 (51.8%) | 39 |

The paired difference is +1.22 percentage points for the repair configuration,
with a task-bootstrap 95% interval of [-8.54, +10.98] points and exact McNemar
`p=0.902`. This is noise, not an advantage. All 328 task bundles passed manifest,
file-inventory, content-hash, and trainer-bank verification.

The separately pinned RunBugRun v0.0.1 task 6581 pilot finished 0/1 after four
rounds. Its ESS remained 8 throughout, confirming that its flat posterior did
not update. See `runbugrun_pilot_6581_summary.json` for its source and artifact
hashes.

The full formal S0-S4 run remains unavailable because the repository does not
ship a production `FormalFactory`, an operator site inventory, immutable formal
dataset snapshots, H200 Slurm resources, or independent evaluator authority.
The formal launcher fails closed before starting a model. The complete local CPU
DAG did pass all 15 synthetic stages; it remains `smoke-only-no-claim`.

Machine-readable aggregate metrics are in `evalplus_diagnostic_metrics.json`.
The two runner-produced task summaries are retained beside this note. Raw task
bundles remain local because they contain thousands of generated program files.
