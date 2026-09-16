# A-PBPF stage-worker contract

This is **prospective infrastructure**, not an experiment result. The repository
does not bundle a complete set of scientifically validated real-stage adapters.
The site template intentionally leaves every command unprovisioned. Existing
research scripts need explicit adapters to satisfy this contract; do not simply
point a stage at an unrelated script and treat its exit status as evidence.

The independent schema is `apbpf-iclr-v1`; legacy `pbpf-iclr` is unchanged.
`local_24gb`, `slurm_h200x16`, and `real` use only real commands. Profile names
describe operator resources; the runner itself does not allocate GPUs or submit
Slurm jobs. Slurm adapters must wait for completion and return the actual worker
result. `local_smoke` is the sole fake backend and is always
`smoke-only-no-claim`.

## Provisioning and invocation

Copy `configs/apbpf/site_local_24gb.example.yaml` to an operator-owned location.
Supply three existing directories (`dataset_root`, `model_cache`,
`artifact_cache`), the working directory, every stage command, and an immutable
SHA-256 worker identity for every command. The identity may be the adapter-file
digest, immutable container digest, or a canonical bundle digest; it enters the
run fingerprint, so changing worker code requires a new run identity. When a
local adapter is declared under `worker_files`, `doctor` recomputes its SHA-256
and refuses a mismatch. Immutable container/bundle identities remain an operator
attestation and must be independently audited. Command
values are argv arrays, never shell snippets. Never put access tokens or other
credentials in argv because commands are recorded in run artifacts; inject them
through the operator's secret manager or inherited environment instead. The exact tokens `{request}` and
`{result}` are replaced by absolute paths. No other interpolation occurs.
Alternatively workers may read the environment variables below.

```bash
pbpf-apbpf doctor --profile local_24gb --site /absolute/apbpf-site.yaml
bash scripts/run_apbpf_iclr.sh --profile local_24gb --site /absolute/apbpf-site.yaml --resume
pbpf-apbpf report --profile local_24gb --site /absolute/apbpf-site.yaml
pbpf-apbpf verify --profile local_24gb --site /absolute/apbpf-site.yaml
pbpf-apbpf rerun-stage --profile local_24gb --site /absolute/apbpf-site.yaml --stage association
```

Set `PBPF_APBPF_PYTHON` to an installed project Python if `.venv/bin/python`
is absent. `PBPF_APBPF_SITE` supplies the site path when `--site` is omitted.
All commands default to `configs/experiments/apbpf_iclr2027.yaml` and
`--output-root runs/apbpf`. `doctor` is read-only: it lists every missing command,
missing directory, and unavailable executable and exits 2 when unprovisioned.
Readiness does not validate adapter scientific correctness or start experiments.

## Worker inputs and outputs

The `hard_bank_lock` stage receives only public materialization and must finish
before `execution_cache` mounts or produces evaluator-hidden outcomes. The
later `hard_bank` stage audits that exact locked source/candidate inventory.

Every worker receives:

- `APBPF_REQUEST`: absolute `request.json`, containing the resolved config,
  stage/fingerprint, declared dependency checksums and result-file paths,
  confirmatory/claim labels, failed-gate lineage, and output directory.
- `APBPF_RESULT`: absolute `worker-result.json` to create before successful exit.
- `APBPF_OUTPUTS`: an empty attempt-owned directory for evidence artifacts.

Read declared inputs; do not fetch another run's outputs, mutate request or
identity files, or expose evaluator-only hidden labels to generator code. The
runner verifies checksums and provenance, but workers remain responsible for
the scientific computation and public/hidden information firewall. Bootstrap
units must be source-problem components with 10,000 draws, not candidates.

Return a JSON object of this shape (the strings below are explanatory):

```json
{
  "schema": "apbpf-stage-result-v1",
  "stage": "the exact request stage",
  "fingerprint": "the exact request fingerprint",
  "dependencies": {"upstream_stage": "exact digest copied from request.dependencies"},
  "summary": {"description": "actual computation and population"},
  "artifacts": [{"path": "outputs/evidence.json", "sha256": "actual SHA-256"}]
}
```

Copy the complete `request.dependencies` mapping verbatim; its values identify
the direct dependencies' immutable completion records and are not a checksum of
the current `request.json`. Do not write `runner_lineage`: after the worker
returns, the runner adds that reserved field and transitively binds every
ancestor's exact completion checksum. Confirmatory repair validates this chain,
not merely equal configuration fingerprints. The focused repair helper also
requires real, unfailed confirmatory gate attempts and binds the supplied belief
checkpoint bytes to the inherited `train_belief` evidence artifact.

Every real stage needs at least one evidence artifact below `outputs/`, including
the paper-tables stage. Artifact paths are relative to the attempt directory.
Do not write symlink evidence or refer to mutable external files. Copy durable
checkpoint/data evidence into the output inventory as appropriate. Workers may
read model/data caches from the explicitly provisioned paths in the request.
The runner captures `stdout.log` and `stderr.log` and enforces the site timeout.
Nonzero exit, timeout, missing result, invalid provenance, or checksum mismatch
is an execution failure, never a reason to fall back to smoke output.

## Gate metrics

Gate results additionally contain
`"gate": {"passed": true, "reason": "...", "metrics": {...}}`.
`passed` must agree with a decision recomputed by the runner from the locked
thresholds. All numerical metrics must be finite. Required keys:

| Stage | Metrics |
| --- | --- |
| `hard_bank_gate` | v2 `lock_schema`/`audit_schema`; lock/candidate/source SHA-256s, with the lock SHA matching an artifact of the exact `hard_bank_lock` dependency; `lock_precedes_hidden_execution`, `candidate_inventory_bound`, `source_inventory_bound`, `pilot_primary_source_disjoint`; group counts and visible/oracle Pass@1 |
| `baseline_fairness_gate` | `baselines`, keyed by all four names in `gates.yaml`, each with `nll_advantage` and `clustered_lower_bound` |
| `association_gate` | `population: full_locked_population`; `domains` rows with `gap_nats_per_test`, `clustered_lower_bound` |
| `pair_invariance_gate` | `domains` rows with `shuffle_gap`, `joint_reversal_degradation`, `presentation_permutation_degradation` |
| `oracle_headroom_gate` | `domains` rows with `nll_advantage_over_fixed`, `nll_advantage_over_random` |
| `active_testing_gate` | `domains` rows with `matched_hidden_quality`, `test_reduction`, `fixed_budget`, `all_policies_execute_exact_budget`, `nll_advantage_at_four_tests` |
| `selection_gate` | `comparator: strongest_cross_fitted_deterministic`; `domains` rows with `absolute_selected_pass1_advantage`, `clustered_lower_bound` |
| `replication_gate` | `cells`: rows with `domain`, `family`, `association_gap`, `selection_advantage` |

`domains` must contain exactly the configured confirmatory domain names
(`runbugrun`, `codearc_replay`); EvalPlus is audit-only. Replication must cover
the full confirmatory-domain/model-family cross product without duplicates.
Report all metrics, including failing values. Baseline advantages and selection
advantages use the positive-is-better direction; association is shuffled NLL
minus aligned NLL. Supporting evidence must preserve full-population results,
ambiguity strata, confidence interval construction, model/data locks, seeds,
and controls; the dispatcher does not recreate them from scalar summaries.

## Immutability, gate stops, and recovery

Resolved scientific configs, source-file hashes, profile, site commands/paths,
and declared worker revisions determine the fingerprint. A change creates a new run directory; it cannot be
resumed into another identity. Runs are serialized by a process lock.

Each `stages/STAGE/attempt-NNNNNN/` retains `identity.json`, `request.json`,
`command.json`, `result.json`, `work.json`, `summary.json`, and a checksum inventory
in `complete.json`. Real workers add their logs, raw result and evidence files.
Publication is create-once. Completed attempts are verified before reuse;
corruption is rejected rather than overwritten. Interrupted/failed attempts are
retained and `--resume` appends a fresh attempt. `rerun-stage` retains all prior
attempts and reruns the target and complete later stage suffix, because a changed
gate can alter downstream global claim labels, not just direct dependency edges.

Execution failures include immutable `failure.json` with exact argv (when
provisioned), return code, error, a same-identity `rerun-stage` command, and a
separate `run` command for the new fingerprint after code/config/worker changes. Missing
commands are explicitly null, never fabricated as executable. Failed scientific
decisions add `gate-failure.json`; a normal run exits 20 before downstream work.
`--continue-exploratory` permits continuation but propagates failed-gate checksums
and a nonconfirmatory label to every subsequent artifact. It cannot restore main
table eligibility. Repair occurs only after association and selection gates,
unless this explicit exploratory override is used.

`report` inspects completed, failed, interrupted, stale and pending stages read-only.
`verify` additionally requires the complete DAG. Neither command creates or
repairs artifacts. `main_table_eligible` is true only for a complete real DAG
with all gates passing and no exploratory lineage; it is an infrastructure
eligibility check, not an independent audit of the worker's scientific validity.

## Explicit offline smoke path

```bash
pbpf-apbpf run --profile local_smoke --output-root runs/apbpf-smoke
pbpf-apbpf run --profile local_smoke --output-root runs/apbpf-smoke --resume
pbpf-apbpf verify --profile local_smoke --output-root runs/apbpf-smoke
```

Do not supply a site file (or `PBPF_APBPF_SITE`) for smoke. Every smoke stage
contains `scientific_measurements: false`; successful verification does not make
it confirmatory. `--fail-smoke-gate association_gate` deliberately fails that
synthetic gate; the run's immutable smoke control is reused on resume. Use a
different output root for a different smoke-control scenario. These commands
are examples, not a record of tests performed in this implementation session.
