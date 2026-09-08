# Immutable artifacts and verification

Every formal run directory is create-once. It records the canonical config hash,
data/model/container revisions, task rows, ordered events, numeric arrays, the
execution/token/compute ledgers, gate decisions, and SHA-256 checksums. Gold
sidecars remain outside trainer-readable directories. If a report or bank byte is
changed, readers fail closed on checksum mismatch.

Verify an independently mounted trainer bank and a run report:

```bash
python -c 'from pbpf.bank import TrajectoryBank; print(TrajectoryBank.read_trainer("artifacts/trainer-bank").trainer_content_hash)'
python -c 'from pbpf.artifacts import read_immutable_report; print(read_immutable_report("artifacts/report"))'
```

The run-bundle manifest rejects missing and extra files and checksums the report,
resolved config/model/data/container, final program, trainer bank, keyed
predictions, round events, and full ledgers. It also records canonical config and
trainer-bank content hashes. The execution ledger records the exact
`(task_id, test_id)` sequence. The token
ledger separates visible and generated tokens. The compute ledger records CPU
seconds, wall seconds, and GPU hours. `assert_equal_budgets` compares all three
before cross-arm results are accepted. Reports also include physical KV bytes,
round-wise/final Pass@1, regressions, patch size, executions, and failure counts.

Run bundles also record a hash reference to evaluator truth, the immutable
prediction-report inputs, runtime model identity, and the explicit stage-gate
status. Stage advancement is disabled until a registered multi-seed/task
aggregator can persist comparison reports and bootstrap inputs; caller-supplied
confidence bounds are rejected.

Do not edit or reuse a run directory. A rerun receives a new directory and new
manifest. Preserve the failed artifact when verification fails so the mismatch is
auditable. Smoke artifacts must retain the `smoke-only` or
`preregistered_configuration_only` status and cannot be promoted to evidence.
