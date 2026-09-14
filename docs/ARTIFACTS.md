# Immutable ICLR artifacts

All commands share a resolved scientific config and run ID. Runs contain
create-once `resolved.yaml`, its SHA-256 and `fingerprint.json`. Recursive registry
resolution binds actual prompts, finite generator specification, scientific
procedures, code/factory bytes, model/data snapshots, container and seeds. Slurm
IDs, paths, prices and wall-clock telemetry are operational, not scientific.

Formal execution has two typed phases. `development_inputs` contains no invented
future learned hashes. Training publishes actual checkpoint, selection and scalar
temperature artifacts; verified hashes create `confirmatory-fingerprint.json`
before sealed prediction/repair/hidden/replication. Changed learned files or a
development-only identity are rejected. The directory retains the stable
development ID for whole-command resume; confirmatory leaves bind the stronger ID.

Each shard writes JSONL, a create-once identity, checksum completion marker and
local lock using temporary write/fsync/atomic rename. Stable SHA-256 work keys use
full-hash modulo shard count. Merge checks every shard identity, inventory,
dependencies, envelope and checksum. No shared NFS SQLite is used. Valid leaves
are reused; corrupt/partial leaves rerun only under the same immutable identity.
Mismatches are never merged. Preserve failed artifacts for diagnosis.

```bash
pbpf-iclr run --config configs/experiments/iclr_pbpf.yaml --profile local_cpu --resume
pbpf-iclr aggregate --config configs/experiments/iclr_pbpf.yaml --profile local_cpu
pbpf-iclr verify --config configs/experiments/iclr_pbpf.yaml --profile local_cpu
pbpf-iclr package --config configs/experiments/iclr_pbpf.yaml --profile local_cpu
```

Use `slurm_h200x16` only with the audited provisioning in EXPERIMENTS.md. Verify is
read-only, not a repair operation. Aggregation verifies first and rejects forced
gate-override output; formal eligibility additionally requires passing B, C and
locked replication gates. Package verifies first and writes deterministic
`verified-run.zip` public metadata and stage records, not a paper ZIP or a
self-contained model/checkpoint archive. Retain referenced learned binaries in
the immutable deployment for full re-verification. Private truth/keys/site
settings are excluded.

Smoke always remains `smoke-only-no-claim`, never main-table evidence. Force after
a failed B is sticky in every downstream artifact; force on a passing B does not
relabel it. Append-only ledgers distinguish actor requests, full and partial
decodes, native input/output tokens, belief forwards, visible/hidden verifier
suites/cases, CPU/GPU/wall time, memory and disk. Retries remain incurred work.

Legacy `pbpf-run` banks/reports and `evalplus_full164_v1` use diagnostic schemas
and cannot populate ICLR main tables.
