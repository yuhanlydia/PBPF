> Final decision: the user re-authorized the complete data/run Release backup.
> Experiment processes are stopped; see release STOP_RECEIPT.json.
> The final stopped-state overlay supersedes the initial live snapshot.

# EESD GPU migration and continuation — 2026-09-21

## Snapshot and scope

Read-only audit at approximately **2026-09-21 06:12 UTC**; the release inventory,
with its own capture timestamp and checksums, takes precedence over these counts.

- Joint generation status: **16 complete, 2 running, 30 pending, 0 failed**,
  out of 48 dataset/model/seed cells. This excludes the separate training bank job.
- Direct execution tree contains **12 mechanism report.json files**. These are
  individual cell artifacts, not a completed four-domain inference family.
- Qwen RunBugRun round-1 original train bank: **321 candidates, sealed**.
  Development bank: **400 planned, not yet sealed** at this audit.
- Training pipeline is waiting for those banks. **No SFT training-report exists**.
- Four models × four mechanism domains × three seeds = 48 cells. Original
  RunBugRun/CodeARC and supplementary APPS-Replay/CodeContests-Replay each retain
  their own statistical families. Expansion does not add SFT training domains.

The user requested shutdown after verified upload. The release therefore has an
initial live snapshot followed by a final stopped-state overlay and STOP_RECEIPT.json.
Use the final receipt for shutdown evidence. Do not start a second writer against
a live shared output tree.

## GitHub delivery layout

Repository: https://github.com/yuhanlydia/PBPF

Restore the exact commit accompanying the release, not whichever branch HEAD is
newest. The release description records its commit; RESTORE.md and the manifests list
all assets. Publication and final shutdown are verified by the release publisher.

Keep source/docs in Git; put large data, sealed banks, caches, reports and setup
receipts in release assets. Do not put model weights, Hugging Face credentials,
Git credentials, `.venv`, or installation caches in Git. Download model weights
again at the **exact configured revisions**. Tokenizer assets must also match.

Required artifact content, with paths relative to `/root/PBPF`:

| Tree | Purpose |
|---|---|
| `runs/eesd-setup/` | frozen manifests/config, execution/statistical locks, provenance, operational scripts, dependency snapshot |
| `runs/eesd-data/` | materialized public/evaluator views, Replay admission/audits and raw provenance needed by verification |
| `runs/eesd-20260920/` | original and Replay candidate banks |
| `runs/eesd-direct-20260921/` | direct caches/reports, training original banks and any subsequently completed corrections/training |

Preserve complete and partial artifacts distinctly. A release inventory should
record per-file SHA-256 and classify sealed versus incomplete directories. A
snapshot of a directory being written is **not** a valid completed bank merely
because some files copied successfully. Frozen original sandbox failures remain
historical evidence; direct results must keep their direct-profile receipts.

## Stage and verify before restoration

Template below requires the release publisher's actual tag/commit/archive names.
It intentionally does not launch experiments.

```bash
export EESD_RELEASE_TAG='eesd-checkpoint-20260921'
export EESD_STAGE=/root/eesd-migration-stage
mkdir -p "$EESD_STAGE"
gh release download "$EESD_RELEASE_TAG" --repo yuhanlydia/PBPF --dir "$EESD_STAGE"
cd "$EESD_STAGE"
sha256sum -c SHA256SUMS
# Inspect each archive and its file manifest; reject unexpected absolute/.. paths.
# Restore only into an inactive, separately staged tree, then verify file hashes.
git clone https://github.com/yuhanlydia/PBPF /root/PBPF
cd /root/PBPF
git checkout --detach "$EESD_RELEASE_TAG"
# Example only; replace the archive name with the release's documented asset.
# tar -xzf "$EESD_STAGE/ARTIFACT_ARCHIVE.tar.gz" -C /root/PBPF
```

For split archives, first verify every part, then concatenate/extract in the
publisher's specified order. Never extract over an existing active checkout.
The release should provide a restore manifest if archive names/layout differ
from this template. A checksum file stored alongside assets verifies transfer
consistency; retain the release commit and independent receipts as provenance.

## Host and runtime requirements

1. Prefer the same absolute checkout path **`/root/PBPF`**. Many manifests,
   receipts and operational scripts embed it. Relocation requires explicit path
   migration receipts and dependency revalidation; global search/replace would
   invalidate hashes and is not a valid restoration.
2. Builder/ML environment is Python **3.11.16**, candidate interpreter is
   **`/usr/bin/python3` Python 3.10.12**. Recreate `.venv`; do not copy it blindly.
   Review `runs/eesd-setup/requirements.lock.txt` and
   `environment-installed.json` and install compatible CUDA/PyTorch wheels for
   the replacement GPU. Broad `pip install -e '.[ml,experiment,test]'` alone does
   not recreate the recorded versions. A new binary hash/platform is a real
   runtime change even if its version string matches.
3. Direct execution is explicitly **not sandboxed**. Its parent requires root,
   keeps `/root` mode 0700 and drops candidates to UID/GID 65534 with no extra
   groups. Preserve six-second deadlines, comparison semantics and declared
   resource limits. Do not describe old bubblewrap readiness as direct readiness.
4. Preserve old execution locks. On a new host build a **new runtime/amendment
   receipt and run trusted probes** before new candidate execution. Old binary
   or platform-bound locks can correctly reject the replacement host. Existing
   result provenance stays attached to the old host; do not rewrite it to make
   validation pass or silently mix execution profiles.
5. Re-download pinned model IDs/revisions from `configs/models/` and verify
   model/tokenizer inventories. Existing `*-verified.json` describes the old
   local installation, not automatic proof of a new download. Recheck GPU
   inference compatibility. SFT with an anchor loads student and reference;
   generation memory measurements do not establish its peak training memory.

## Resume in this order

1. **Verify artifacts first.** Re-run complete-bank checksums/identity validators,
   admission checks, and cache/report provenance checks. Reuse valid completed
   banks and reports; do not regenerate them merely because the GPU changed.
   Keep a migration ledger recording original source hashes, host/GPU/runtime,
   which completed artifacts were reused, and newly executed work.
2. **Create a fresh operational inventory.** Old PID/starttime/session/boot ID,
   `adoption*.json`, lock ownership and `running` status are historical. Never
   blindly start v4 with the old adoption receipt: it is explicitly bound to the
   old boot and live processes. Rebuild pending/completed state from verified
   artifacts, not guessed exit codes. The original recovery wrapper only fixes
   permission errors while scanning local direct candidates; it is not a
   cross-host adoption tool.
3. **Reconcile partial banks individually.** Preserve `run.json`, candidate JSON
   and per-record checksums. Original generators can resume matching banks;
   Replay rejects `.partial`, orphan checksum or identity conflicts rather than
   overwriting them. Resolve interrupted publication explicitly. Hardware
   changes can alter generated tokens despite the same seed, so record a mixed
   hardware continuation and never select which candidates to rerun by outcome.
4. **Resume training prerequisites.** The old operational training-bank preparer
   raises `FileExistsError` on any existing output, including the already sealed
   321-source train bank. Do not rerun it unchanged. Verify that bank, resume only
   the incomplete 400-source development generator with its recorded arguments,
   then create a new verified readiness/status handoff. The training pipeline
   also consumes PID-bearing upstream status and create-once outputs; resume by
   verified stage rather than restarting from its first command.
5. **Run CPU scoring alongside GPU generation/training.** Each completed pair
   of development/primary mechanism banks can be scored immediately; no need to
   wait for all 48. Training corrections use train/development public tests only,
   never the primary mechanism outcomes. Preserve the adopted response-token
   budget and zero-positive non-estimable policy. Fresh evaluation remains a
   separate required phase after training.
6. **Rebuild scheduling limits for the new device.** Current queues use GPU 0,
   `device_map={"":0}`, a single-device memory query and device-specific floors.
   They do **not** automatically support multi-GPU scheduling. Multiple GPUs need
   explicit per-process device assignment and coordinated, non-overlapping tasks;
   merely increasing `max_jobs` is insufficient. Keep one writer per bank/cell.

The old host's `pause-queue` may belong to training bank preparation or training.
Do not remove it on that host as part of upload. On the replacement host preserve
its snapshot as history, then establish a fresh owner after confirming there are
no local workers; copying the old pause file unchanged can stall a new queue.

## Published handoff details

The release description and STOP_RECEIPT.json record the verified source commit,
asset checksums, final cutoff and original-host shutdown requested by the user.
No migration is claimed complete by this document alone.

## Published checkpoint location

- Branch: `research/eesd-iclr2027`
- Release/tag: `eesd-checkpoint-20260921`
- Release URL: https://github.com/yuhanlydia/PBPF/releases/tag/eesd-checkpoint-20260921
- Snapshot assets: `eesd-runs-20260921.tar.gz.part-000`, subsequent numbered parts,
  `ARTIFACT_FILES.jsonl`, `SHA256SUMS`, `RESTORE.md`, and `verify_artifacts.py`.
- The release description records the exact source commit and verification outcome.
- Restore command after verifying `SHA256SUMS`:

```bash
cat "$EESD_STAGE"/eesd-runs-20260921.tar.gz.part-* | tar -xzf - -C /root/PBPF
# Apply the final run-state overlay following RESTORE.md before file verification.
```

The snapshot briefly stopped eight related processes for 4.31 seconds while
copying mutable run trees, then resumed every surviving process. Dataset files
were immutable and archived directly. The snapshot is a point-in-time backup;
the release additionally includes a final stopped-state overlay. Follow RESTORE.md
to apply that overlay before validating FINAL_ARTIFACT_FILES.jsonl. The final stop
receipt distinguishes interrupted jobs from completed experiments.
