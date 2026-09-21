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

## Final delivery — supersedes the original full-release plan

The user cancelled large backup uploads. No full run/data Release is published.
Source, all future experiment plans and the twelve completed eval outputs are
preserved on branch `research/eesd-iclr2027`.

See `artifacts/eesd-eval-20260921/README.md` and `STOP_RECEIPT.json` there.
All experiment processes were stopped after verifying those twelve evals.
The snapshot counts above are historical; partial generation does not imply eval.

Raw data, candidate banks, full setup/run logs, model weights and the environment
remain on the original machine and have NOT been uploaded. Downloading Git alone
is not a full run-state restoration. Preserve the original disk if those partial
runs must be reused. There is no unique trained SFT adapter to migrate.

## Continuing on a replacement GPU

1. Read EESD_REMAINING_EXPERIMENTS_20260921.md and its machine-readable task ledger.
2. Download base models/tokenizers at configured revisions and prepare the datasets.
3. Preserve historical receipts. Recreate runtime/environment locks through a new
   documented execution profile; old locks bind the previous host/binaries.
4. Old PID/adoption files cannot be reused across hosts. Rebuild scheduling state
   from actual verified banks; do not infer completed work from an old running flag.
5. Current scheduling scripts assume GPU 0. Multiple GPUs require explicit device
   allocation and one writer per bank. Replay direct evaluation still needs its
   documented adapter work; the complete future pipeline is not one-click ready.
6. The old training-bank preparer rejects existing output directories. Resume a
   partial bank stage explicitly rather than rerunning its whole setup unchanged.
