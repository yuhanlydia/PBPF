# EESD backup complete — 2026-09-21

- Full backup: https://huggingface.co/datasets/humanlong/PBPF
- Restoration instructions: https://huggingface.co/datasets/humanlong/PBPF/blob/main/RESTORE.md
- Verification receipt: https://huggingface.co/datasets/humanlong/PBPF/blob/main/REMOTE_VERIFICATION.json
- 306 required files verified against remote sizes and LFS SHA-256 / Git blob SHA-1;
  missing 0, mismatches 0. Data revision: `b7c78b0d90ff5813b94dd1638780cdafc910242a`.
- Full source and future experiment plans are in the chunked Git bundle. Small
  files remain browsable on this GitHub branch. The snapshot bundle commit
  `ed4bdd7` and GitHub snapshot commit `913b343` have the same source tree.
- All experiment controllers/workers are stopped; the GPU compute-process list
  is empty. Twelve existing direct eval cells finished and were checksum-verified.
- The full four-model/four-domain study and SFT are **not complete**; continuation
  tasks remain in EESD_REMAINING_EXPERIMENTS_20260921.md.
- Base model weights and package environments are excluded. Download the exact
  locked revisions; no unique trained SFT adapter existed at shutdown.

This final delivery supersedes provisional GitHub Release/browsable HF mirror
instructions elsewhere. Use HF RESTORE.md: join the base archive chunks and then
apply the final stopped-state overlay. The source Git history is also chunked.
Do not reuse old PIDs/adoption receipts on a replacement host.
