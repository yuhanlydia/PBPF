
## 2026-09-20 — Replay admission implementation (not yet data admission)

- Previous continuation made progress: caught CC normalized-IO export conflicting with the raw-preserving spec; old joint/probe artifacts were explicitly rejected and archived. Revised joint receipt `2d965273dd5383e9ca5e95d35f765f9742c88667a047fbac6f6d5d49d616dd9a` restores raw native IO; independent Arrow comparison covers 10,290 selected CC tests. Final four-tokenizer probe is rerunning.
- Added pure public/evaluator projection (explicit public allowlist, exact raw text, four/ten test alignment, provenance, target exposure exclusion), receipt-bound admission validation and `scripts/materialize_eesd_replay_extension.py`. Receipt validation checks all upstream input/output checksums, full component membership, the probe source/template and per-task message hashes.
- Independent review identified missing selected-token publication coverage and an empty-directory rename race. Both reproduced in failing tests and fixed: exact 1,400-task/four-model token coverage; Linux atomic RENAME_NOREPLACE.
- Focused verification: 49 tests passed across `test_replay_materialization.py`, `test_replay_projection.py`, `test_replay_admission.py`. This is implementation verification, not formal data admission or an experimental result. No new Replay domain has been queued yet.
- Existing generation supervisor PID 105654 revalidated live. Latest queue observation: four candidate-generation tasks complete, three running, seventeen pending. Namespace blocker still prevents candidate scoring/training dependencies.

### Replay production data admission completed

- Corrected raw-preserving four-tokenizer audit: APPS 1,988/2,004 fitting members → 1,758 independent allocated sources; CC 1,011/1,029 → 995. Token receipt SHA `8257ad4e5b03daf1e176baf6b249059d142ad26be08637bf2fddc9d8156921b4`.
- Production materializer exited 0 and atomically published `runs/eesd-data/replay-extension-locked-20260920`; admission receipt SHA `73d96f7b0054726aef27b007e8a789ea22a739cfd338e9d096360261dbfbd40f`. Each domain has 200 development and 500 primary sources, with separated public/evaluator views and selected-token audit. Fixed thresholds unchanged.
- Independent reviewer confirmed the admission fixes and actual upstream receipt validation. Final independent bundle verification is in progress. Replay generation integration is being designed separately; no new-domain inference or scoring has occurred.

## 2026-09-20 — Replay public-only generation integration

- Final data admission independently verified in `runs/eesd-setup/replay-extension-independent-verification-20260920.json`: exact selection and all 1,400 original statements / 14,000 raw IO pairs, source separation and hashes verified. Materialization plan complete; this remains preparation, not experimental results.
- Added `replay_prompt.py`, matching sealed template exactly. Seven focused tests pass. All 1,400 message hashes match selected audits; four real pinned tokenizers × two domain samples match rendered/token IDs/counts (`replay-generation-prompt-check.json`).
- Added independent `generate_eesd_replay_bank.py` with full-split CPU token preflight before GPU loading, explicit FP4/bfloat16, fixed one-candidate sampling, immutable completion helpers, public-only inputs, external writer lock and no-overwrite artifact publication. Eight CLI tests pass; independent bounded prompt/CLI review found no definite protocol deviation.
- Review limitation: interruption after decoding but before any record/partial file exists can cause a same-seed retry. Publication interruptions fail closed. Do not describe generation attempts as strictly exactly-once.
- Real CLI CPU preflight: Seed-Coder/CodeContests development, 200 tasks, exit 0; `runs/eesd-setup/replay-cli-preflight-seed-cc.json`. No generation output directory or GPU model was created. Subsequent source/proof-binding changes require regenerated final freeze/preflight evidence.
- New public loader/bank verification and 24-cell manifest planner are under implementation/review. No Replay GPU task has launched; existing supervisor PID 105654 and all three current worker PIDs were revalidated live.

## 2026-09-20 — Replay generation frozen; queue integration pending

- Root independently verified 44 focused generation tests; then independently ran the new real-library/CLI integration test (CPU fake model), 1 passed. It seals 200 fixture records, proves complete resume makes no model load, restores exactly one missing record, and rejects an orphan record missing its checksum. These are tests, not generated study candidates.
- Frozen production manifest: `runs/eesd-setup/replay-generation-manifest-20260920.json`, SHA `f0fcced94556984ec554f272a5adcf2df1f05d1d2d62f09a3f5120fa82f81f52`, 24 cells / 48 split commands / 16,800 planned candidates. Public-only read audit recorded separately. No frozen scientific source changed during integration tests.
- Joint v4 scheduler/Replay-cell wrapper are being prepared separately, with old 24 tasks prioritized before 24 Replay tasks. Planned handoff uses the same exclusive queue lock, witnesses PID/start-time/session identities and preserves live workers. No scheduler has yet been stopped or replaced.
- New sealed-record token ledger: `runs/eesd-setup/generation-ledger-20260920-replay-freeze.json`: 3,571 verified candidate records, 2,037,013 input tokens, 767,150 generated tokens, 47 cap hits, 14 legacy clipped prompts. Partial population snapshot, not unique study sources, billing totals or scientific scoring.

## 2026-09-20 — Joint GPU queue v4 is now authoritative

- Root independently ran all 16 queue handoff/recovery tests successfully, then performed the live handoff. Old scheduler PID 105654 was frozen to stabilize its status, witnessed in a create-once adoption receipt, and terminated individually. No worker/process group was stopped.
- New scheduler PID **187702** owns the same queue lock. Worker PIDs **105943, 165910, 173047** retained their original PID/start ticks/commands. Root verified the 48-task status before removing its own temporary pause marker.
- **Authoritative live status is now `runs/eesd-setup/joint-generation-v4-status.json`. `generation-status.json` is historical after this handoff.** Initial v4 snapshot: 4 generation tasks complete, 3 running, 41 pending, 0 failed. Original 24 tasks remain prioritized; new 24 Replay tasks may fill compatible unused capacity.
- Handoff receipts: `joint-generation-v4-adoption-20260920.json`, `joint-generation-v4-handoff-pre.json`, `joint-generation-v4-handoff.json`; log `joint-generation-v4.log`. The same max-three-worker / measured reservation / 2560 MiB headroom / 90-second launch interval rules apply.
- Frozen v4 SHA `42b7469fea0c98c8747e634fb6b95aa8ba124af4fcc820745ae901174bfe9c7a`; wrapper SHA `bfdf9b24d8e9468f94dea2184ecff92c67a04c7a90be083457e59201bb36fc23`. Old scheduler source remained SHA `68fbec86bd05fac4374d79379f9e7565fe5c1dec9a6b44d32d652cf6e584d569`.
- This transition schedules candidate generation only. Namespace support still blocks scientific execution/scoring and dependent training. Replay cache adapter implementation is underway with mocks only; no actual cache/outcome was published.

## 2026-09-20 — Replay execution adapter and fresh namespace blocker receipt

- Added trusted `replay_execution_cache.py`, fixed-profile `replay_runtime.py`, and `scripts/build_eesd_replay_mechanism_cache.py`. Both banks must verify before evaluator reads; raw output strings map narrowly to executor `expected`; ten outcomes retain their order and caches redact code/expected outputs/diagnostics.
- Fresh trusted readiness precedes every dataset execution, even compile-invalid candidates. A bounded spawn process pool admits at most the locked worker count and cancels pending work on infrastructure failure. Input/source/runtime drift prevents publication; completed caches require external bindings and a validated historical readiness receipt.
- Focused tests: cache 20, runtime 6, CLI 13 passed. Independent review reproduced and prompted fixes for a missing sealed record being misreported as pending when another bank was incomplete, and insufficient historical readiness metadata validation. No tests ran real candidate programs.
- Execution lock `runs/eesd-setup/replay-execution-lock-20260920.json`, SHA `dbb246e394b47462df42e2056ef8710ba0d184aef33f6193f954912132e54d6c`, binds seven execution sources, builder/candidate/bubblewrap binary identities, Python 3.11.16/3.10.12, bubblewrap 0.6.1, six-second deadline and max 12 workers. Lock creation is not readiness.
- One actual fixed trusted-print probe exited **3**, status **infrastructure_blocked**, with `bwrap: No permissions to create new namespace`. Saved `runs/eesd-setup/replay-execution-probe-20260920.json`. No dataset candidate was executed and no outcome cache or scientific result was published.
- Joint GPU generation remains independent and live under PID 187702; the latest checked state was fresh, 4 complete / 3 running / 41 pending / 0 failed. Cache matrix preparation is continuing without execution. Host namespace support remains the external prerequisite for scoring and dependent training.

## 2026-09-21 — Direct scoring and migration checkpoint

- User explicitly authorized execution without bubblewrap. New `direct_execution` and runtime lock keep original scoring semantics, timeout and resource bounds, with UID/GID 65534 and no namespace isolation. Old sandbox locks/failure receipts remain historical.
- Twelve original-domain cells (Qwen/DeepSeek × RunBugRun/CodeARC × three seeds) have completed and sealed direct execution caches and mechanism reports. This is not all 48 cells, not final cross-family inference, and not a trained-model result.
- Generation checkpoint: 16 complete, two running, 30 pending, no reported generation failures. The original v4 supervisor exited on PermissionError while inspecting a nobody candidate's `/proc/.../cwd`; a source-bound recovery wrapper now skips only identified direct candidates and adopted existing workers without duplication.
- Qwen RunBugRun round-1 original training bank has 321 sealed candidates; its 400-source development bank is still generating. The separately launched training pipeline waits for independent training data. No SFT model/adapter has been trained yet.
- Diversity analysis and direct-profile inference adapters are implemented with focused tests; full reports remain dependent on complete verified inputs. Their scientific conclusions have not been fabricated from software tests.
- User requested GitHub backup before changing GPUs, including all future experiments. See `docs/EESD_GPU_MIGRATION_20260921.md` and `docs/EESD_REMAINING_EXPERIMENTS_20260921.md`. Source lives on `research/eesd-iclr2027`; run/data snapshot is attached to release `eesd-checkpoint-20260921`. Release publication and verification details are recorded by the upload step.

## Final shutdown and reduced GitHub handoff — 2026-09-21

User cancelled large data/run uploads. Twelve existing direct eval cells were
verified complete (all sealed file SHA-256 values match); their artifacts are
checked in under `artifacts/eesd-eval-20260921`. Experiment controllers/workers
were stopped and GPU compute-process inventory is empty. Full 4×4 experiment
and SFT remain incomplete. No full-data Release was published.

## Final user decision: full backup re-authorized

User requested the large files after all. Publish the complete base archive plus
final stopped-state overlay at Release `eesd-checkpoint-20260921`. The earlier
cancellation is superseded. Experiments remain stopped; do not resume generation
or training during upload. Final receipt/checksums in that Release are authoritative.
