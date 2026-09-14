# ICLR data and evaluator boundary

`configs/iclr/datasets.yaml` is the authoritative Task-4 registry. Never reuse
legacy `configs/benchmarks` pins for the formal experiment. Immutable raw/split
and adapter hashes accompany upstream commits; preparation reports exact
raw/kept/excluded counts and reasons.

| Protocol | Formal partition |
|---|---|
| Finite audit | 50,000/5,000/5,000 disjoint latent family/template cases; four visible/six future tests |
| PBPF-RBR | Python, >=10 active tests, fixed passes all, buggy fails at least one, <=2,048 actor tokens |
| CodeARC-Replay | Anonymous only; invocations 0–3 visible, 4–9 evaluator; not the interactive leaderboard |
| PBPF-EvalPlus | HumanEval+ 164 and MBPP+ 378; base visible, deduplicated plus-only hidden; repair protocol |
| LiveCodeBench | Release-v6 minus release-v5, locked contamination-reduced replication |

RBR preserves official boundaries and removes leakage-connected source components
by problem/code/test signatures. The prospective 12,000 training cap reserves at
most 1,000 source-disjoint tasks for scalar-temperature calibration, leaving at
most 11,000 fitted-training tasks. Component boundaries outrank exact caps; report
realized counts. The 2,000 development cap splits into tune/select halves; the
2,000 test cap never enters fitting. The separate 64-task resource pilot is
disjoint from all of these scientific splits.

`configs/iclr/finite.yaml` versions the full generator contract: PCG64 stream
derivation, family/template allocation and categorical probability recipe. Its
content is hashed along with the actual versioned prompt bundle. The current
local handler is a tiny synthetic finite audit, not the 60,000-case production
generator. A production factory must materialize/hash that contract for S1.

Credential-free offline preparation:

```bash
pbpf-iclr prepare --config configs/experiments/iclr_pbpf.yaml --profile local_cpu --resume
```

Production preparation reads operator-provisioned pinned public snapshots through
`pbpf.data.prepare_dataset`. The optional `pbpf.runner.hf.load_dataset_snapshot`
requires exact revisions and defaults to local files. EvalPlus/RunBugRun repository
snapshots need their specific adapters, not guessed generic Hub IDs.

Generator records contain public task text, candidate code, visible identifiers
and bounded feedback only. Hidden tests/outputs, gold code/patches and final
outcomes remain separately mounted evaluator-side. Final hashes are sealed before
private tests open. GPU/controller jobs receive no private overlay. Same-UID
subprocesses are smoke only, not formal secrecy. Formal services require distinct
UIDs, immutable sandbox dependencies and signed one-use capabilities.

SWE-bench and trace-rich debugger data are outside core S0–S4. No private sidecars,
keys, hidden canaries or gold targets belong in public run packages.
