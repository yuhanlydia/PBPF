# EESD ICLR 2027 branch

The active ICLR paper direction on branch `research/eesd-iclr2027` is
**Effective-Evidence Self-Distillation (EESD)** rather than the legacy PBPF
repair-conditioning claim. Historical PBPF/A-PBPF results are preserved as
diagnostics; they were never deleted from `results/`.

## Start here

- experiment lock: [docs/EESD_ICLR2027_EXPERIMENTS.md](docs/EESD_ICLR2027_EXPERIMENTS.md)
- locked hyperparameters/matrix: [configs/experiments/eesd_iclr2027.yaml](configs/experiments/eesd_iclr2027.yaml)
- external-path manifest template: [configs/experiments/eesd_cache_manifest.example.yaml](configs/experiments/eesd_cache_manifest.example.yaml)
- measured + planned paper tables: [paper/eesd_tables.tex](paper/eesd_tables.tex)
- evidence matrix: [scripts/run_eesd_evidence_matrix.py](scripts/run_eesd_evidence_matrix.py)
- public correction firewall: [scripts/prepare_eesd_recursive_corrections.py](scripts/prepare_eesd_recursive_corrections.py)
- correction generation/re-execution: [scripts/generate_eesd_corrections.py](scripts/generate_eesd_corrections.py)
- all-rule correction scoring: [scripts/score_eesd_corrections.py](scripts/score_eesd_corrections.py)
- shared QLoRA trainer: [scripts/run_eesd_weighted_sft.py](scripts/run_eesd_weighted_sft.py)
- fresh Pass@1 evaluator: [scripts/evaluate_eesd_fresh_bank.py](scripts/evaluate_eesd_fresh_bank.py)
- closed-loop recursive study: [scripts/run_eesd_recursive.py](scripts/run_eesd_recursive.py)
- official EvalPlus transfer: [scripts/run_eesd_evalplus_transfer.py](scripts/run_eesd_evalplus_transfer.py)
- results-to-LaTeX renderer: [scripts/render_eesd_tables.py](scripts/render_eesd_tables.py)
- resumable orchestrator: [scripts/run_eesd_matrix.py](scripts/run_eesd_matrix.py)

## Locked empirical breadth

The mechanism study is **2 execution domains x 6 model settings**:
RunBugRun and CodeARC-Replay crossed with Qwen2.5-Coder-1.5B,
Qwen2.5-Coder-7B, Qwen3-8B (non-thinking), DeepSeek-Coder-6.7B,
Seed-Coder-8B, and Qwen3-Coder-30B-A3B. Each cell uses 200 development
source components and the full 500-source primary population, with one sealed
candidate per source.

The mechanism runner includes same-alpha comparisons, full alpha x relevance
strength factorials, a tuned global-mass baseline, constant-mean-mass control,
ten aligned-vs-permuted-mass controls, history-size sweeps, binary-outcome
sensitivity, concentration-bin analysis, NLL/Brier/ECE/accuracy, and 10,000-draw
paired source-cluster bootstrap intervals.

Downstream evidence is separate from the transductive probability mechanism:
- one-round fresh all-tests Pass@1 on RunBugRun and CodeARC;
- cross-family RunBugRun replication with DeepSeek-Coder-6.7B;
- official HumanEval+ and MBPP+ Base+Extra Pass@1 transfer after RunBugRun distillation;
- three-round closed-loop equal-weight-vs-EESD studies on RunBugRun and CodeARC;
- fixes, regressions, retained correctness, and net gain are reported with Pass@1.

Training/fresh/transfer/recursive runs use the fixed seeds 1701, 1702, 1703.
Experience collection is stochastic under a fixed seed; fresh primary evaluation
is deterministic greedy one-candidate Pass@1.

## Run

```bash
git fetch origin research/eesd-iclr2027
git switch research/eesd-iclr2027
python -m pip install -e '.[test,ml,experiment]'

cp configs/experiments/eesd_cache_manifest.example.yaml \
   configs/experiments/eesd_cache_manifest.yaml
# Edit only public_root/evaluator_root paths. Keep every declared cell.

# CPU/public preparation + mechanism work; resumes completed artifacts.
bash scripts/run_eesd_iclr.sh

# Explicit GPU stages can be distributed across workers.
CUDA_VISIBLE_DEVICES=0 python scripts/run_eesd_matrix.py \
  --config configs/experiments/eesd_iclr2027.yaml \
  --manifest configs/experiments/eesd_cache_manifest.yaml \
  --output runs/eesd-iclr2027 --stage generate-corrections

CUDA_VISIBLE_DEVICES=0 python scripts/run_eesd_matrix.py \
  --config configs/experiments/eesd_iclr2027.yaml \
  --manifest configs/experiments/eesd_cache_manifest.yaml \
  --output runs/eesd-iclr2027 --stage train

# Then run fresh / transfer / recursive and render tables.
python scripts/run_eesd_matrix.py --config configs/experiments/eesd_iclr2027.yaml \
  --manifest configs/experiments/eesd_cache_manifest.yaml \
  --output runs/eesd-iclr2027 --stage render
```

The old `paper/pbpf_iclr2027.tex` is preserved for provenance. It was a
prospective manuscript from its first commit, which is why it did not contain
the later completed tables. Those measurements lived in `results/`, JSON
reports, and the EED blueprint. The EESD branch now centralizes measured and
planned tables in `paper/eesd_tables.tex`; completed new artifacts can be
rendered automatically to `paper/generated_eesd_results.tex`.


---

# PBPF

Candidate-specific particle beliefs for execution-conditioned Python repair.
This repository contains mathematical/data/arm primitives, an executable synthetic
runner, and a prospective ICLR 2027 S0–S4 experiment contract. **No formal PBPF
results are included.** The historical `50 -> 84 / 164` EvalPlus screen is a
legacy diagnostic, not PBPF evidence.

The first real-data prediction diagnostic is recorded in
[`results/RBR_GATE_B_2026-09-15.md`](results/RBR_GATE_B_2026-09-15.md). On
source-problem-disjoint RunBugRun data, learned candidate-specific states improve
future-outcome NLL over no evidence, an outcome-rate baseline, wrong-candidate
evidence, and random latents across three seeds. The stricter causal gate still
fails because shuffled test/outcome associations retain the gain; actor repair
conditioning has not yet been tested by this diagnostic.

The subsequent frozen-Qwen actor pilot is recorded in
[`results/RBR_REPAIR_GATE_2026-09-15.md`](results/RBR_REPAIR_GATE_2026-09-15.md).
After repairing the missing-problem-statement input and bounding the prefix to
the actor embedding scale, PBPF posterior conditions solved 0/8 candidates
versus 1/8 for no latent and regressed future-test pass fraction from 35.4% to
22.9%. This small gate is negative and was not expanded.

The proposed candidate-selector follow-up and its data requirements are recorded
in [`docs/PBPF_SELECTOR_GATE.md`](docs/PBPF_SELECTOR_GATE.md). An audit of the
existing 164-task GOAV EvalPlus bank found that four-test pass rate already
matches Oracle Pass@8, so that bank cannot identify an additional PBPF selector
gain; a harder grouped candidate bank is required.

## 2026-09-16 local execution results

The [local execution report and artifacts](results/local_20260916/README.md) record
completed runnable diagnostics (EvalPlus frozen 86/164; repair 72/164), training
logs, reproduction patches, and 674 passing tests on the recorded run version.
Formal S0–S4 remains unexecuted. These are not APBPF results or evidence of
learned PBPF repair gains.

## Install and run offline

After the reviewed feature branch is published:

```bash
git clone https://github.com/yuhanlydia/PBPF.git
cd PBPF
git fetch origin codex/pbpf-iclr-formal
git switch --track origin/codex/pbpf-iclr-formal
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
pbpf-iclr doctor --config configs/experiments/iclr_pbpf.yaml --profile local_cpu --dry-run
pbpf-iclr prepare --config configs/experiments/iclr_pbpf.yaml --profile local_cpu --resume
bash scripts/run_iclr.sh --config configs/experiments/iclr_pbpf.yaml --profile local_cpu --resume
pbpf-iclr aggregate --config configs/experiments/iclr_pbpf.yaml --profile local_cpu
pbpf-iclr verify --config configs/experiments/iclr_pbpf.yaml --profile local_cpu
pbpf-iclr package --config configs/experiments/iclr_pbpf.yaml --profile local_cpu
```

An existing reviewed checkout can start at environment creation. Python 3.11 is
also supported. `local_cpu` runs the complete 15-stage synthetic DAG without
Torch, Transformers, datasets, sockets, downloads, GPU time, or projected GPU
cost. Its label is always `smoke-only-no-claim`; synthetic S0–S4 control-flow
coverage is not an empirical finite audit, training run, or replication.

## Formal entrypoint and deployment limitation

```bash
bash scripts/run_iclr.sh --config configs/experiments/iclr_pbpf.yaml --profile slurm_h200x16 --resume
```

This exact command is the single formal launcher, **not a way to bypass cluster
provisioning**. It fails closed without an audited production `FormalFactory`,
immutable snapshots, H200 resource/pricing overlay, container, and independently
provisioned evaluator authority. A complete production scientific factory is not
shipped yet. The lazy native-chat HF actor/partial-decode adapter, Datasets loader,
neural/arm primitives and phase-boundary checks are available for integration;
no formal path falls back to smoke. H200 execution has not been verified here.

See [operator workflow](docs/EXPERIMENTS.md), [data contracts](docs/DATA.md),
[arms/provenance](docs/BASELINES.md), and [verification](docs/ARTIFACTS.md).
SWE-bench and trace-rich LDB are outside the core S0–S4 claim.

## A-PBPF information-inference workflow

A-PBPF is a separate prospective experiment path with its own
`apbpf-iclr-v1` identity. It does not change the frozen `pbpf-iclr` contract or
turn the historical prediction/repair diagnostics into positive evidence. The
feature branch can be checked out with:

```bash
git fetch origin codex/apbpf-information-inference
git switch --track origin/codex/apbpf-information-inference
```

After provisioning the real stage-worker overlay described in
[`docs/APBPF_WORKERS.md`](docs/APBPF_WORKERS.md), the resumable launcher is:

```bash
bash scripts/run_apbpf_iclr.sh --config configs/experiments/apbpf_iclr2027.yaml --profile local_24gb --site /path/to/apbpf-site.yaml --output-root runs/apbpf --resume
```

This command requires an audited real `local_24gb` site overlay; the example
overlay deliberately contains null commands and therefore fails closed. A
provisioned profile must first pass the association and selection gates. A failed upstream
gate blocks repair; an explicit
`--continue-exploratory` records downstream output as nonconfirmatory rather
than bypassing the failure. The explicit `local_smoke` profile is smoke-only and
cannot make an empirical claim. See [A-PBPF experiments](docs/APBPF_EXPERIMENTS.md)
and [A-PBPF artifacts](docs/APBPF_ARTIFACTS.md) for the exact command set,
conditioning controls, and verification inventory.

## Verify the checkout

```bash
python -m pytest -q -m 'not gpu and not network'
python -m compileall -q src scripts tests
for script in scripts/*.sh scripts/slurm/*.sbatch; do bash -n "$script"; done
git diff --check
```

CPU CI covers Python 3.11/3.12 and all six local commands. GPU/network tests are
opt-in. `.[ml,experiment]` is for a provisioned experiment host, not CPU smoke.
Legacy `pbpf-run` and its configurations remain a separate diagnostic interface.
