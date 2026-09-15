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
