# PBPF

PBPF is a standalone experiment package for execution-conditioned particle
beliefs in coding repair. It implements the preregistered finite audit, immutable
trajectory protocol, frozen-prediction interfaces, coherent particle-conditioned
generation, fixed-order four-round repair, and optional Transformers/PEFT QLoRA
hooks. This repository contains software and preregistered configurations only;
it makes no empirical performance or repair-gain claim.

## Install and validate

```bash
git clone https://github.com/yuhanlydia/PBPF.git
cd PBPF
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pbpf-run doctor
pbpf-run plan configs/experiments/exact_smoke.yaml
scripts/run_smoke.sh
```

The default install is CPU-only and never imports or downloads model weights.
Install `.[ml]` only on an experiment host. `doctor` uses module discovery rather
than importing Torch, Transformers, PEFT, bitsandbytes, or datasets. The smoke
launcher executes the full protocol against an in-memory Transformers-compatible
model; the 7B launchers perform fail-closed plan validation before an operator
supplies authorized model/data mounts and available baseline adapters.

## Protocol invariants

- The five modeled outcomes are PASS, WRONG_OUTPUT, RUNTIME_EXCEPTION, TIMEOUT,
  and COMPILE_ERROR; infrastructure failures are separate.
- Tests execute in manifest order with G=8 shared initial candidates and exactly
  four repair rounds. Arms cannot skip, repeat, reorder, or early-stop tests.
- Future outcomes, gold solutions/patches, expected outputs, and SWE-bench
  `test_patch`, `FAIL_TO_PASS`, and `PASS_TO_PASS` remain evaluator-side.
- Evaluator outcomes are keyed by candidate content hash and test ID; task-global
  outcomes are not accepted for prediction scoring.
- A posterior particle is sampled once per continuation. Token-wise re-mixture is
  available only as the explicitly faulty `tokenwise_remixture_fault` ablation.
- Reports and banks are create-once and checksum-verified.

See [EXPERIMENTS](docs/EXPERIMENTS.md), [BASELINES](docs/BASELINES.md),
[DATA](docs/DATA.md), and [ARTIFACTS](docs/ARTIFACTS.md) for executable protocols.
