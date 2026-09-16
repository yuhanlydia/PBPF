#!/usr/bin/env bash
set -Eeuo pipefail
unset PYTHONPATH PYTHONHOME
umask 077
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
export PATH="$root/.venv/bin:$PATH"
args=(--config configs/experiments/iclr_pbpf.yaml --profile local_cpu --output-root "${PBPF_SMOKE_OUTPUT_ROOT:-local/results/iclr}")
pbpf-iclr doctor "${args[@]}" --dry-run
pbpf-iclr prepare "${args[@]}" --resume
bash scripts/run_iclr.sh "${args[@]}" --resume
pbpf-iclr aggregate "${args[@]}"
pbpf-iclr verify "${args[@]}"
pbpf-iclr package "${args[@]}"
