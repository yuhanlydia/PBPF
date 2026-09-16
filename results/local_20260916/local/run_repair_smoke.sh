#!/usr/bin/env bash
# Two training steps and one task only: integration check, not a repair-effectiveness experiment.
set -Eeuo pipefail
unset PYTHONPATH PYTHONHOME
umask 077
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
export HF_HOME="$root/local/model-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
local/sandbox.sh .venv/bin/python -u scripts/run_rbr_repair_gate.py \
  --data local/results/rbr_real_gate_dataset.json \
  --data-root local/data/runbugrun-v0.0.1 \
  --belief-checkpoint local/results/rbr_gate_b_seed1703.pt \
  --projector-checkpoint local/results/rbr_soft_prefix_smoke.pt \
  --output local/results/rbr_repair_smoke.json \
  --train-projector --learning-rate 3e-5 --steps 2 --tasks 1 --max-sequence-tokens 1024 \
  --maximum-token-rms 0.02 --maximum-delta-rms 0.002 --seed 2701 \
  2>&1 | tee local/logs/repair_smoke.log
