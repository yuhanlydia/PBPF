#!/usr/bin/env bash
set -Eeuo pipefail
unset PYTHONPATH PYTHONHOME
umask 077
cd /data/cwj/PBPF
export PATH="$PWD/.venv/bin:$PATH"
export HF_HOME="$PWD/local/model-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HUMANEVAL_OVERRIDE_PATH="$PWD/local/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"
mode="${1:?frozen or repair}"
case "$mode" in
 frozen) script=scripts/run_frozen_7b.sh; config=configs/diagnostics/evalplus_frozen_7b_16gb.yaml ;;
 repair) script=scripts/run_repair.sh; config=configs/diagnostics/evalplus_repair_7b_24gb.yaml ;;
 *) exit 2 ;;
esac
output="/dev/shm/pbpf-additional/evalplus_${mode}_164"
local/sandbox.sh bash "$script" "$config" "$output" 2>&1 | tee "local/logs/evalplus_${mode}_164.log"
cp -a "$output" "local/results/"
