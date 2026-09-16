#!/usr/bin/env bash
set -Eeuo pipefail
unset PYTHONPATH PYTHONHOME
umask 077
cd /data/cwj/PBPF
export HF_HOME="$PWD/local/model-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
local/sandbox.sh .venv/bin/python -u scripts/run_real_pilot.py --parquet local/data/runbugrun-v0.0.1/task6581.parquet --output /dev/shm/pbpf-additional/legacy_rbr_6581 2>&1 | tee local/logs/legacy_rbr_6581.log
cp -a /dev/shm/pbpf-additional/legacy_rbr_6581 local/results/
