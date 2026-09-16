#!/usr/bin/env bash
set -Eeuo pipefail
unset PYTHONPATH PYTHONHOME
umask 077
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Extract the verified small HTML archive in sandbox tmpfs to avoid random HDD reads.
for seed in 1701 1702 1703; do
  local/sandbox.sh /usr/bin/bash -c '
    tar --no-same-owner -xzf local/data/project_codenet/problem_descriptions.tar.gz -C /tmp
    exec .venv/bin/python -u scripts/run_rbr_prediction_gate.py "$@"
  ' prediction \
    --data-root local/data/runbugrun-v0.0.1 \
    --descriptions-root /tmp/problem_descriptions \
    --descriptions-archive local/data/project_codenet/problem_descriptions.tar.gz \
    --cache local/results/rbr_real_gate_dataset.json \
    --steps 1000 --seed "$seed" --workers 16 \
    --output "local/results/rbr_gate_b_seed${seed}.json" \
    2>&1 | tee "local/logs/prediction_seed${seed}.log"
done
