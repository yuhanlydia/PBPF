#!/usr/bin/env bash
set -Eeuo pipefail
unset PYTHONPATH PYTHONHOME
umask 077
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python -m pytest -q -m 'not gpu and not network'
.venv/bin/python -m compileall -q src scripts tests
for script in scripts/*.sh scripts/slurm/*.sbatch local/*.sh; do bash -n "$script"; done
git diff --check
