#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
config=${1:-"$repo_dir/configs/experiments/frozen_7b_16gb.yaml"}
python -m pbpf plan "$config"
