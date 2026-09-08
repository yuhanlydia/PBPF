#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pbpf plan "$repo_dir/configs/experiments/exact_smoke.yaml"
python -m pbpf smoke
