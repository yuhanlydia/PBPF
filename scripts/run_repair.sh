#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
config=${1:-"$repo_dir/configs/diagnostics/evalplus_repair_7b_24gb.yaml"}
output=${2:-"$repo_dir/results/real_repair_7b"}
num_tasks=${PBPF_NUM_TASKS:-164}
start_index=${PBPF_START_INDEX:-0}
max_new_tokens=${PBPF_MAX_NEW_TOKENS:-192}
python "$repo_dir/scripts/run_evalplus_screen.py" \
  --config "$config" --output "$output" --num-tasks "$num_tasks" \
  --start-index "$start_index" --max-new-tokens "$max_new_tokens"
