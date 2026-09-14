#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
fail() { printf 'evaluator worker: %s\n' "$1" >&2; exit 2; }
(($# >= 4)) || fail 'expected interpreter, absolute run directory, stage, shard count'
pbpf_python="$1"; pbpf_directory="$2"; pbpf_stage="$3"; pbpf_count="$4"
shift 4
[[ -x "$pbpf_python" && "$pbpf_directory" == /* ]] || fail 'absolute executable and run directory required'
case "$pbpf_stage" in prepare|visible_execute|prediction_eval|hidden_eval|replication_eval) ;; *) fail 'invalid evaluator stage' ;; esac
pbpf_index="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$pbpf_count" =~ ^[0-9]{1,9}$ && "$pbpf_index" =~ ^[0-9]{1,9}$ ]] || fail 'invalid shard integer'
((10#$pbpf_count > 0 && 10#$pbpf_index < 10#$pbpf_count)) || fail 'invalid shard range'
pbpf_args=()
declare -A pbpf_seen=()
while (($#)); do
  [[ -z "${pbpf_seen[$1]+x}" ]] || fail "duplicate option: $1"
  pbpf_seen["$1"]=1
  case "$1" in
    --resume|--force-after-failed-gate) pbpf_args+=("$1"); shift ;;
    --evaluator-site)
      (($# >= 2)) && [[ "$2" == /* ]] || fail 'missing absolute evaluator overlay'
      pbpf_args+=("$1" "$2"); shift 2 ;;
    *) fail "unknown option: $1" ;;
  esac
done
[[ -n "${pbpf_seen[--evaluator-site]+x}" ]] || fail 'private evaluator overlay required'
unset PBPF_ICLR_SITE
exec "$pbpf_python" -m pbpf.runner.formal "$pbpf_directory" "$pbpf_stage" "$pbpf_count" --shard-index "$pbpf_index" "${pbpf_args[@]}"
