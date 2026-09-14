#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
fail() { printf 'controller: %s\n' "$1" >&2; exit 2; }
(($# >= 4)) || fail 'expected interpreter, absolute run directory, stage, shard count'
pbpf_python="$1"; pbpf_directory="$2"; pbpf_stage="$3"; pbpf_count="$4"
shift 4
[[ -x "$pbpf_python" && "$pbpf_directory" == /* ]] || fail 'absolute executable and run directory required'
[[ "$pbpf_count" == 1 ]] || fail 'controller requires exactly one shard'
case "$pbpf_stage" in doctor|manifest|finite|gate_b|seal|gate_c|tables) ;; *) fail 'invalid controller stage' ;; esac
pbpf_args=()
declare -A pbpf_seen=()
while (($#)); do
  [[ -z "${pbpf_seen[$1]+x}" ]] || fail "duplicate option: $1"
  pbpf_seen["$1"]=1
  case "$1" in
    --resume|--force-after-failed-gate) pbpf_args+=("$1"); shift ;;
    *) fail "unknown/private option on controller: $1" ;;
  esac
done
unset PBPF_ICLR_SITE
exec "$pbpf_python" -m pbpf.runner.formal "$pbpf_directory" "$pbpf_stage" 1 --shard-index 0 "${pbpf_args[@]}"
