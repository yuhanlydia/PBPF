#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
pbpf_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
declare -A pbpf_seen=()
pbpf_args=()
pbpf_profile=""
pbpf_count=""
pbpf_index=0
fail() { printf 'run_iclr.sh: %s\n' "$1" >&2; exit 2; }
while (($#)); do
  pbpf_option="$1"
  [[ -z "${pbpf_seen[$pbpf_option]+x}" ]] || fail "duplicate option: $pbpf_option"
  pbpf_seen["$pbpf_option"]=1
  case "$pbpf_option" in
    --config|--profile|--site|--output-root|--shard-count|--shard-index)
      (($# >= 2)) && [[ "$2" != --* && -n "$2" ]] || fail "missing value: $pbpf_option"
      case "$pbpf_option" in
        --profile) pbpf_profile="$2" ;;
        --shard-count) pbpf_count="$2" ;;
        --shard-index) pbpf_index="$2" ;;
      esac
      pbpf_args+=("$1" "$2")
      shift 2
      ;;
    --resume|--force-after-failed-gate|--dry-run|--calibration-only)
      pbpf_args+=("$1")
      shift
      ;;
    *) fail "unknown option: $pbpf_option" ;;
  esac
done
[[ -n "${pbpf_seen[--config]+x}" ]] || fail '--config is required'
[[ "$pbpf_profile" == local_cpu || "$pbpf_profile" == slurm_h200x16 ]] || fail 'profile must be local_cpu or slurm_h200x16'
if [[ -z "$pbpf_count" ]]; then
  pbpf_count=16
  [[ "$pbpf_profile" != local_cpu ]] || pbpf_count=1
fi
[[ "$pbpf_count" =~ ^[0-9]{1,9}$ && "$pbpf_index" =~ ^[0-9]{1,9}$ ]] || fail 'shards require nonnegative decimal integers'
((10#$pbpf_count > 0 && 10#$pbpf_index < 10#$pbpf_count)) || fail 'shard index must be less than positive shard count'
pbpf_python="${PBPF_ICLR_PYTHON:-$pbpf_root/.venv/bin/python}"
[[ -x "$pbpf_python" ]] || fail 'create .venv and install .[test], or set PBPF_ICLR_PYTHON to one executable'
cd -- "$pbpf_root"
exec "$pbpf_python" -m pbpf.iclr_cli run "${pbpf_args[@]}"
