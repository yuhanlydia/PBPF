#!/usr/bin/env bash
# Restrict model-generation workers to public task data and their own outputs.
set -Eeuo pipefail
if [[ $# -lt 3 ]]; then
  printf 'Usage: %s PUBLIC_ROOT OUTPUT_ROOT COMMAND [ARGS...]\n' "$0" >&2
  exit 2
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
public_root="$(realpath -- "$1")"
output_root="$(realpath -- "$2")"
shift 2
model_cache="$(readlink -f -- "$root/local/model-cache")"
unset PYTHONPATH PYTHONHOME
exec bwrap --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/sbin /sbin \
  --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /etc/ld.so.cache /etc/ld.so.cache \
  --ro-bind /sys /sys --dev-bind /dev /dev --proc /proc --tmpfs /tmp \
  --ro-bind "$root" "$root" --tmpfs "$root/local" \
  --ro-bind "$model_cache" "$model_cache" --ro-bind "$public_root" /input --bind "$output_root" /output \
  --unshare-net --die-with-parent --setenv HOME /tmp --setenv HF_HOME "$model_cache" \
  --setenv HF_HUB_OFFLINE 1 --setenv TRANSFORMERS_OFFLINE 1 \
  --setenv TOKENIZERS_PARALLELISM false --chdir "$root" "$@"
