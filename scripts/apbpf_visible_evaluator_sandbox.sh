#!/usr/bin/env bash
# Public execution phase: neither hidden tests nor reference code are mounted.
set -Eeuo pipefail
if [[ $# -lt 4 ]]; then
  printf 'Usage: %s PUBLIC_ROOT BANK_ROOT OUTPUT_ROOT COMMAND [ARGS...]\n' "$0" >&2
  exit 2
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
public_root="$(realpath -- "$1")"
bank_root="$(realpath -- "$2")"
output_root="$(realpath -- "$3")"
shift 3
unset PYTHONPATH PYTHONHOME
exec bwrap --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/sbin /sbin \
  --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /etc/ld.so.cache /etc/ld.so.cache \
  --ro-bind /etc/alternatives /etc/alternatives --dev /dev --proc /proc --tmpfs /tmp \
  --ro-bind "$root" "$root" --tmpfs "$root/local" \
  --ro-bind "$public_root" /input --ro-bind "$bank_root" /bank --bind "$output_root" /output \
  --unshare-net --die-with-parent --setenv HOME /tmp --chdir "$root" "$@"
