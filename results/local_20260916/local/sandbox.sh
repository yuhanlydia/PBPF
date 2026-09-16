#!/usr/bin/env bash
set -Eeuo pipefail
unset PYTHONPATH PYTHONHOME
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
actor_cache="$(readlink -f -- "$root/local/model-cache")"
exec bwrap --ro-bind /usr /usr --symlink usr/bin /bin --symlink usr/sbin /sbin --ro-bind /lib /lib --ro-bind /lib64 /lib64 \
  --ro-bind /etc/ld.so.cache /etc/ld.so.cache --ro-bind /sys /sys \
  --dev-bind /dev /dev --proc /proc --tmpfs /tmp \
  --ro-bind "$root" "$root" --bind "$root/local" "$root/local" \
  --ro-bind /home/root123/.cache/pbpf /home/root123/.cache/pbpf \
  --ro-bind "$actor_cache" "$actor_cache" \
  --bind /home/root123/.cache/pbpf/runs /home/root123/.cache/pbpf/runs \
  --unshare-net --die-with-parent --setenv HOME /tmp --chdir "$root" "$@"
