#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
apbpf_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
apbpf_python="${PBPF_APBPF_PYTHON:-$apbpf_root/.venv/bin/python}"
if [[ ! -x "$apbpf_python" ]]; then
  printf 'Set PBPF_APBPF_PYTHON to an executable Python with this project installed, or install into .venv.\n' >&2
  exit 2
fi
cd -- "$apbpf_root"
exec "$apbpf_python" -m pbpf.apbpf.cli run "$@"
