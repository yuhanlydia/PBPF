#!/usr/bin/env bash
set -Eeuo pipefail
unset PYTHONPATH PYTHONHOME
umask 077
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
command -v uv >/dev/null
if [[ ! -x .venv/bin/python ]]; then uv venv --python /usr/bin/python3.12 .venv; fi
uv pip install --python .venv/bin/python --link-mode copy 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python --link-mode copy -r local/requirements.lock.txt
uv pip check --python .venv/bin/python
