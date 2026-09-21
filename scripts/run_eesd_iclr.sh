#!/usr/bin/env bash
set -euo pipefail

CONFIG="${EESD_CONFIG:-configs/experiments/eesd_iclr2027.yaml}"
MANIFEST="${EESD_MANIFEST:-configs/experiments/eesd_cache_manifest.yaml}"
OUTPUT="${EESD_OUTPUT:-runs/eesd-iclr2027}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing $MANIFEST" >&2
  echo "Copy configs/experiments/eesd_cache_manifest.example.yaml, bind immutable cache paths, and retry." >&2
  exit 2
fi

python scripts/run_eesd_matrix.py --config "$CONFIG" --manifest "$MANIFEST" --output "$OUTPUT" --stage mechanism
python scripts/run_eesd_matrix.py --config "$CONFIG" --manifest "$MANIFEST" --output "$OUTPUT" --stage prepare-corrections

cat <<EOF
Mechanism and correction preparation stages are complete/resumed.
Continue with correction generation/scoring and the GPU stages:
  python scripts/run_eesd_matrix.py --config "$CONFIG" --manifest "$MANIFEST" --output "$OUTPUT" --stage generate-corrections
  python scripts/run_eesd_matrix.py --config "$CONFIG" --manifest "$MANIFEST" --output "$OUTPUT" --stage score
  python scripts/run_eesd_matrix.py --config "$CONFIG" --manifest "$MANIFEST" --output "$OUTPUT" --stage train
  python scripts/run_eesd_matrix.py --config "$CONFIG" --manifest "$MANIFEST" --output "$OUTPUT" --stage fresh
  python scripts/run_eesd_matrix.py --config "$CONFIG" --manifest "$MANIFEST" --output "$OUTPUT" --stage transfer
  python scripts/run_eesd_matrix.py --config "$CONFIG" --manifest "$MANIFEST" --output "$OUTPUT" --stage recursive
  python scripts/run_eesd_matrix.py --config "$CONFIG" --manifest "$MANIFEST" --output "$OUTPUT" --stage render
Set CUDA_VISIBLE_DEVICES before launching each GPU worker.
EOF
