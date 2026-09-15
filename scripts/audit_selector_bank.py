#!/usr/bin/env python3
"""Audit whether a frozen grouped candidate bank can identify a PBPF selector."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pbpf.real_gate import selector_bank_audit


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(bank_jsonl: Path, bank_npz: Path, sidecar_json: Path) -> dict:
    groups = [json.loads(line) for line in bank_jsonl.read_text().splitlines() if line.strip()]
    sidecar = json.loads(sidecar_json.read_text())
    labels = {(row["problem_id"], row["checkpoint_id"]): row["values"]
              for row in sidecar["labels"]}
    matrices, trusted = [], []
    with np.load(bank_npz, allow_pickle=False) as archive:
        for group in groups:
            key = (group["problem_id"], group["checkpoint_id"])
            if group["matrix_key"] not in archive or key not in labels:
                raise ValueError(f"candidate group is missing outcomes or trusted labels: {key}")
            matrix = np.asarray(archive[group["matrix_key"]])
            label = np.asarray(labels[key])
            if matrix.shape[0] != len(group["candidate_ids"]) or label.shape != (matrix.shape[0],):
                raise ValueError(f"candidate inventory does not align for {key}")
            matrices.append(matrix)
            trusted.append(label)
    if not matrices or len({matrix.shape for matrix in matrices}) != 1:
        raise ValueError("selector bank must contain nonempty fixed-shape candidate groups")
    report = selector_bank_audit(np.stack(matrices), np.stack(trusted), prefixes=(1, 2, 4, 8, 12, 24, 48))
    return {
        "schema": "pbpf-selector-bank-audit-v1",
        "source_bank_hash": sidecar.get("bank_hash"),
        "source_files": {path.name: _sha256(path) for path in (bank_jsonl, bank_npz, sidecar_json)},
        "audit": report,
        "decision": {
            "bank_identifies_pbpf_beyond_visible_pass_rate":
                report["visible_pass_rate_selected_pass_at_1"]["4"] < report["oracle_pass_at_k"],
            "status": "saturated-baseline-no-pbpf-selector-test"
                if report["visible_pass_rate_selected_pass_at_1"]["4"] >= report["oracle_pass_at_k"]
                else "candidate-bank-has-selector-headroom",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-jsonl", type=Path, required=True)
    parser.add_argument("--bank-npz", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bank_jsonl, args.bank_npz, args.sidecar)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
