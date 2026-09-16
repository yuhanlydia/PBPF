#!/usr/bin/env python3
"""Build a pre-hidden population lock, then separately audit a hard bank.

Inputs are caller-provided candidate banks, not candidate generators. `lock`
accepts only public fields; `audit` is the evaluator-side hidden-label phase.
`pilot` may select mixed DEVELOPMENT groups but never primary/test groups.
All outputs are create-once and carry a content hash.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from pbpf.apbpf.hard_bank import (
    HardBankPopulationLock, audit_hard_bank, lock_hard_bank_population,
    select_development_eligible_groups,
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "content_sha256": _digest(payload)}
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _read_sealed(path):
    value = json.loads(Path(path).read_text())
    digest = value.pop("content_sha256", None)
    if digest != _digest(value):
        raise ValueError("bank artifact checksum mismatch")
    return value


def lock_bank(records, *, provenance, output):
    """Lock all supplied primary groups without reading any hidden labels."""
    if not records:
        raise ValueError("bank requires at least one group")
    required = {"group_id", "source_component_id", "split", "candidate_ids", "visible_outcomes"}
    for row in records:
        if set(row) != required or row["split"] not in {"primary", "test"}:
            raise ValueError("lock accepts exactly public primary/test group fields; hidden fields forbidden")
        ids = row["candidate_ids"]
        if (not isinstance(ids, list) or len(ids) < 2 or any(not isinstance(v, str) or not v for v in ids)
                or len(set(ids)) != len(ids) or len(ids) != len(row["visible_outcomes"])):
            raise ValueError("candidate identities must be unique and aligned with outcomes")
    if len({row["split"] for row in records}) != 1:
        raise ValueError("one split per locked bank")
    sources = [row["source_component_id"] for row in records]
    if (any(not isinstance(value, str) or not value for value in sources)
            or len(set(sources)) != len(sources)):
        raise ValueError("source_component_id must uniquely bind each source-problem group")
    outcomes = np.asarray([row["visible_outcomes"] for row in records])
    lock = lock_hard_bank_population([row["group_id"] for row in records], outcomes,
                                     split=records[0]["split"], provenance=provenance,
                                     candidate_ids=[row["candidate_ids"] for row in records],
                                     source_component_ids=sources)
    return _write_once(output, {"schema": "apbpf-hard-bank-lock-v2", "population_lock": asdict(lock),
        "groups": records, "candidate_inventory_sha256": lock.candidate_inventory_digest,
        "source_inventory_sha256": _digest([
            {"group_id": row["group_id"], "source_component_id": row["source_component_id"]}
            for row in records]),
        "population_rule": "full caller-supplied public bank; no hidden-label selection"})


def audit_bank(lock_path, hidden_records, *, output, pilot_report=None):
    """Score only the already locked inventory, retaining all-pass/all-fail groups."""
    lock_bytes = Path(lock_path).read_bytes()
    sealed = _read_sealed(lock_path)
    if sealed.get("schema") != "apbpf-hard-bank-lock-v2":
        raise ValueError("a hard-bank population lock is required")
    rows = sealed["groups"]
    if len(hidden_records) != len(rows):
        raise ValueError("hidden inventory must cover the complete locked bank")
    hidden = {}
    for row in hidden_records:
        if set(row) != {"group_id", "candidate_ids", "hidden_labels"} or row["group_id"] in hidden:
            raise ValueError("invalid or duplicate hidden group record")
        hidden[row["group_id"]] = row
    labels = []
    for row in rows:
        value = hidden.get(row["group_id"])
        if value is None or value["candidate_ids"] != row["candidate_ids"]:
            raise ValueError("hidden candidate ordering differs from the pre-hidden lock")
        labels.append(value["hidden_labels"])
    fields = dict(sealed["population_lock"])
    fields["group_ids"] = tuple(fields["group_ids"])
    if fields.get("source_component_ids") is not None:
        fields["source_component_ids"] = tuple(fields["source_component_ids"])
    audit = audit_hard_bank([row["visible_outcomes"] for row in rows], labels,
        group_ids=[row["group_id"] for row in rows], population_lock=HardBankPopulationLock(**fields),
        candidate_ids=[row["candidate_ids"] for row in rows],
        source_component_ids=[row["source_component_id"] for row in rows])
    pilot = _read_sealed(pilot_report) if pilot_report is not None else None
    pilot_passes = bool(pilot and pilot.get("schema") == "apbpf-hard-bank-pilot-v2"
                        and pilot.get("mixed_groups", 0) >= 300)
    disjoint = bool(pilot and not (set(pilot.get("all_source_component_ids", ()))
                                  & {row["source_component_id"] for row in rows}))
    size_passes = 500 <= audit.groups <= 1000
    return _write_once(output, {"schema": "apbpf-hard-bank-audit-v2", "audit": asdict(audit),
        "lock_artifact_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "lock_payload_sha256": _digest(sealed),
        "gate": {"pilot_300_mixed_groups": pilot_passes, "pilot_primary_source_disjoint": disjoint,
                 "confirmatory_500_to_1000_groups": size_passes, "visible_below_hidden_oracle": audit.has_headroom,
                 "passes": bool(pilot_passes and disjoint and size_passes and audit.has_headroom)},
        "status": "confirmatory" if pilot_passes and disjoint and size_passes and audit.has_headroom else "audit_only"})


def pilot_bank(records, *, output, minimum_groups=300):
    if minimum_groups < 300:
        raise ValueError("pilot minimum cannot weaken the 300 mixed-group requirement")
    eligible = select_development_eligible_groups(records, minimum_groups)
    sources = [row.get("source_component_id") for row in records]
    if any(not isinstance(value, str) or not value for value in sources):
        raise ValueError("development groups require source_component_id for disjointness auditing")
    return _write_once(output, {"schema": "apbpf-hard-bank-pilot-v2", "mixed_groups": len(eligible),
        "total_development_groups": len(records), "all_source_component_ids": sources,
        "selected_development_group_ids": [r["group_id"] for r in eligible],
        "selection_scope": "development only; hidden labels explicitly permitted for pilot selection"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("lock")
    lock.add_argument("--visible-bank", type=Path, required=True)
    lock.add_argument("--provenance", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--lock", type=Path, required=True)
    audit.add_argument("--hidden-bank", type=Path, required=True)
    audit.add_argument("--pilot-report", type=Path)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--development-bank", type=Path, required=True)
    pilot.add_argument("--minimum-groups", type=int, default=300)
    for command in (lock, audit, pilot):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock":
        result = lock_bank(json.loads(args.visible_bank.read_text()), provenance=args.provenance, output=args.output)
    elif args.command == "audit":
        result = audit_bank(args.lock, json.loads(args.hidden_bank.read_text()),
                            output=args.output, pilot_report=args.pilot_report)
    else:
        result = pilot_bank(json.loads(args.development_bank.read_text()), output=args.output,
                            minimum_groups=args.minimum_groups)
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}, sort_keys=True))


if __name__ == "__main__":
    main()
