"""Validate generation inventories and enforce pre-hidden CodeARC bank locks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bank(root):
    root = Path(root)
    complete = json.loads((root / "complete.json").read_bytes())
    if file_sha(root / "run.json") != complete["run_sha256"]:
        raise ValueError("bank run identity checksum mismatch")
    run = json.loads((root / "run.json").read_bytes())
    files = complete["files"]
    expected_files = [task.replace("/", "-") + ".json" for task in run["task_ids"]]
    if set(files) != set(expected_files) or len(files) != len(run["task_ids"]):
        raise ValueError("completion inventory differs from pre-generation inventory")
    rows = []
    for name, task_id, source in zip(expected_files, run["task_ids"], run["source_component_ids"], strict=True):
        if Path(name).name != name or file_sha(root / name) != files[name]:
            raise ValueError("bank artifact checksum mismatch")
        row = json.loads((root / name).read_bytes())
        candidates = row["candidates"]
        ids = [c["candidate_id"] for c in candidates]
        if (row["task_id"] != task_id or row["source_component_id"] != source
                or row["split"] != run["split"] or len(candidates) != run["candidates"]
                or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids)):
            raise ValueError("candidate inventory differs from pre-generation inventory")
        rows.append(row)
    if len({r["source_component_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source components in generation bank")
    return run, rows, file_sha(root / "complete.json")


def verify_hidden_lock(path, bank, bank_digest):
    """Check the entire bank binding before opening any evaluator task data."""
    payload = json.loads(Path(path).read_bytes())
    digest = payload.pop("content_sha256", None)
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != actual or payload.get("schema") != "apbpf-hard-bank-lock-v2":
        raise ValueError("invalid pre-hidden lock checksum or schema")
    if payload["population_lock"]["provenance"] != "codearc-bank-sha256:" + bank_digest:
        raise ValueError("pre-hidden lock binds a different candidate bank")
    groups = payload["groups"]
    if len(groups) != len(bank):
        raise ValueError("pre-hidden lock must cover the full generated bank")
    for group, row in zip(groups, bank, strict=True):
        if (group["group_id"] != row["task_id"] or group["split"] != row["split"]
                or group["source_component_id"] != row["source_component_id"]
                or group["candidate_ids"] != [c["candidate_id"] for c in row["candidates"]]):
            raise ValueError("pre-hidden lock inventory mismatch")
        outcomes = group["visible_outcomes"]
        if (len(outcomes) != len(row["candidates"])
                or any(len(v) != 4 or any(x not in (0, 1) for x in v) for v in outcomes)):
            raise ValueError("pre-hidden lock requires four visible outcomes per candidate")
    return file_sha(path)


def select_tests(task, phase):
    if phase == "visible":
        tests = task["visible_tests"]
        expected = [str(i) for i in range(4)]
    else:
        if phase == "all" and task["split"] == "primary":
            raise ValueError("primary tests require separate visible and locked hidden phases")
        tests = task["tests"][4:] if phase == "hidden" else task["tests"]
        expected = [str(i) for i in range(4, 10)] if phase == "hidden" else [str(i) for i in range(10)]
    if [t["id"] for t in tests] != expected:
        raise ValueError("test IDs do not match the fixed replay protocol")
    return tests
