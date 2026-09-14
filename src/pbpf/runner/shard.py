"""Canonical work identities and durable, checksum-validated publications."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def work_id(key):
    if not isinstance(key, (list, tuple, dict)) or not key:
        raise ValueError("work key must be a nonempty structured identity")
    def validate(value):
        if isinstance(value, dict):
            if any(type(k) is not str for k in value):
                raise ValueError("work-key mapping keys must be strings")
            for child in value.values():
                validate(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                validate(child)
        elif type(value) not in (str, int, float) or (type(value) is float and not math.isfinite(value)):
            raise ValueError("work keys reject boolean, null, nonfinite and ambiguous types")
    validate(key)
    return hashlib.sha256(canonical_bytes(key)).hexdigest()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def shard_for(key, count):
    if type(count) is not int or count < 1:
        raise ValueError("positive integer shard count required")
    return int(work_id(key), 16) % count


def atomic_write(path, payload, *, create_once=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if create_once:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_leaf(result):
    marker = json.loads(result.completion.read_bytes())
    if not isinstance(marker, dict):
        raise ValueError("corrupt completion body")
    if marker.get("identity") != result.identity:
        raise ValueError("leaf identity mismatch")
    payload = result.artifact.read_bytes()
    if hashlib.sha256(payload).hexdigest() != marker.get("checksum"):
        raise ValueError("leaf checksum mismatch")
    rows = [json.loads(line) for line in payload.splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("corrupt artifact body")
    actual = [row.get("work_id") for row in rows]
    expected = result.identity["leaf_keys"]
    if (len(rows) != marker.get("row_count") or actual != expected
            or marker.get("expected_keys") != expected or len(actual) != len(set(actual))):
        raise ValueError("leaf expected inventory mismatch")
    for row in rows:
        if work_id(row["key"]) != row["work_id"] or shard_for(row["key"], result.identity["shard_count"]) != result.identity["shard_index"]:
            raise ValueError("corrupt or mis-sharded work key")
    return rows


def merge_leaves(results, *, expected_keys):
    results = tuple(results)
    expected = sorted(work_id(key) for key in expected_keys)
    if not results or len(expected) != len(set(expected)):
        raise ValueError("unique nonempty expected work inventory required")
    identity = {k: v for k, v in results[0].identity.items() if k not in {"shard_index", "leaf_keys"}}
    if identity["expected_keys"] != expected:
        raise ValueError("merge expected inventory mismatch")
    seen_shards, rows = set(), []
    for result in results:
        other = {k: v for k, v in result.identity.items() if k not in {"shard_index", "leaf_keys"}}
        if other != identity or result.identity["shard_index"] in seen_shards:
            raise ValueError("duplicate shard or different leaf identity")
        seen_shards.add(result.identity["shard_index"])
        rows.extend(read_leaf(result))
    if seen_shards != set(range(identity["shard_count"])) or sorted(row["work_id"] for row in rows) != expected:
        raise ValueError("missing, duplicate or extra work inventory")
    return sorted(rows, key=lambda row: row["work_id"])
