"""Create-once identities with replaceable corrupt/partial matching leaf bodies."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import re

from .shard import atomic_write, canonical_bytes, digest, read_leaf, shard_for, work_id

STAGES = ("doctor", "manifest", "prepare", "finite", "bank", "visible_execute", "train_belief",
          "predict", "gate_b", "repair", "seal", "hidden_eval", "gate_c", "replication", "tables")
DEPENDENCIES = {
    "doctor": (), "manifest": ("doctor",), "prepare": ("manifest",), "finite": ("prepare",),
    "bank": ("prepare", "finite"), "visible_execute": ("prepare", "bank"),
    "train_belief": ("visible_execute", "prepare"), "predict": ("bank", "train_belief", "visible_execute"),
    "gate_b": ("predict",), "repair": ("bank", "visible_execute", "train_belief", "gate_b"),
    "seal": ("repair",), "hidden_eval": ("seal",), "gate_c": ("hidden_eval", "gate_b"),
    "replication": ("gate_c",), "tables": ("replication", "gate_c", "gate_b")}


class GateFailure(RuntimeError):
    exit_code = 20

    def __init__(self, decision):
        self.decision = decision
        super().__init__("Gate B failed: confirmatory repair stopped (exit code 20)")


class StageContext:
    def __init__(self, name, dependencies, config, confirmatory, reason, failed_gate_hash):
        self.name, self.config = name, config
        self.confirmatory, self.reason, self.failed_gate_hash = confirmatory, reason, failed_gate_hash
        self._dependencies, self._consumed = dependencies, set()

    def input(self, name):
        result = self._dependencies[name]
        self._consumed.add(name)
        return result.rows()[0]["value"]


def run_pipeline(*, root, fingerprint, handlers, config, force_after_failed_gate=False,
                 confirmatory=True, reason=None):
    """Execute injected scientific handlers; caches bind all consumed artifacts.

    Model/data backends are injected, not silently replaced by smoke backends.
    A handler receives only its declared dependency artifacts and must consume
    every one. The dispatcher never invokes an arm after hidden evaluation.
    """
    from .aggregate import validate_decision
    if set(handlers) != set(STAGES) or any(not callable(h) for h in handlers.values()):
        raise ValueError("complete executable DAG handler inventory required")
    if not confirmatory and not reason:
        raise ValueError("nonconfirmatory execution requires an explicit reason")
    results, failed_gate_hash = {}, None
    for name in STAGES:
        dependencies = {dep: results[dep] for dep in DEPENDENCIES[name]}
        def invoke(key, checksums, stage_name=name, deps=dependencies):
            context = StageContext(stage_name, deps, config, confirmatory, reason, failed_gate_hash)
            value = handlers[stage_name](context)
            if context._consumed != set(deps):
                raise ValueError(f"{stage_name} handler must consume every declared dependency")
            return value
        results[name] = run_stage(root=root, name=name, fingerprint=fingerprint, keys=[["pipeline", name]],
            dependencies={dep: result.checksum for dep, result in dependencies.items()}, config=config,
            handler=invoke, confirmatory=confirmatory, reason=reason, failed_gate_hash=failed_gate_hash)
        if name == "gate_b":
            decision = results[name].rows()[0]["value"]
            validate_decision(decision)
            if decision.get("gate") != "B":
                raise ValueError("Gate B handler returned wrong decision")
            if not decision["passed"]:
                if not force_after_failed_gate:
                    raise GateFailure(decision)
                confirmatory, reason = False, "forced_after_failed_gate_b"
                failed_gate_hash = decision["decision_hash"]
    return results


@dataclass(frozen=True)
class StageResult:
    name: str
    artifact: Path
    completion: Path
    identity: dict
    checksum: str
    cached: bool = False

    def rows(self):
        return read_leaf(self)


def run_stage(*, root, name, fingerprint, keys, dependencies, config, handler,
              shard_index=0, shard_count=1, confirmatory=True, reason=None, failed_gate_hash=None):
    scientific_config = fingerprint.to_dict()["config"]
    if canonical_bytes(config) != canonical_bytes(scientific_config):
        raise ValueError("active scientific config identity differs from fingerprint")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("invalid stage name")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index")
    keyed = sorted((work_id(key), key) for key in keys)
    if not keyed or len({key for key, _ in keyed}) != len(keyed):
        raise ValueError("unique nonempty work inventory required")
    selected = [(key_id, key) for key_id, key in keyed if shard_for(key, shard_count) == shard_index]
    identity = {"format": "pbpf-stage-v1", "stage": name, "fingerprint": fingerprint.digest,
        "schema": fingerprint.schema, "config": digest(config), "dependencies": dict(dependencies),
        "expected_keys": [key_id for key_id, _ in keyed], "leaf_keys": [key_id for key_id, _ in selected],
        "shard_index": shard_index, "shard_count": shard_count, "confirmatory": confirmatory,
        "reason": reason, "failed_gate_hash": failed_gate_hash}
    identity = json.loads(canonical_bytes(identity))
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"shard-{shard_index:05d}"
    artifact, completion = base.with_suffix(".jsonl"), base.with_suffix(".complete.json")
    identity_path = base.with_suffix(".identity.json")
    with base.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if identity_path.exists():
            if json.loads(identity_path.read_bytes()) != identity:
                raise ValueError("stage identity mismatch; refusing resume")
        else:
            if artifact.exists() or completion.exists():
                raise ValueError("stage identity absent for preexisting artifact")
            atomic_write(identity_path, canonical_bytes(identity), create_once=True)
        existing = StageResult(name, artifact, completion, identity, "", True)
        if completion.exists():
            try:
                marker = json.loads(completion.read_bytes())
                marker_identity = marker.get("identity") if isinstance(marker, dict) else None
            except (ValueError, UnicodeError):
                marker_identity = None
            if marker_identity is not None and marker_identity != identity:
                raise ValueError("completion identity mismatch; refusing resume")
        if artifact.exists() and completion.exists():
            try:
                read_leaf(existing)
                checksum = json.loads(completion.read_bytes())["checksum"]
                return StageResult(name, artifact, completion, identity, checksum, True)
            except (ValueError, KeyError, OSError, UnicodeError):
                pass  # Only this exact identity may replace a damaged leaf.
        rows = []
        for key_id, key in selected:
            value = handler(key, dict(dependencies))
            if not isinstance(value, dict) or not value:
                raise ValueError("stage handler must produce a nonempty artifact record")
            rows.append({"work_id": key_id, "key": key, "value": value,
                         "confirmatory": confirmatory, "reason": reason, "failed_gate_hash": failed_gate_hash})
        payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
        checksum = hashlib.sha256(payload).hexdigest()
        atomic_write(artifact, payload)
        atomic_write(completion, canonical_bytes({"identity": identity, "checksum": checksum,
            "row_count": len(rows), "schema": fingerprint.schema, "expected_keys": identity["leaf_keys"]}))
        result = StageResult(name, artifact, completion, identity, checksum)
        read_leaf(result)
        return result
