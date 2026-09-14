from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

from .schema import EvaluatorTask, EvaluatorTest, LoadedEvaluator

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PROTOCOL_PROVENANCE: dict[str, tuple[str, dict[str, str]]] = {
    "PBPF-RBR": (
        "giganticode/run_bug_run",
        {"repository": "374251a9d65410f37e1136049cb7ff5dcca3d0ae"},
    ),
    "CodeARC-Replay": (
        "anjiangwei/CodeARC-Problems+anjiangwei/CodeARC-Invocations",
        {
            "repository": "32f2a1e1ba3bdf8a7d5057aacbd665edaf441ce0",
            "problems_dataset": "d2ee0ad485724668eabb893a965c681a9ce78031",
            "invocations_dataset": "119512573bf96148cbb95e99cf7cdd21c06153e4",
        },
    ),
    "PBPF-EvalPlus": (
        "evalplus/humanevalplus+evalplus/mbppplus",
        {"repository": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"},
    ),
    "LiveCodeBench": (
        "livecodebench/code_generation_lite",
        {
            "repository": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
            "lite_dataset": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        },
    ),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path, expected_format: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != expected_format:
        raise ValueError(f"unsupported or invalid {expected_format} manifest")
    root = path.parent
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError("manifest file inventory is missing")
    ignored = {path.name, "publication.json"}
    actual_files = {
        child.relative_to(root).as_posix()
        for child in root.rglob("*")
        if child.is_file() and child.relative_to(root).as_posix() not in ignored
    }
    if actual_files != set(expected_files):
        raise ValueError("manifest file inventory mismatch")
    for name, expected in expected_files.items():
        if not isinstance(expected, str) or file_hash(root / name) != expected:
            raise ValueError(f"checksum mismatch for {name}")
    return manifest


def seal_candidates(
    path: str | Path,
    candidate_hashes: Iterable[str],
    *,
    experiment_fingerprint: str,
    expected_experiment_fingerprint: str | None = None,
) -> Path:
    """Atomically create the only input that authorizes hidden evaluation."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"candidate seal is create-once: {target}")
    if not experiment_fingerprint:
        raise ValueError("experiment fingerprint is required")
    if (
        expected_experiment_fingerprint is not None
        and experiment_fingerprint != expected_experiment_fingerprint
    ):
        raise ValueError("experiment fingerprint mismatch")
    hashes = tuple(sorted(candidate_hashes))
    if not hashes or len(hashes) != len(set(hashes)) or any(
        not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in hashes
    ):
        raise ValueError("candidate hashes must be unique lowercase SHA-256 values")
    payload = {
        "format": "pbpf-candidate-seal-v1",
        "experiment_fingerprint": experiment_fingerprint,
        "candidate_hashes": list(hashes),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((canonical_json(payload) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f"candidate seal is create-once: {target}") from error
        temporary.unlink()
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _load_candidate_seal(path: Path, fingerprint: str) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"sealed candidate manifest is required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"format", "experiment_fingerprint", "candidate_hashes"}:
        raise ValueError("invalid sealed candidate manifest fields")
    if payload["format"] != "pbpf-candidate-seal-v1":
        raise ValueError("invalid sealed candidate manifest format")
    if payload["experiment_fingerprint"] != fingerprint:
        raise ValueError("experiment fingerprint mismatch")
    hashes = payload["candidate_hashes"]
    if (
        not isinstance(hashes, list)
        or not hashes
        or hashes != sorted(hashes)
        or len(hashes) != len(set(hashes))
        or any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in hashes)
    ):
        raise ValueError("invalid sealed candidate hashes")
    return tuple(hashes)


def load_evaluator_tasks(
    evaluator_manifest_path: str | Path,
    sealed_candidates_path: str | Path,
    *,
    experiment_fingerprint: str,
) -> LoadedEvaluator:
    """Load secrets only after candidate code has crossed the one-way hash seal."""

    hashes = _load_candidate_seal(Path(sealed_candidates_path), experiment_fingerprint)
    manifest_path = Path(evaluator_manifest_path)
    publication_path = manifest_path.parent / "publication.json"
    if not publication_path.is_file():
        raise FileNotFoundError("completed generator/evaluator publication is required")
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    if set(publication) != {
        "format",
        "experiment_fingerprint",
        "generator_manifest_hash",
        "evaluator_manifest_hash",
    } or publication.get("format") != "pbpf-data-publication-v1":
        raise ValueError("invalid data publication completion record")
    if publication.get("experiment_fingerprint") != experiment_fingerprint:
        raise ValueError("experiment fingerprint mismatch")
    if publication.get("evaluator_manifest_hash") != file_hash(manifest_path):
        raise ValueError("evaluator manifest does not match completion record")
    manifest = _read_manifest(manifest_path, "pbpf-evaluator-data-v1")
    if manifest.get("experiment_fingerprint") != experiment_fingerprint:
        raise ValueError("experiment fingerprint mismatch")
    if publication.get("generator_manifest_hash") != manifest.get(
        "generator_manifest_hash"
    ):
        raise ValueError("generator manifest does not match completion record")
    rows = [
        json.loads(line)
        for line in (manifest_path.parent / "tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    tasks = tuple(
        EvaluatorTask(
            task_id=row["task_id"],
            protocol=row["protocol"],
            tests=tuple(EvaluatorTest(**case) for case in row["tests"]),
            gold_code=row.get("gold_code"),
            gold_patch=row.get("gold_patch"),
            hidden_outcomes=tuple(
                (str(key), str(value)) for key, value in row.get("hidden_outcomes", ())
            ),
        )
        for row in rows
    )
    return LoadedEvaluator(tasks, hashes, experiment_fingerprint)


def checked_revisions(
    configured_dataset_id: str,
    runtime_dataset_id: str,
    configured_revisions: Mapping[str, str],
    runtime_revisions: Mapping[str, str],
) -> dict[str, str]:
    if not configured_dataset_id or configured_dataset_id != runtime_dataset_id:
        raise ValueError("configured dataset id does not match runtime dataset id")
    if not configured_revisions or set(configured_revisions) != set(runtime_revisions):
        raise ValueError("configured revision inventory does not match runtime revision inventory")
    resolved: dict[str, str] = {}
    for name in sorted(configured_revisions):
        configured = configured_revisions[name]
        runtime = runtime_revisions[name]
        if not isinstance(configured, str) or not re.fullmatch(r"[0-9a-f]{40}", configured):
            raise ValueError("dataset revisions must be full immutable 40-character commits")
        if not isinstance(runtime, str) or not re.fullmatch(r"[0-9a-f]{40}", runtime):
            raise ValueError("runtime revisions must be full immutable 40-character commits")
        if configured != runtime:
            raise ValueError(f"configured revision mismatch for {name}")
        resolved[name] = runtime
    return resolved


def checked_protocol_provenance(
    protocol: str,
    configured_dataset_id: str,
    runtime_dataset_id: str,
    configured_revisions: Mapping[str, str],
    runtime_revisions: Mapping[str, str],
) -> dict[str, str]:
    resolved = checked_revisions(
        configured_dataset_id,
        runtime_dataset_id,
        configured_revisions,
        runtime_revisions,
    )
    try:
        authoritative_id, authoritative_revisions = PROTOCOL_PROVENANCE[protocol]
    except KeyError as error:
        raise ValueError(f"unknown adapted protocol: {protocol}") from error
    if runtime_dataset_id != authoritative_id:
        raise ValueError(
            f"{protocol} requires authoritative dataset id {authoritative_id}"
        )
    if set(resolved) != set(authoritative_revisions):
        raise ValueError(
            f"{protocol} requires revisions {sorted(authoritative_revisions)}"
        )
    if resolved != authoritative_revisions:
        raise ValueError(f"{protocol} runtime revisions are not authoritative pins")
    return resolved
