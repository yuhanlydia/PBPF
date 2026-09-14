from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from .codearc import adapt_codearc
from .evalplus import adapt_evalplus
from .firewall import canonical_json, checked_protocol_provenance, file_hash
from .livecodebench import adapt_livecodebench
from .runbugrun import adapt_runbugrun
from .schema import AdapterResult, PreparedSplit, PublicTask

ADAPTER_VERSION = "1"
_ADAPTERS = {
    "PBPF-RBR": adapt_runbugrun,
    "CodeARC-Replay": adapt_codearc,
    "PBPF-EvalPlus": adapt_evalplus,
    "LiveCodeBench": adapt_livecodebench,
}


def _require_disjoint_roots(generator_root: Path, evaluator_root: Path) -> None:
    generator = generator_root.resolve()
    evaluator = evaluator_root.resolve()
    if generator == evaluator or generator in evaluator.parents or evaluator in generator.parents:
        raise ValueError("generator and evaluator roots must be disjoint process mount boundaries")


def _write_bytes(path: Path, payload: bytes, mode: int) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, mode)


def _write_json(path: Path, value: Any, mode: int) -> None:
    _write_bytes(path, (canonical_json(value) + "\n").encode("utf-8"), mode)


def _write_jsonl(path: Path, values: Iterable[Any], mode: int) -> None:
    payload = "".join(canonical_json(value) + "\n" for value in values).encode("utf-8")
    _write_bytes(path, payload, mode)


def _split_hashes(tasks: tuple[PublicTask, ...]) -> dict[str, str]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_split.setdefault(task.split, []).append(asdict(task))
    return {
        split: hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()
        for split, rows in sorted(by_split.items())
    }


def _inventory(root: Path, *, exclude: set[str]) -> dict[str, str]:
    return {
        child.relative_to(root).as_posix(): file_hash(child)
        for child in sorted(root.rglob("*"))
        if child.is_file() and child.relative_to(root).as_posix() not in exclude
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_temporary_root(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))


def _publish_or_validate(
    temporary: Path,
    target: Path,
    expected_manifest_hash: str,
    *,
    allowed_unmanifested: frozenset[str],
) -> None:
    if target.exists():
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file() or file_hash(manifest_path) != expected_manifest_hash:
            raise ValueError(
                "incomplete prepared root does not match the requested fingerprint/content"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("incomplete prepared root has an invalid manifest") from error
        expected_files = manifest.get("files")
        actual_files = _inventory(
            target, exclude={"manifest.json", *allowed_unmanifested}
        )
        if not isinstance(expected_files, dict) or actual_files != expected_files:
            raise ValueError("incomplete prepared root inventory/checksum mismatch")
        os.chmod(temporary, 0o700)
        shutil.rmtree(temporary)
        return
    os.replace(temporary, target)


def _write_completion(
    evaluator_root: Path,
    *,
    experiment_fingerprint: str,
    generator_manifest_hash: str,
    evaluator_manifest_hash: str,
) -> Path:
    completion = evaluator_root / "publication.json"
    payload = {
        "format": "pbpf-data-publication-v1",
        "experiment_fingerprint": experiment_fingerprint,
        "generator_manifest_hash": generator_manifest_hash,
        "evaluator_manifest_hash": evaluator_manifest_hash,
    }
    os.chmod(evaluator_root, 0o700)
    try:
        if completion.exists():
            existing = json.loads(completion.read_text(encoding="utf-8"))
            if existing != payload:
                raise FileExistsError("data publication completion record conflicts")
            return completion
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{evaluator_root.name}.publication.json.tmp-",
            dir=evaluator_root.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write((canonical_json(payload) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o400)
            try:
                os.link(temporary, completion)
            except FileExistsError:
                existing = json.loads(completion.read_text(encoding="utf-8"))
                if existing != payload:
                    raise FileExistsError("data publication completion record conflicts")
            temporary.unlink()
            _fsync_directory(evaluator_root)
        finally:
            if temporary.exists():
                temporary.unlink()
    finally:
        os.chmod(evaluator_root, 0o500)
    return completion


def _adapt(protocol: str, rows: list[Mapping[str, Any]], options: Mapping[str, Any]) -> AdapterResult:
    try:
        adapter = _ADAPTERS[protocol]
    except KeyError as error:
        raise ValueError(f"unknown adapted protocol: {protocol}") from error
    return adapter(rows, **dict(options))


def prepare_dataset(
    protocol: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    generator_root: str | Path,
    evaluator_root: str | Path,
    configured_dataset_id: str,
    runtime_dataset_id: str,
    configured_revisions: Mapping[str, str],
    runtime_revisions: Mapping[str, str],
    licenses: Mapping[str, str],
    experiment_fingerprint: str,
    adapter_options: Mapping[str, Any] | None = None,
) -> PreparedSplit:
    generator_target = Path(generator_root)
    evaluator_target = Path(evaluator_root)
    _require_disjoint_roots(generator_target, evaluator_target)
    completion_path = evaluator_target / "publication.json"
    if completion_path.exists():
        raise FileExistsError("prepared dataset roots are create-once")
    revisions = checked_protocol_provenance(
        protocol,
        configured_dataset_id,
        runtime_dataset_id,
        configured_revisions,
        runtime_revisions,
    )
    if not experiment_fingerprint:
        raise ValueError("experiment fingerprint is required")
    if not licenses or any(not isinstance(value, str) or not value for value in licenses.values()):
        raise ValueError("non-empty dataset licenses are required")
    materialized = list(rows)
    result = _adapt(protocol, materialized, adapter_options or {})
    counts = {
        "raw": result.raw_count,
        "kept": len(result.public_tasks),
        "excluded": len(result.exclusions),
    }
    split_hashes = _split_hashes(result.public_tasks)
    generator_temp = _make_temporary_root(generator_target)
    evaluator_temp = _make_temporary_root(evaluator_target)
    try:
        public_rows = [asdict(task) for task in result.public_tasks]
        _write_jsonl(generator_temp / "tasks.jsonl", public_rows, 0o444)
        _write_jsonl(
            generator_temp / "prompt_inputs.jsonl",
            (
                {
                    "task_id": task.task_id,
                    "task_text": task.task_text,
                    "candidate_code": task.candidate_code,
                    "visible_tests": [asdict(test) for test in task.visible_tests],
                }
                for task in result.public_tasks
            ),
            0o444,
        )
        _write_jsonl(
            generator_temp / "training_batches.jsonl",
            ({"task": asdict(task), "bounded_visible_outcomes": []} for task in result.public_tasks),
            0o444,
        )
        _write_json(
            generator_temp / "resume.json",
            {
                "experiment_fingerprint": experiment_fingerprint,
                "protocol": protocol,
                "split_hashes": split_hashes,
                "task_ids": [task.task_id for task in result.public_tasks],
            },
            0o444,
        )
        _write_jsonl(
            generator_temp / "events.jsonl",
            ({"event": "prepared", "kept": counts["kept"], "protocol": protocol},),
            0o444,
        )
        _write_json(
            generator_temp / "exclusions.json",
            {
                "counts": counts,
                "reason_histogram": result.exclusion_reasons,
                "records": [
                    {"task_id": task_id, "reason": reason}
                    for task_id, reason in result.exclusions
                ],
            },
            0o444,
        )
        generator_manifest = {
            "format": "pbpf-generator-data-v1",
            "protocol": protocol,
            "dataset_id": runtime_dataset_id,
            "revisions": revisions,
            "licenses": dict(sorted(licenses.items())),
            "adapter_version": ADAPTER_VERSION,
            "experiment_fingerprint": experiment_fingerprint,
            "counts": counts,
            "exclusion_reasons": result.exclusion_reasons,
            "resolved_options": dict(result.resolved_options),
            "split_hashes": split_hashes,
            "files": _inventory(generator_temp, exclude={"manifest.json"}),
        }
        _write_json(generator_temp / "manifest.json", generator_manifest, 0o444)
        generator_manifest_hash = file_hash(generator_temp / "manifest.json")

        _write_jsonl(
            evaluator_temp / "tasks.jsonl",
            (asdict(task) for task in result.evaluator_tasks),
            0o400,
        )
        _write_json(
            evaluator_temp / "preparation.json",
            {"records": list(result.trusted_metadata)},
            0o400,
        )
        evaluator_manifest = {
            "format": "pbpf-evaluator-data-v1",
            "protocol": protocol,
            "dataset_id": runtime_dataset_id,
            "revisions": revisions,
            "licenses": dict(sorted(licenses.items())),
            "adapter_version": ADAPTER_VERSION,
            "experiment_fingerprint": experiment_fingerprint,
            "generator_manifest_hash": generator_manifest_hash,
            "split_hashes": split_hashes,
            "resolved_options": dict(result.resolved_options),
            "files": _inventory(evaluator_temp, exclude={"manifest.json"}),
        }
        _write_json(evaluator_temp / "manifest.json", evaluator_manifest, 0o400)
        evaluator_manifest_hash = file_hash(evaluator_temp / "manifest.json")

        _fsync_directory(generator_temp)
        _fsync_directory(evaluator_temp)
        os.chmod(generator_temp, 0o555)
        os.chmod(evaluator_temp, 0o500)
        _publish_or_validate(
            generator_temp,
            generator_target,
            generator_manifest_hash,
            allowed_unmanifested=frozenset(),
        )
        _publish_or_validate(
            evaluator_temp,
            evaluator_target,
            evaluator_manifest_hash,
            allowed_unmanifested=frozenset({"publication.json"}),
        )
        completion_path = _write_completion(
            evaluator_target,
            experiment_fingerprint=experiment_fingerprint,
            generator_manifest_hash=generator_manifest_hash,
            evaluator_manifest_hash=evaluator_manifest_hash,
        )
        _fsync_directory(generator_target.parent)
        if evaluator_target.parent != generator_target.parent:
            _fsync_directory(evaluator_target.parent)
    finally:
        for temporary in (generator_temp, evaluator_temp):
            if temporary.exists():
                os.chmod(temporary, 0o700)
                shutil.rmtree(temporary)

    return PreparedSplit(
        protocol=protocol,
        public_tasks=result.public_tasks,
        generator_root=generator_target,
        evaluator_root=evaluator_target,
        generator_manifest_path=generator_target / "manifest.json",
        evaluator_manifest_path=evaluator_target / "manifest.json",
        generator_manifest_hash=generator_manifest_hash,
        evaluator_manifest_hash=evaluator_manifest_hash,
        completion_path=completion_path,
        counts=counts,
        exclusion_reasons=result.exclusion_reasons,
    )
