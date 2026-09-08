from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_immutable_report(path: str | Path, report: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"artifact is create-once: {target}")
    target.mkdir(parents=True)
    payload = (_canonical(dict(report)) + "\n").encode("utf-8")
    (target / "report.json").write_bytes(payload)
    manifest = {
        "format": "pbpf-report-v1",
        "files": {"report.json": _digest(payload)},
    }
    (target / "manifest.json").write_text(_canonical(manifest) + "\n", encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Any) -> None:
    path.write_text(
        "".join(_canonical(row) + "\n" for row in rows), encoding="utf-8"
    )


def write_run_bundle(
    path: str | Path,
    *,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    bank: Any,
    prediction: Any,
    repair: Any,
) -> None:
    """Write the exact trainer-safe inventory needed to audit one arm."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"artifact is create-once: {target}")
    target.mkdir(parents=True)
    bank.write_trainer(target / "bank")
    _write_json(target / "report.json", dict(report))
    _write_json(target / "config.json", dict(config))
    _write_json(target / "model.json", dict(config["model"]))
    _write_json(target / "data.json", dict(config["data"]))
    _write_json(target / "container.json", dict(config["container"]))
    _write_jsonl(
        target / "programs.jsonl",
        ({"program": program} for program in repair.final_programs),
    )
    _write_jsonl(
        target / "keyed_predictions.jsonl",
        (asdict(item) for item in prediction.keyed_predictions),
    )
    _write_jsonl(target / "events.jsonl", repair.events)
    _write_json(
        target / "ledgers.json",
        {
            "executions": repair.execution_ledger.executions,
            "tokens": asdict(repair.token_ledger),
            "compute": asdict(repair.compute_ledger),
            "model_calls": repair.model_calls,
            "budget": asdict(repair.budget),
        },
    )
    decision_evidence = {
        "stage_gate_status": report.get("stage_gate_status"),
        "advance_stage": False,
        "prediction_report": {
            "bank_hash": prediction.bank_hash,
            "task_count": prediction.task_count,
            "future_nll": prediction.future_nll,
            "brier": prediction.brier,
            "task_scores": [asdict(item) for item in prediction.task_scores],
            "sample_count": len(prediction.keyed_predictions),
        },
        "evaluator_truth_ref": report.get("evaluator_truth_ref"),
        "evaluator_truth_hash": report.get("evaluator_truth_hash"),
        "comparison_reports": None,
        "bootstrap_inputs": None,
        "gate_evidence": None,
    }
    _write_json(target / "decision_evidence.json", decision_evidence)
    files = sorted(
        file.relative_to(target).as_posix()
        for file in target.rglob("*")
        if file.is_file() and file.name != "manifest.json"
    )
    # The nested bank manifest is data in the outer inventory and must be retained.
    files.append("bank/manifest.json")
    files = sorted(set(files))
    manifest = {
        "format": "pbpf-run-bundle-v1",
        "files": {name: _digest((target / name).read_bytes()) for name in files},
        "content_hashes": {
            "config": hashlib.sha256(
                _canonical(dict(config)).encode("utf-8")
            ).hexdigest(),
            "trainer_bank": bank.trainer_content_hash,
            "model": _digest(_canonical(dict(config["model"])).encode("utf-8")),
            "prediction_report": _digest(_canonical(decision_evidence["prediction_report"]).encode("utf-8")),
            "evaluator_truth": report.get("evaluator_truth_hash"),
        },
    }
    _write_json(target / "manifest.json", manifest)


def read_immutable_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") not in {"pbpf-report-v1", "pbpf-run-bundle-v1"}:
        raise ValueError("unsupported report format")
    expected_files = set(manifest.get("files", {}))
    actual_files = {
        file.relative_to(source).as_posix()
        for file in source.rglob("*")
        if file.is_file() and file.relative_to(source).as_posix() != "manifest.json"
    }
    if actual_files != expected_files:
        raise ValueError("artifact file inventory mismatch")
    for filename, expected in manifest["files"].items():
        if _digest((source / filename).read_bytes()) != expected:
            raise ValueError(f"artifact checksum mismatch for {filename}")
    if manifest.get("format") == "pbpf-run-bundle-v1":
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        if _digest(_canonical(config).encode("utf-8")) != manifest.get(
            "content_hashes", {}
        ).get("config"):
            raise ValueError("run config content hash mismatch")
        from .bank import TrajectoryBank

        if TrajectoryBank.read_trainer(source / "bank").trainer_content_hash != manifest.get(
            "content_hashes", {}
        ).get("trainer_bank"):
            raise ValueError("trainer bank content hash mismatch")
        content_hashes = manifest.get("content_hashes", {})
        model = json.loads((source / "model.json").read_text(encoding="utf-8"))
        if _digest(_canonical(model).encode("utf-8")) != content_hashes.get("model"):
            raise ValueError("run model content hash mismatch")
        evidence = json.loads(
            (source / "decision_evidence.json").read_text(encoding="utf-8")
        )
        if _digest(
            _canonical(evidence["prediction_report"]).encode("utf-8")
        ) != content_hashes.get("prediction_report"):
            raise ValueError("prediction report content hash mismatch")
        if evidence.get("evaluator_truth_hash") != content_hashes.get("evaluator_truth"):
            raise ValueError("evaluator truth reference hash mismatch")
    payload = (source / "report.json").read_bytes()
    report = json.loads(payload)
    if not isinstance(report, dict):
        raise ValueError("report payload must be an object")
    return report
