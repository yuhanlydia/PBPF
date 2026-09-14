from dataclasses import asdict
import json
import os
import sys
from pathlib import Path

import pytest

from pbpf.arms.base import ArmCandidate, source_hash
from pbpf.data.schema import PublicTask, PublicTest
from pbpf.runner.evaluator import (VisibleVerifier, HiddenEvaluator, seal_predictions,
    seal_final, IsolationCapability, score_predictions)
from pbpf.runner.shard import atomic_write, canonical_bytes, digest

# Offline execution backend; the trust-boundary verifier itself is a real RPC
# process. No candidate code is executed with smoke privileges.
SANDBOX = [sys.executable, "-I", "-c", "import json,sys; r=json.load(sys.stdin); print(json.dumps({'outcome': 'PASS' if \"print('ok')\" in r['source'] else 'WRONG_OUTPUT'}))"]


def candidate(source="print('ok')", slot=0):
    return ArmCandidate("t", slot, 0, source, "actor_sample", "fake", "a"*40, "prompt")


def private_manifest(tmp_path):
    path = tmp_path / "private" / "manifest.json"
    atomic_write(path, canonical_bytes({"format": "pbpf-rpc-evaluator-v1", "tasks": [{"task_id": "t",
        "gold_code": "GOLD_CANARY", "tests": [
            {"test_id": "v", "hidden": False, "source": "", "expected_output": "ok\n"},
            {"test_id": "h", "hidden": True, "source": "# HIDDEN_SOURCE_CANARY", "expected_output": "ok\n"}]}]}))
    return path


def test_visible_verifier_returns_only_allowlisted_feedback_without_final_seal(tmp_path):
    manifest = private_manifest(tmp_path)
    task = PublicTask("t", "fake", "visible", "", (PublicTest("v"),), "test", ("s",))
    verifier = VisibleVerifier(manifest, manifest_hash=digest(json.loads(manifest.read_bytes())), sandbox_command=SANDBOX)
    events = verifier.execute(task, candidate())
    assert events[0].outcome == "PASS"
    assert events[0].feedback == "pass"
    assert events[0].test_id == "v"
    assert not any(secret in json.dumps([asdict(e) for e in events]) for secret in ("GOLD_CANARY", "HIDDEN_SOURCE_CANARY"))
    with pytest.raises(ValueError):
        verifier.execute(PublicTask("t", "fake", "", "", (PublicTest("h"),), "test", ("s",)), candidate())


def test_final_map_allows_duplicate_hashes_and_validates_bytes_before_manifest_open(tmp_path):
    manifest = private_manifest(tmp_path)
    c = candidate()
    mapping = [{"key": ["fake", "fake", "t", 1701, arm], "source_hash": c.source_hash,
                "version_id": c.version_id} for arm in ("pbpf", "self_debug")]
    seal = seal_final(tmp_path / "seal.json", mapping=mapping, expected_keys=[r["key"] for r in mapping],
        fingerprint="f"*64, code={c.source_hash: c.source})
    evaluator = HiddenEvaluator(manifest, fingerprint="f"*64, output_root=tmp_path / "results",
        mode="smoke", manifest_hash=digest(json.loads(manifest.read_bytes())), sandbox_command=SANDBOX)
    result = evaluator.evaluate(seal, code={c.source_hash: c.source})
    assert len(result["results"]) == 2
    assert result["confirmatory"] is False
    assert {r["outcome"] for r in result["results"]} == {"PASS"}
    assert result["worker_pid"] != os.getpid()
    before = sorted(p.read_bytes() for p in (tmp_path / "results").glob("*.json"))
    assert evaluator.evaluate(seal, code={c.source_hash: c.source})["results"] == result["results"]
    assert sorted(p.read_bytes() for p in (tmp_path / "results").glob("*.json")) == before
    manifest.unlink()
    with pytest.raises(ValueError, match="code"):
        evaluator.evaluate(seal, code={c.source_hash: "print('changed')"})
    with pytest.raises(FileExistsError):
        seal_final(seal, mapping=mapping, expected_keys=[r["key"] for r in mapping],
            fingerprint="f"*64, code={c.source_hash: c.source})


def test_formal_rejects_same_uid_or_unverified_isolation(tmp_path):
    manifest = private_manifest(tmp_path)
    with pytest.raises(ValueError, match="verified"):
        HiddenEvaluator(manifest, fingerprint="f"*64, output_root=tmp_path / "result", mode="formal",
            manifest_hash=digest(json.loads(manifest.read_bytes())))
    with pytest.raises(ValueError, match="distinct UID"):
        IsolationCapability.verify(generator_pid=os.getpid(), private_root=manifest.parent,
                                   public_root=tmp_path / "public")


def test_prediction_seal_binds_keyed_probabilities_model_lock_before_future_scoring(tmp_path):
    manifest = private_manifest(tmp_path)
    c = candidate()
    prediction = {"key": ["t", 1701, "pbpf", c.version_id, "h"],
                  "source_hash": c.source_hash, "probabilities": [.8, .05, .05, .05, .05]}
    seal = seal_predictions(tmp_path / "predictions.json", predictions=[prediction],
        expected_keys=[prediction["key"]], candidate_inventory=[c.source_hash],
        model_lock="a"*64, fingerprint="f"*64)
    result = score_predictions(manifest, seal, code={c.source_hash: c.source}, fingerprint="f"*64,
        model_lock="a"*64, manifest_hash=digest(json.loads(manifest.read_bytes())), sandbox_command=SANDBOX)
    assert result["scores"][0]["nll"] == pytest.approx(-__import__("math").log(.8))
    assert result["scores"][0]["brier"] == pytest.approx(.05)
    assert result["worker_pid"] != os.getpid()
    with pytest.raises(ValueError, match="model lock"):
        score_predictions(manifest, seal, code={c.source_hash: c.source}, fingerprint="f"*64,
            model_lock="b"*64, manifest_hash=digest(json.loads(manifest.read_bytes())))


def test_tampered_seal_inventory_rejected_even_with_recomputed_checksum(tmp_path):
    manifest = private_manifest(tmp_path)
    c = candidate()
    mapping = [{"key": ["fake", "fake", "t", 1701, "pbpf"], "source_hash": c.source_hash, "version_id": c.version_id}]
    seal = seal_final(tmp_path / "seal.json", mapping=mapping, expected_keys=[mapping[0]["key"]], fingerprint="f"*64,
                      code={c.source_hash: c.source})
    row = json.loads(seal.read_bytes())
    row["selections"].append(row["selections"][0])
    row["seal_hash"] = digest({k: v for k, v in row.items() if k != "seal_hash"})
    seal.write_bytes(canonical_bytes(row))
    evaluator = HiddenEvaluator(manifest, fingerprint="f"*64, output_root=tmp_path / "out", mode="smoke",
        manifest_hash=digest(json.loads(manifest.read_bytes())), sandbox_command=SANDBOX)
    with pytest.raises(ValueError, match="inventory"):
        evaluator.evaluate(seal, code={c.source_hash: c.source})


def test_concurrent_hidden_evaluation_has_one_accepted_receipt_per_work(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    manifest = private_manifest(tmp_path)
    c = candidate()
    mapping = [{"key": ["fake", "fake", "t", 1701, "pbpf"], "source_hash": c.source_hash, "version_id": c.version_id}]
    seal = seal_final(tmp_path / "seal.json", mapping=mapping, expected_keys=[mapping[0]["key"]], fingerprint="f"*64,
                      code={c.source_hash: c.source})
    evaluator = HiddenEvaluator(manifest, fingerprint="f"*64, output_root=tmp_path / "out", mode="smoke",
        manifest_hash=digest(json.loads(manifest.read_bytes())), sandbox_command=SANDBOX)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: evaluator.evaluate(seal, code={c.source_hash: c.source}), range(2)))
    assert results[0]["results"] == results[1]["results"]
    assert len(list((tmp_path / "out").glob("*.json"))) == 1
    changed = HiddenEvaluator(manifest, fingerprint="f"*64, output_root=tmp_path / "out", mode="smoke",
        manifest_hash=digest(json.loads(manifest.read_bytes())), sandbox_command=SANDBOX + ["changed-backend"])
    with pytest.raises(ValueError):
        changed.evaluate(seal, code={c.source_hash: c.source})


def test_failed_verifier_attempt_is_charged_and_hidden_cache_is_not(tmp_path):
    from pbpf.runner.budget import BudgetLedger
    manifest = private_manifest(tmp_path)
    task = PublicTask("t", "fake", "", "", (PublicTest("v"),), "test", ("s",))
    failed = [sys.executable, "-I", "-c", "raise SystemExit(1)"]
    verifier = VisibleVerifier(manifest, manifest_hash=digest(json.loads(manifest.read_bytes())),
        sandbox_command=failed, budget_root=tmp_path / "budget")
    with pytest.raises(ValueError):
        verifier.execute(task, candidate())
    assert BudgetLedger(tmp_path / "budget").totals()["visible_verifier_suites"] == 1
    assert BudgetLedger(tmp_path / "budget").totals()["visible_verifier_cases"] == 1
    c = candidate()
    mapping = [{"key": ["fake", "fake", "t", 1701, "pbpf"], "source_hash": c.source_hash, "version_id": c.version_id}]
    seal = seal_final(tmp_path / "seal.json", mapping=mapping, expected_keys=[mapping[0]["key"]], fingerprint="f"*64,
        code={c.source_hash: c.source})
    evaluator = HiddenEvaluator(manifest, fingerprint="f"*64, output_root=tmp_path / "out", mode="smoke",
        manifest_hash=digest(json.loads(manifest.read_bytes())), sandbox_command=SANDBOX, budget_root=tmp_path / "budget")
    evaluator.evaluate(seal, code={c.source_hash: c.source})
    evaluator.evaluate(seal, code={c.source_hash: c.source})
    assert BudgetLedger(tmp_path / "budget").totals()["hidden_verifier_suites"] == 1
    assert BudgetLedger(tmp_path / "budget").totals()["hidden_verifier_cases"] == 1


def test_future_scoring_formal_mode_requires_verified_isolation_before_private_read(tmp_path):
    with pytest.raises(ValueError, match="verified"):
        score_predictions(tmp_path / "absent.json", tmp_path / "absent-seal.json", code={},
            fingerprint="f"*64, model_lock="a"*64, manifest_hash="b"*64, mode="formal")
