import json

import pytest

from pbpf.runner.stage import STAGES, GateFailure, run_pipeline
from pbpf.runner.repair import OfflineExperiment
from pbpf.runner.shard import digest
from test_resume import fingerprint


def artifact_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file() and p.suffix != ".lock"}


def test_fake_offline_dag_executes_all_consuming_handlers_with_canary_firewall_and_resume(tmp_path):
    backend = OfflineExperiment(tmp_path / "trusted", fingerprint=fingerprint())
    result = run_pipeline(root=tmp_path / "run", fingerprint=fingerprint(), handlers=backend.handlers(),
        config={"mode": "smoke"}, confirmatory=False, reason="same_uid_smoke")
    assert tuple(result) == STAGES
    assert backend.calls == list(STAGES)
    assert result["tables"].rows()[0]["value"]["main_table_eligible"] is False
    assert result["gate_b"].rows()[0]["value"]["passed"] is True
    assert result["gate_c"].rows()[0]["value"]["passed"] is True
    assert backend.actor_requests == 60  # 3 seeds * (8 shared roots + 3*4 repair)
    assert len(backend.arm_inputs) > 0
    arm_data = json.dumps(backend.arm_inputs)
    public_artifacts = b"\n".join(artifact_bytes(tmp_path / "run").values()).decode()
    for secret in ("HIDDEN_SOURCE_CANARY", "HIDDEN_OUTPUT_CANARY", "GOLD_CODE_CANARY", "GOLD_PATCH_CANARY"):
        assert secret not in arm_data
        assert secret not in public_artifacts
    assert all("hidden_outcome" not in row for row in backend.arm_inputs)
    assert len(result["hidden_eval"].rows()[0]["value"]["results"]) == 9
    before = artifact_bytes(tmp_path / "run")
    resumed = run_pipeline(root=tmp_path / "run", fingerprint=fingerprint(), handlers=backend.handlers(),
        config={"mode": "smoke"}, confirmatory=False, reason="same_uid_smoke")
    assert all(value.cached for value in resumed.values())
    assert before == artifact_bytes(tmp_path / "run")
    assert backend.calls == list(STAGES)
    result["bank"].artifact.write_bytes(b"interrupted")
    run_pipeline(root=tmp_path / "run", fingerprint=fingerprint(), handlers=backend.handlers(),
        config={"mode": "smoke"}, confirmatory=False, reason="same_uid_smoke")
    assert backend.ledger.totals()["full_decodes"] == 84


def test_failed_b_has_immutable_decision_zero_repair_then_force_is_sticky(tmp_path):
    backend = OfflineExperiment(tmp_path / "trusted", fingerprint=fingerprint(), fail_gate_b=True)
    args = dict(root=tmp_path / "run", fingerprint=fingerprint(), handlers=backend.handlers(),
                config={"mode": "smoke"}, confirmatory=False, reason="same_uid_smoke")
    with pytest.raises(GateFailure) as error:
        run_pipeline(**args)
    assert error.value.exit_code == 20
    assert "repair" not in backend.calls
    assert backend.actor_requests == 24
    assert not (tmp_path / "run" / "repair").exists()
    decision = (tmp_path / "run" / "gate_b" / "shard-00000.jsonl").read_bytes()
    result = run_pipeline(**args, force_after_failed_gate=True)
    assert decision == (tmp_path / "run" / "gate_b" / "shard-00000.jsonl").read_bytes()
    failed_hash = result["gate_b"].rows()[0]["value"]["decision_hash"]
    for stage in STAGES[STAGES.index("repair"):]:
        assert result[stage].identity["confirmatory"] is False
        assert result[stage].identity["failed_gate_hash"] == failed_hash
        assert result[stage].identity["reason"] == "forced_after_failed_gate_b"


def test_force_on_passing_gate_does_not_relabel_and_handlers_must_consume_inputs(tmp_path):
    from pbpf.runner.aggregate import require_main_tables
    handlers = {name: (lambda context: {"artifact": "ignored"}) for name in STAGES}
    with pytest.raises(ValueError, match="consume"):
        run_pipeline(root=tmp_path / "empty", fingerprint=fingerprint(), handlers=handlers, config={"mode": "smoke"})
    with pytest.raises(ValueError, match="nonconfirmatory"):
        require_main_tables([{"confirmatory": False}])


def test_locked_eval_training_labels_rejected():
    from pbpf.runner.repair import training_records
    with pytest.raises(ValueError, match="locked"):
        training_records([{"split": "test", "labels": ["PASS"]}])
    assert training_records([{"split": "train", "labels": ["PASS"]}])
