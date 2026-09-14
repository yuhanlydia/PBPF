import json
import math
import os
from pathlib import Path
import sys

import pytest

from pbpf.runner.aggregate import aggregate_stage_b, aggregate_stage_c
from pbpf.runner.budget import Usage, validate_matched
from pbpf.runner.doctor import doctor, validate_calibration
from pbpf.runner.evaluator import HiddenEvaluator, _rpc
from pbpf.runner.shard import canonical_bytes, digest, work_id
from pbpf.runner.stage import run_stage
from pbpf.runner.stats import ClusterPlan, PlanLock, formal_interval
from test_evaluator import private_manifest, candidate, SANDBOX
from test_resume import fingerprint
from test_stats import passing_metrics


@pytest.mark.parametrize("key", [{1: "x"}, {"nested": {1: "x"}}, [True], [float("nan")], [float("inf")]])
def test_work_key_rejects_ambiguous_or_noncanonical_types(key):
    with pytest.raises(ValueError):
        work_id(key)


@pytest.mark.parametrize("domain,value", [("code", {"tree": "main"}), ("datasets", {"data": {"revision": "main"}}),
    ("prompt", {"hash": "main"}), ("locks", {"brier_margin": .01}), ("container_digest", "sha256:"+"z"*64)])
def test_complete_scientific_identity_rejects_weak_domains(domain, value):
    with pytest.raises(ValueError):
        fingerprint(**{domain: value})


def test_active_scientific_config_cannot_differ_from_fingerprint(tmp_path):
    with pytest.raises(ValueError, match="config"):
        run_stage(root=tmp_path, name="bank", fingerprint=fingerprint(config={"temperature": .8}), keys=[[1]],
            dependencies={}, config={"temperature": .9}, handler=lambda k, d: {"result": 1})


def test_marker_identity_is_authoritative_without_payload_and_list_body_is_recoverable(tmp_path):
    args = dict(root=tmp_path, name="bank", fingerprint=fingerprint(), keys=[[1]],
        dependencies={}, config=fingerprint().to_dict()["config"], handler=lambda k, d: {"result": 1})
    result = run_stage(**args)
    result.artifact.unlink()
    marker = json.loads(result.completion.read_bytes())
    marker["identity"]["fingerprint"] = "0"*64
    result.completion.write_bytes(canonical_bytes(marker))
    with pytest.raises(ValueError, match="identity"):
        run_stage(**args)
    result.completion.write_bytes(b"[]")
    assert not run_stage(**args).cached


def test_common_plan_rejects_changed_valid_draws_with_same_seed(tmp_path):
    plan = ClusterPlan.create(task_keys=["a", "b"], source_clusters=["a", "b"], strata=["s", "s"], seed=7, upstream_identity="a"*64)
    path = tmp_path / "plan.json"
    plan.save(path)
    row = json.loads(path.read_bytes())
    row["draws"] = [["a", "a"]]*10000
    if "digest" in row:
        row["digest"] = digest({k: v for k, v in row.items() if k != "digest"})
    path.write_bytes(canonical_bytes(row))
    with pytest.raises(ValueError):
        ClusterPlan.load(path, lock=PlanLock.from_plan(plan))


def test_zero_individual_nll_and_infinite_aggregate_policy():
    plan = ClusterPlan.create(task_keys=["a", "b"], source_clusters=["a", "a"], strata=["s", "s"], seed=7, upstream_identity="a"*64)
    result = formal_interval([0., 2.], [1., 3.], plan, statistic="log_nll_ratio")
    assert result.estimate == pytest.approx(math.log(2))
    result = formal_interval([0., 0.], [1., 3.], plan, statistic="log_nll_ratio")
    assert result.estimate == math.inf and result.lower == math.inf
    with pytest.raises(ValueError, match="undefined"):
        formal_interval([0., 0.], [0., 0.], plan, statistic="log_nll_ratio")


def test_c_uses_its_own_locked_plan_and_accepts_exact_point_boundary():
    bplan = ClusterPlan.create(task_keys=["b"], source_clusters=["b"], strata=["s"], seed=7, upstream_identity="a"*64)
    cplan = ClusterPlan.create(task_keys=["c"], source_clusters=["c"], strata=["replication"], seed=8, upstream_identity="b"*64)
    metrics = {name: {metric: values[:1] for metric, values in row.items()} for name, row in passing_metrics().items()}
    b = aggregate_stage_b(metrics=metrics, plan=bplan, comparator="det", brier_margin=.011,
        selection_hash="a"*64, prediction_seal_hash="b"*64)
    c = aggregate_stage_c(metrics={"pbpf": [.7], "self_debug": [.67], "det": [.67]}, plan=cplan,
        comparator="det", stage_b=b, expected_stage_b_hash=b["decision_hash"], endpoint="final_selected_pass1")
    assert c["passed"]


@pytest.mark.parametrize("change", [{"max_new_tokens": True}, {"max_new_tokens": float("nan")}, {"max_new_tokens": 1.5}])
def test_doctor_rejects_invalid_caps(change):
    with pytest.raises(ValueError):
        doctor([dict(dataset="d", model="m", tasks=1, arms=1, visible_cases=1, hidden_cases=1, seeds=3)], **change)


def test_calibration_counts_and_matched_visible_inventory_are_strict():
    with pytest.raises(ValueError):
        validate_calibration(expected_work=100.5, failed_work=1, projected_cost=1, measured_cost=1, allocated_memory=1, available_memory=2)
    row = dict(bank_hash="a"*64, roots=8, genuine=True, arm="pbpf", actor_requests=4, full_decodes=4,
        multiplicities=[True]*4, visible_opportunities=[[]]*4, max_new_tokens=1024, label="matched_four_decodes")
    with pytest.raises(ValueError):
        validate_matched([row])


def test_mutable_client_mode_and_raw_stdio_cannot_mint_formal(tmp_path):
    manifest = private_manifest(tmp_path)
    evaluator = HiddenEvaluator(manifest, fingerprint="f"*64, output_root=tmp_path / "out", mode="smoke",
        manifest_hash=digest(json.loads(manifest.read_bytes())), sandbox_command=SANDBOX)
    with pytest.raises((AttributeError, TypeError)):
        evaluator.mode = "formal"
    with pytest.raises(ValueError):
        _rpc({"operation": "visible", "confirmatory": True, "manifest": str(manifest),
            "manifest_hash": digest(json.loads(manifest.read_bytes())), "task_id": "t", "test_ids": ["v"],
            "source": candidate().source, "sandbox_command": SANDBOX, "work": "x"})


def test_backend_symlink_retarget_invalidates_dispatch_and_accepted_cache(tmp_path):
    from pbpf.runner.evaluator import seal_final
    script = tmp_path / "backend.py"
    script.write_text("import json; print(json.dumps({'outcome':'PASS'}))")
    other = tmp_path / "other.py"
    other.write_text("import json; print(json.dumps({'outcome':'WRONG_OUTPUT'}))")
    link = tmp_path / "current.py"
    link.symlink_to(script)
    manifest = private_manifest(tmp_path)
    c = candidate()
    mapping = [{"key": ["fake", "fake", "t", 1701, "pbpf"], "source_hash": c.source_hash, "version_id": c.version_id}]
    seal = seal_final(tmp_path / "seal.json", mapping=mapping, expected_keys=[mapping[0]["key"]], fingerprint="f"*64, code={c.source_hash:c.source})
    evaluator = HiddenEvaluator(manifest, fingerprint="f"*64, output_root=tmp_path / "out", mode="smoke",
        manifest_hash=digest(json.loads(manifest.read_bytes())), sandbox_command=[sys.executable, str(link)])
    evaluator.evaluate(seal, code={c.source_hash:c.source})
    link.unlink()
    link.symlink_to(other)
    with pytest.raises(ValueError, match="backend"):
        evaluator.evaluate(seal, code={c.source_hash:c.source})


def test_rpc_deadline_applies_to_individual_cases_not_whole_inventory(tmp_path, monkeypatch):
    import pbpf.runner.evaluator as module
    from pbpf.runner.evaluator import seal_final
    manifest = private_manifest(tmp_path)
    row = json.loads(manifest.read_bytes())
    row["tasks"][0]["tests"] = [dict(row["tasks"][0]["tests"][1], test_id=f"h{i}") for i in range(3)]
    manifest.write_bytes(canonical_bytes(row))
    c = candidate()
    mapping = [{"key": ["fake", "fake", "t", 1701, "pbpf"], "source_hash": c.source_hash, "version_id": c.version_id}]
    seal = seal_final(tmp_path / "seal.json", mapping=mapping, expected_keys=[mapping[0]["key"]], fingerprint="f"*64, code={c.source_hash:c.source})
    slow = [sys.executable, "-I", "-c", "import time,json; time.sleep(.15); print(json.dumps({'outcome':'PASS'}))"]
    calls = []
    original_rpc = module._rpc
    def traced_rpc(request):
        calls.append(request)
        return original_rpc(request)
    monkeypatch.setattr(module, "_rpc", traced_rpc)
    evaluator = HiddenEvaluator(manifest, fingerprint="f"*64, output_root=tmp_path / "out", mode="smoke",
        manifest_hash=digest(row), sandbox_command=slow, case_seconds=.5, rpc_overhead_seconds=2.)
    assert evaluator.evaluate(seal, code={c.source_hash:c.source})["results"][0]["outcome"] == "PASS"
    case_calls = [request for request in calls if request["action"] == "case"]
    assert [request["case_id"] for request in case_calls] == ["h0", "h1", "h2"]
    assert all(request["case_seconds"] == .5 and request["rpc_overhead_seconds"] == 2. for request in calls)


def test_over_limit_unit_kills_its_process_group(tmp_path):
    from pbpf.runner.evaluator import _bounded_process
    child_file = tmp_path / "child.pid"
    script = ("import os,time,pathlib; child=os.fork(); "
        f"pathlib.Path({str(child_file)!r}).write_text(str(os.getpid())) if child == 0 else None; "
        "time.sleep(30)")
    with pytest.raises(ValueError, match="deadline"):
        _bounded_process([sys.executable, "-I", "-c", script], payload=b"", env={"PATH": os.defpath}, timeout=1.)
    child_pid = int(child_file.read_text())
    status = Path(f"/proc/{child_pid}/stat")
    # An adopted zombie can remain until init reaps it, but cannot execute.
    assert not status.exists() or status.read_text().rsplit(")", 1)[1].split()[0] == "Z"


def test_runner_meter_reserves_failed_and_multiple_decodes_with_native_telemetry(tmp_path):
    from pbpf.runner.budget import BudgetLedger, MeteredActor
    from pbpf.runner.repair import RootRequest
    from pbpf.data.schema import PublicTask, PublicTest
    class Actor:
        def generate_root(self, request):
            raise RuntimeError("backend failure")
        def native_usage(self):
            return Usage(input_token_ids=(11,), output_token_ids=(22, 23), wall_seconds=.1)
    ledger = BudgetLedger(tmp_path)
    actor = MeteredActor(Actor(), ledger=ledger, work_prefix="bank")
    task = PublicTask("t", "fake", "", "", (PublicTest("v"),), "test", ("s",))
    for _ in range(2):
        with pytest.raises(RuntimeError):
            actor.generate_root(RootRequest(task, 1701, 0, num_return_sequences=8))
    assert ledger.totals()["actor_requests"] == 2
    assert ledger.totals()["full_decodes"] == 16
    assert ledger.totals()["input_tokens"] == 2
    assert ledger.totals()["output_tokens"] == 4


def test_external_authority_rejects_forgery_same_uid_expiry_and_replay(tmp_path, monkeypatch):
    """Crypto/replay unit fixture, not a positive real-OS formal integration."""
    import hashlib
    import hmac
    import time
    from pbpf.runner.evaluator import TrustedEvaluatorService, ExternalSandboxSpec
    key = tmp_path / "authority.key"
    key.write_bytes(b"k"*32)
    key.chmod(0o400)
    private, public, nonces = (tmp_path / n for n in ("private", "public", "nonces"))
    for directory in (private, public, nonces):
        directory.mkdir(mode=0o700)
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(SANDBOX[-1])
    spec = ExternalSandboxSpec(command=("/usr/bin/python3", "-I", str(wrapper)),
        dependency_files=("/usr/bin/python3", str(wrapper)), container_digest="sha256:"+"a"*64)
    service = TrustedEvaluatorService(key_file=key, private_root=private, public_root=public, nonce_root=nonces, sandbox_spec=spec)
    request = {"manifest": str(private / "missing.json"), "backend_identity": service.backend.to_dict()}
    with pytest.raises(ValueError, match="distinct UID"):
        service.binding(request, generator_pid=os.getpid(), nonce="a"*64, expires=time.time()+30)
    def signed(body):
        return {"body": body, "signature": hmac.new(b"k"*32, canonical_bytes(body), hashlib.sha256).hexdigest()}
    body = {"generator_pid": 123, "nonce": "a"*64, "expires": time.time()+30}
    with pytest.raises(ValueError, match="signature"):
        service.authorize(request, {"body": body, "signature": "0"*64})
    with pytest.raises(ValueError, match="expired"):
        service.authorize(request, signed(dict(body, expires=time.time()-1)))
    # Only the OS-evidence seam is simulated for the cryptographic replay unit.
    monkeypatch.setattr(service, "binding", lambda *a, **k: body)
    service.authorize(request, signed(body))
    with pytest.raises(ValueError, match="replay"):
        service.authorize(request, signed(body))


def test_cluster_plan_requires_explicit_upstream_identity_and_binds_it():
    a = ClusterPlan.create(task_keys=["a"], source_clusters=["a"], strata=["s"], seed=1, upstream_identity="a"*64)
    b = ClusterPlan.create(task_keys=["a"], source_clusters=["a"], strata=["s"], seed=1, upstream_identity="b"*64)
    assert a.digest != b.digest


def test_zero_nll_gate_has_canonical_non_nan_infinite_bounds():
    plan = ClusterPlan.create(task_keys=["a"], source_clusters=["a"], strata=["s"], seed=7, upstream_identity="a"*64)
    metrics = {name: {metric: values[:1] for metric, values in row.items()} for name, row in passing_metrics().items()}
    metrics["pbpf"]["nll"] = [0.]
    result = aggregate_stage_b(metrics=metrics, plan=plan, comparator="det", brier_margin=.011,
        selection_hash="a"*64, prediction_seal_hash="b"*64)
    assert result["passed"]
    assert result["log_ratio_lower"] == "+inf"
    assert b"NaN" not in canonical_bytes(result)


def test_doctor_rejects_nonfinite_projected_resource_cost():
    with pytest.raises(ValueError):
        doctor([dict(dataset="d", model="m", tasks=1, arms=1, visible_cases=1, hidden_cases=1, seeds=3, projected_cost=float("nan"))])
