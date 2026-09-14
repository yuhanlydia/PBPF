"""Residual review counterexamples; mocked OS evidence is never formal proof."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import pytest

from pbpf.arms.repair import DecodeResult
from pbpf.runner.budget import BudgetLedger, MeteredActor, Usage, validate_matched
from pbpf.runner.evaluator import TrustedEvaluatorService, ExternalSandboxSpec, seal_final
from pbpf.runner.repair import RootRequest
from pbpf.runner.shard import canonical_bytes, digest, work_id
from pbpf.runner.stats import ClusterPlan
from test_evaluator import SANDBOX, candidate, private_manifest
from test_resume import fingerprint


TRUSTED_SANDBOX = ["/usr/bin/python3", *SANDBOX[1:]]


def service_fixture(tmp_path, monkeypatch=None, sandbox=TRUSTED_SANDBOX):
    key = tmp_path / "authority.key"
    key.write_bytes(b"k"*32)
    key.chmod(0o400)
    private, public, nonces = (tmp_path / n for n in ("private", "public", "nonces"))
    for path in (private, public, nonces):
        path.mkdir(mode=0o700, exist_ok=True)
    if sandbox is TRUSTED_SANDBOX:
        wrapper = tmp_path / "trusted-wrapper.py"
        wrapper.write_text(SANDBOX[-1])
        sandbox = ["/usr/bin/python3", "-I", str(wrapper)]
    if type(sandbox) is not ExternalSandboxSpec and sandbox is not None:
        dependencies = [str(Path(arg).resolve()) for arg in sandbox if Path(arg).is_file()]
        sandbox = ExternalSandboxSpec(command=tuple(sandbox), dependency_files=tuple(dependencies), container_digest="sha256:"+"a"*64)
    service = TrustedEvaluatorService(key_file=key, private_root=private, public_root=public,
        nonce_root=nonces, sandbox_spec=sandbox)
    if monkeypatch:
        monkeypatch.setattr("pbpf.runner.evaluator.IsolationCapability.verify",
            lambda **kw: SimpleNamespace(generator_uid=65534, process_start="fixture"))
    return service, key


def hidden_request(tmp_path, service):
    manifest = private_manifest(tmp_path)
    c = candidate()
    mapping = [{"key": ["fake", "fake", "t", 1701, "pbpf"], "source_hash": c.source_hash, "version_id": c.version_id}]
    seal = seal_final(tmp_path / "public" / "seal.json", mapping=mapping,
        expected_keys=[mapping[0]["key"]], fingerprint="f"*64, code={c.source_hash:c.source})
    return dict(operation="hidden", manifest=str(manifest), manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        seal=str(seal), seal_digest=hashlib.sha256(seal.read_bytes()).hexdigest(), expected_keys=[work_id(mapping[0]["key"])],
        fingerprint="f"*64, code={c.source_hash:c.source}, backend_identity=service.backend.to_dict(),
        sandbox_command=list(service.backend.command), budget_root=str(tmp_path / "budget"), output_root=str(tmp_path / "out"),
        case_seconds=2., rpc_overhead_seconds=5., action="inventory", work_id=work_id(mapping[0]["key"]))


def signed(service, request, nonce="a"*64):
    body = service.binding(request, generator_pid=os.getpid(), nonce=nonce, expires=time.time()+30)
    return {"body": body, "signature": hmac.new(b"k"*32, canonical_bytes(body), hashlib.sha256).hexdigest()}


def test_invalid_capability_reads_no_request_selected_seal(tmp_path, monkeypatch):
    service, key = service_fixture(tmp_path)
    reads = []
    original = Path.read_bytes
    def observed(path):
        reads.append(path)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", observed)
    with pytest.raises(ValueError):
        service.execute(dict(operation="hidden", seal=str(key), fingerprint="f"*64),
            {"body": {}, "signature": "0"*64})
    assert reads == []


def test_capability_binds_seal_bytes_and_executes_one_validated_snapshot(tmp_path, monkeypatch):
    service, _ = service_fixture(tmp_path, monkeypatch)
    request = hidden_request(tmp_path, service)
    capability = signed(service, request)
    seal = Path(request["seal"])
    row = json.loads(seal.read_bytes())
    row["selections"][0]["version_id"] = "e"*64
    row["seal_hash"] = digest({k:v for k,v in row.items() if k != "seal_hash"})
    seal.write_bytes(canonical_bytes(row))
    with pytest.raises(ValueError, match="seal"):
        service.execute(request, capability)


def test_validated_seal_snapshot_is_not_reread_after_path_swap(tmp_path, monkeypatch):
    service, _ = service_fixture(tmp_path, monkeypatch)
    request = hidden_request(tmp_path, service)
    capability = signed(service, request)
    snapshot = service.seal_snapshot
    def swapped(request):
        content = snapshot(request)
        Path(request["seal"]).write_bytes(b"invalid replacement")
        return content
    monkeypatch.setattr(service, "seal_snapshot", swapped)
    assert service.execute(request, capability)["case_ids"] == ["h"]


def test_parser_error_has_no_secret_exception_context(tmp_path, monkeypatch):
    service, _ = service_fixture(tmp_path, monkeypatch)
    request = hidden_request(tmp_path, service)
    content = b'"PRIVATE_PARSE_CANARY\x00'
    Path(request["seal"]).write_bytes(content)
    request["seal_digest"] = hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError) as error:
        service.execute(request, signed(service, request))
    assert "PRIVATE_PARSE_CANARY" not in str(error.value)
    assert error.value.__context__ is None


@pytest.mark.parametrize("attack", ["traversal", "symlink", "root_replacement"])
def test_seal_root_is_pinned_and_rejects_traversal_or_symlink_leaf(tmp_path, monkeypatch, attack):
    service, key = service_fixture(tmp_path, monkeypatch)
    request = hidden_request(tmp_path, service)
    if attack == "traversal":
        request["seal"] = str(service.public_root / ".." / "authority.key")
    elif attack == "symlink":
        alias = service.public_root / "alias.json"
        alias.symlink_to(key)
        request["seal"] = str(alias)
    else:
        service.public_root.rename(tmp_path / "old-public")
        service.public_root.mkdir(mode=0o700)
    capability = signed(service, request)
    key_inode = key.stat().st_ino
    original_read = os.read
    def no_secret_read(fd, size):
        assert os.fstat(fd).st_ino != key_inode, "request selected trust key was opened"
        return original_read(fd, size)
    monkeypatch.setattr(os, "read", no_secret_read)
    with pytest.raises(ValueError, match="seal"):
        service.execute(request, capability)


def test_original_backend_symlink_swap_after_validation_cannot_change_execution(tmp_path, monkeypatch):
    import pbpf.runner.evaluator as module
    good, bad, link = (tmp_path / name for name in ("good.py", "bad.py", "current.py"))
    good.write_text("import json; print(json.dumps({'outcome':'PASS'}))")
    bad.write_text("import json; print(json.dumps({'outcome':'WRONG_OUTPUT'}))")
    link.symlink_to(good)
    service, _ = service_fixture(tmp_path, monkeypatch, sandbox=["/usr/bin/python3", str(link)])
    request = dict(hidden_request(tmp_path, service), action="case", case_id="h")
    capability = signed(service, request)
    original = module._private_tasks
    def swapped(request, **kwargs):
        link.unlink()
        link.symlink_to(bad)
        return original(request, **kwargs)
    monkeypatch.setattr(module, "_private_tasks", swapped)
    assert service.execute(request, capability)["accepted_case"] == "h"
    receipt = json.loads(next((tmp_path / "out" / "cases").glob("*.json")).read_bytes())
    assert receipt["result"]["outcome"] == "PASS"


def test_formal_backend_rejects_generator_writable_inputs(tmp_path):
    script = tmp_path / "backend.py"
    script.write_text("pass")
    script.chmod(0o666)
    with pytest.raises(ValueError, match="writable"):
        service_fixture(tmp_path, sandbox=["/usr/bin/python3", str(script)])


def test_nonce_directory_replacement_cannot_reset_replay(tmp_path, monkeypatch):
    service, _ = service_fixture(tmp_path, monkeypatch)
    request = hidden_request(tmp_path, service)
    capability = signed(service, request)
    service.authorize(request, capability)
    service.nonce_root.rename(tmp_path / "old-nonces")
    service.nonce_root.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="nonce"):
        service.authorize(request, capability)


def test_formal_service_requires_external_sandbox(tmp_path):
    with pytest.raises(ValueError, match="sandbox"):
        service_fixture(tmp_path, sandbox=None)


@pytest.mark.parametrize("config", [{"output": "/tmp/x"}, {"created_at": "2026-01-01"},
    {"job_id": "123"}, {"arbitrary": {"temp": "/tmp/x"}}, {"temperature": "/tmp/x"}])
def test_scientific_config_uses_explicit_typed_allowlist(config):
    with pytest.raises(ValueError):
        fingerprint(config=config)


def test_external_plan_lock_rejects_rewritten_inventory_and_upstream(tmp_path):
    from pbpf.runner.stats import PlanLock
    original = ClusterPlan.create(task_keys=["a", "b"], source_clusters=["a", "b"], strata=["s", "s"], seed=7, upstream_identity="a"*64)
    lock = PlanLock.from_plan(original)
    path = tmp_path / "plan.json"
    changed = ClusterPlan.create(task_keys=["a", "b"], source_clusters=["a", "a"], strata=["s", "s"], seed=7, upstream_identity="b"*64)
    changed.save(path)
    with pytest.raises(ValueError, match="lock"):
        ClusterPlan.load(path, lock=lock)


def test_bad_native_reconciliation_still_charges_resources(tmp_path):
    class Actor:
        def generate_root(self, request):
            return DecodeResult("pass", 2)
        def native_usage(self):
            return Usage(input_token_ids=(1,), output_token_ids=(2, 3, 4), wall_seconds=.7)
    ledger = BudgetLedger(tmp_path)
    actor = MeteredActor(Actor(), ledger=ledger, work_prefix="bank")
    with pytest.raises(ValueError, match="token"):
        actor.generate_root(RootRequest(None, 1701, 0))
    assert ledger.totals()["output_tokens"] == 3
    assert ledger.totals()["wall_seconds"] == .7
    receipt = json.loads(next((tmp_path / "actor-receipts").glob("*.json")).read_bytes())
    assert receipt["status"] == "failed" and receipt["reconciliation_failure"] == "output_token_count_mismatch"


def test_partial_complete_requires_exact_bool_before_dispatch(tmp_path):
    class Actor:
        calls = 0
        def generate_partial(self, request, **kwargs):
            self.calls += 1
            return DecodeResult("pass", 1)
        def native_usage(self):
            return Usage(output_token_ids=(1,))
    backend = Actor()
    actor = MeteredActor(backend, ledger=BudgetLedger(tmp_path), work_prefix="partial")
    with pytest.raises(ValueError, match="complete"):
        actor.generate_partial(RootRequest(None, 1701, 0), prefix=(), token_budget=512, complete=1)
    assert backend.calls == 0


@pytest.mark.parametrize("complete,source", [(False, "pass"), (True, None)])
def test_partial_result_must_match_requested_completion_semantics(tmp_path, complete, source):
    from pbpf.arms.repair import PartialResult
    class Actor:
        def generate_partial(self, request, **kwargs):
            return PartialResult(prefix=("token",), source=source, output_tokens=1)
        def native_usage(self):
            return Usage(output_token_ids=(1,), wall_seconds=.2)
    ledger = BudgetLedger(tmp_path)
    actor = MeteredActor(Actor(), ledger=ledger, work_prefix="partial")
    with pytest.raises(ValueError, match="completion"):
        actor.generate_partial(RootRequest(None, 1701, 0), prefix=(), token_budget=512, complete=complete)
    assert ledger.totals()["output_tokens"] == 1
    assert ledger.totals()["full_decodes"] == int(complete)


@pytest.mark.parametrize("suite", ["v1", {"v1"}])
def test_matched_visible_suites_are_ordered_sequences_not_iterables(suite):
    row = dict(bank_hash="a"*64, roots=8, genuine=True, arm="pbpf", actor_requests=4, full_decodes=4,
        multiplicities=[1]*4, visible_opportunities=[suite]*4, max_new_tokens=1024, label="matched_four_decodes")
    with pytest.raises(ValueError):
        validate_matched([row])


def test_outer_timeout_stops_nested_sandbox_heartbeat(tmp_path):
    from pbpf.runner.evaluator import _bounded_process
    heartbeat, pidfile = tmp_path / "heartbeat", tmp_path / "pid"
    child = (f"import os,time,pathlib; pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); "
        f"f=open({str(heartbeat)!r},'a'); "
        "\nwhile True: f.write('x'); f.flush(); time.sleep(.02)")
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    worker = (f"import sys; sys.path.insert(0,{source_root!r}); import pbpf.runner.evaluator as e; "
        f"e._IN_WORKER=True; e._bounded_process({[sys.executable, '-I', '-c', child]!r}, payload=b'', env={{}}, timeout=10)")
    try:
        with pytest.raises(ValueError, match="deadline"):
            _bounded_process([sys.executable, "-I", "-c", worker], payload=b"", env={}, timeout=1.)
        before = heartbeat.stat().st_size
        time.sleep(.1)
        assert heartbeat.stat().st_size == before
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
