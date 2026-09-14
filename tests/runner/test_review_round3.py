"""Private snapshots and auditable formal launch dependencies."""
import hashlib
import json
from pathlib import Path

import pytest

from pbpf.runner.evaluator import TrustedEvaluatorService
from pbpf.runner.shard import canonical_bytes
from test_review_round2 import service_fixture, hidden_request, signed


def test_private_root_swap_between_hash_and_parse_never_uses_replacement(tmp_path, monkeypatch):
    service, _ = service_fixture(tmp_path, monkeypatch)
    request = hidden_request(tmp_path, service)
    capability = signed(service, request)
    original_bytes = Path(request["manifest"]).read_bytes()
    original_hash = hashlib.sha256
    swapped = []
    def hash_then_swap(content=b"", *args, **kwargs):
        result = original_hash(content, *args, **kwargs)
        if content == original_bytes and not swapped:
            swapped.append(True)
            service.private_root.rename(tmp_path / "old-private")
            service.private_root.mkdir(mode=0o700)
            replacement = json.loads(original_bytes)
            replacement["tasks"][0]["tests"][1]["test_id"] = "REPLACEMENT_CANARY"
            Path(request["manifest"]).write_bytes(canonical_bytes(replacement))
        return result
    monkeypatch.setattr(hashlib, "sha256", hash_then_swap)
    try:
        result = service.execute(request, capability)
    except ValueError:
        result = None
    assert swapped
    assert result is None or result["case_ids"] == ["h"]


def test_private_manifest_symlink_is_rejected_even_inside_private_root(tmp_path, monkeypatch):
    service, _ = service_fixture(tmp_path, monkeypatch)
    request = hidden_request(tmp_path, service)
    manifest = Path(request["manifest"])
    manifest.rename(manifest.with_name("actual.json"))
    manifest.symlink_to("actual.json")
    with pytest.raises(ValueError):
        service.execute(request, signed(service, request))


def test_private_root_has_no_generator_writable_ancestor(tmp_path):
    service, key = service_fixture(tmp_path)
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    private = unsafe / "private"
    private.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="writable"):
        TrustedEvaluatorService(key_file=key, private_root=private, public_root=service.public_root,
            nonce_root=service.nonce_root, sandbox_spec=service.sandbox_spec)


@pytest.mark.parametrize("command", [["/usr/bin/python3", "-c", "exec(open('dynamic_backend.py').read())"],
    ["/usr/bin/python3", "-m", "dynamic_backend"], ["/bin/sh", "-c", "eval echo PASS"]])
def test_formal_rejects_inline_module_and_dynamic_launch_before_start(tmp_path, command):
    with pytest.raises(ValueError):
        service_fixture(tmp_path, sandbox=command)


def test_declared_dependency_mutation_invalidates_formal_dispatch(tmp_path, monkeypatch):
    from pbpf.runner.evaluator import ExternalSandboxSpec
    wrapper, config = tmp_path / "wrapper.py", tmp_path / "config.json"
    wrapper.write_text("import json; print(json.dumps({'outcome':'PASS'}))")
    config.write_text('{"locked": 1}')
    spec = ExternalSandboxSpec(command=("/usr/bin/python3", "-I", str(wrapper)),
        dependency_files=("/usr/bin/python3", str(wrapper), str(config)), container_digest="sha256:"+"a"*64)
    service, _ = service_fixture(tmp_path, monkeypatch, sandbox=spec)
    request = hidden_request(tmp_path, service)
    capability = signed(service, request)
    config.write_text('{"locked": 2}')
    with pytest.raises(ValueError):
        service.execute(request, capability)


def test_declared_dependency_symlink_retarget_invalidates_formal_dispatch(tmp_path, monkeypatch):
    from pbpf.runner.evaluator import ExternalSandboxSpec
    wrapper, config, other, alias = (tmp_path / name for name in ("wrapper.py", "one.json", "two.json", "config.json"))
    wrapper.write_text("import json; print(json.dumps({'outcome':'PASS'}))")
    config.write_text("1")
    other.write_text("2")
    alias.symlink_to(config)
    spec = ExternalSandboxSpec(command=("/usr/bin/python3", "-I", str(wrapper)),
        dependency_files=("/usr/bin/python3", str(wrapper), str(alias)), container_digest="sha256:"+"a"*64)
    service, _ = service_fixture(tmp_path, monkeypatch, sandbox=spec)
    request = hidden_request(tmp_path, service)
    capability = signed(service, request)
    alias.unlink()
    alias.symlink_to(other)
    with pytest.raises(ValueError):
        service.execute(request, capability)


@pytest.mark.parametrize("change", [None, "checksum", "inventory", "publication", "symlink"])
def test_formal_task4_publication_validates_descriptor_snapshots(tmp_path, monkeypatch, change):
    service, _ = service_fixture(tmp_path, monkeypatch)
    request = hidden_request(tmp_path, service)
    manifest_path = Path(request["manifest"])
    embedded = json.loads(manifest_path.read_bytes())
    tasks_path = service.private_root / "tasks.jsonl"
    tasks_bytes = b"\n".join(canonical_bytes(task) for task in embedded["tasks"]) + b"\n"
    tasks_path.write_bytes(tasks_bytes)
    manifest = dict(format="pbpf-evaluator-data-v1", experiment_fingerprint="f"*64,
        generator_manifest_hash="a"*64, files={"tasks.jsonl": hashlib.sha256(tasks_bytes).hexdigest()})
    manifest_bytes = canonical_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    request["manifest_hash"] = hashlib.sha256(manifest_bytes).hexdigest()
    publication = dict(format="pbpf-data-publication-v1", experiment_fingerprint="f"*64,
        generator_manifest_hash="a"*64, evaluator_manifest_hash=request["manifest_hash"])
    if change == "publication":
        publication["generator_manifest_hash"] = "b"*64
    (service.private_root / "publication.json").write_bytes(canonical_bytes(publication))
    if change == "checksum":
        tasks_path.write_bytes(tasks_bytes + b" ")
    elif change == "inventory":
        (service.private_root / "unexpected.txt").write_text("extra")
    elif change == "symlink":
        elsewhere = tmp_path / "tasks-copy.jsonl"
        elsewhere.write_bytes(tasks_bytes)
        tasks_path.unlink()
        tasks_path.symlink_to(elsewhere)
    capability = signed(service, request)
    if change:
        with pytest.raises(ValueError):
            service.execute(request, capability)
    else:
        assert service.execute(request, capability)["case_ids"] == ["h"]
