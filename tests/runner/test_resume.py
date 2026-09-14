import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from pbpf.runner.fingerprint import RunFingerprint, code_identity
from pbpf.runner.shard import canonical_bytes, work_id, shard_for, merge_leaves
from pbpf.runner.stage import run_stage


def fingerprint(**changes):
    values = dict(code={"tree": "a" * 64, "content_hash": "b"*64}, models={"actor": {"model_id": "fake", "revision": "b" * 40,
        "tokenizer_id": "fake", "tokenizer_revision": "c" * 40, "tokenizer_hash": "6" * 64,
        "chat_template_hash": "d" * 64, "weights_hash": "e"*64}},
        datasets={"data": {"dataset_id": "fake", "revision": "e" * 40, "adapter_hash": "1"*64, "split_hash": "f" * 64, "raw_hash": "2"*64}},
        prompt={"revision": "1" * 64, "template_hash": "2"*64}, container_digest="sha256:" + "2" * 64,
        config={"mode": "smoke"}, seeds=(1701, 1702, 1703), locks={"brier_margin": .01,
            "checkpoint_hash": "3"*64, "selection_hash": "4"*64, "calibration_hash": "5"*64}, schema="v1")
    values.update(changes)
    return RunFingerprint(**values)


def test_fingerprint_binds_all_scientific_inputs_and_dirty_bytes(tmp_path):
    base = fingerprint()
    for name in ("code", "models", "datasets", "prompt", "config", "locks"):
        with pytest.raises(ValueError):
            fingerprint(**{name: {"output_path": "/tmp/leak"}})
    for name, change in [("code", {"tree": "b"*64, "content_hash": "c"*64}), ("prompt", {"revision": "3"*64, "template_hash": "2"*64}),
                         ("config", {"temperature": .7}), ("schema", "v2"),
                         ("container_digest", "sha256:"+"9"*64)]:
        assert fingerprint(**{name: change}).digest != base.digest
    (tmp_path / "a.py").write_text("one")
    before = code_identity(tmp_path, ["a.py"])
    (tmp_path / "a.py").write_text("two")
    assert code_identity(tmp_path, ["a.py"]) != before
    assert fingerprint(config={"roots": 8, "temperature": .8}).digest == fingerprint(config={"temperature": .8, "roots": 8}).digest


def test_fingerprint_changes_when_tokenizer_artifact_content_changes():
    base = fingerprint()
    changed = base.to_dict()["models"]
    changed["actor"]["tokenizer_hash"] = "7" * 64
    assert fingerprint(models=changed).digest != base.digest


def test_structured_key_and_full_hash_modulo_across_processes():
    key = ["a|b", "c", 1701]
    assert work_id(key) != work_id(["a", "b|c", 1701])
    expected = hashlib.sha256(b'["a|b","c",1701]').hexdigest()
    assert work_id(key) == expected
    assert shard_for(key, 17) == int(expected, 16) % 17
    result = subprocess.check_output([sys.executable, "-c",
        "from pbpf.runner.shard import work_id; print(work_id(['a|b','c',1701]))"], text=True)
    assert result.strip() == expected


def test_resume_validates_identity_inventory_and_corruption(tmp_path):
    calls = []
    def handler(key, deps):
        calls.append(key)
        return {"value": key}
    args = dict(root=tmp_path, name="bank", fingerprint=fingerprint(config={"roots": 8}), keys=[["a"], ["b"]],
                dependencies={"prepare": "a"*64}, config={"roots": 8}, handler=handler)
    first = run_stage(**args)
    original = first.artifact.read_bytes()
    assert len(calls) == 2
    assert run_stage(**args).cached
    assert len(calls) == 2
    first.artifact.write_bytes(b"truncated")
    assert not run_stage(**args).cached
    assert first.artifact.read_bytes() == original
    assert len(calls) == 4
    for change in ({"fingerprint": fingerprint(schema="v2")}, {"dependencies": {"prepare": "b"*64}},
                   {"keys": [["a"]]}, {"config": {"roots": 7}}):
        with pytest.raises(ValueError, match="identity"):
            run_stage(**dict(args, **change))


def test_partial_completion_reruns_and_concurrent_publication_accepts_once(tmp_path):
    calls = []
    def handler(key, deps):
        calls.append(key)
        return {"result": 1}
    args = dict(root=tmp_path, name="bank", fingerprint=fingerprint(), keys=[[1]],
                dependencies={}, config=fingerprint().to_dict()["config"], handler=handler)
    with ThreadPoolExecutor(2) as pool:
        outputs = list(pool.map(lambda _: run_stage(**args), range(2)))
    assert len(calls) == 1
    assert sum(result.cached for result in outputs) == 1
    outputs[0].completion.unlink()
    assert not run_stage(**args).cached
    assert len(calls) == 2


def test_merge_rejects_wrong_inventory_corrupt_and_missharded(tmp_path):
    keys = [[i] for i in range(9)]
    results = [run_stage(root=tmp_path, name="bank", fingerprint=fingerprint(), keys=keys,
        dependencies={}, config=fingerprint().to_dict()["config"], handler=lambda key, deps: {"value": key}, shard_index=i,
        shard_count=2) for i in range(2)]
    assert len(merge_leaves(results, expected_keys=keys)) == 9
    with pytest.raises(ValueError):
        merge_leaves(results[:1], expected_keys=keys)
    with pytest.raises(ValueError):
        merge_leaves(results + results[:1], expected_keys=keys)
    results[0].artifact.write_bytes(b"[]")
    with pytest.raises(ValueError, match="checksum"):
        merge_leaves(results, expected_keys=keys)


def test_completion_identity_mutation_fails_closed(tmp_path):
    args = dict(root=tmp_path, name="prepare", fingerprint=fingerprint(), keys=[[1]],
                dependencies={}, config=fingerprint().to_dict()["config"], handler=lambda key, deps: {"result": 1})
    result = run_stage(**args)
    marker = json.loads(result.completion.read_bytes())
    marker["identity"]["fingerprint"] = "0"*64
    result.completion.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="identity"):
        run_stage(**args)


def test_crash_after_payload_before_marker_reexecutes_only_matching_work(tmp_path, monkeypatch):
    import pbpf.runner.stage as module
    calls = []
    def handler(key, deps):
        calls.append(key)
        return {"output": 9}
    args = dict(root=tmp_path, name="bank", fingerprint=fingerprint(), keys=[[1]],
                dependencies={}, config=fingerprint().to_dict()["config"], handler=handler)
    real = module.atomic_write
    def crash(path, payload, **options):
        if str(path).endswith(".complete.json"):
            raise RuntimeError("simulated crash")
        return real(path, payload, **options)
    monkeypatch.setattr(module, "atomic_write", crash)
    with pytest.raises(RuntimeError, match="crash"):
        run_stage(**args)
    before = (tmp_path / "bank" / "shard-00000.jsonl").read_bytes()
    monkeypatch.setattr(module, "atomic_write", real)
    result = run_stage(**args)
    assert result.artifact.read_bytes() == before
    assert len(calls) == 2
