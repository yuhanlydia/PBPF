import hashlib
import json

import pytest

from scripts.evaluate_eesd_fresh_bank import load_fresh_banks


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_shard(root, task, source, *, revision="r" * 40):
    root.mkdir()
    run = {"schema": "apbpf-rbr-generation-v1", "split": "primary", "candidates": 1,
           "model": "example/model", "revision": revision, "adapter_sha256": "a" * 64,
           "public_tasks_sha256": "p" * 64, "seed": 1701,
           "max_input_tokens": 4096, "max_new_tokens": 1024,
           "temperature": .8, "top_p": .95, "decode_policy": "temperature0.8-top_p0.95",
           "task_ids": [task], "source_component_ids": [source]}
    run_sha = write_json(root / "run.json", run)
    filename = task + ".json"
    row = {"task_id": task, "source_component_id": source, "split": "primary",
           "candidates": [{"candidate_id": task + "/0", "code": "print(1)"}]}
    record_sha = write_json(root / filename, row)
    write_json(root / "complete.json", {"run_sha256": run_sha,
                                         "files": {filename: record_sha}})
    return root


def test_fresh_evaluator_loads_disjoint_sealed_shards(tmp_path):
    a = sealed_shard(tmp_path / "a", "task-a", "source-a")
    b = sealed_shard(tmp_path / "b", "task-b", "source-b")
    run, rows, _, digests = load_fresh_banks([a, b], "rbr")
    assert run["seed"] == 1701
    assert {row["source_component_id"] for row in rows} == {"source-a", "source-b"}
    assert len(digests) == 2


def test_fresh_evaluator_rejects_mixed_revision_and_duplicate_task(tmp_path):
    a = sealed_shard(tmp_path / "a", "task-a", "source-a")
    b = sealed_shard(tmp_path / "b", "task-b", "source-b", revision="s" * 40)
    with pytest.raises(ValueError, match="different model or sampling"):
        load_fresh_banks([a, b], "rbr")
    c = sealed_shard(tmp_path / "c", "task-a", "source-a")
    with pytest.raises(ValueError, match="duplicate task"):
        load_fresh_banks([a, c], "rbr")
