import hashlib
import json

import pytest

from pbpf.apbpf.codearc_bank import file_sha, load_bank, select_tests, verify_hidden_lock


def write(path, value):
    path.write_text(json.dumps(value))


def bank_fixture(root):
    row = {"task_id": "CodeARC/1", "source_component_id": "source1", "split": "primary",
           "candidates": [{"candidate_id": f"c{i}", "code": f"def solution(): return {i}"} for i in range(2)]}
    write(root / "CodeARC-1.json", row)
    write(root / "run.json", {"task_ids": [row["task_id"]], "source_component_ids": ["source1"],
                              "candidates": 2, "split": "primary"})
    write(root / "complete.json", {"run_sha256": file_sha(root / "run.json"),
                                  "files": {"CodeARC-1.json": file_sha(root / "CodeARC-1.json")}})
    return row


def seal(path, payload):
    payload = dict(payload)
    payload["content_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write(path, payload)


def test_bank_rejects_modified_code_and_omitted_inventory(tmp_path):
    bank_fixture(tmp_path)
    _, rows, _ = load_bank(tmp_path)
    assert len(rows) == 1
    (tmp_path / "CodeARC-1.json").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_bank(tmp_path)
    bank_fixture(tmp_path)
    complete = json.loads((tmp_path / "complete.json").read_text())
    complete["files"] = {}
    write(tmp_path / "complete.json", complete)
    with pytest.raises(ValueError, match="inventory"):
        load_bank(tmp_path)


def test_hidden_lock_rejects_population_and_bank_replacement(tmp_path):
    row = bank_fixture(tmp_path)
    _, rows, digest = load_bank(tmp_path)
    payload = {"schema": "apbpf-hard-bank-lock-v2",
               "population_lock": {"provenance": "codearc-bank-sha256:" + digest},
               "groups": [{"group_id": row["task_id"], "source_component_id": "source1", "split": "primary",
                           "candidate_ids": ["c0", "c1"], "visible_outcomes": [[1, 0, 1, 0], [0, 1, 0, 1]]}]}
    path = tmp_path / "lock.json"
    seal(path, payload)
    assert verify_hidden_lock(path, rows, digest) == file_sha(path)
    with pytest.raises(ValueError, match="different candidate bank"):
        verify_hidden_lock(path, rows, "different")
    payload["groups"][0]["candidate_ids"].reverse()
    seal(path, payload)
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_hidden_lock(path, rows, digest)
    payload["groups"] = []
    seal(path, payload)
    with pytest.raises(ValueError, match="full generated bank"):
        verify_hidden_lock(path, rows, digest)


def test_visible_phase_does_not_access_hidden_data_and_primary_all_forbidden():
    public = {"split": "primary", "visible_tests": [{"id": str(i)} for i in range(4)]}
    assert len(select_tests(public, "visible")) == 4
    with pytest.raises(ValueError, match="separate visible"):
        select_tests(public, "all")
    private = {"split": "primary", "tests": [{"id": str(i)} for i in range(10)]}
    assert [t["id"] for t in select_tests(private, "hidden")] == [str(i) for i in range(4, 10)]
