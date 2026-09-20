from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from threading import Barrier

import pytest

from pbpf.data import (
    EvaluatorTask,
    EvaluatorTest,
    PublicTask,
    PublicTest,
    load_evaluator_tasks,
    prepare_dataset,
    seal_candidates,
)
import pbpf.data.prepare as prepare_module


CANARIES = {
    "hidden_source": "CANARY_HIDDEN_SOURCE_7cbb",
    "expected_output": "CANARY_EXPECTED_OUTPUT_e143",
    "gold_code": "CANARY_GOLD_CODE_f838",
    "gold_patch": "CANARY_GOLD_PATCH_92fa",
    "hidden_outcome": "CANARY_HIDDEN_OUTCOME_65dd",
}


def _evalplus_row() -> dict[str, object]:
    return {
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b):\n    pass",
        "candidate_code": "def add(a, b):\n    return a - b",
        "base_tests": [
            {
                "id": "base-0",
                "source": "assert add(1, 1) == 2",
                "expected_output": "2",
            }
        ],
        "plus_tests": [
            {
                "id": "plus-0",
                "source": CANARIES["hidden_source"],
                "expected_output": CANARIES["expected_output"],
            }
        ],
        "gold_code": CANARIES["gold_code"],
        "gold_patch": CANARIES["gold_patch"],
        "hidden_outcomes": {"plus-0": CANARIES["hidden_outcome"]},
        "official_split": "test",
    }


def _prepared(tmp_path: Path):
    return prepare_dataset(
        "PBPF-EvalPlus",
        [_evalplus_row()],
        generator_root=tmp_path / "generator",
        evaluator_root=tmp_path / "evaluator",
        configured_dataset_id="evalplus/humanevalplus+evalplus/mbppplus",
        runtime_dataset_id="evalplus/humanevalplus+evalplus/mbppplus",
        configured_revisions={
            "repository": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
        },
        runtime_revisions={
            "repository": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
        },
        licenses={"dataset": "Apache-2.0"},
        experiment_fingerprint="experiment-17",
    )


def _prepared_lcb(tmp_path: Path, tests: list[dict[str, object]]):
    return prepare_dataset(
        "LiveCodeBench",
        [
            {
                "task_id": "lcb-new",
                "prompt": "implement solve",
                "candidate_code": "def solve(x): return x",
                "tests": tests,
            }
        ],
        generator_root=tmp_path / "generator",
        evaluator_root=tmp_path / "evaluator",
        configured_dataset_id="livecodebench/code_generation_lite",
        runtime_dataset_id="livecodebench/code_generation_lite",
        configured_revisions={
            "repository": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
            "lite_dataset": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        },
        runtime_revisions={
            "repository": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
            "lite_dataset": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
        },
        licenses={"dataset": "Apache-2.0"},
        experiment_fingerprint="experiment-lcb-order",
        adapter_options={"release_v5_task_ids": set()},
    )


def _all_file_bytes(root: Path) -> bytes:
    return b"\n".join(
        path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()
    )


def test_generator_objects_and_every_serialized_artifact_exclude_canaries(tmp_path):
    prepared = _prepared(tmp_path)

    public_bytes = json.dumps(
        [asdict(task) for task in prepared.public_tasks], sort_keys=True
    ).encode()
    serialized_generator = _all_file_bytes(prepared.generator_root)
    for canary in CANARIES.values():
        assert canary.encode() not in public_bytes
        assert canary.encode() not in serialized_generator

    generator_inventory = {
        path.relative_to(prepared.generator_root).as_posix()
        for path in prepared.generator_root.rglob("*")
        if path.is_file()
    }
    assert generator_inventory == {
        "events.jsonl",
        "exclusions.json",
        "manifest.json",
        "prompt_inputs.jsonl",
        "resume.json",
        "tasks.jsonl",
        "training_batches.jsonl",
    }
    manifest = json.loads(prepared.generator_manifest_path.read_text())
    assert "evaluator" not in json.dumps(manifest).lower()
    assert prepared.evaluator_root.as_posix() not in serialized_generator.decode()


def test_rbr_generator_has_no_private_signature_or_gold_hash_equality_oracle(tmp_path):
    gold = "def solve(x):\n    return x"
    hidden_tests = [
        {
            "id": f"hidden-{index}",
            "active": True,
            "source": f"assert solve({index}) == {index}  # PRIVATE_{index}",
            "expected_output": f"PRIVATE_EXPECTED_{index}",
            "fixed_outcome": "PASS",
            "buggy_outcome": "WRONG_OUTPUT" if index == 0 else "PASS",
        }
        for index in range(10)
    ]
    prepared = prepare_dataset(
        "PBPF-RBR",
        [
            {
                "task_id": "rbr-private",
                "language": "python",
                "problem_id": "problem-private",
                "task_text": "repair solve",
                "buggy_code": "def solve(x): return x + 1",
                "fixed_code": gold,
                "gold_patch": "PRIVATE_PATCH",
                "tests": hidden_tests,
                "official_split": "train",
                "actor_token_count": 8,
            }
        ],
        generator_root=tmp_path / "generator",
        evaluator_root=tmp_path / "evaluator",
        configured_dataset_id="giganticode/run_bug_run",
        runtime_dataset_id="giganticode/run_bug_run",
        configured_revisions={
            "repository": "374251a9d65410f37e1136049cb7ff5dcca3d0ae"
        },
        runtime_revisions={
            "repository": "374251a9d65410f37e1136049cb7ff5dcca3d0ae"
        },
        licenses={"dataset": "upstream-project-specific"},
        experiment_fingerprint="experiment-rbr-private",
    )
    gold_signature = hashlib.sha256(
        ast.dump(ast.parse(gold), annotate_fields=True, include_attributes=False).encode()
    ).hexdigest()
    serialized_generator = _all_file_bytes(prepared.generator_root)
    assert gold_signature.encode() not in serialized_generator
    assert b"PRIVATE_EXPECTED" not in serialized_generator
    assert all(
        group.startswith("rbr:train:component:")
        for group in prepared.public_tasks[0].group_ids
    )
    trusted = json.loads(
        (prepared.evaluator_root / "preparation.json").read_text(encoding="utf-8")
    )
    assert gold_signature in json.dumps(trusted)


def test_public_schema_cannot_accept_evaluator_only_fields():
    with pytest.raises(TypeError):
        PublicTask(
            task_id="x",
            protocol="PBPF-EvalPlus",
            task_text="repair",
            candidate_code="pass",
            visible_tests=(PublicTest("base-0", None),),
            split="test",
            group_ids=("x",),
            gold_code=CANARIES["gold_code"],
        )


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_evaluator_roundtrip_preserves_unicode_in_expected_output(tmp_path, separator):
    expected = "left" + separator + "right"
    prepared = _prepared_lcb(tmp_path, [
        {"id": f"case-{index}", "source": "input", "expected_output": expected}
        for index in range(5)
    ])
    seal_path = tmp_path / "seal.json"
    seal_candidates(seal_path, [hashlib.sha256(b"candidate").hexdigest()],
                    experiment_fingerprint="experiment-lcb-order")
    loaded = load_evaluator_tasks(prepared.evaluator_manifest_path, seal_path,
                                  experiment_fingerprint="experiment-lcb-order")
    assert loaded.tasks[0].tests[0].expected_output == expected


def test_evaluator_loading_requires_matching_immutable_candidate_seal(tmp_path):
    prepared = _prepared(tmp_path)
    seal_path = tmp_path / "seals" / "candidates.json"
    with pytest.raises(FileNotFoundError, match="sealed candidate"):
        load_evaluator_tasks(
            prepared.evaluator_manifest_path,
            seal_path,
            experiment_fingerprint="experiment-17",
        )
    hashes = (
        hashlib.sha256(b"candidate-a").hexdigest(),
        hashlib.sha256(b"candidate-b").hexdigest(),
    )
    seal_candidates(
        seal_path,
        hashes,
        experiment_fingerprint="experiment-17",
        expected_experiment_fingerprint="experiment-17",
    )
    seal_payload = json.loads(seal_path.read_text())
    assert set(seal_payload) == {
        "candidate_hashes",
        "experiment_fingerprint",
        "format",
    }
    assert tuple(seal_payload["candidate_hashes"]) == tuple(sorted(hashes))
    assert "candidate-a" not in seal_path.read_text()

    loaded = load_evaluator_tasks(
        prepared.evaluator_manifest_path,
        seal_path,
        experiment_fingerprint="experiment-17",
    )
    assert loaded.candidate_hashes == tuple(sorted(hashes))
    assert loaded.tasks[0].gold_code == CANARIES["gold_code"]
    assert loaded.tasks[0].tests[-1].source == CANARIES["hidden_source"]

    with pytest.raises(FileExistsError, match="create-once"):
        seal_candidates(
            seal_path,
            hashes,
            experiment_fingerprint="experiment-17",
            expected_experiment_fingerprint="experiment-17",
        )
    with pytest.raises(ValueError, match="fingerprint"):
        seal_candidates(
            tmp_path / "wrong.json",
            hashes,
            experiment_fingerprint="wrong",
            expected_experiment_fingerprint="experiment-17",
        )
    with pytest.raises(ValueError, match="fingerprint"):
        load_evaluator_tasks(
            prepared.evaluator_manifest_path,
            seal_path,
            experiment_fingerprint="another-experiment",
        )


def test_concurrent_differing_candidate_seals_publish_exactly_one_immutable_value(tmp_path):
    for round_index in range(50):
        seal_path = tmp_path / f"round-{round_index}" / "candidates.json"
        barrier = Barrier(16)

        def attempt(index):
            candidate_hash = hashlib.sha256(f"candidate-{index}".encode()).hexdigest()
            barrier.wait()
            try:
                seal_candidates(
                    seal_path,
                    [candidate_hash],
                    experiment_fingerprint="concurrent-experiment",
                )
                return candidate_hash
            except FileExistsError:
                return None

        with ThreadPoolExecutor(max_workers=16) as executor:
            winners = [value for value in executor.map(attempt, range(16)) if value]
        assert len(winners) == 1
        original_bytes = seal_path.read_bytes()
        assert json.loads(original_bytes)["candidate_hashes"] == winners
        assert seal_path.read_bytes() == original_bytes


def test_preparation_is_atomic_create_once_and_uses_independent_permissions(tmp_path):
    prepared = _prepared(tmp_path)
    assert prepared.generator_root not in prepared.evaluator_root.parents
    assert prepared.evaluator_root not in prepared.generator_root.parents
    assert prepared.generator_manifest_path.stat().st_mode & 0o777 == 0o444
    assert prepared.evaluator_manifest_path.stat().st_mode & 0o777 == 0o400
    assert prepared.generator_root.stat().st_mode & 0o777 == 0o555
    assert prepared.evaluator_root.stat().st_mode & 0o777 == 0o500
    assert not list(tmp_path.glob(".*.tmp-*"))

    with pytest.raises(FileExistsError, match="create-once"):
        _prepared(tmp_path)


@pytest.mark.parametrize(
    ("configured_id", "runtime_id", "configured_revision", "runtime_revision", "message"),
    [
        ("evalplus/humanevalplus+evalplus/mbppplus", "wrong/dataset", "a" * 40, "a" * 40, "dataset id"),
        ("evalplus/humanevalplus+evalplus/mbppplus", "evalplus/humanevalplus+evalplus/mbppplus", "a" * 40, "b" * 40, "revision"),
        ("evalplus/humanevalplus+evalplus/mbppplus", "evalplus/humanevalplus+evalplus/mbppplus", "short", "short", "full immutable"),
    ],
)
def test_runtime_identity_and_full_revision_are_validated_before_writing(
    tmp_path,
    configured_id,
    runtime_id,
    configured_revision,
    runtime_revision,
    message,
):
    with pytest.raises(ValueError, match=message):
        prepare_dataset(
            "PBPF-EvalPlus",
            [object()],  # Would fail inside the adapter if validation happened late.
            generator_root=tmp_path / "generator",
            evaluator_root=tmp_path / "evaluator",
            configured_dataset_id=configured_id,
            runtime_dataset_id=runtime_id,
            configured_revisions={"repository": configured_revision},
            runtime_revisions={"repository": runtime_revision},
            licenses={"dataset": "Apache-2.0"},
            experiment_fingerprint="experiment-17",
        )
    assert not (tmp_path / "generator").exists()
    assert not (tmp_path / "evaluator").exists()


@pytest.mark.parametrize(
    ("protocol", "dataset_id", "revisions", "adapter_options"),
    [
        (
            "PBPF-EvalPlus",
            "giganticode/run_bug_run",
            {"repository": "374251a9d65410f37e1136049cb7ff5dcca3d0ae"},
            {},
        ),
        (
            "CodeARC-Replay",
            "anjiangwei/CodeARC-Problems+anjiangwei/CodeARC-Invocations",
            {"repository": "32f2a1e1ba3bdf8a7d5057aacbd665edaf441ce0"},
            {},
        ),
        (
            "LiveCodeBench",
            "livecodebench/code_generation_lite",
            {"repository": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"},
            {"release_v5_task_ids": set()},
        ),
    ],
)
def test_protocol_rejects_non_authoritative_dataset_or_revision_inventory(
    tmp_path, protocol, dataset_id, revisions, adapter_options
):
    with pytest.raises(ValueError, match="authoritative|requires revisions"):
        prepare_dataset(
            protocol,
            [],
            generator_root=tmp_path / protocol / "generator",
            evaluator_root=tmp_path / protocol / "evaluator",
            configured_dataset_id=dataset_id,
            runtime_dataset_id=dataset_id,
            configured_revisions=revisions,
            runtime_revisions=revisions,
            licenses={"dataset": "test-only"},
            experiment_fingerprint="experiment-protocol-mismatch",
            adapter_options=adapter_options,
        )


def test_manifest_records_provenance_counts_exclusions_and_split_hashes(tmp_path):
    prepared = _prepared(tmp_path)
    manifest = json.loads(prepared.generator_manifest_path.read_text())
    assert manifest["protocol"] == "PBPF-EvalPlus"
    assert manifest["dataset_id"] == "evalplus/humanevalplus+evalplus/mbppplus"
    assert manifest["revisions"] == {
        "repository": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
    }
    assert manifest["licenses"] == {"dataset": "Apache-2.0"}
    assert manifest["adapter_version"] == "1"
    assert manifest["counts"] == {"excluded": 0, "kept": 1, "raw": 1}
    assert manifest["exclusion_reasons"] == {}
    assert manifest["resolved_options"] == {"exact_case_deduplication": True}
    assert set(manifest["split_hashes"]) == {"test"}
    assert len(manifest["split_hashes"]["test"]) == 64
    assert prepared.generator_manifest_hash == hashlib.sha256(
        prepared.generator_manifest_path.read_bytes()
    ).hexdigest()


def test_evaluator_schema_round_trip_keeps_secrets_on_evaluator_side_only(tmp_path):
    prepared = _prepared(tmp_path)
    evaluator_rows = [
        json.loads(line)
        for line in (prepared.evaluator_root / "tasks.jsonl").read_text().splitlines()
    ]
    task = EvaluatorTask(
        **{
            **evaluator_rows[0],
            "tests": tuple(EvaluatorTest(**case) for case in evaluator_rows[0]["tests"]),
        }
    )
    assert task.gold_patch == CANARIES["gold_patch"]
    assert task.hidden_outcomes == (("plus-0", CANARIES["hidden_outcome"]),)
    plus = next(case for case in task.tests if case.test_id == "plus-0")
    assert plus.payload == _evalplus_row()["plus_tests"][0]
    assert CANARIES["gold_patch"].encode() in _all_file_bytes(prepared.evaluator_root)


def test_evaluator_rejects_published_roots_without_completion_record(tmp_path):
    prepared = _prepared(tmp_path)
    seal_path = tmp_path / "seal.json"
    seal_candidates(
        seal_path,
        [hashlib.sha256(b"candidate").hexdigest()],
        experiment_fingerprint="experiment-17",
    )
    prepared.evaluator_root.chmod(0o700)
    prepared.completion_path.unlink()
    prepared.evaluator_root.chmod(0o500)
    with pytest.raises(FileNotFoundError, match="completed.*publication"):
        load_evaluator_tasks(
            prepared.evaluator_manifest_path,
            seal_path,
            experiment_fingerprint="experiment-17",
        )


def test_failed_pair_publication_is_incomplete_and_matching_retry_recovers(
    tmp_path, monkeypatch
):
    generator_root = tmp_path / "generator"
    evaluator_root = tmp_path / "evaluator"
    real_replace = prepare_module.os.replace
    failed = False

    def fail_between_publications(source, target):
        nonlocal failed
        if Path(target) == evaluator_root and not failed:
            failed = True
            raise OSError("injected evaluator publication failure")
        return real_replace(source, target)

    monkeypatch.setattr(prepare_module.os, "replace", fail_between_publications)
    with pytest.raises(OSError, match="injected evaluator publication failure"):
        _prepared(tmp_path)
    assert generator_root.is_dir()
    assert not evaluator_root.exists()
    assert not (evaluator_root / "publication.json").exists()

    monkeypatch.setattr(prepare_module.os, "replace", real_replace)
    prepared = _prepared(tmp_path)
    completion = json.loads(
        (prepared.evaluator_root / "publication.json").read_text(encoding="utf-8")
    )
    assert completion["experiment_fingerprint"] == "experiment-17"
    assert completion["generator_manifest_hash"] == prepared.generator_manifest_hash
    assert completion["evaluator_manifest_hash"] == prepared.evaluator_manifest_hash


def test_retry_never_completes_a_tampered_incomplete_publication(tmp_path, monkeypatch):
    evaluator_root = tmp_path / "evaluator"
    real_replace = prepare_module.os.replace

    def fail_between_publications(source, target):
        if Path(target) == evaluator_root:
            raise OSError("injected evaluator publication failure")
        return real_replace(source, target)

    monkeypatch.setattr(prepare_module.os, "replace", fail_between_publications)
    with pytest.raises(OSError, match="injected evaluator publication failure"):
        _prepared(tmp_path)

    task_store = tmp_path / "generator" / "tasks.jsonl"
    task_store.chmod(0o644)
    task_store.write_bytes(task_store.read_bytes() + b"{}\n")
    task_store.chmod(0o444)
    monkeypatch.setattr(prepare_module.os, "replace", real_replace)

    with pytest.raises(ValueError, match="inventory|checksum"):
        _prepared(tmp_path)
    assert not (evaluator_root / "publication.json").exists()


def test_generator_retry_rejects_an_unmanifested_publication_record(
    tmp_path, monkeypatch
):
    evaluator_root = tmp_path / "evaluator"
    real_replace = prepare_module.os.replace

    def fail_between_publications(source, target):
        if Path(target) == evaluator_root:
            raise OSError("injected evaluator publication failure")
        return real_replace(source, target)

    monkeypatch.setattr(prepare_module.os, "replace", fail_between_publications)
    with pytest.raises(OSError, match="injected evaluator publication failure"):
        _prepared(tmp_path)

    generator_root = tmp_path / "generator"
    generator_root.chmod(0o700)
    unexpected = generator_root / "publication.json"
    unexpected.write_text('{"not":"a generator artifact"}\n', encoding="utf-8")
    unexpected.chmod(0o444)
    generator_root.chmod(0o555)
    monkeypatch.setattr(prepare_module.os, "replace", real_replace)

    with pytest.raises(ValueError, match="inventory"):
        _prepared(tmp_path)
    assert not evaluator_root.exists()


@pytest.mark.parametrize(("boundary", "exit_code"), (("before", 91), ("after", 92)))
def test_abrupt_completion_crash_leaves_clean_evaluator_inventory_and_recovers(
    tmp_path, boundary, exit_code
):
    script = r'''
import json
import os
from pathlib import Path
import sys

from pbpf.data import prepare_dataset
import pbpf.data.prepare as prepare_module

boundary = sys.argv[1]
root = Path(sys.argv[2])
row = json.loads(sys.argv[3])
real_link = prepare_module.os.link

def crash_at_completion_link(source, target):
    if Path(target).name == "publication.json":
        if boundary == "before":
            os._exit(91)
        real_link(source, target)
        os._exit(92)
    return real_link(source, target)

prepare_module.os.link = crash_at_completion_link
prepare_dataset(
    "PBPF-EvalPlus",
    [row],
    generator_root=root / "generator",
    evaluator_root=root / "evaluator",
    configured_dataset_id="evalplus/humanevalplus+evalplus/mbppplus",
    runtime_dataset_id="evalplus/humanevalplus+evalplus/mbppplus",
    configured_revisions={"repository": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"},
    runtime_revisions={"repository": "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"},
    licenses={"dataset": "Apache-2.0"},
    experiment_fingerprint="experiment-17",
)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, boundary, str(tmp_path), json.dumps(_evalplus_row())],
        check=False,
    )
    assert completed.returncode == exit_code
    evaluator_root = tmp_path / "evaluator"
    assert not list(evaluator_root.glob(".publication.json.tmp-*"))

    if boundary == "before":
        prepared = _prepared(tmp_path)
        assert prepared.completion_path.is_file()
        evaluator_manifest_path = prepared.evaluator_manifest_path
    else:
        evaluator_manifest_path = evaluator_root / "manifest.json"
    seal_path = tmp_path / "seal.json"
    seal_candidates(
        seal_path,
        [hashlib.sha256(b"candidate").hexdigest()],
        experiment_fingerprint="experiment-17",
    )
    loaded = load_evaluator_tasks(
        evaluator_manifest_path,
        seal_path,
        experiment_fingerprint="experiment-17",
    )
    assert [task.task_id for task in loaded.tasks] == ["HumanEval/0"]


def test_evaluator_loader_still_rejects_unrelated_unmanifested_files(tmp_path):
    prepared = _prepared(tmp_path)
    seal_path = tmp_path / "seal.json"
    seal_candidates(
        seal_path,
        [hashlib.sha256(b"candidate").hexdigest()],
        experiment_fingerprint="experiment-17",
    )
    prepared.evaluator_root.chmod(0o700)
    (prepared.evaluator_root / "unrelated.tmp").write_text("unexpected", encoding="utf-8")
    prepared.evaluator_root.chmod(0o500)
    with pytest.raises(ValueError, match="inventory"):
        load_evaluator_tasks(
            prepared.evaluator_manifest_path,
            seal_path,
            experiment_fingerprint="experiment-17",
        )


def test_lcb_unpartitioned_test_permutations_produce_identical_manifests(tmp_path):
    tests = [
        {"id": test_id, "input": [test_id], "expected_output": test_id}
        for test_id in ("z", "a", "m", "q", "b", "y")
    ]
    forward = _prepared_lcb(tmp_path / "forward", tests)
    reverse = _prepared_lcb(tmp_path / "reverse", list(reversed(tests)))
    assert forward.public_tasks == reverse.public_tasks
    assert (
        forward.generator_manifest_path.read_bytes()
        == reverse.generator_manifest_path.read_bytes()
    )
    assert (
        forward.evaluator_manifest_path.read_bytes()
        == reverse.evaluator_manifest_path.read_bytes()
    )
