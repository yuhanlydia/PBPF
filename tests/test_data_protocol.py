from dataclasses import FrozenInstanceError, replace
import json
import os
import resource
import sys

import pytest

from pbpf.bank import TrajectoryBank
from pbpf.ledger import ComputeLedger, ExecutionLedger, TokenLedger, assert_equal_budgets
from pbpf.manifest import (
    CandidateVersion,
    Observation,
    TaskRecord,
    build_grouped_split,
    canonical_hash,
    evaluator_view,
    trainer_view,
    validate_initial_candidate_bank,
)
from pbpf.sandbox import FakeSandbox, LocalPythonSandbox, SandboxResult


def task(task_id, groups=("source:1",), test_order=("t1", "t2")):
    return TaskRecord(
        task_id=task_id,
        dataset="runbugrun",
        public_prompt="repair me",
        test_order=test_order,
        group_ids=groups,
        gold_solution="gold code",
        gold_patch="secret patch",
    )


def test_grouped_split_keeps_transitive_lineage_together():
    tasks = [
        task("a", ("source:1", "user:7")),
        task("b", ("user:7", "translation:9")),
        task("c", ("source:other",)),
        task("d", ("shared-tests:z",)),
    ]
    split = build_grouped_split(tasks, seed=4)
    assert split["a"] == split["b"]
    assert set(split) == {"a", "b", "c", "d"}


def test_trainer_view_excludes_evaluator_only_fields():
    record = task("a")
    visible = trainer_view(record)
    assert set(visible) == {"task_id", "dataset", "public_prompt", "test_order", "group_ids"}
    assert "gold_solution" not in visible
    assert evaluator_view(record)["gold_patch"] == "secret patch"


def test_candidate_patches_create_fresh_immutable_versions():
    candidate = CandidateVersion.create("c1", "a", "print(1)")
    repaired = candidate.with_patch("print(2)")
    assert repaired.version == 1
    assert repaired.parent_hash == candidate.content_hash
    assert repaired.content_hash != candidate.content_hash
    assert candidate.code == "print(1)"
    with pytest.raises(FrozenInstanceError):
        candidate.code = "mutated"


def test_bank_enforces_fixed_test_order_and_detects_tampering(tmp_path):
    record = task("a")
    candidate = CandidateVersion.create("c1", "a", "print(1)")
    first = Observation(candidate.content_hash, "t1", "PASS")
    bank = TrajectoryBank((record,), (candidate,), (first,))
    with pytest.raises(ValueError, match="fixed test order"):
        bank.with_observation(Observation(candidate.content_hash, "t1", "PASS"))

    trainer_path = tmp_path / "trainer-bank"
    evaluator_path = tmp_path / "evaluator-sidecar"
    outcomes = {
        (candidate.content_hash, test_id): outcome
        for test_id, outcome in zip(record.test_order, ("PASS", "WRONG_OUTPUT"), strict=True)
    }
    bank.write(trainer_path, evaluator_path, candidate_outcomes=outcomes)
    tasks_path = trainer_path / "tasks.jsonl"
    tasks_path.write_text(tasks_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        TrajectoryBank.read_trainer(trainer_path)


def test_serialized_bank_separates_trainer_rows_from_evaluator_sidecars(tmp_path):
    record = task("a")
    candidate = CandidateVersion.create("c1", "a", "print(1)")
    trainer_path = tmp_path / "trainer-bank"
    evaluator_path = tmp_path / "evaluator-sidecar"
    outcomes = {
        (candidate.content_hash, test_id): outcome
        for test_id, outcome in zip(record.test_order, ("PASS", "WRONG_OUTPUT"), strict=True)
    }
    TrajectoryBank((record,), (candidate,), ()).write(
        trainer_path, evaluator_path, candidate_outcomes=outcomes
    )
    public = json.loads((trainer_path / "tasks.jsonl").read_text(encoding="utf-8"))
    assert "future_outcomes" not in public
    assert "gold_solution" not in public
    sidecar = json.loads(
        (evaluator_path / "task_sidecars.jsonl").read_text(encoding="utf-8")
    )
    assert "future_outcomes" not in sidecar
    candidate_rows = [
        json.loads(line)
        for line in (evaluator_path / "candidate_outcomes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {
        (row["candidate_hash"], row["test_id"]): row["outcome"]
        for row in candidate_rows
    } == outcomes
    trainer_bank = TrajectoryBank.read_trainer(trainer_path)
    assert trainer_bank.tasks[0].gold_solution is None
    assert not (trainer_path / "evaluator").exists()
    evaluator = TrajectoryBank.read_evaluator(trainer_path, evaluator_path)
    assert evaluator.bank.tasks == (record,)
    assert evaluator.future_outcomes_by_candidate == outcomes


@pytest.mark.parametrize("field", ["public_prompt", "gold_solution", "test_order", "candidate_code"])
@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_bank_roundtrip_preserves_unicode_separators(tmp_path, field, separator):
    value = "left" + separator + "right"
    record = task("a")
    if field == "test_order":
        record = replace(record, test_order=(value, "t2"))
    elif field != "candidate_code":
        record = replace(record, **{field: value})
    candidate = CandidateVersion.create("c1", "a", value if field == "candidate_code" else "print(1)")
    outcomes = {(candidate.content_hash, test_id): "PASS" for test_id in record.test_order}
    trainer, evaluator = tmp_path / "trainer", tmp_path / "evaluator"
    TrajectoryBank((record,), (candidate,), ()).write(trainer, evaluator, candidate_outcomes=outcomes)
    public = TrajectoryBank.read_trainer(trainer)
    assert public.tasks[0].public_prompt == record.public_prompt
    assert public.tasks[0].gold_solution is None
    assert public.candidates == (candidate,)
    private = TrajectoryBank.read_evaluator(trainer, evaluator)
    assert private.bank.tasks == (record,)
    assert private.future_outcomes_by_candidate == outcomes


def test_trainer_writer_rejects_raw_evaluator_feedback(tmp_path):
    record = task("a")
    candidate = CandidateVersion.create("c1", "a", "print(1)")
    bank = TrajectoryBank(
        (record,),
        (candidate,),
        (Observation(candidate.content_hash, "t1", "WRONG_OUTPUT", "SECRET"),),
    )
    with pytest.raises(ValueError, match="raw evaluator feedback"):
        bank.write_trainer(tmp_path / "trainer")
    assert not (tmp_path / "trainer").exists()


@pytest.mark.parametrize("layout", ("evaluator-under-trainer", "trainer-under-evaluator"))
def test_trainer_and_evaluator_roots_must_be_disjoint_mount_boundaries(tmp_path, layout):
    record = task("a")
    candidate = CandidateVersion.create("c1", "a", "print(1)")
    bank = TrajectoryBank((record,), (candidate,), ())
    outcomes = {
        (candidate.content_hash, test_id): outcome
        for test_id, outcome in zip(record.test_order, ("PASS", "PASS"), strict=True)
    }
    if layout == "evaluator-under-trainer":
        trainer_path = tmp_path / "root"
        evaluator_path = trainer_path / "evaluator"
    else:
        evaluator_path = tmp_path / "root"
        trainer_path = evaluator_path / "trainer"
    with pytest.raises(ValueError, match="disjoint mount boundaries"):
        bank.write(trainer_path, evaluator_path, candidate_outcomes=outcomes)
    assert not trainer_path.exists()
    assert not evaluator_path.exists()


def test_initial_bank_requires_six_samples_two_mutants_and_real_logprobs():
    mutant_registry = {
        f"mutation-{index}": canonical_hash({"code": f"mutant-{index}"})
        for index in range(2)
    }
    registry_hash = canonical_hash(mutant_registry)
    stored_logprobs = (-0.13, -0.29, -0.41, -0.62, -0.74, -0.93)
    sampled = tuple(
        CandidateVersion.create(
            f"s{index}",
            "a",
            f"code-{index}",
            provenance="model_sample",
            generation_logprob=stored_logprobs[index],
            visible_tokens=5,
            generated_tokens=2,
            model_calls=1,
            sampling_temperature=0.8,
            sampling_top_p=0.95,
            model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            model_revision="a" * 40,
            generation_trace_hash=canonical_hash(
                {"code": f"code-{index}", "stored_logprob": stored_logprobs[index]}
            ),
        )
        for index in range(6)
    )
    mutants = tuple(
        CandidateVersion.create(
            f"m{index}",
            "a",
            f"mutant-{index}",
            provenance="deterministic_mutant",
            generation_logprob=sampled[index].generation_logprob,
            source_ref=f"mutation-{index}",
            mutation_registry_hash=registry_hash,
            generation_trace_hash=sampled[index].generation_trace_hash,
            source_candidate_hash=sampled[index].content_hash,
            model_id=sampled[index].model_id,
            model_revision=sampled[index].model_revision,
        )
        for index in range(2)
    )
    validate_initial_candidate_bank(sampled + mutants, mutant_registry)
    validate_initial_candidate_bank(
        sampled + mutants,
        mutant_registry,
        expected_model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        expected_model_revision="a" * 40,
    )
    with pytest.raises(ValueError, match="configured model"):
        validate_initial_candidate_bank(
            sampled + mutants,
            mutant_registry,
            expected_model_id="different/model",
            expected_model_revision="a" * 40,
        )
    with pytest.raises(ValueError, match="six model samples and two deterministic mutants"):
        validate_initial_candidate_bank(sampled[:5] + mutants + (mutants[0],), mutant_registry)
    missing = CandidateVersion.create("bad", "a", "bad")
    with pytest.raises(ValueError, match="generation log-probability"):
        validate_initial_candidate_bank(sampled[:5] + mutants + (missing,), mutant_registry)
    missing_usage = CandidateVersion.create(
        "bad-usage", "a", "bad", provenance="model_sample", generation_logprob=-2.0
    )
    with pytest.raises(ValueError, match="initial generation usage"):
        validate_initial_candidate_bank(sampled[:5] + mutants + (missing_usage,), mutant_registry)
    nan_usage = CandidateVersion.create(
        "nan",
        "a",
        "nan-code",
        provenance="model_sample",
        generation_logprob=-2.0,
        visible_tokens=2,
        generated_tokens=1,
        cpu_seconds=float("nan"),
        model_calls=1,
        sampling_temperature=0.8,
        sampling_top_p=0.95,
        model_revision="a" * 40,
    )
    with pytest.raises(ValueError, match="finite"):
        validate_initial_candidate_bank(sampled[:5] + mutants + (nan_usage,), mutant_registry)
    with pytest.raises(ValueError, match="verified.*registry"):
        validate_initial_candidate_bank(sampled + mutants, {"mutation-0": "0" * 64})


def test_equal_budget_check_covers_execution_token_and_compute_ledgers():
    arms = {}
    for arm in ("pbpf", "gru"):
        executions = ExecutionLedger()
        executions.record("a", "t1")
        tokens = TokenLedger()
        tokens.record(visible=10, generated=3)
        compute = ComputeLedger()
        compute.record(cpu_seconds=0.5, wall_seconds=0.7, gpu_hours=0.0)
        arms[arm] = (executions, tokens, compute)
    assert_equal_budgets(arms)
    arms["gru"][1].record(visible=1, generated=0)
    with pytest.raises(ValueError, match="budget mismatch"):
        assert_equal_budgets(arms)


def test_fake_sandbox_is_deterministic_and_infrastructure_failure_is_separate():
    sandbox = FakeSandbox({("print(1)", "t1"): "PASS"})
    assert sandbox.execute("print(1)", "t1").outcome == "PASS"
    assert sandbox.execute("unknown", "t1").outcome == "COMPILE_ERROR"
    sandbox = FakeSandbox({}, infrastructure_failures={("x", "t2")})
    result = sandbox.execute("x", "t2")
    assert result.outcome is None
    assert result.infrastructure_failure is True


def test_local_python_sandbox_checks_hidden_stdio_and_classifies_failures(monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 1)
    sandbox = LocalPythonSandbox(
        {
            "double": {"input": "3\n", "output": "6\n"},
            "empty": {"input": "", "output": "ok\n"},
        },
        timeout_seconds=0.2,
    )
    assert sandbox.execute("print(int(input()) * 2)", "double").outcome == "PASS"
    assert sandbox.execute("print(7)", "double").outcome == "WRONG_OUTPUT"
    assert sandbox.execute("raise RuntimeError('x')", "empty").outcome == "RUNTIME_EXCEPTION"
    assert sandbox.execute("def broken(:\n pass", "empty").outcome == "COMPILE_ERROR"
    assert sandbox.execute("while True: pass", "empty").outcome == "TIMEOUT"


def test_local_python_sandbox_runs_function_assertion_harness(monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 1)
    sandbox = LocalPythonSandbox(
        {"function": {"harness": "\nassert add_one(2) == 3\n"}},
        timeout_seconds=0.5,
    )
    assert sandbox.execute("def add_one(x):\n    return x + 1\n", "function").outcome == "PASS"
    assert sandbox.execute("def add_one(x):\n    return x\n", "function").outcome == "WRONG_OUTPUT"


def test_sandbox_skips_unsupported_child_limits(monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 1)
    original_setrlimit = resource.setrlimit

    def selective_permission_error(limit, values):
        if limit == resource.RLIMIT_NPROC:
            raise PermissionError("limit unavailable")
        original_setrlimit(limit, values)

    monkeypatch.setattr(resource, "setrlimit", selective_permission_error)
    result = LocalPythonSandbox(
        {"t": {"output": f"{1024 * 1024}\nok\n"}},
        python_executable=sys.executable,
    ).execute(
        "import resource\nprint(resource.getrlimit(resource.RLIMIT_FSIZE)[0])\nprint('ok')",
        "t",
    )
    assert result == SandboxResult("PASS", "pass", False)


def test_sandbox_uses_configured_python_executable(monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 1)
    result = LocalPythonSandbox(
        {"t": {"output": f"{sys.executable}\n"}},
        python_executable=sys.executable,
    ).execute("import sys\nprint(sys.executable)", "t")
    assert result == SandboxResult("PASS", "pass", False)


@pytest.mark.parametrize("operation", ("setgroups", "setgid", "setuid"))
def test_sandbox_fails_closed_when_privilege_drop_fails(monkeypatch, operation):
    monkeypatch.setattr(os, "getuid", lambda: 0)
    for name in ("setgroups", "setgid", "setuid"):
        monkeypatch.setattr(os, name, lambda *args: None)

    def reject_transition(*args):
        raise PermissionError("credential transition unavailable")

    monkeypatch.setattr(os, operation, reject_transition)
    result = LocalPythonSandbox(
        {"t": {"output": "ok\n"}}, python_executable=sys.executable
    ).execute("print('ok')", "t")
    assert result == SandboxResult(
        None, "Exception occurred in preexec_fn.", True
    )


def test_sandbox_classifies_real_preexec_failure_as_infrastructure(monkeypatch):
    def fail_preexec(self):
        raise RuntimeError("preexec failed")

    monkeypatch.setattr(LocalPythonSandbox, "_limits", fail_preexec)
    result = LocalPythonSandbox(
        {"t": {"output": "ok\n"}}, python_executable=sys.executable
    ).execute("print('ok')", "t")
    assert result == SandboxResult(
        None, "Exception occurred in preexec_fn.", True
    )
