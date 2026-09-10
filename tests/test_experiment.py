import json
from pathlib import Path

import numpy as np
import pytest

from pbpf.artifacts import read_immutable_report, write_immutable_report
from pbpf.bank import TrajectoryBank
from pbpf.experiment import PredictionReport, run_prediction
from pbpf.loss import actor_loss, belief_loss
from pbpf.manifest import CandidateVersion, TaskRecord, canonical_hash
from pbpf.repair import CandidateProposal, run_repair, run_repair_arms, select_active_slot
from pbpf.repair import BudgetExceeded, RepairBudget
from pbpf.orchestrator import GateEvidence, derive_stage_gates, run_deterministic_smoke
from pbpf.sandbox import SandboxResult
from pbpf.statistics import paired_cluster_bootstrap, paired_seed_task_bootstrap
from pbpf.config import load_experiment, validate_experiment


def repair_task():
    return TaskRecord(
        task_id="task-a",
        dataset="runbugrun",
        public_prompt="fix",
        test_order=("t1", "t2"),
        group_ids=("source:a",),
    )


def initial_bank():
    task = repair_task()
    mutant_registry = {
        f"registered-mutant-{index - 6}": canonical_hash({"code": f"code-{index}"})
        for index in range(6, 8)
    }
    registry_hash = canonical_hash(mutant_registry)
    stored_logprobs = (-0.17, -0.31, -0.46, -0.58, -0.73, -0.89)
    samples = tuple(
        CandidateVersion.create(
            f"c{index}",
            task.task_id,
            f"code-{index}",
            provenance="model_sample",
            generation_logprob=stored_logprobs[index],
            sampling_temperature=0.8,
            sampling_top_p=0.95,
            model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            model_revision="a" * 40,
            generation_trace_hash=canonical_hash(
                {"code": f"code-{index}", "stored_logprob": stored_logprobs[index]}
            ),
            visible_tokens=4,
            generated_tokens=2,
            cpu_seconds=0.001,
            wall_seconds=0.001,
            model_calls=1,
        )
        for index in range(6)
    )
    mutants = tuple(
        CandidateVersion.create(
            f"c{index + 6}",
            task.task_id,
            f"code-{index + 6}",
            provenance="deterministic_mutant",
            generation_logprob=samples[index].generation_logprob,
            source_ref=f"registered-mutant-{index}",
            mutation_registry_hash=registry_hash,
            generation_trace_hash=samples[index].generation_trace_hash,
            source_candidate_hash=samples[index].content_hash,
            model_id=samples[index].model_id,
            model_revision=samples[index].model_revision,
            cpu_seconds=0.001,
            wall_seconds=0.001,
        )
        for index in range(2)
    )
    candidates = samples + mutants
    return TrajectoryBank((task,), candidates, (), tuple(sorted(mutant_registry.items())))


def candidate_truth(bank, *, values=("PASS", "WRONG_OUTPUT")):
    return {
        (candidate.content_hash, test_id): outcome
        for candidate in bank.candidates
        for test_id, outcome in zip(bank.tasks[0].test_order, values, strict=True)
    }


def repair_budget(**overrides):
    values = {
        "max_visible_tokens": 10000,
        "max_generated_tokens": 10000,
        "max_model_calls": 100,
        "max_executions": 100,
        "max_cpu_seconds": 60.0,
        "max_wall_seconds": 60.0,
        "max_gpu_hours": 1.0,
    }
    values.update(overrides)
    return RepairBudget(**values)


class RecordingSandbox:
    def __init__(self):
        self.calls = []

    def execute(self, code, test_id):
        self.calls.append((code, test_id))
        return SandboxResult("PASS", "ok")


class RepairBackend:
    def __init__(self):
        self.repair_calls = []
        self.bank_ids = []
        self._last_generated_tokens = 0

    def predict_success(self, task, candidate, history, *, request_text):
        assert "future_outcomes" not in task
        assert all("visible_feedback" not in item for item in history)
        return 0.5

    def count_tokens(self, text):
        return max(1, len(text.split()))

    def repair(
        self,
        task,
        candidate,
        history,
        round_index,
        *,
        request_text,
        max_generated_tokens,
    ):
        self.repair_calls.append((candidate.content_hash, round_index))
        self._last_generated_tokens = 1
        return CandidateProposal(candidate.code + f"-r{round_index}", -0.25)

    def generated_tokens_for_last_call(self):
        return self._last_generated_tokens


def test_select_active_slot_uses_generation_logprob_as_fixed_tie_break():
    bank = initial_bank()
    scores = {candidate.content_hash: 0.4 for candidate in bank.candidates}
    tie_break_values = (-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.1)
    generation = {
        candidate.content_hash: value
        for candidate, value in zip(bank.candidates, tie_break_values, strict=True)
    }
    assert select_active_slot(bank.candidates, scores, generation).candidate_id == "c7"


def test_repair_runs_four_rounds_without_early_stop_or_test_reordering():
    bank = initial_bank()
    backend = RepairBackend()
    sandbox = RecordingSandbox()
    result = run_repair(
        bank, backend=backend, sandbox=sandbox, rounds=4, budget=repair_budget()
    )
    assert len(backend.repair_calls) == 4
    assert len(result.final_programs) == 1
    assert len(sandbox.calls) == (8 + 4) * 2
    assert [test_id for _, test_id in sandbox.calls] == ["t1", "t2"] * 12
    assert result.rounds_completed == 4
    assert result.model_calls == 50
    assert result.token_ledger.visible >= 24
    assert result.compute_ledger.cpu_seconds > 0


def test_fixed_rounds_preserve_passing_elite_and_repair_unsolved_slots():
    class OnePassSandbox(RecordingSandbox):
        def execute(self, code, test_id):
            self.calls.append((code, test_id))
            return SandboxResult("PASS" if code == "code-0" else "WRONG_OUTPUT", "status")

    class HistoryScorer(RepairBackend):
        def predict_success(self, task, candidate, history, *, request_text):
            return float(all(item["outcome"] == "PASS" for item in history))

    result = run_repair(
        initial_bank(), backend=HistoryScorer(), sandbox=OnePassSandbox(),
        rounds=4, budget=repair_budget(),
    )
    assert result.final_candidate.code == "code-0"
    assert all(event["selected_candidate"] != initial_bank().candidates[0].content_hash for event in result.events)


def test_multiple_arms_receive_the_same_immutable_candidate_bank():
    bank = initial_bank()
    backends = {"pbpf": RepairBackend(), "gru": RepairBackend()}
    sandboxes = {"pbpf": RecordingSandbox(), "gru": RecordingSandbox()}
    results = run_repair_arms(
        bank,
        backends=backends,
        sandboxes=sandboxes,
        rounds=4,
        budget=repair_budget(),
    )
    assert results["pbpf"].initial_bank_hash == results["gru"].initial_bank_hash == bank.content_hash


def test_repair_rejects_missing_initial_provenance_and_generation_logprob():
    task = repair_task()
    invalid = tuple(CandidateVersion.create(f"x{i}", task.task_id, str(i)) for i in range(8))
    with pytest.raises(ValueError, match="generation log-probability"):
        run_repair(
            TrajectoryBank((task,), invalid, ()),
            backend=RepairBackend(),
            sandbox=RecordingSandbox(),
            rounds=4,
            budget=repair_budget(),
        )


def test_repair_centrally_enforces_generated_token_budget():
    class VerboseBackend(RepairBackend):
        def repair(self, task, candidate, history, round_index, **kwargs):
            self._last_generated_tokens = 7
            return CandidateProposal("too many generated tokens in this repair", -0.2)

    with pytest.raises(BudgetExceeded, match="generated token"):
        run_repair(
            initial_bank(),
            backend=VerboseBackend(),
            sandbox=RecordingSandbox(),
            rounds=4,
            budget=repair_budget(max_generated_tokens=13),
        )


def test_repair_counts_the_exact_request_and_preflights_execution_and_generation():
    class ExactRequestBackend(RepairBackend):
        def __init__(self):
            super().__init__()
            self.counted = []
            self.repair_requests = []

        def count_tokens(self, text):
            self.counted.append(text)
            return 1

        def predict_success(self, task, candidate, history, *, request_text):
            assert self.counted[-1] == request_text
            return 0.5

        def repair(
            self,
            task,
            candidate,
            history,
            round_index,
            *,
            request_text,
            max_generated_tokens,
        ):
            assert self.counted[-1] == request_text
            assert max_generated_tokens > 0
            self.repair_requests.append(request_text)
            self._last_generated_tokens = 1
            return CandidateProposal("fixed", -0.2)

        def gpu_hours_for_last_call(self):
            return 0.001

    backend = ExactRequestBackend()
    result = run_repair(
        initial_bank(),
        backend=backend,
        sandbox=RecordingSandbox(),
        budget=repair_budget(),
    )
    assert len(backend.repair_requests) == 4
    assert result.compute_ledger.gpu_hours == pytest.approx(44 * 0.001)

    sandbox = RecordingSandbox()
    with pytest.raises(BudgetExceeded, match="execution"):
        run_repair(
            initial_bank(),
            backend=ExactRequestBackend(),
            sandbox=sandbox,
            budget=repair_budget(max_executions=0),
        )
    assert sandbox.calls == []


def test_prediction_keeps_future_outcomes_behind_evaluator_firewall():
    bank = initial_bank()

    class Predictor:
        def predict(self, task, candidate, history):
            assert "future_outcomes" not in task
            assert "gold_patch" not in task
            return {
                "t1": np.array([0.8, 0.05, 0.05, 0.05, 0.05]),
                "t2": np.array([0.1, 0.6, 0.1, 0.1, 0.1]),
            }

    report = run_prediction(
        bank,
        Predictor(),
        prefix_cutoffs={"task-a": 0},
        future_outcomes_by_candidate=candidate_truth(bank),
    )
    assert report.task_count == 1
    assert report.future_nll == pytest.approx(-(np.log(0.8) + np.log(0.6)) / 2)
    assert len(report.keyed_predictions) == 16


def test_prediction_rejects_evaluated_future_overlap():
    bank = initial_bank()
    candidate = bank.candidates[0]
    from pbpf.manifest import Observation

    bank = bank.with_observation(Observation(candidate.content_hash, "t1", "PASS"))
    bank = bank.with_observation(Observation(candidate.content_hash, "t2", "PASS"))

    class Predictor:
        def predict(self, task, candidate, history):
            return {"t2": np.array([1.0, 0.0, 0.0, 0.0, 0.0])}

    with pytest.raises(ValueError, match="history/evaluation overlap"):
        run_prediction(
            bank,
            Predictor(),
            prefix_cutoffs={"task-a": 1},
            future_outcomes_by_candidate=candidate_truth(bank),
        )


def test_prediction_truth_is_candidate_test_keyed_and_prefix_is_not_reexecuted():
    from pbpf.manifest import Observation

    bank = initial_bank()
    for candidate in bank.candidates:
        bank = bank.with_observation(
            Observation(candidate.content_hash, "t1", "PASS", "prefix")
        )
    truth = candidate_truth(bank)

    class Predictor:
        def predict(self, task, candidate, history):
            assert [item["test_id"] for item in history] == ["t1"]
            assert all("visible_feedback" not in item for item in history)
            return {"t2": np.array([0.1, 0.6, 0.1, 0.1, 0.1])}

    report = run_prediction(
        bank,
        Predictor(),
        prefix_cutoffs={"task-a": 1},
        future_outcomes_by_candidate=truth,
    )
    assert len(report.keyed_predictions) == 8
    sandbox = RecordingSandbox()
    result = run_repair(
        bank,
        backend=RepairBackend(),
        sandbox=sandbox,
        rounds=4,
        budget=repair_budget(max_executions=24),
    )
    assert len(sandbox.calls) == 16
    assert sandbox.calls[:8] == [(candidate.code, "t2") for candidate in bank.candidates]
    assert len(result.execution_ledger.executions) == 24
    assert result.execution_ledger.executions[:8] == [("task-a", "t1")] * 8


def test_status_only_callback_and_trainer_bank_never_receive_raw_evaluator_feedback():
    class SecretSandbox(RecordingSandbox):
        def execute(self, code, test_id):
            self.calls.append((code, test_id))
            return SandboxResult("PASS", "SECRET RAW TEST OUTPUT")

    seen = []
    result = run_repair(
        initial_bank(),
        backend=RepairBackend(),
        sandbox=SecretSandbox(),
        rounds=4,
        feedback_mode="status_only",
        observation_callback=seen.append,
        budget=repair_budget(),
    )
    assert seen
    assert all(set(item) == {"candidate_hash", "test_id", "outcome"} for item in seen)
    assert all(observation.visible_feedback is None for observation in result.bank.observations)
    assert any(
        observation.visible_feedback == "SECRET RAW TEST OUTPUT"
        for observation in result.evaluator_observations
    )


def test_generated_budget_uses_backend_token_ids_not_decoded_text():
    class TokenIdBackend(RepairBackend):
        def __init__(self):
            super().__init__()
            self.counts = iter((4, 3, 2, 1))

        def repair(self, task, candidate, history, round_index, **kwargs):
            self._last_generated_tokens = next(self.counts)
            return CandidateProposal("one-decoded-token", -0.2)

    result = run_repair(
        initial_bank(),
        backend=TokenIdBackend(),
        sandbox=RecordingSandbox(),
        rounds=4,
        budget=repair_budget(),
    )
    assert result.token_ledger.generated == 22


def test_cpu_budget_includes_observation_callbacks_and_stops_mid_protocol():
    import time

    def expensive_belief_update(_observation):
        deadline = time.process_time() + 0.004
        while time.process_time() < deadline:
            pass

    with pytest.raises(BudgetExceeded, match="CPU-second"):
        run_repair(
            initial_bank(),
            backend=RepairBackend(),
            sandbox=RecordingSandbox(),
            rounds=4,
            observation_callback=expensive_belief_update,
            budget=repair_budget(max_cpu_seconds=0.02),
        )


def test_losses_separate_belief_and_actor_gradients():
    torch = pytest.importorskip("torch")
    belief = torch.tensor([0.2, -0.1], requires_grad=True)
    actor = torch.tensor([0.5, 0.25], requires_grad=True)
    b_loss = belief_loss(torch.stack((belief, -belief), dim=1), torch.tensor([0, 1]))
    b_loss.backward()
    assert belief.grad is not None
    belief.grad = None
    a_loss = actor_loss(lambda state: actor + state.sum(), torch.tensor([1.0, 0.5]), belief)
    a_loss.backward()
    assert actor.grad is not None
    assert belief.grad is None


def test_actor_loss_detaches_conditioning_without_optional_torch():
    class State:
        def __init__(self):
            self.detached = False

        def detach(self):
            self.detached = True
            return np.array([2.0])

    state = State()
    value = actor_loss(lambda detached: detached, np.array([0.5]), state)
    assert state.detached is True
    assert value == pytest.approx(-1.0)


def test_cluster_bootstrap_and_create_once_report_tamper_detection(tmp_path):
    stats = paired_cluster_bootstrap(
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        clusters=["a", "a", "b", "b"],
        rng=np.random.default_rng(2),
        replicates=200,
    )
    assert stats.estimate == pytest.approx(0.5)
    path = tmp_path / "report"
    write_immutable_report(path, {"status": "smoke-only", "metric": stats.estimate})
    assert read_immutable_report(path)["metric"] == 0.5
    with pytest.raises(FileExistsError):
        write_immutable_report(path, {"changed": True})
    payload = path / "report.json"
    payload.write_text(json.dumps({"metric": 9}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_immutable_report(path)
    extra_path = tmp_path / "extra-report"
    write_immutable_report(extra_path, {"status": "smoke-only"})
    (extra_path / "untracked.txt").write_text("not inventoried", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        read_immutable_report(extra_path)


def test_seed_task_hierarchical_bootstrap_weights_sources_equally():
    # Two seeds x three tasks; source A has two tasks, source B one.
    treatment = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    control = np.zeros_like(treatment)
    stats = paired_seed_task_bootstrap(
        treatment,
        control,
        source_clusters=("source-a", "source-a", "source-b"),
        rng=np.random.default_rng(4),
        replicates=200,
    )
    assert stats.estimate == pytest.approx(0.5)


def test_seed_task_bootstrap_resamples_one_shared_seed_vector_per_replicate():
    class CountingRng:
        def __init__(self):
            self.inner = np.random.default_rng(12)
            self.seed_draws = 0

        def choice(self, *args, **kwargs):
            return self.inner.choice(*args, **kwargs)

        def integers(self, *args, **kwargs):
            self.seed_draws += 1
            return self.inner.integers(*args, **kwargs)

    rng = CountingRng()
    paired_seed_task_bootstrap(
        np.ones((3, 4)),
        np.zeros((3, 4)),
        source_clusters=("a", "a", "b", "b"),
        rng=rng,
        replicates=25,
    )
    assert rng.seed_draws == 25


def test_all_experiment_configs_validate_and_cover_profiles_and_primary_arms():
    root = Path(__file__).parents[1]
    paths = sorted((root / "configs" / "experiments").glob("*.yaml"))
    assert {path.stem for path in paths} == {
        "exact_smoke",
        "frozen_7b_16gb",
        "repair_7b_24gb",
        "formal_h200",
    }
    configs = [validate_experiment(load_experiment(path)) for path in paths]
    assert {config["profile"] for config in configs} == {"cpu", "16gb", "24gb", "h200_formal"}
    required = {
        "raw_transcript",
        "last_observation",
        "window_1",
        "window_2",
        "window_4",
        "orderless_set",
        "pass_rate",
        "matched_gru",
        "matched_exchangeable",
        "map",
        "posterior_mean",
        "p_way_ensemble",
        "random_matched_norm",
        "shared_kv_delta",
        "pbpf_soft_prompt",
        "pbpf_low_rank_kv",
        "tokenwise_remixture_fault",
    }
    assert all(required <= set(config["arms"]) for config in configs)


def test_baseline_and_method_fields_use_same_execution_availability_registry():
    config = load_experiment(
        Path(__file__).parents[1] / "configs" / "experiments" / "exact_smoke.yaml"
    )
    config["arms"] = []
    config["baseline"] = {
        "name": "ladi_rl_audit_required",
        "provenance_mode": "paper_spec_reimplementation",
    }
    validate_experiment(config)
    with pytest.raises(ValueError, match="unavailable experiment method.*ladi_rl"):
        validate_experiment(config, for_execution=True)


def test_runtime_model_binding_rejects_unmarked_injected_backend():
    from types import SimpleNamespace
    from pbpf.orchestrator import validate_runtime_model_binding

    configured = {
        "id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "revision": "a" * 40,
    }
    with pytest.raises(ValueError, match="actual repair backend model"):
        validate_runtime_model_binding(
            configured,
            SimpleNamespace(model_id="local/fake", revision="a" * 40),
            execution_mode=None,
        )
    smoke = SimpleNamespace(
        model_id=configured["id"], revision="a" * 40, smoke_only=True
    )
    assert validate_runtime_model_binding(
        configured, smoke, execution_mode="deterministic_smoke"
    ) == (configured["id"], "a" * 40, True)


def test_evaluator_truth_binding_rejects_saved_execution_disagreement():
    from pbpf.manifest import Observation
    from pbpf.orchestrator import validate_evaluator_truth_binding

    candidate = initial_bank().candidates[0]
    expected = {
        (candidate.content_hash, "t1"): "PASS",
        (candidate.content_hash, "t2"): "WRONG_OUTPUT",
    }
    observations = (
        Observation(candidate.content_hash, "t1", "COMPILE_ERROR"),
        Observation(candidate.content_hash, "t2", "COMPILE_ERROR"),
    )
    with pytest.raises(ValueError, match="inconsistent with executed"):
        validate_evaluator_truth_binding(
            expected, observations, {candidate.content_hash}
        )


def test_runnable_deterministic_arm_smoke_connects_protocol_and_artifact(tmp_path):
    result = run_deterministic_smoke(tmp_path / "smoke")
    assert result["status"] == "smoke-only-no-claim"
    assert result["conditioned_repairs"] == 4
    assert result["sequence_backend"] == "TransformersRepairBackend(fake-local)"
    assert result["particle_updates"] == 24
    assert len(result["particle_ess_trace"]) == 24
    assert len({round(value, 10) for value in result["particle_ess_trace"]}) > 1
    assert result["particle_resampled"] is True
    assert result["prediction_gate"] is False
    assert result["repair_gate"] is False
    assert result["advance_stage"] is False
    assert result["stage_gate_status"].startswith("disabled")
    assert result["runtime_model"]["smoke_only"] is True
    assert result["evaluator_truth_ref"].startswith("sha256:")
    assert result["rounds_completed"] == 4
    assert result["final_program_count"] == 1
    run_root = tmp_path / "smoke"
    assert (run_root / "report.json").is_file()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "report.json",
        "config.json",
        "model.json",
        "data.json",
        "container.json",
        "programs.jsonl",
        "keyed_predictions.jsonl",
        "events.jsonl",
        "ledgers.json",
        "decision_evidence.json",
        "bank/manifest.json",
        "bank/tasks.jsonl",
        "bank/candidates.jsonl",
        "bank/observations.jsonl",
        "bank/arrays.npz",
        "bank/mutant_registry.json",
    }
    assert set(manifest["files"]) == expected
    assert set(manifest["content_hashes"]) == {
        "config",
        "trainer_bank",
        "model",
        "prediction_report",
        "evaluator_truth",
    }
    assert "future_outcomes" not in (run_root / "bank/tasks.jsonl").read_text(encoding="utf-8")
    assert len((run_root / "programs.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len(
        (run_root / "keyed_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ) == 16
    assert read_immutable_report(run_root)["model_calls"] == 50
    decision = json.loads(
        (run_root / "decision_evidence.json").read_text(encoding="utf-8")
    )
    assert decision["advance_stage"] is False
    assert decision["comparison_reports"] is None
    assert decision["bootstrap_inputs"] is None
    assert decision["prediction_report"]["sample_count"] == 16


def test_stage_gates_reject_caller_fabricated_or_zero_sample_evidence():
    empty = PredictionReport(0, 0.0, 0.0, (), (), ())
    with pytest.raises(ValueError, match="caller-supplied gate evidence"):
        GateEvidence(
            baseline_prediction=empty,
            shuffled_prediction=empty,
            random_prediction=empty,
            baseline_repair=None,
            prediction_improvement_ci_lower=1.0,
            repair_improvement_ci_lower=1.0,
        )
    assert derive_stage_gates(None, None, None) == (False, False)
