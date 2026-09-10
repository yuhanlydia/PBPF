from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .artifacts import write_run_bundle
from .bank import TrajectoryBank
from .config import canonical_config_hash, validate_experiment
from .experiment import PredictionReport, run_prediction
from .history import feedback_view
from .manifest import (
    CandidateVersion,
    TaskRecord,
    canonical_hash,
    validate_initial_candidate_bank,
)
from .kv import build_kv_delta
from .models import ParticleConditioner, TransformersRepairBackend
from .particles import particle_update
from .repair import CandidateProposal, RepairBudget, RepairResult, run_repair
from .sandbox import FakeSandbox


class _MetadataBackend:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.last_logprob: float | None = None
        self.last_generated_tokens: int | None = None

    def generate(self, prompt: str, *, particle_state: Any, **generation: Any) -> str:
        method = getattr(self.backend, "generate_with_logprob", None)
        if not callable(method):
            raise TypeError("conditioned sequence backend requires generate_with_logprob")
        text, logprob = method(prompt, particle_state=particle_state, **generation)
        self.last_logprob = float(logprob)
        meter = getattr(self.backend, "generated_tokens_for_last_call", None)
        if not callable(meter):
            raise TypeError("sequence backend must expose generated token-ID telemetry")
        self.last_generated_tokens = meter()
        return str(text)


class ConditionedRepairBackend:
    """Repair protocol adapter that samples one belief particle per continuation."""

    def __init__(
        self,
        *,
        scorer: Any,
        sequence_backend: Any,
        particles: Sequence[Any],
        log_weights: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.scorer = scorer
        self.sequence_backend = sequence_backend
        self.particles = tuple(particles)
        self.log_weights = np.asarray(log_weights, dtype=np.float64)
        self.rng = rng
        self._metadata_backend = _MetadataBackend(sequence_backend)
        self.conditioner = ParticleConditioner(self._metadata_backend)
        self.conditioned_repairs = 0
        self._last_component: Any | None = None

    def update_belief(
        self,
        particles: Sequence[Any],
        log_weights: np.ndarray,
    ) -> None:
        self.particles = tuple(particles)
        self.log_weights = np.asarray(log_weights, dtype=np.float64)

    def count_tokens(self, text: str) -> int:
        counter = getattr(self.sequence_backend, "count_tokens", None)
        if not callable(counter):
            raise TypeError("sequence backend requires count_tokens")
        return int(counter(text))

    def predict_success(self, task, candidate, history, *, request_text: str) -> float:
        self._last_component = self.scorer
        return float(
            self.scorer.predict_success(
                task, candidate, history, request_text=request_text
            )
        )

    def repair(
        self,
        task,
        candidate,
        history,
        round_index,
        *,
        request_text: str,
        max_generated_tokens: int,
    ) -> CandidateProposal:
        self._last_component = self.sequence_backend
        generated = self.conditioner.generate(
            request_text,
            particles=self.particles,
            log_weights=self.log_weights,
            rng=self.rng,
            max_new_tokens=max_generated_tokens,
        )
        if self._metadata_backend.last_logprob is None:
            raise ValueError("conditioned generation did not report a sequence log-probability")
        self.conditioned_repairs += 1
        if self._metadata_backend.last_generated_tokens is None:
            raise ValueError("conditioned generation did not report generated token IDs")
        return CandidateProposal(
            generated.text,
            self._metadata_backend.last_logprob,
            self._metadata_backend.last_generated_tokens,
        )

    def generated_tokens_for_last_call(self) -> int:
        if self._metadata_backend.last_generated_tokens is None:
            raise RuntimeError("no conditioned generation token telemetry is available")
        return self._metadata_backend.last_generated_tokens

    def gpu_hours_for_last_call(self) -> float:
        meter = getattr(self._last_component, "gpu_hours_for_last_call", None)
        return float(meter()) if callable(meter) else 0.0


class GateEvidence:
    """Disabled external evidence type retained solely for API fail-closure."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError(
            "caller-supplied gate evidence is forbidden; stage advancement requires a future "
            "registered immutable multi-seed/task report aggregator"
        )


def _pass_at_one(result: RepairResult) -> float:
    task = result.bank.tasks[0]
    outcomes = tuple(
        observation.outcome
        for observation in result.bank.observations
        if observation.candidate_hash == result.final_candidate.content_hash
    )
    if tuple(
        observation.test_id
        for observation in result.bank.observations
        if observation.candidate_hash == result.final_candidate.content_hash
    ) != task.test_order:
        raise ValueError("final candidate lacks a complete fixed-order evaluation")
    return float(bool(outcomes) and all(outcome == "PASS" for outcome in outcomes))


def derive_stage_gates(
    prediction: PredictionReport | None,
    repair: RepairResult | None,
    evidence: GateEvidence | None,
) -> tuple[bool, bool]:
    if evidence is not None:
        raise ValueError("caller-supplied gate evidence is forbidden")
    return False, False


def validate_runtime_model_binding(
    configured_model: Mapping[str, Any],
    sequence_backend: Any,
    *,
    execution_mode: str | None,
) -> tuple[str | None, str | None, bool]:
    runtime_identity = (
        getattr(sequence_backend, "model_id", None),
        getattr(sequence_backend, "revision", None),
    )
    configured_identity = (configured_model.get("id"), configured_model.get("revision"))
    smoke_backend = bool(getattr(sequence_backend, "smoke_only", False))
    if runtime_identity != configured_identity:
        raise ValueError("actual repair backend model id/revision does not match config")
    if smoke_backend != (execution_mode == "deterministic_smoke"):
        raise ValueError("injected fake backends require explicit deterministic_smoke execution")
    return runtime_identity[0], runtime_identity[1], smoke_backend


def validate_evaluator_truth_binding(
    expected: Mapping[tuple[str, str], str],
    evaluator_observations: Sequence[Any],
    initial_candidate_hashes: set[str],
) -> None:
    observed = {
        (observation.candidate_hash, observation.test_id): observation.outcome
        for observation in evaluator_observations
        if observation.candidate_hash in initial_candidate_hashes
    }
    if observed != dict(expected):
        raise ValueError(
            "prediction evaluator truth is inconsistent with executed initial-candidate outcomes"
        )


def run_arm(
    *,
    config: Mapping[str, Any],
    bank: TrajectoryBank,
    predictor: Any,
    scorer: Any,
    sequence_backend: Any,
    sandbox: Any,
    prefix_cutoffs: Mapping[str, int],
    future_outcomes_by_candidate: Mapping[tuple[str, str], str],
    particles: Sequence[Any],
    previous_log_weights: np.ndarray,
    log_likelihood: Callable[[Any, Sequence[Any], np.ndarray], np.ndarray],
    log_transition: Callable[[Any, Sequence[Any], np.ndarray], np.ndarray],
    log_proposal: Callable[[Any, Sequence[Any], np.ndarray], np.ndarray],
    budget: RepairBudget,
    gate_evidence: GateEvidence | None,
    rng: np.random.Generator,
    artifact_path: str | Path,
) -> dict[str, Any]:
    orchestration_cpu_start = time.process_time()
    orchestration_wall_start = time.perf_counter()
    if gate_evidence is not None:
        raise ValueError("caller-supplied gate evidence is forbidden")
    validated = validate_experiment(config, for_execution=True)
    configured_budget = RepairBudget(**validated["budget"])
    if budget != configured_budget:
        raise ValueError("run budget must exactly match the validated config budget")
    configured_model = validated["model"]
    validate_initial_candidate_bank(
        bank.candidates,
        dict(bank.mutant_registry),
        expected_model_id=configured_model["id"],
        expected_model_revision=configured_model["revision"],
    )
    runtime_model_id, runtime_revision, smoke_backend = validate_runtime_model_binding(
        configured_model,
        sequence_backend,
        execution_mode=validated.get("execution_mode"),
    )
    if not all(callable(item) for item in (log_likelihood, log_transition, log_proposal)):
        raise TypeError(
            "sequential particle terms must be callables over each observation and current particles"
        )
    backend = ConditionedRepairBackend(
        scorer=scorer,
        sequence_backend=sequence_backend,
        particles=particles,
        log_weights=np.asarray(previous_log_weights, dtype=np.float64),
        rng=rng,
    )
    posterior_trace: list[dict[str, Any]] = []

    def term(value: Any, observation: Any) -> np.ndarray:
        resolved = value(observation, backend.particles, backend.log_weights)
        return np.asarray(resolved, dtype=np.float64)

    def observe(observation: Any) -> None:
        update = particle_update(
            backend.log_weights,
            term(log_likelihood, observation),
            term(log_transition, observation),
            term(log_proposal, observation),
            rng=rng,
            ess_fraction=float(validated["filtering"]["ess_fraction"]),
        )
        updated_particles = backend.particles
        if update.resampled:
            updated_particles = tuple(
                backend.particles[int(index)] for index in update.ancestors
            )
        backend.update_belief(updated_particles, update.log_weights)
        posterior_trace.append(
            {
                "candidate_hash": observation["candidate_hash"],
                "test_id": observation["test_id"],
                "ess": update.ess,
                "resampled": update.resampled,
                "ancestors": update.ancestors.tolist(),
            }
        )

    for existing_observation in bank.observations:
        observe(
            feedback_view(
                (existing_observation,), validated.get("feedback_mode", "status_only")
            )[0]
        )
    prediction = run_prediction(
        bank,
        predictor,
        prefix_cutoffs=prefix_cutoffs,
        future_outcomes_by_candidate=future_outcomes_by_candidate,
        feedback_mode=validated.get("feedback_mode", "status_only"),
    )
    repair = run_repair(
        bank,
        backend=backend,
        sandbox=sandbox,
        budget=budget,
        rounds=validated["protocol"]["rounds"],
        feedback_mode=validated.get("feedback_mode", "status_only"),
        observation_callback=observe,
        max_visible_tokens_per_call=validated["protocol"]["visible_token_budget"],
        _prior_cpu_seconds=time.process_time() - orchestration_cpu_start,
        _prior_wall_seconds=time.perf_counter() - orchestration_wall_start,
    )
    initial_candidate_hashes = {candidate.content_hash for candidate in bank.candidates}
    validate_evaluator_truth_binding(
        future_outcomes_by_candidate,
        repair.evaluator_observations,
        initial_candidate_hashes,
    )
    prediction_passed, repair_passed = derive_stage_gates(
        prediction, repair, gate_evidence
    )
    observations_by_candidate = {
        candidate.content_hash: tuple(
            observation
            for observation in repair.bank.observations
            if observation.candidate_hash == candidate.content_hash
        )
        for candidate in repair.bank.candidates
    }
    outcomes_by_candidate = {
        candidate_hash: {item.test_id: item.outcome for item in observations}
        for candidate_hash, observations in observations_by_candidate.items()
    }
    round_pass_at_one = [
        float(
            len(outcomes_by_candidate[event["new_candidate"]])
            == len(repair.bank.tasks[0].test_order)
            and all(
                outcome == "PASS"
                for outcome in outcomes_by_candidate[event["new_candidate"]].values()
            )
        )
        for event in repair.events
    ]
    regressions = sum(
        1
        for event in repair.events
        for test_id in repair.bank.tasks[0].test_order
        if outcomes_by_candidate[event["selected_candidate"]].get(test_id) == "PASS"
        and outcomes_by_candidate[event["new_candidate"]].get(test_id) != "PASS"
    )
    candidate_by_hash = {
        candidate.content_hash: candidate for candidate in repair.bank.candidates
    }
    patch_size = sum(
        abs(
            len(candidate_by_hash[event["new_candidate"]].code)
            - len(candidate_by_hash[event["selected_candidate"]].code)
        )
        for event in repair.events
    )
    failure_counts = {
        outcome: sum(
            observation.outcome == outcome for observation in repair.bank.observations
        )
        for outcome in (
            "WRONG_OUTPUT",
            "RUNTIME_EXCEPTION",
            "TIMEOUT",
            "COMPILE_ERROR",
        )
    }
    physical_kv_bytes = sum(
        int(item["kv_delta"].physical_bytes)
        for item in particles
        if isinstance(item, Mapping) and "kv_delta" in item
    )
    evaluator_truth_hash = canonical_hash(
        [
            (candidate_hash, test_id, outcome)
            for (candidate_hash, test_id), outcome in sorted(
                future_outcomes_by_candidate.items()
            )
        ]
    )
    payload = {
        "status": validated.get("claim_status", "unlabeled-no-claim"),
        "config_hash": canonical_config_hash(validated),
        "prediction_bank_hash": prediction.bank_hash,
        "evaluator_truth_ref": f"sha256:{evaluator_truth_hash}",
        "evaluator_truth_hash": evaluator_truth_hash,
        "runtime_model": {
            "id": runtime_model_id,
            "revision": runtime_revision,
            "smoke_only": smoke_backend,
        },
        "stage_gate_status": "disabled-until-registered-multiseed-aggregator",
        "particle_updates": len(posterior_trace),
        "particle_ess": posterior_trace[-1]["ess"] if posterior_trace else None,
        "particle_ess_trace": [item["ess"] for item in posterior_trace],
        "particle_resampled": any(item["resampled"] for item in posterior_trace),
        "conditioned_repairs": backend.conditioned_repairs,
        "sequence_backend": getattr(
            sequence_backend, "audit_label", type(sequence_backend).__name__
        ),
        "prediction_future_nll": prediction.future_nll,
        "prediction_brier": prediction.brier,
        "prediction_gate": prediction_passed,
        "repair_gate": repair_passed,
        "advance_stage": bool(prediction_passed and repair_passed),
        "rounds_completed": repair.rounds_completed,
        "final_program_count": len(repair.final_programs),
        "round_pass_at_1": round_pass_at_one,
        "final_pass_at_1": _pass_at_one(repair),
        "regressions": regressions,
        "patch_size": patch_size,
        "failure_counts": failure_counts,
        "physical_kv_bytes": physical_kv_bytes,
        "model_calls": repair.model_calls,
        "execution_count": len(repair.execution_ledger.executions),
        "visible_tokens": repair.token_ledger.visible,
        "generated_tokens": repair.token_ledger.generated,
        "cpu_seconds": repair.compute_ledger.cpu_seconds,
        "wall_seconds": repair.compute_ledger.wall_seconds,
        "gpu_hours": repair.compute_ledger.gpu_hours,
    }
    write_run_bundle(
        artifact_path,
        report=payload,
        config=validated,
        bank=repair.bank,
        prediction=prediction,
        repair=repair,
    )
    return payload


def run_deterministic_smoke(artifact_path: str | Path) -> dict[str, Any]:
    task = TaskRecord(
        task_id="smoke-task",
        dataset="finite",
        public_prompt="repair the deterministic fake",
        test_order=("t1", "t2"),
        group_ids=("smoke-source",),
    )
    mutant_registry = {
        f"registered-mutant-{index - 6}": canonical_hash(
            {"code": f"candidate {index}"}
        )
        for index in range(6, 8)
    }
    registry_hash = canonical_hash(mutant_registry)
    stored_logprobs = (-0.12, -0.27, -0.44, -0.61, -0.79, -0.96)
    samples = tuple(
        CandidateVersion.create(
            f"c{index}",
            task.task_id,
            f"candidate {index}",
            provenance="model_sample",
            generation_logprob=stored_logprobs[index],
            sampling_temperature=0.8,
            sampling_top_p=0.95,
            model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            model_revision="a" * 40,
            generation_trace_hash=canonical_hash(
                {
                    "code": f"candidate {index}",
                    "stored_logprob": stored_logprobs[index],
                }
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
            f"candidate {index + 6}",
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
    bank = TrajectoryBank(
        (task,), candidates, (), tuple(sorted(mutant_registry.items()))
    )

    class Predictor:
        def predict(self, public_task, candidate, history):
            return {
                "t1": np.asarray([0.7, 0.1, 0.05, 0.1, 0.05]),
                "t2": np.asarray([0.1, 0.6, 0.1, 0.1, 0.1]),
            }

    class Scorer:
        def predict_success(self, public_task, candidate, history, *, request_text):
            return 0.5

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def __call__(self, prompt, return_tensors):
            return {
                "input_ids": np.asarray([[1, 2, 3]]),
                "attention_mask": np.ones((1, 3), dtype=np.int64),
            }

        def decode(self, tokens, skip_special_tokens):
            values = np.asarray(tokens).reshape(-1)
            return "repair-token-" + "-".join(str(int(value)) for value in values)

    class FakeTransformersModel:
        device = None

        def __call__(self, **inputs):
            from types import SimpleNamespace

            cache = tuple(
                (
                    np.ones((1, 1, 2, 3), dtype=np.float64),
                    np.ones((1, 1, 2, 3), dtype=np.float64),
                )
                for _ in range(2)
            )
            return SimpleNamespace(past_key_values=cache)

        def generate(self, **inputs):
            from types import SimpleNamespace

            cache_signal = float(np.asarray(inputs["past_key_values"][0][0]).sum())
            token = 4 if cache_signal < 6.0 else 5
            scores = np.zeros((1, 8), dtype=np.float64)
            scores[0, token] = 3.0
            return SimpleNamespace(
                sequences=np.asarray([[3, token]]), scores=(scores,)
            )

    sequence_backend = TransformersRepairBackend(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "a" * 40,
        model=FakeTransformersModel(),
        tokenizer=FakeTokenizer(),
        success_scorer=Scorer(),
        generation={"max_new_tokens": 1},
        smoke_only=True,
    )
    sequence_backend.audit_label = "TransformersRepairBackend(fake-local)"
    delta_options = {
        "layers": 2,
        "heads": 1,
        "head_dim": 3,
        "rank": 1,
        "mode": "kv",
        "active_layers": 2,
        "target_norm": 1.0,
    }
    particles = (
        {
            "id": "particle-0",
            "kv_delta": build_kv_delta(
                np.asarray([1.0, 0.0]), rng=np.random.default_rng(21), **delta_options
            ),
        },
        {
            "id": "particle-1",
            "kv_delta": build_kv_delta(
                np.asarray([0.0, 1.0]), rng=np.random.default_rng(21), **delta_options
            ),
        },
    )

    config = {
        "name": "deterministic-smoke",
        "formal": False,
        "profile": "cpu",
        "model": {"id": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "revision": "a" * 40},
        "data": {"id": "finite", "revision": "b" * 40},
        "container": {"digest": "sha256:" + "c" * 64},
        "hardware": {"context": 512},
        "protocol": {
            "fixed_test_order": True,
            "rounds": 4,
            "early_stop": False,
            "candidates": 8,
            "visible_token_budget": 512,
        },
        "feedback_mode": "status_only",
        "budget": {
            "max_visible_tokens": 50_000,
            "max_generated_tokens": 500,
            "max_model_calls": 50,
            "max_executions": 24,
            "max_cpu_seconds": 60.0,
            "max_wall_seconds": 60.0,
            "max_gpu_hours": 0.0,
        },
        "candidate_bank": {
            "model_samples": 6,
            "deterministic_mutants": 2,
            "temperature": 0.8,
            "top_p": 0.95,
            "mutation_registry_required": True,
        },
        "filtering": {"ess_fraction": 0.75},
        "execution_mode": "deterministic_smoke",
        "claim_status": "smoke-only-no-claim",
    }
    return run_arm(
        config=config,
        bank=bank,
        predictor=Predictor(),
        scorer=Scorer(),
        sequence_backend=sequence_backend,
        sandbox=FakeSandbox(
            {
                (candidate.code, test_id): outcome
                for candidate in candidates
                for test_id, outcome in zip(
                    task.test_order, ("PASS", "WRONG_OUTPUT"), strict=True
                )
            }
        ),
        prefix_cutoffs={task.task_id: 0},
        future_outcomes_by_candidate={
            (candidate.content_hash, test_id): outcome
            for candidate in candidates
            for test_id, outcome in zip(
                task.test_order, ("PASS", "WRONG_OUTPUT"), strict=True
            )
        },
        particles=particles,
        previous_log_weights=np.log([0.5, 0.5]),
        log_likelihood=lambda observation, current_particles, weights: np.log(
            [0.8 if item["id"] == "particle-0" else 0.2 for item in current_particles]
        ),
        log_transition=lambda observation, current_particles, weights: np.log(
            [0.7 if item["id"] == "particle-0" else 0.3 for item in current_particles]
        ),
        log_proposal=lambda observation, current_particles, weights: np.log(
            [0.6 if item["id"] == "particle-0" else 0.4 for item in current_particles]
        ),
        budget=RepairBudget(50_000, 500, 50, 24, 60.0, 60.0, 0.0),
        gate_evidence=None,
        rng=np.random.default_rng(17),
        artifact_path=artifact_path,
    )
