from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .bank import TrajectoryBank
from .history import feedback_view
from .ledger import ComputeLedger, ExecutionLedger, TokenLedger
from .manifest import CandidateVersion, Observation, trainer_view, validate_initial_candidate_bank


class BudgetExceeded(RuntimeError):
    """Raised when a centrally measured repair budget is exhausted."""


@dataclass(frozen=True)
class RepairBudget:
    max_visible_tokens: int
    max_generated_tokens: int
    max_model_calls: int
    max_executions: int
    max_cpu_seconds: float
    max_wall_seconds: float
    max_gpu_hours: float

    def __post_init__(self) -> None:
        values = (
            self.max_visible_tokens,
            self.max_generated_tokens,
            self.max_model_calls,
            self.max_executions,
            self.max_cpu_seconds,
            self.max_wall_seconds,
            self.max_gpu_hours,
        )
        if any(
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or value < 0
            for value in values
        ):
            raise ValueError("repair budget limits must be finite and non-negative")
        if any(
            type(value) is not int
            for value in (
                self.max_visible_tokens,
                self.max_generated_tokens,
                self.max_model_calls,
                self.max_executions,
            )
        ):
            raise ValueError("token, call, and execution budgets must be integers")


@dataclass(frozen=True)
class CandidateProposal:
    code: str
    generation_logprob: float
    generated_tokens: int | None = None


@dataclass(frozen=True)
class RepairResult:
    initial_bank_hash: str
    final_programs: tuple[str, ...]
    final_candidate: CandidateVersion
    rounds_completed: int
    events: tuple[dict[str, Any], ...]
    bank: TrajectoryBank
    execution_ledger: ExecutionLedger
    token_ledger: TokenLedger
    compute_ledger: ComputeLedger
    model_calls: int
    budget: RepairBudget
    evaluator_observations: tuple[Observation, ...]


def select_active_slot(
    candidates: Sequence[CandidateVersion],
    predicted_success: Mapping[str, float],
    generation_logprobs: Mapping[str, float],
) -> CandidateVersion:
    if not candidates:
        raise ValueError("cannot select from an empty candidate set")
    for candidate in candidates:
        if candidate.content_hash not in predicted_success or candidate.content_hash not in generation_logprobs:
            raise ValueError("every candidate requires a score and generation log-probability")
    return max(
        candidates,
        key=lambda candidate: (
            float(predicted_success[candidate.content_hash]),
            float(generation_logprobs[candidate.content_hash]),
            candidate.candidate_id,
        ),
    )


def _check_budget(
    budget: RepairBudget,
    execution_ledger: ExecutionLedger,
    token_ledger: TokenLedger,
    compute_ledger: ComputeLedger,
    model_calls: int,
) -> None:
    checks = (
        (token_ledger.visible, budget.max_visible_tokens, "visible token"),
        (token_ledger.generated, budget.max_generated_tokens, "generated token"),
        (model_calls, budget.max_model_calls, "model call"),
        (len(execution_ledger.executions), budget.max_executions, "execution"),
        (compute_ledger.cpu_seconds, budget.max_cpu_seconds, "CPU-second"),
        (compute_ledger.wall_seconds, budget.max_wall_seconds, "wall-second"),
        (compute_ledger.gpu_hours, budget.max_gpu_hours, "GPU-hour"),
    )
    for actual, limit, name in checks:
        if actual > limit:
            raise BudgetExceeded(f"{name} budget exceeded: {actual} > {limit}")


def _count_tokens(backend: Any, text: str) -> int:
    counter = getattr(backend, "count_tokens", None)
    if not callable(counter):
        raise TypeError("repair backend must expose count_tokens for central accounting")
    count = counter(text)
    if type(count) is not int or count < 0:
        raise ValueError("backend count_tokens must return a non-negative integer")
    return count


def _gpu_hours(backend: Any) -> float:
    meter = getattr(backend, "gpu_hours_for_last_call", None)
    value = float(meter()) if callable(meter) else 0.0
    if not math.isfinite(value) or value < 0:
        raise ValueError("backend GPU-hour telemetry must be finite and non-negative")
    return value


def _generated_tokens(backend: Any, proposal: CandidateProposal) -> int:
    value = proposal.generated_tokens
    if value is None:
        meter = getattr(backend, "generated_tokens_for_last_call", None)
        if not callable(meter):
            raise TypeError(
                "repair backend must report generated token IDs through CandidateProposal "
                "or generated_tokens_for_last_call"
            )
        value = meter()
    if type(value) is not int or value < 0:
        raise ValueError("generated token telemetry must be a non-negative integer")
    return value


def _request_text(
    operation: str,
    task: Mapping[str, Any],
    candidate: CandidateVersion,
    history: Sequence[Mapping[str, Any]],
    round_index: int | None = None,
) -> str:
    payload = json.dumps(
        {
            "operation": operation,
            "task": dict(task),
            "candidate": {"hash": candidate.content_hash, "code": candidate.code},
            "history": list(history),
            "round": round_index,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    if operation == "repair":
        return (
            "Repair the candidate described in this protocol record. Return only the complete "
            "corrected Python source, with no Markdown fence or explanation.\n"
            + payload
            + "\nCorrected Python source:\n"
        )
    return payload


def run_repair(
    initial_bank: TrajectoryBank,
    *,
    backend: Any,
    sandbox: Any,
    budget: RepairBudget,
    rounds: int = 4,
    feedback_mode: str = "status_only",
    observation_callback: Callable[[Mapping[str, Any]], None] | None = None,
    max_visible_tokens_per_call: int | None = None,
    _prior_cpu_seconds: float = 0.0,
    _prior_wall_seconds: float = 0.0,
) -> RepairResult:
    runtime_cpu_start = time.process_time()
    runtime_wall_start = time.perf_counter()
    if rounds != 4:
        raise ValueError("the preregistered repair protocol requires exactly four rounds")
    if len(initial_bank.tasks) != 1:
        raise ValueError("run_repair operates on one task bank at a time")
    if max_visible_tokens_per_call is not None and (
        type(max_visible_tokens_per_call) is not int or max_visible_tokens_per_call <= 0
    ):
        raise ValueError("per-call visible token limit must be a positive integer")
    task = initial_bank.tasks[0]
    slots = [candidate for candidate in initial_bank.candidates if candidate.task_id == task.task_id]
    validate_initial_candidate_bank(slots, dict(initial_bank.mutant_registry))
    initial_hash = initial_bank.content_hash
    evaluator_observations = list(initial_bank.observations)
    projected_initial = tuple(
        Observation(
            item["candidate_hash"],
            item["test_id"],
            item["outcome"],
        )
        for item in feedback_view(initial_bank.observations, feedback_mode)
    )
    bank = TrajectoryBank(
        initial_bank.tasks,
        initial_bank.candidates,
        projected_initial,
        initial_bank.mutant_registry,
    )
    execution_ledger = ExecutionLedger()
    token_ledger = TokenLedger()
    compute_ledger = ComputeLedger()
    if any(
        not math.isfinite(value) or value < 0
        for value in (_prior_cpu_seconds, _prior_wall_seconds)
    ):
        raise ValueError("prior orchestration compute must be finite and non-negative")
    compute_ledger.record(
        cpu_seconds=_prior_cpu_seconds,
        wall_seconds=_prior_wall_seconds,
        gpu_hours=0.0,
    )
    model_calls = 0
    for observation in initial_bank.observations:
        execution_ledger.record(task.task_id, observation.test_id)
    for candidate in slots:
        token_ledger.record(visible=candidate.visible_tokens, generated=candidate.generated_tokens)
        compute_ledger.record(
            cpu_seconds=candidate.cpu_seconds,
            wall_seconds=candidate.wall_seconds,
            gpu_hours=candidate.gpu_hours,
        )
        model_calls += candidate.model_calls
    runtime_cpu_last = runtime_cpu_start
    runtime_wall_last = runtime_wall_start

    def sync_runtime() -> None:
        nonlocal runtime_cpu_last, runtime_wall_last
        cpu_now = time.process_time()
        wall_now = time.perf_counter()
        compute_ledger.record(
            cpu_seconds=max(cpu_now - runtime_cpu_last, 0.0),
            wall_seconds=max(wall_now - runtime_wall_last, 0.0),
            gpu_hours=0.0,
        )
        runtime_cpu_last = cpu_now
        runtime_wall_last = wall_now

    sync_runtime()
    _check_budget(budget, execution_ledger, token_ledger, compute_ledger, model_calls)

    def backend_call(
        request: str,
        call: Callable[[], Any],
        generated: Callable[[Any], int] | None = None,
    ) -> tuple[Any, dict[str, float | int]]:
        nonlocal model_calls
        sync_runtime()
        _check_budget(budget, execution_ledger, token_ledger, compute_ledger, model_calls)
        visible = _count_tokens(backend, request)
        if max_visible_tokens_per_call is not None and visible > max_visible_tokens_per_call:
            raise BudgetExceeded(
                f"per-call visible token budget exceeded: {visible} > {max_visible_tokens_per_call}"
            )
        if visible > budget.max_visible_tokens - token_ledger.visible:
            raise BudgetExceeded("visible token budget would be exceeded by next request")
        if model_calls >= budget.max_model_calls:
            raise BudgetExceeded("model call budget would be exceeded by next request")
        if compute_ledger.cpu_seconds >= budget.max_cpu_seconds:
            raise BudgetExceeded("CPU-second budget exhausted before next request")
        if compute_ledger.wall_seconds >= budget.max_wall_seconds:
            raise BudgetExceeded("wall-second budget exhausted before next request")
        token_ledger.record(visible=visible, generated=0)
        model_calls += 1
        _check_budget(budget, execution_ledger, token_ledger, compute_ledger, model_calls)
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        value = call()
        cpu_seconds = max(time.process_time() - cpu_start, 1e-12)
        wall_seconds = max(time.perf_counter() - wall_start, 1e-12)
        generated_count = generated(value) if generated else 0
        token_ledger.record(visible=0, generated=generated_count)
        gpu_hours = _gpu_hours(backend)
        compute_ledger.record(
            cpu_seconds=0.0,
            wall_seconds=0.0,
            gpu_hours=gpu_hours,
        )
        sync_runtime()
        _check_budget(budget, execution_ledger, token_ledger, compute_ledger, model_calls)
        return value, {
            "visible_tokens": visible,
            "generated_tokens": generated_count,
            "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "gpu_hours": gpu_hours,
        }

    histories: dict[str, tuple[Observation, ...]] = {
        candidate.content_hash: tuple(
            observation
            for observation in bank.observations
            if observation.candidate_hash == candidate.content_hash
        )
        for candidate in slots
    }

    def execute(candidate: CandidateVersion) -> tuple[Observation, ...]:
        nonlocal bank
        observations = list(histories.get(candidate.content_hash, ()))
        for test_id in task.test_order[len(observations):]:
            if len(execution_ledger.executions) >= budget.max_executions:
                raise BudgetExceeded("execution budget would be exceeded by next sandbox call")
            if compute_ledger.cpu_seconds >= budget.max_cpu_seconds:
                raise BudgetExceeded("CPU-second budget exhausted before next sandbox call")
            if compute_ledger.wall_seconds >= budget.max_wall_seconds:
                raise BudgetExceeded("wall-second budget exhausted before next sandbox call")
            sync_runtime()
            _check_budget(budget, execution_ledger, token_ledger, compute_ledger, model_calls)
            result = sandbox.execute(candidate.code, test_id)
            sync_runtime()
            if result.infrastructure_failure:
                raise RuntimeError(f"infrastructure failure for {task.task_id}/{test_id}")
            raw_observation = Observation(
                candidate.content_hash, test_id, result.outcome, result.feedback
            )
            evaluator_observations.append(raw_observation)
            projected = feedback_view((raw_observation,), feedback_mode)[0]
            observation = Observation(
                projected["candidate_hash"],
                projected["test_id"],
                projected["outcome"],
            )
            bank = bank.with_observation(observation)
            observations.append(observation)
            execution_ledger.record(task.task_id, test_id)
            if observation_callback is not None:
                observation_callback(projected)
                sync_runtime()
            _check_budget(budget, execution_ledger, token_ledger, compute_ledger, model_calls)
        return tuple(observations)

    generation = {candidate.content_hash: float(candidate.generation_logprob) for candidate in slots}
    for candidate in slots:
        histories[candidate.content_hash] = execute(candidate)
    events: list[dict[str, Any]] = []
    public_task = trainer_view(task)

    def score(candidate: CandidateVersion) -> float:
        visible_history = feedback_view(histories[candidate.content_hash], feedback_mode)
        request = _request_text("predict_success", public_task, candidate, visible_history)
        value, _ = backend_call(
            request,
            lambda: backend.predict_success(
                public_task,
                candidate,
                visible_history,
                request_text=request,
            ),
        )
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("predicted success must be finite")
        return result

    for round_index in range(1, rounds + 1):
        scores = {candidate.content_hash: score(candidate) for candidate in slots}
        unsolved = [
            candidate
            for candidate in slots
            if not (
                len(histories[candidate.content_hash]) == len(task.test_order)
                and all(
                    observation.outcome == "PASS"
                    for observation in histories[candidate.content_hash]
                )
            )
        ]
        # Fixed-round evaluation continues after success, but already-correct
        # candidates remain immutable elites while an unsolved slot exists.
        active = select_active_slot(unsolved or slots, scores, generation)
        visible_history = feedback_view(histories[active.content_hash], feedback_mode)
        request = _request_text("repair", public_task, active, visible_history, round_index)
        remaining_generated = budget.max_generated_tokens - token_ledger.generated
        if remaining_generated <= 0:
            raise BudgetExceeded("generated token budget exhausted before repair generation")
        proposal, usage = backend_call(
            request,
            lambda: backend.repair(
                public_task,
                active,
                visible_history,
                round_index,
                request_text=request,
                max_generated_tokens=remaining_generated,
            ),
            generated=lambda value: _generated_tokens(backend, value),
        )
        if not isinstance(proposal, CandidateProposal):
            raise TypeError("repair backend must return CandidateProposal")
        if not math.isfinite(float(proposal.generation_logprob)):
            raise ValueError("repair proposal requires a finite generation log-probability")
        repaired = active.with_patch(
            proposal.code,
            generation_logprob=proposal.generation_logprob,
            **usage,
        )
        bank = bank.with_candidate(repaired)
        histories[repaired.content_hash] = execute(repaired)
        generation[repaired.content_hash] = proposal.generation_logprob
        slots[slots.index(active)] = repaired
        events.append(
            {
                "round": round_index,
                "selected_candidate": active.content_hash,
                "new_candidate": repaired.content_hash,
                "test_order": task.test_order,
                "feedback_mode": feedback_mode,
                "usage": usage,
            }
        )
    final_scores = {candidate.content_hash: score(candidate) for candidate in slots}
    final_candidate = select_active_slot(slots, final_scores, generation)
    sync_runtime()
    _check_budget(budget, execution_ledger, token_ledger, compute_ledger, model_calls)
    return RepairResult(
        initial_hash,
        (final_candidate.code,),
        final_candidate,
        rounds,
        tuple(events),
        bank,
        execution_ledger,
        token_ledger,
        compute_ledger,
        model_calls,
        budget,
        tuple(evaluator_observations),
    )


def run_repair_arms(
    initial_bank: TrajectoryBank,
    *,
    backends: Mapping[str, Any],
    sandboxes: Mapping[str, Any],
    budget: RepairBudget,
    rounds: int = 4,
    feedback_mode: str = "status_only",
) -> dict[str, RepairResult]:
    if set(backends) != set(sandboxes):
        raise ValueError("backend and sandbox arms must match")
    results = {
        arm: run_repair(
            initial_bank,
            backend=backend,
            sandbox=sandboxes[arm],
            rounds=rounds,
            budget=budget,
            feedback_mode=feedback_mode,
        )
        for arm, backend in backends.items()
    }
    if {result.initial_bank_hash for result in results.values()} != {initial_bank.content_hash}:
        raise ValueError("arms did not share the immutable initial candidate bank")
    if len({result.execution_ledger.snapshot() for result in results.values()}) != 1:
        raise ValueError("arms did not execute the same candidate/test budget")
    if {result.model_calls for result in results.values()} != {50}:
        raise ValueError("arms did not use the fixed model-call budget")
    return results
