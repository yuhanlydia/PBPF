"""Diagnostic-component mutual information for active public-test selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np


_NORMALIZATION_ATOL = 1e-8
_MI_ROUNDOFF_ATOL = 1e-12


def _probability_vector(values: Any, *, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError(f"{name} must be a nonempty probability vector")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.isclose(probabilities.sum(), 1.0, atol=_NORMALIZATION_ATOL, rtol=0.0):
        raise ValueError(f"{name} must sum to one")
    return probabilities


def _component_outcome_matrix(values: Any, *, components: int) -> np.ndarray:
    outcomes = np.asarray(values, dtype=np.float64)
    if outcomes.ndim != 2 or outcomes.shape[0] != components or outcomes.shape[1] == 0:
        raise ValueError("component_outcomes must have shape [components, outcomes]")
    if not np.isfinite(outcomes).all() or (outcomes < 0).any():
        raise ValueError("component_outcomes must be finite and non-negative")
    if not np.allclose(outcomes.sum(axis=1), 1.0, atol=_NORMALIZATION_ATOL, rtol=0.0):
        raise ValueError("each diagnostic component outcome prediction must sum to one")
    return outcomes


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return float(-(positive * np.log(positive)).sum())


def expected_information_gain(component_probs: Any, component_outcomes: Any) -> float:
    """Return ``I(component; outcome)`` for one prospective public test.

    ``component_probs`` is the posterior over an explicit discrete diagnostic
    component, while ``component_outcomes[k]`` is that component's predicted
    outcome distribution.  This deliberately has no particle-weight or latent
    entropy input: its value is invariant to particle coordinates and measures
    only diagnostic information in the observable prospective outcome.
    """
    components = _probability_vector(component_probs, name="component_probs")
    outcome_predictions = _component_outcome_matrix(component_outcomes, components=len(components))
    marginal = components @ outcome_predictions
    information = _entropy(marginal) - float(
        sum(weight * _entropy(prediction) for weight, prediction in zip(components, outcome_predictions, strict=True))
    )
    # Numerical cancellation can produce a tiny negative value for independent
    # predictions; a material negative value indicates invalid computation.
    if information < -_MI_ROUNDOFF_ATOL:
        raise ValueError("mutual information is materially negative")
    return float(max(0.0, information))


def predictive_information_gain(component_probs: Any, query_outcomes: Any,
                                target_outcomes: Any) -> float:
    """Mean I(query outcome; target outcome), in nats, under the model.

    Targets have shape [components, targets, outcomes]. Query and target
    outcomes are conditionally independent given a component. Inputs are
    predictions only: no observed query or target labels enter this score.
    This is expected target log-loss reduction for exact Bayesian updating
    under this finite mixture, not information about a diagnostic identity.
    """
    weights = _probability_vector(component_probs, name='component_probs')
    query = _component_outcome_matrix(query_outcomes, components=len(weights))
    targets = np.asarray(target_outcomes, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[0] != len(weights) or targets.shape[1] == 0:
        raise ValueError('target_outcomes must have shape [components, nonempty targets, outcomes]')
    for index in range(targets.shape[1]):
        _component_outcome_matrix(targets[:, index], components=len(weights))
    joint = np.einsum('k,ka,ktb->tab', weights, query, targets)
    query_marginal = weights @ query
    target_marginal = np.einsum('k,ktb->tb', weights, targets)
    positive = joint > 0
    t, q, y = np.nonzero(positive)
    log_ratio = np.log(joint[positive]) - np.log(query_marginal[q]) - np.log(target_marginal[t, y])
    information = float(np.sum(joint[positive] * log_ratio) / targets.shape[1])
    if information < -_MI_ROUNDOFF_ATOL:
        raise ValueError('predictive mutual information is materially negative')
    return max(0., information)


def predictive_information_scores(component_probs: Any, query_outcomes: Any,
                                  target_outcomes: Any, *, budget: int = 1) -> np.ndarray:
    """Score each remaining query with a one- or two-step predictive horizon.

    Queries have shape [components, queries, outcomes]; targets have shape
    [components, targets, outcomes]. For budget two, add expected best remaining
    target information after each possible first outcome. This plans under the
    finite mixture without reading actual outcomes. Callers must pass only
    unexecuted queries, observe the chosen query, update weights, then call with
    budget one. All query and target outcomes must be conditionally independent
    given the component. Optimality is model-relative, not a data guarantee.
    """
    if type(budget) is not int or budget not in (1, 2):
        raise ValueError('predictive planning supports budget one or two')
    weights = _probability_vector(component_probs, name='component_probs')
    queries = np.asarray(query_outcomes, dtype=np.float64)
    if queries.ndim != 3 or queries.shape[0] != len(weights) or queries.shape[1] < budget:
        raise ValueError('not enough remaining query predictions for the budget')
    count = queries.shape[1]
    scores = np.array([predictive_information_gain(weights, queries[:, i], target_outcomes)
                       for i in range(count)])
    if budget == 1:
        return scores
    for first in range(count):
        for outcome in range(queries.shape[2]):
            unnormalized = weights * queries[:, first, outcome]
            probability = float(unnormalized.sum())
            if probability == 0:
                continue
            posterior = unnormalized / probability
            continuation = max(predictive_information_gain(posterior, queries[:, second], target_outcomes)
                               for second in range(count) if second != first)
            scores[first] += probability * continuation
    return scores


def pool_predictive_particles(component_probs: Any, query_outcomes: Any,
                              target_outcomes: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool equally weighted posterior draws before computing predictive MI.

    Weights are [draws, particles]; predictions are [draws, particles, tests,
    outcomes]. Each draw is an independently normalized posterior estimate.
    Returns a flat mixture, not averaged MI scores or one pooled IS estimate.
    After observing a query, update the returned joint weights globally; do not
    reset each draw's mass to uniform. Pooling is not a calibration guarantee.
    """
    weights = np.asarray(component_probs, dtype=np.float64)
    if weights.ndim != 2 or min(weights.shape) == 0:
        raise ValueError('posterior draws require nonempty [draws, particles] weights')
    for draw in weights:
        _probability_vector(draw, name='draw weights')
    predictions = []
    for value in (query_outcomes, target_outcomes):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 4 or value.shape[:2] != weights.shape or value.shape[2] == 0:
            raise ValueError('predictions must match draws and particles with nonempty tests')
        flattened = value.reshape((-1, *value.shape[2:]))
        for test in range(flattened.shape[1]):
            _component_outcome_matrix(flattened[:, test], components=weights.size)
        predictions.append(flattened)
    return weights.reshape(-1) / len(weights), predictions[0], predictions[1]


def _state_value(state: Any, name: str, default: Any = ()) -> Any:
    if isinstance(state, Mapping):
        return state.get(name, default)
    return getattr(state, name, default)


def _remaining_predictions(remaining: Any) -> tuple[tuple[Any, Any], ...]:
    if isinstance(remaining, Mapping):
        entries = tuple(remaining.items())
    elif isinstance(remaining, Sequence) and not isinstance(remaining, (str, bytes)):
        entries_list: list[tuple[Any, Any]] = []
        for entry in remaining:
            if isinstance(entry, Mapping):
                if "test_id" not in entry or "component_outcomes" not in entry:
                    raise ValueError("remaining records require test_id and component_outcomes")
                entries_list.append((entry["test_id"], entry["component_outcomes"]))
            else:
                try:
                    test_id, outcomes = entry
                except (TypeError, ValueError) as error:
                    raise ValueError("remaining entries must be (test_id, component_outcomes) pairs") from error
                entries_list.append((test_id, outcomes))
        entries = tuple(entries_list)
    else:
        raise TypeError("remaining must map test IDs to component outcome predictions")
    if not entries:
        raise ValueError("at least one remaining public test is required")
    ids = tuple(test_id for test_id, _ in entries)
    if len(set(ids)) != len(ids):
        raise ValueError("remaining test IDs must be unique")
    return entries


def _fixed_budget(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError("budget must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class TestDecision:
    """Immutable record of one fixed-budget public-test decision."""

    test_id: Any
    information_gain: float
    step: int
    budget: int
    observed_outcome: Any | None = None


@dataclass(frozen=True)
class AcquisitionTrace:
    """Sequential decisions plus the posterior needed by downstream scoring."""

    decisions: tuple[TestDecision, ...]
    final_state: Any
    remaining: Any


class ActiveTestPolicy:
    """Select public tests by diagnostic outcome mutual information.

    Callers update ``state.component_probs`` after observing a selected test and
    pass the reduced ``remaining`` inventory on the next call.  This makes the
    policy adaptive without any hidden mutable selection state.
    """

    def choose(self, state: Any, remaining: Any, budget: int) -> tuple[TestDecision, ...]:
        """Choose exactly ``budget`` distinct, not-yet-executed public tests.

        Fixed-budget evaluation fails closed when fewer eligible tests remain;
        threshold stopping belongs in a separately reported policy/curve.
        """
        requested = _fixed_budget(budget)
        candidates = self._rank(state, remaining, executed=())
        if len(candidates) < requested:
            raise ValueError("budget exceeds the number of unexecuted remaining tests")
        # Python's stable sort preserves declared inventory order for an exact
        # EIG tie, so replay does not depend on hidden labels or random state.
        ranked = sorted(candidates, key=lambda item: -item[2])[:requested]
        return tuple(
            TestDecision(test_id=test_id, information_gain=information, step=step, budget=requested)
            for step, (_, test_id, information) in enumerate(ranked)
        )

    def choose_adaptively(
        self,
        state: Any,
        remaining: Any,
        budget: int,
        observe: Callable[[Any], Any],
        update: Callable[[Any, Any, Any, Any], tuple[Any, Any]],
    ) -> tuple[TestDecision, ...]:
        """Execute a sequential fixed-budget diagnostic acquisition policy.

        ``observe`` runs the selected public test.  ``update`` receives the
        previous state, test ID, observed outcome, and current remaining
        predictions, and must return the updated diagnostic state and refreshed
        remaining predictions.  EIG is recomputed after every update.
        """
        return self.acquire_trace(state, remaining, budget, observe, update).decisions

    def acquire_trace(
        self,
        state: Any,
        remaining: Any,
        budget: int,
        observe: Callable[[Any], Any],
        update: Callable[[Any, Any, Any, Any], tuple[Any, Any]],
    ) -> AcquisitionTrace:
        """Run adaptive acquisition and retain its final diagnostic posterior."""
        requested = _fixed_budget(budget)
        current_state, current_remaining = state, remaining
        executed = set(_state_value(state, "executed", ()))
        if len(self._rank(current_state, current_remaining, executed=executed)) < requested:
            raise ValueError("budget exceeds the number of unexecuted remaining tests")
        decisions: list[TestDecision] = []
        for step in range(requested):
            ranked = self._rank(current_state, current_remaining, executed=executed)
            if not ranked:
                raise ValueError("budget exceeds the number of unexecuted remaining tests")
            _, test_id, information = ranked[0]
            outcome = observe(test_id)
            decision = replace(
                TestDecision(test_id=test_id, information_gain=information, step=step, budget=requested),
                observed_outcome=outcome,
            )
            decisions.append(decision)
            executed.add(test_id)
            try:
                current_state, current_remaining = update(
                    current_state, test_id, outcome, current_remaining
                )
            except (TypeError, ValueError) as error:
                raise ValueError("update must return (updated_state, updated_remaining)") from error
        return AcquisitionTrace(tuple(decisions), current_state, current_remaining)

    # A short alias supports callers that name this lifecycle operation rather
    # than its selection result.
    acquire = choose_adaptively

    @staticmethod
    def _rank(state: Any, remaining: Any, *, executed: set[Any] | tuple[Any, ...]) -> list[tuple[int, Any, float]]:
        component_probs = _probability_vector(
            _state_value(state, "component_probs", None), name="state.component_probs"
        )
        state_executed = tuple(_state_value(state, "executed", ()))
        if len(set(state_executed)) != len(state_executed):
            raise ValueError("state.executed must not contain duplicate test IDs")
        all_executed = set(executed) | set(state_executed)
        entries = _remaining_predictions(remaining)
        candidates = [
            (position, test_id, expected_information_gain(component_probs, outcomes))
            for position, (test_id, outcomes) in enumerate(entries)
            if test_id not in all_executed
        ]
        # Python's stable sort preserves declared inventory order for an exact
        # EIG tie, so replay does not depend on hidden labels or random state.
        return sorted(candidates, key=lambda item: -item[2])


@dataclass(frozen=True)
class DiagnosticParticleState:
    """Mean-field diagnosis components with a separately marginalized nuisance.

    ``component_probs`` weights diagnosis particles. ``difficulty_probs`` and
    ``difficulty_particles`` form a nuisance marginal that is deliberately
    crossed with every diagnosis component as an explicit interventional
    approximation, rather than being mislabeled as the original correlated SMC
    posterior. This prevents candidate difficulty variation from masquerading
    as diagnostic mutual information; a joint-particle MI baseline remains a
    required ablation.
    """

    component_probs: np.ndarray
    diagnosis_particles: np.ndarray
    difficulty_probs: np.ndarray
    difficulty_particles: np.ndarray
    executed: tuple[Any, ...] = ()

    def __post_init__(self):
        diagnosis = np.asarray(self.diagnosis_particles, dtype=np.float64).copy()
        difficulty = np.asarray(self.difficulty_particles, dtype=np.float64).copy()
        component_probs = _probability_vector(self.component_probs, name="component_probs").copy()
        difficulty_probs = _probability_vector(self.difficulty_probs, name="difficulty_probs").copy()
        if (diagnosis.ndim != 2 or difficulty.ndim != 2 or diagnosis.shape[1] == 0
                or difficulty.shape[1] == 0 or len(diagnosis) != len(component_probs)
                or len(difficulty) != len(difficulty_probs)
                or not np.isfinite(diagnosis).all() or not np.isfinite(difficulty).all()):
            raise ValueError("diagnosis/difficulty particles require finite [components,dimensions] arrays")
        executed = tuple(self.executed)
        if len(set(executed)) != len(executed):
            raise ValueError("executed public-test IDs must be unique")
        for value in (diagnosis, difficulty, component_probs, difficulty_probs):
            value.setflags(write=False)
        object.__setattr__(self, "diagnosis_particles", diagnosis)
        object.__setattr__(self, "difficulty_particles", difficulty)
        object.__setattr__(self, "component_probs", component_probs)
        object.__setattr__(self, "difficulty_probs", difficulty_probs)
        object.__setattr__(self, "executed", executed)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalized_weights(log_weights: Any) -> np.ndarray:
    values = _numpy(log_weights).astype(np.float64, copy=False)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("log_weights must be a finite nonempty vector")
    values = np.exp(values - values.max())
    return values / values.sum()


class NeuralDiagnosticAdapter:
    """Connect a factored neural belief posterior to adaptive public testing.

    Each diagnosis particle is paired interventionally with the complete
    difficulty marginal. Consequently ``p_k(y|t,H)`` varies across diagnosis
    components while integrating out the test-invariant nuisance. This is a
    declared mean-field policy, not the original correlated SMC posterior.
    After an outcome, both marginals receive a parallel mean-field Bayes update.
    """

    def __init__(self, model: Any, task: Any, candidate: Any, tests: Mapping[Any, Any],
                 *, outcome_labels: Sequence[Any] | None = None):
        import torch
        if outcome_labels is None:
            from pbpf.registry import OUTCOMES
            outcome_labels = OUTCOMES

        if getattr(model, "difficulty_dim", None) is None:
            raise ValueError("diagnostic acquisition requires a factored A-PBPF model")
        if not isinstance(tests, Mapping) or not tests:
            raise ValueError("tests must be a nonempty ordered mapping of public test IDs to features")
        parameter = next(model.parameters())
        self.model = model
        self.device, self.dtype = parameter.device, parameter.dtype

        def vector(value: Any, name: str):
            result = torch.as_tensor(value, device=self.device, dtype=self.dtype)
            if result.ndim == 2 and result.shape[0] == 1:
                result = result[0]
            if result.shape != (model.feature_dim,) or not torch.isfinite(result).all():
                raise ValueError(f"{name} must be a finite feature_dim vector")
            return result

        self.task = vector(task, "task")
        self.candidate = vector(candidate, "candidate")
        self.tests = {test_id: vector(features, f"test {test_id!r}")
                      for test_id, features in tests.items()}
        self.outcome_labels = tuple(outcome_labels)
        if (len(self.outcome_labels) != 5 or len(set(self.outcome_labels)) != 5
                or any(isinstance(value, bool) for value in self.outcome_labels)):
            raise ValueError("outcome_labels must contain five unique categorical values")

    def state(self, particles: Any, log_weights: Any, *, executed: Sequence[Any] = ()) -> DiagnosticParticleState:
        values = _numpy(particles).astype(np.float64, copy=False)
        if (values.ndim != 2 or values.shape[1] != self.model.latent_dim
                or not np.isfinite(values).all()):
            raise ValueError("particles must have finite shape [components,latent_dim]")
        weights = _normalized_weights(log_weights)
        if len(values) != len(weights):
            raise ValueError("particles and log_weights require the same component count")
        cut = self.model.difficulty_dim
        return DiagnosticParticleState(weights, values[:, cut:], weights, values[:, :cut], tuple(executed))

    def _joint_outcomes(self, state: DiagnosticParticleState, test_id: Any) -> np.ndarray:
        """Return interventional [diagnosis,difficulty,outcome] predictions."""
        import torch

        if test_id not in self.tests:
            raise KeyError(f"unknown public test ID: {test_id!r}")
        diagnosis = torch.as_tensor(state.diagnosis_particles, device=self.device, dtype=self.dtype)
        difficulty = torch.as_tensor(state.difficulty_particles, device=self.device, dtype=self.dtype)
        diagnosis_count, difficulty_count = len(diagnosis), len(difficulty)
        joint = torch.cat((
            difficulty[None].expand(diagnosis_count, -1, -1),
            diagnosis[:, None].expand(-1, difficulty_count, -1),
        ), dim=-1)
        task = self.task[None].expand(diagnosis_count, -1)
        candidate = self.candidate[None].expand(diagnosis_count, -1)
        test = self.tests[test_id][None].expand(diagnosis_count, -1)
        with torch.no_grad():
            conditional = self.model.likelihood(joint, task, candidate, test).exp()
        predictions = conditional.detach().double().cpu().numpy()
        predictions = np.maximum(predictions, 0.0)
        return predictions / predictions.sum(axis=-1, keepdims=True)

    def component_outcomes(self, state: DiagnosticParticleState, test_id: Any) -> np.ndarray:
        """Return diagnosis-specific outcomes after marginalizing difficulty."""
        joint = self._joint_outcomes(state, test_id)
        predictions = np.einsum("j,kjc->kc", state.difficulty_probs, joint)
        predictions = np.maximum(predictions, 0.0)
        return predictions / predictions.sum(axis=-1, keepdims=True)

    def remaining(self, state: DiagnosticParticleState) -> dict[Any, np.ndarray]:
        return {test_id: self.component_outcomes(state, test_id)
                for test_id in self.tests if test_id not in set(state.executed)}

    def update(self, state: DiagnosticParticleState, test_id: Any, outcome: Any,
               _remaining: Any = None) -> tuple[DiagnosticParticleState, dict[Any, np.ndarray]]:
        """Apply one parallel mean-field Bayes update and refresh all EIG inputs."""
        if test_id in state.executed:
            raise ValueError("a public test cannot update the posterior twice")
        try:
            outcome_index = self.outcome_labels.index(outcome)
        except ValueError as error:
            raise ValueError("observed outcome is outside the configured outcome registry") from error
        joint = self._joint_outcomes(state, test_id)[..., outcome_index]
        diagnosis_likelihood = joint @ state.difficulty_probs
        difficulty_likelihood = state.component_probs @ joint
        diagnosis_posterior = state.component_probs * diagnosis_likelihood
        difficulty_posterior = state.difficulty_probs * difficulty_likelihood
        diagnosis_normalizer = diagnosis_posterior.sum()
        difficulty_normalizer = difficulty_posterior.sum()
        if (not np.isfinite(diagnosis_normalizer) or diagnosis_normalizer <= 0
                or not np.isfinite(difficulty_normalizer) or difficulty_normalizer <= 0):
            raise ValueError("observed outcome has zero posterior predictive probability")
        updated = DiagnosticParticleState(
            diagnosis_posterior / diagnosis_normalizer,
            state.diagnosis_particles,
            difficulty_posterior / difficulty_normalizer,
            state.difficulty_particles,
            (*state.executed, test_id),
        )
        return updated, self.remaining(updated)
