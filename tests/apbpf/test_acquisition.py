from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pbpf.apbpf import expected_information_gain as package_expected_information_gain
from pbpf.apbpf.acquisition import ActiveTestPolicy, expected_information_gain


def test_expected_information_gain_matches_hand_computed_diagnostic_mutual_information():
    information = expected_information_gain(
        component_probs=np.array([0.5, 0.5]),
        component_outcomes=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    assert information == pytest.approx(np.log(2.0))


def test_diagnostic_information_gain_is_part_of_the_apbpf_public_contract():
    assert package_expected_information_gain is expected_information_gain


def test_expected_information_gain_is_zero_when_components_predict_the_same_outcome():
    information = expected_information_gain(
        component_probs=np.array([0.2, 0.8]),
        component_outcomes=np.array([[0.3, 0.7], [0.3, 0.7]]),
    )

    assert information == pytest.approx(0.0)


def test_active_policy_selects_highest_diagnostic_information_without_repeating_executed_tests():
    policy = ActiveTestPolicy()
    decisions = policy.choose(
        state={"component_probs": np.array([0.5, 0.5]), "executed": ("already-run",)},
        remaining={
            "already-run": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "uninformative": np.array([[0.4, 0.6], [0.4, 0.6]]),
            "diagnostic": np.array([[1.0, 0.0], [0.0, 1.0]]),
        },
        budget=1,
    )

    assert [decision.test_id for decision in decisions] == ["diagnostic"]
    assert decisions[0].information_gain == pytest.approx(np.log(2.0))
    with pytest.raises(FrozenInstanceError):
        decisions[0].test_id = "mutated"  # type: ignore[misc]


def test_active_policy_fixed_budget_selects_exactly_that_many_distinct_tests_or_fails_closed():
    state = {"component_probs": np.array([0.5, 0.5])}
    remaining = {
        "first": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "second": np.array([[0.9, 0.1], [0.1, 0.9]]),
    }

    decisions = ActiveTestPolicy().choose(state, remaining, budget=2)

    assert len(decisions) == 2
    assert len({decision.test_id for decision in decisions}) == 2
    with pytest.raises(ValueError, match="budget"):
        ActiveTestPolicy().choose(state, remaining, budget=3)


def test_active_policy_recomputes_after_each_observation_before_selecting_the_next_test():
    initial_remaining = {
        "first": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "second": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "third": np.array([[0.6, 0.4], [0.4, 0.6]]),
    }
    observed_ids: list[str] = []
    updates: list[tuple[str, str]] = []

    def observe(test_id: str) -> str:
        observed_ids.append(test_id)
        return "PASS" if test_id == "first" else "FAIL"

    def update(state, test_id: str, outcome: str, remaining):
        updates.append((test_id, outcome))
        if test_id == "first":
            return (
                {"component_probs": np.array([0.5, 0.5])},
                {
                    "second": np.array([[0.4, 0.6], [0.4, 0.6]]),
                    "third": np.array([[1.0, 0.0], [0.0, 1.0]]),
                },
            )
        return state, remaining

    decisions = ActiveTestPolicy().choose_adaptively(
        state={"component_probs": np.array([0.5, 0.5])},
        remaining=initial_remaining,
        budget=2,
        observe=observe,
        update=update,
    )

    assert [decision.test_id for decision in decisions] == ["first", "third"]
    assert [decision.observed_outcome for decision in decisions] == ["PASS", "FAIL"]
    assert [decision.budget for decision in decisions] == [2, 2]
    assert observed_ids == ["first", "third"]
    assert updates == [("first", "PASS"), ("third", "FAIL")]


def test_adaptive_policy_rejects_insufficient_initial_inventory_before_executing_any_test():
    observed_ids: list[str] = []

    with pytest.raises(ValueError, match="budget"):
        ActiveTestPolicy().choose_adaptively(
            state={"component_probs": np.array([0.5, 0.5])},
            remaining={"only-test": np.array([[1.0, 0.0], [0.0, 1.0]])},
            budget=2,
            observe=lambda test_id: observed_ids.append(test_id),
            update=lambda state, test_id, outcome, remaining: (state, remaining),
        )

    assert observed_ids == []
