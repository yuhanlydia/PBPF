from __future__ import annotations

import numpy as np

from pbpf.apbpf.counterfactual import (
    association_eligibility,
    joint_permutation,
    outcome_derangement,
)


def test_outcome_derangement_preserves_each_visible_histogram_and_changes_duplicate_history():
    outcomes = np.array([[0, 0, 1, 1, 4], [2, 2, 2, 2, 3]], dtype=np.int64)

    shuffled = outcome_derangement(outcomes, visible_steps=4, seed=1701)

    for original, changed in zip(outcomes, shuffled, strict=True):
        np.testing.assert_array_equal(
            np.bincount(changed[:4], minlength=5), np.bincount(original[:4], minlength=5)
        )
    assert not np.array_equal(shuffled[0, :4], outcomes[0, :4])
    np.testing.assert_array_equal(shuffled[:, 4:], outcomes[:, 4:])


def test_constant_visible_history_is_ineligible_and_is_not_forced_to_change():
    outcomes = np.array([[1, 1, 1, 0], [0, 1, 0, 2]], dtype=np.int64)

    eligible = association_eligibility(outcomes, visible_steps=3)
    shuffled = outcome_derangement(outcomes, visible_steps=3, seed=10)

    np.testing.assert_array_equal(eligible, np.array([False, True]))
    np.testing.assert_array_equal(shuffled[0], outcomes[0])


def test_outcome_derangement_changes_periodic_duplicate_history():
    outcomes = np.array([0, 1, 0, 1], dtype=np.int64)

    shuffled = outcome_derangement(outcomes, visible_steps=4, seed=1)

    assert not np.array_equal(shuffled, outcomes)
    np.testing.assert_array_equal(
        np.bincount(shuffled, minlength=2), np.bincount(outcomes, minlength=2)
    )


def test_joint_permutation_preserves_visible_test_outcome_pairs():
    tests = np.array([["a", "b", "c", "future"]], dtype=object)
    outcomes = np.array([[0, 1, 2, 3]], dtype=np.int64)

    permuted_tests, permuted_outcomes = joint_permutation(
        tests, outcomes, visible_steps=3, seed=8
    )

    assert set(zip(permuted_tests[0, :3], permuted_outcomes[0, :3], strict=True)) == {
        ("a", 0),
        ("b", 1),
        ("c", 2),
    }
    np.testing.assert_array_equal(permuted_tests[:, 3:], tests[:, 3:])
    np.testing.assert_array_equal(permuted_outcomes[:, 3:], outcomes[:, 3:])


def test_counterfactual_permutations_replay_from_the_same_seed():
    tests = np.array([[10, 11, 12], [20, 21, 22]], dtype=np.int64)
    outcomes = np.array([[0, 1, 0], [2, 3, 4]], dtype=np.int64)

    np.testing.assert_array_equal(
        outcome_derangement(outcomes, visible_steps=3, seed=99),
        outcome_derangement(outcomes, visible_steps=3, seed=99),
    )
    first = joint_permutation(tests, outcomes, visible_steps=3, seed=99)
    second = joint_permutation(tests, outcomes, visible_steps=3, seed=99)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
