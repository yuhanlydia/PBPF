import numpy as np
import pytest

from pbpf.eesd.shapley_relevance import (
    absolute_relevance,
    build_causal_labels,
    coalition_masks,
    edited_token_masks,
    exact_shapley_values,
    leave_one_out_values,
    select_trajectory_indices,
    subset_execution_messages,
)


def test_exact_shapley_recovers_additive_player_contributions():
    contributions = np.array([1.5, -0.25, 0.0, 2.0])
    values = {
        mask: float(sum(contributions[i] for i in range(4) if mask & (1 << i)))
        for mask in range(1 << 4)
    }
    actual = exact_shapley_values(values, 4)
    assert np.allclose(actual, contributions)


def test_exact_shapley_splits_pure_pair_interaction_equally():
    values = {mask: float(bool(mask & 1) and bool(mask & 2)) for mask in range(4)}
    actual = exact_shapley_values(values, 2)
    assert np.allclose(actual, [0.5, 0.5])


def test_leave_one_out_is_not_mislabeled_as_exact_shapley():
    values = {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0}
    assert np.allclose(exact_shapley_values(values, 2), [0.5, 0.5])
    assert np.allclose(leave_one_out_values(values, 2), [1.0, 1.0])


def test_absolute_relevance_preserves_influence_magnitude_and_uniform_fallback():
    assert np.allclose(absolute_relevance([2.0, -1.0, 0.0]), [2.0, 1.0, 0.0])
    assert np.allclose(absolute_relevance([0.0, 0.0, 0.0]), np.ones(3))


def test_edited_token_masks_score_only_changed_regions_on_both_candidates():
    original = [10, 20, 30, 40, 50]
    correction = [10, 20, 31, 40, 60, 50]
    original_mask, correction_mask = edited_token_masks(original, correction)
    assert original_mask.tolist() == [False, False, True, False, False]
    assert correction_mask.tolist() == [False, False, True, False, True, False]


def test_deleted_only_edit_has_original_targets_and_no_correction_targets():
    original_mask, correction_mask = edited_token_masks([1, 2, 3], [1, 3])
    assert original_mask.tolist() == [False, True, False]
    assert correction_mask.tolist() == [False, False]


def test_subset_execution_messages_preserves_order_and_non_evidence_content():
    messages = [
        {"role": "system", "content": "repair"},
        {"role": "user", "content": "Prefix\n" + '{"program":"x","public_executions":[{"id":"0"},{"id":"1"},{"id":"2"},{"id":"3"}],"task":"t"}'},
    ]
    reduced = subset_execution_messages(messages, [0, 2])
    assert reduced[0] == messages[0]
    prefix, payload = reduced[1]["content"].split("\n", 1)
    assert prefix == "Prefix"
    import json
    body = json.loads(payload)
    assert body["program"] == "x"
    assert body["task"] == "t"
    assert [row["id"] for row in body["public_executions"]] == ["0", "2"]


def test_subset_execution_messages_rejects_duplicate_or_out_of_range_indices():
    messages = [
        {"role": "system", "content": "repair"},
        {"role": "user", "content": "P\n" + '{"public_executions":[1,2,3,4]}'},
    ]
    with pytest.raises(ValueError):
        subset_execution_messages(messages, [1, 1])
    with pytest.raises(ValueError):
        subset_execution_messages(messages, [4])


def test_coalition_masks_exact_and_leave_one_out_inventory():
    assert coalition_masks(3, "exact") == list(range(8))
    assert coalition_masks(3, "loo") == [7, 6, 5, 3]


def test_build_causal_labels_marks_only_selected_response_tokens():
    labels = build_causal_labels(3, [10, 11, 12, 13], [False, True, False, True])
    assert labels.tolist() == [-100, -100, -100, -100, 11, -100, 13]


def test_exact_shapley_efficiency_matches_full_minus_empty_value():
    values = {0: -2.0, 1: -1.0, 2: -1.5, 3: 0.25}
    phi = exact_shapley_values(values, 2)
    assert np.isclose(phi.sum(), values[3] - values[0])


def test_select_trajectory_indices_is_deterministic_and_without_replacement():
    a = select_trajectory_indices(100, 16, seed=1701)
    b = select_trajectory_indices(100, 16, seed=1701)
    assert a == b
    assert len(a) == 16
    assert len(set(a)) == 16
    assert all(0 <= i < 100 for i in a)


def test_select_trajectory_indices_full_population_preserves_order():
    assert select_trajectory_indices(5, None, seed=1701) == [0, 1, 2, 3, 4]
    assert select_trajectory_indices(5, 5, seed=999) == [0, 1, 2, 3, 4]


def test_select_trajectory_indices_rejects_invalid_sample_size():
    with pytest.raises(ValueError):
        select_trajectory_indices(4, 5, seed=1701)
    with pytest.raises(ValueError):
        select_trajectory_indices(4, 0, seed=1701)
