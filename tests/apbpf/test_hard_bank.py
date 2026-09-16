from __future__ import annotations

import numpy as np
import pytest

from pbpf.apbpf.hard_bank import (
    audit_hard_bank,
    lock_hard_bank_population,
    select_development_eligible_groups,
)


def _lock(group_ids, visible_outcomes):
    return lock_hard_bank_population(
        group_ids,
        visible_outcomes,
        split="test",
        provenance="pre-hidden-bank-manifest-v1",
    )


def test_hard_bank_audit_marks_visible_baseline_saturation_and_reports_zero_headroom():
    visible = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]]])
    report = audit_hard_bank(
        visible_outcomes=visible,
        hidden_labels=np.array([[1, 0], [0, 1]]),
        group_ids=("task-a", "task-b"),
        population_lock=_lock(("task-a", "task-b"), visible),
    )

    assert report.saturated
    assert report.visible_selected_pass_at_1 == pytest.approx(1.0)
    assert report.hidden_oracle_pass_at_k == pytest.approx(1.0)
    assert report.headroom == pytest.approx(0.0)


def test_hard_bank_audit_reports_non_saturated_visible_headroom_over_full_population():
    visible = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]]])
    report = audit_hard_bank(
        visible_outcomes=visible,
        hidden_labels=np.array([[0, 0], [1, 0]]),
        group_ids=("task-a", "task-b"),
        population_lock=_lock(("task-a", "task-b"), visible),
    )

    assert report.groups == 2
    assert report.mixed_groups == 1
    assert not report.saturated
    assert report.visible_selected_pass_at_1 == pytest.approx(0.0)
    assert report.hidden_oracle_pass_at_k == pytest.approx(0.5)
    assert report.headroom == pytest.approx(0.5)


def test_hard_bank_audit_rejects_hidden_label_prefiltered_or_reordered_primary_population():
    visible = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]], [[1, 0], [0, 1]]])
    labels = np.array([[0, 0], [1, 0], [1, 0]])
    lock = _lock(("task-a", "task-b", "task-c"), visible)

    with pytest.raises(ValueError, match="population"):
        audit_hard_bank(
            visible[1:], labels[1:], group_ids=("task-b", "task-c"), population_lock=lock
        )
    with pytest.raises(ValueError, match="ordered"):
        audit_hard_bank(
            visible[[1, 0, 2]], labels[[1, 0, 2]],
            group_ids=("task-b", "task-a", "task-c"), population_lock=lock,
        )


def test_hard_bank_audit_reports_groups_with_tied_visible_scores():
    visible = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
    report = audit_hard_bank(
        visible,
        np.array([[0, 1], [1, 0]]),
        group_ids=("tie", "clear"),
        population_lock=_lock(("tie", "clear"), visible),
    )

    assert report.visible_score_tie_groups == 1


def test_development_eligibility_may_use_hidden_labels_only_for_development_groups():
    records = [
        {"group_id": "dev-mixed", "split": "development", "hidden_labels": [0, 1]},
        {"group_id": "dev-flat", "split": "development", "hidden_labels": [1, 1]},
    ]

    selected = select_development_eligible_groups(records, minimum_groups=1)

    assert [record["group_id"] for record in selected] == ["dev-mixed"]
    with pytest.raises(ValueError, match="development"):
        select_development_eligible_groups(
            [{"group_id": "test-mixed", "split": "test", "hidden_labels": [0, 1]}],
            minimum_groups=1,
        )
