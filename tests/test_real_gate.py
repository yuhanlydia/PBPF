import numpy as np

from pbpf.real_gate import (
    FrozenTextEncoder,
    classify_execution,
    compare_predictions,
    select_disjoint_problem_ids,
)


def test_frozen_text_encoder_depends_on_content_and_is_deterministic():
    encoder = FrozenTextEncoder(64)
    first = encoder("for value in values: total += value")
    again = encoder("for value in values: total += value")
    changed = encoder("for value in values: total -= value")
    assert first.shape == (64,)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, changed)


def test_execution_classification_uses_exact_stdout_and_failure_kind():
    assert classify_execution(0, "42\n", "", "42\n") == "PASS"
    assert classify_execution(0, "42", "", "41") == "WRONG_OUTPUT"
    assert classify_execution(1, "", "boom", "") == "RUNTIME_EXCEPTION"
    assert classify_execution(None, "", "", "", timed_out=True) == "TIMEOUT"


def test_problem_selection_is_source_disjoint_and_deterministic():
    selected = select_disjoint_problem_ids(
        {"train": {"a", "b", "shared"}, "development": {"c", "shared"}, "test": {"d", "shared"}},
        {"train": 2, "development": 1, "test": 1},
        seed=17,
    )
    assert set(selected["train"]).isdisjoint(selected["development"])
    assert set(selected["train"]).isdisjoint(selected["test"])
    assert set(selected["development"]).isdisjoint(selected["test"])
    assert "shared" not in set().union(*map(set, selected.values()))
    assert selected == select_disjoint_problem_ids(
        {"train": {"a", "b", "shared"}, "development": {"c", "shared"}, "test": {"d", "shared"}},
        {"train": 2, "development": 1, "test": 1},
        seed=17,
    )


def test_prediction_comparison_reports_paired_delta():
    labels = np.array([0, 1])
    baseline = np.array([[0.6, 0.4], [0.6, 0.4]])
    method = np.array([[0.8, 0.2], [0.2, 0.8]])
    report = compare_predictions(labels, {"baseline": baseline, "pbpf": method})
    assert report["pbpf"]["accuracy"] == 1.0
    assert report["pbpf"]["nll"] < report["baseline"]["nll"]
    assert report["pbpf"]["nll_gain_vs_baseline"] > 0
