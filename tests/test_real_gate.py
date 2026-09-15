import numpy as np

from pbpf.real_gate import (
    FrozenTextEncoder,
    classify_execution,
    compare_predictions,
    fit_histogram_latent,
    histogram_latent,
    html_to_text,
    render_repair_prompt,
    selector_bank_audit,
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


def test_histogram_latent_is_a_fitted_visible_only_control():
    histories = [["PASS"] * 4, ["WRONG_OUTPUT"] * 4, ["PASS", "WRONG_OUTPUT"] * 2]
    targets = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    coefficients = fit_histogram_latent(histories, targets, ridge=1e-8)
    np.testing.assert_allclose(histogram_latent(histories, coefficients), targets, atol=1e-6)


def test_repair_prompt_excludes_future_outcomes_and_is_stable():
    tests = [{"input": f"VISIBLE_{i}"} for i in range(4)] + [{"input": "FUTURE_SECRET"}]
    outcomes = ["PASS", "WRONG_OUTPUT", "PASS", "TIMEOUT", "FUTURE_OUTCOME_SECRET"]
    prompt = render_repair_prompt("print('bug')", tests, outcomes, visible=4, task_text="Add two integers.")
    assert "VISIBLE_3" in prompt and "TIMEOUT" in prompt
    assert "FUTURE_SECRET" not in prompt and "FUTURE_OUTCOME_SECRET" not in prompt
    assert "Add two integers" in prompt
    assert prompt == render_repair_prompt("print('bug')", tests, outcomes, visible=4, task_text="Add two integers.")


def test_html_problem_description_becomes_bounded_plain_text():
    source = "<h1>A &amp; B</h1><script>SECRET()</script><p>Add <b>two</b> values.</p>"
    assert html_to_text(source) == "A & B\nAdd two values."


def test_selector_bank_audit_reports_rankable_space_and_prefix_saturation():
    outcomes = np.array([
        [[1, 1, 1, 1], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0]],
        [[1, 0, 0, 0], [0, 1, 1, 1]],
    ])
    labels = np.array([[1, 0], [0, 0], [0, 1]])
    report = selector_bank_audit(outcomes, labels, prefixes=(1, 4))
    assert report["tasks"] == 3 and report["candidates"] == 6
    assert report["rankable_tasks"] == 2 and report["all_fail_tasks"] == 1
    assert report["oracle_pass_at_k"] == 2 / 3
    assert report["visible_pass_rate_selected_pass_at_1"]["4"] == 2 / 3
    assert report["rankable_visible_pass_rate_selected_pass_at_1"]["4"] == 1.0
