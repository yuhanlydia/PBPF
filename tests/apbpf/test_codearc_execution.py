from pbpf.apbpf.codearc_execution import _dictionary_order_only, execute_call, score_result


def test_codearc_runner_scores_returned_output_and_expected_exceptions():
    result = execute_call("def solution(x): return x + 1", {
        "input": "print(solution(2))", "expected": "3", "expected_error": False})
    assert result["outcome"] == "PASS"
    result = execute_call("def solution(x): raise ValueError('example')", {
        "input": "print(solution(2))", "expected": "example", "expected_error": True})
    assert result["outcome"] == "PASS"


def test_candidate_cannot_read_evaluator_files_and_timeout_is_not_pass(tmp_path):
    secret = tmp_path / "evaluator_secret"
    secret.write_text("hidden-label-canary")
    result = execute_call(f"import os\ndef solution(): return os.path.exists({str(secret)!r})", {
        "input": "print(solution())", "expected": "False", "expected_error": False})
    assert result["outcome"] == "PASS"
    result = execute_call("while True: pass", {
        "input": "", "expected": "", "expected_error": True}, timeout=.1)
    assert result["outcome"] == "TIMEOUT"


def test_numpy_dependencies_and_cross_python_exception_text_are_supported():
    result = execute_call("from numpy import prod\ndef solution(): return prod([2, 3])", {
        "input": "print(solution())", "expected": "6", "expected_error": False})
    assert result["outcome"] == "PASS", result

    result = execute_call("def solution(): return max([])", {
        "input": "print(solution())", "expected": "max() arg is an empty sequence", "expected_error": True})
    assert result["outcome"] == "PASS", result


def test_dictionary_order_diagnostic_does_not_relax_primary_scoring():
    actual = {"stdout": "Result 1: {'b': 2, 'a': 1}", "errored": False, "error": ""}
    test = {"expected": "Result 1: {'a': 1, 'b': 2}", "expected_error": False}
    assert _dictionary_order_only(actual["stdout"], test["expected"])
    assert score_result(actual, test) == "WRONG_OUTPUT"
    assert not _dictionary_order_only("[1, 2]", "[2, 1]")
    assert not _dictionary_order_only("{'a': 1}", "{'a': 2}")


def test_syntax_and_runtime_errors_remain_distinct_prediction_classes():
    test = {"input": "print(solution())", "expected": "3", "expected_error": False}
    assert execute_call("def solution(:", test)["outcome"] == "COMPILE_ERROR"
    assert execute_call("def solution(): return missing_name", test)["outcome"] == "RUNTIME_EXCEPTION"
