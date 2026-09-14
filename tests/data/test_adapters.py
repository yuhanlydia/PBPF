from __future__ import annotations

import hashlib

import pytest

from pbpf.data.codearc import adapt_codearc
from pbpf.data.evalplus import adapt_evalplus
from pbpf.data.livecodebench import adapt_livecodebench
from pbpf.data.runbugrun import adapt_runbugrun


def _rbr_tests(*, failing_buggy: bool = True, active: int = 10):
    return [
        {
            "id": f"test-{index}",
            "active": index < active,
            "source": f"assert solve({index}) == {index}",
            "expected_output": str(index),
            "fixed_outcome": "PASS",
            "buggy_outcome": "WRONG_OUTPUT" if failing_buggy and index == 0 else "PASS",
        }
        for index in range(10)
    ]


def _rbr_row(task_id="rbr-1", **updates):
    row = {
        "task_id": task_id,
        "language": "python",
        "problem_id": f"problem-{task_id}",
        "task_text": "repair solve",
        "buggy_code": "def solve(x): return x + 1",
        "fixed_code": "def solve(x): return x",
        "gold_patch": "- x + 1\n+ x",
        "tests": _rbr_tests(),
        "official_split": "train",
        "actor_token_count": 12,
        "user_id": "shared-user",
    }
    row.update(updates)
    return row


def _lcb_tests(count=5):
    return [
        {"id": f"test-{index}", "source": f"assert solve({index}) == {index}"}
        for index in range(count)
    ]


def test_runbugrun_filters_every_preregistered_eligibility_condition():
    rows = [
        _rbr_row("kept"),
        _rbr_row("language", language="java"),
        _rbr_row("tests", tests=_rbr_tests(active=9)),
        _rbr_row(
            "fixed-fails",
            tests=[
                {**case, "fixed_outcome": "WRONG_OUTPUT" if index == 0 else "PASS"}
                for index, case in enumerate(_rbr_tests())
            ],
        ),
        _rbr_row("buggy-passes", tests=_rbr_tests(failing_buggy=False)),
        _rbr_row("too-long", actor_token_count=2049),
        _rbr_row("unknown-length", actor_token_count=None),
    ]
    result = adapt_runbugrun(rows)
    assert [task.task_id for task in result.public_tasks] == ["kept"]
    assert result.exclusion_reasons == {
        "buggy_passes_all": 1,
        "fixed_fails_active_test": 1,
        "insufficient_active_tests": 1,
        "non_python": 1,
        "over_2048_actor_tokens": 1,
        "unverified_actor_token_count": 1,
    }


def test_runbugrun_orders_tests_by_sha256_of_public_id():
    result = adapt_runbugrun([_rbr_row()])
    expected = tuple(
        sorted(
            (f"test-{index}" for index in range(10)),
            key=lambda test_id: (hashlib.sha256(test_id.encode()).hexdigest(), test_id),
        )
    )
    assert tuple(test.test_id for test in result.public_tasks[0].visible_tests) == expected[:4]
    assert tuple(test.test_id for test in result.evaluator_tasks[0].tests) == expected


def test_runbugrun_leakage_components_preserve_boundaries_and_ignore_user_id():
    rows = [
        _rbr_row("train-a", official_split="train", user_id="same"),
        _rbr_row(
            "test-b",
            official_split="test",
            user_id="same",
            problem_id="other-problem",
            buggy_code="def solve(x): return x - 1",
            fixed_code="def solve(x): return x * 1",
            tests=[
                {**case, "source": f"assert solve({index}) == ({index} * 1)"}
                for index, case in enumerate(_rbr_tests())
            ],
        ),
    ]
    result = adapt_runbugrun(rows)
    assert {task.task_id for task in result.public_tasks} == {"train-a", "test-b"}
    assert all("user" not in group for task in result.public_tasks for group in task.group_ids)
    assert all(
        group.startswith(f"rbr:{task.split}:")
        for task in result.public_tasks
        for group in task.group_ids
    )

    leaking = adapt_runbugrun(
        [
            _rbr_row("train-copy", official_split="train", problem_id="duplicate"),
            _rbr_row("test-copy", official_split="test", problem_id="duplicate"),
        ]
    )
    assert leaking.public_tasks == ()
    assert leaking.exclusion_reasons == {"cross_boundary_leakage": 2}


def test_runbugrun_normalized_signatures_group_formatting_variants_within_boundary():
    first = _rbr_row("a", user_id="one")
    second = _rbr_row(
        "b",
        user_id="two",
        problem_id="different",
        buggy_code="def solve( x ):\n    return x+1 # comment",
        fixed_code="def solve( x ):\n    return x",
    )
    result = adapt_runbugrun([first, second])
    by_id = {task.task_id: task for task in result.public_tasks}
    assert set(by_id["a"].group_ids) & set(by_id["b"].group_ids)


def test_runbugrun_normalizes_test_source_syntax_not_only_whitespace():
    first_tests = _rbr_tests()
    second_tests = [
        {
            **case,
            "source": str(case["source"]).replace(" == ", "==") + "  # formatting",
        }
        for case in _rbr_tests()
    ]
    result = adapt_runbugrun(
        [
            _rbr_row(
                "syntax-a",
                problem_id="problem-a",
                buggy_code="def buggy_a(x): return x + 1",
                fixed_code="def fixed_a(x): return x",
                tests=first_tests,
            ),
            _rbr_row(
                "syntax-b",
                problem_id="problem-b",
                buggy_code="def buggy_b(x): return x + 1",
                fixed_code="def fixed_b(x): return x",
                tests=second_tests,
            ),
        ]
    )
    by_id = {task.task_id: task for task in result.public_tasks}
    assert set(by_id["syntax-a"].group_ids) & set(by_id["syntax-b"].group_ids)


def test_runbugrun_removes_an_entire_transitive_component_crossing_a_boundary():
    tests_a = _rbr_tests()
    tests_b = [
        {**case, "source": f"assert other({index}) == {index}"}
        for index, case in enumerate(_rbr_tests())
    ]
    tests_c = [
        {**case, "source": f"assert third({index}) == {index}"}
        for index, case in enumerate(_rbr_tests())
    ]
    rows = [
        _rbr_row(
            "a",
            problem_id="problem-a",
            buggy_code="def shared(x): return x + 1",
            fixed_code="def fixed_a(x): return x",
            tests=tests_a,
            official_split="train",
        ),
        _rbr_row(
            "b",
            problem_id="problem-b",
            buggy_code="def shared(x): return x+1",
            fixed_code="def fixed_b(x): return x",
            tests=tests_b,
            official_split="train",
        ),
        _rbr_row(
            "c",
            problem_id="problem-b",
            buggy_code="def third(x): return x - 1",
            fixed_code="def fixed_c(x): return x",
            tests=tests_c,
            official_split="test",
        ),
    ]
    result = adapt_runbugrun(rows)
    assert result.public_tasks == ()
    assert result.exclusion_reasons == {"cross_boundary_leakage": 3}


def test_runbugrun_unions_code_identity_across_buggy_and_fixed_roles():
    result = adapt_runbugrun(
        [
            _rbr_row(
                "train-fixed",
                problem_id="train-problem",
                buggy_code="def train_bug(x): return x - 1",
                fixed_code="def bridge(x): return x + 1",
                tests=_rbr_tests(),
                official_split="train",
            ),
            _rbr_row(
                "test-buggy",
                problem_id="test-problem",
                buggy_code="def bridge(x): return x+1",
                fixed_code="def test_fix(x): return x",
                tests=[
                    {**case, "source": f"assert other({index}) == {index}"}
                    for index, case in enumerate(_rbr_tests())
                ],
                official_split="test",
            ),
        ]
    )
    assert result.public_tasks == ()
    assert result.exclusion_reasons == {"cross_boundary_leakage": 2}


def test_runbugrun_removes_transitive_cross_role_leakage_component():
    result = adapt_runbugrun(
        [
            _rbr_row(
                "a",
                problem_id="a",
                buggy_code="def a_bug(x): return x - 1",
                fixed_code="def bridge_one(x): return x",
                tests=[{**case, "source": f"assert a({i})"} for i, case in enumerate(_rbr_tests())],
                official_split="train",
            ),
            _rbr_row(
                "b",
                problem_id="b",
                buggy_code="def bridge_one(x): return x",
                fixed_code="def bridge_two(x): return x",
                tests=[{**case, "source": f"assert b({i})"} for i, case in enumerate(_rbr_tests())],
                official_split="train",
            ),
            _rbr_row(
                "c",
                problem_id="c",
                buggy_code="def bridge_two(x): return x",
                fixed_code="def c_fix(x): return x",
                tests=[{**case, "source": f"assert c({i})"} for i, case in enumerate(_rbr_tests())],
                official_split="test",
            ),
        ]
    )
    assert result.public_tasks == ()
    assert result.exclusion_reasons == {"cross_boundary_leakage": 3}


def test_runbugrun_rejects_missing_invalid_and_infrastructure_only_buggy_outcomes():
    missing = [{key: value for key, value in case.items() if key != "buggy_outcome"} for case in _rbr_tests()]
    invalid = [{**case, "buggy_outcome": "NOT_AN_OUTCOME"} for case in _rbr_tests()]
    infrastructure = [{**case, "buggy_outcome": "INFRASTRUCTURE_FAILURE"} for case in _rbr_tests()]
    result = adapt_runbugrun(
        [
            _rbr_row("missing-outcomes", tests=missing),
            _rbr_row("invalid-outcomes", tests=invalid),
            _rbr_row("infra-outcomes", tests=infrastructure),
        ]
    )
    assert result.public_tasks == ()
    assert result.exclusion_reasons == {"invalid_buggy_outcomes": 3}


def test_runbugrun_keeps_a_development_component_in_one_tune_select_half():
    rows = [
        _rbr_row(
            "a",
            official_split="development",
            problem_id="problem-a",
            fixed_code="def fixed_a(x): return x",
        ),
        _rbr_row(
            "c",
            official_split="development",
            problem_id="problem-c",
            fixed_code="def fixed_c(x): return x",
        ),
    ]
    result = adapt_runbugrun(rows)
    assert len({task.split for task in result.public_tasks}) == 1


def test_codearc_replay_requires_anonymous_rows_and_fixed_invocation_partition():
    invocations = [
        {"id": f"call-{index}", "input": [index], "expected_output": [index + 1]}
        for index in range(10)
    ]
    result = adapt_codearc(
        [
            {
                "task_id": "arc-1",
                "anonymous": True,
                "task_text": "infer the transformation",
                "candidate_code": "lambda grid: grid",
                "target_code": "lambda grid: [[x + 1 for x in row] for row in grid]",
                "invocations": invocations,
                "official_split": "test",
            },
            {
                "task_id": "named",
                "anonymous": False,
                "task_text": "not eligible",
                "candidate_code": "pass",
                "target_code": "secret",
                "invocations": invocations,
                "official_split": "test",
            },
        ]
    )
    assert [task.task_id for task in result.public_tasks] == ["arc-1"]
    public = result.public_tasks[0]
    evaluator = result.evaluator_tasks[0]
    assert public.protocol == "CodeARC-Replay"
    assert [case.invocation_index for case in public.visible_tests] == [0, 1, 2, 3]
    assert all(not hasattr(case, "expected_output") for case in public.visible_tests)
    assert [case.invocation_index for case in evaluator.tests if case.hidden] == list(range(4, 10))
    assert "target_code" not in public.__dataclass_fields__
    assert evaluator.gold_code.startswith("lambda grid")
    assert result.exclusion_reasons == {"non_anonymous": 1}


def test_evalplus_keeps_base_visible_plus_hidden_and_deduplicates_exact_cases():
    duplicate = {"id": "different-id", "source": "assert f(1) == 2", "expected_output": "2"}
    result = adapt_evalplus(
        [
            {
                "task_id": "HumanEval/1",
                "prompt": "def f(x): pass",
                "candidate_code": "def f(x): return x",
                "base_tests": [
                    {"id": "base", "source": "assert f(1) == 2", "expected_output": "2"},
                    duplicate,
                ],
                "plus_tests": [
                    duplicate,
                    {"id": "plus", "source": "assert f(2) == 3", "expected_output": "3"},
                ],
                "gold_code": "def f(x): return x + 1",
                "official_split": "test",
            }
        ]
    )
    public = result.public_tasks[0]
    evaluator = result.evaluator_tasks[0]
    assert public.protocol == "PBPF-EvalPlus"
    assert [case.test_id for case in public.visible_tests] == ["base"]
    assert [(case.test_id, case.hidden) for case in evaluator.tests] == [
        ("base", False),
        ("plus", True),
    ]
    assert result.exclusion_reasons == {}


def test_evalplus_sha_orders_public_ids_within_atomic_partitions():
    result = adapt_evalplus(
        [
            {
                "task_id": "HumanEval/order",
                "prompt": "def f(x): pass",
                "candidate_code": "def f(x): return x",
                "base_tests": [
                    {"id": test_id, "source": f"assert f('{test_id}')", "expected_output": None}
                    for test_id in ("z", "a", "m")
                ],
                "plus_tests": [
                    {"id": test_id, "source": f"assert f('{test_id}')", "expected_output": None}
                    for test_id in ("q", "b", "y")
                ],
                "official_split": "test",
            }
        ]
    )
    key = lambda value: (hashlib.sha256(value.encode()).hexdigest(), value)
    assert [case.test_id for case in result.public_tasks[0].visible_tests] == sorted(
        ("z", "a", "m"), key=key
    )
    evaluator = result.evaluator_tasks[0]
    assert [case.test_id for case in evaluator.tests if case.hidden] == sorted(
        ("q", "b", "y"), key=key
    )


def test_evalplus_exact_deduplication_does_not_merge_distinct_structured_cases():
    result = adapt_evalplus(
        [
            {
                "task_id": "MBPP/structured",
                "prompt": "def f(x): pass",
                "candidate_code": "def f(x): return x",
                "base_tests": [
                    {"id": "base", "input": [1], "expected_output": 1},
                ],
                "plus_tests": [
                    {"id": "plus-a", "input": [2], "expected_output": 1},
                    {"id": "plus-b", "input": [3], "expected_output": 1},
                ],
                "official_split": "test",
            }
        ]
    )
    tests = result.evaluator_tasks[0].tests
    assert len(tests) == 3
    assert {case.test_id: case.payload["input"] for case in tests} == {
        "base": [1],
        "plus-a": [2],
        "plus-b": [3],
    }


def test_evalplus_deduplicates_execution_semantics_with_base_priority():
    result = adapt_evalplus(
        [
            {
                "task_id": "HumanEval/metadata",
                "prompt": "def f(x): pass",
                "candidate_code": "def f(x): return x",
                "base_tests": [
                    {
                        "id": "base",
                        "input": [1],
                        "expected_output": 2,
                        "visibility": "base",
                        "provenance": "official",
                    }
                ],
                "plus_tests": [
                    {
                        "id": "duplicate-plus",
                        "input": [1],
                        "expected_output": 2,
                        "visibility": "plus",
                        "provenance": "augmented",
                        "public_source": False,
                    },
                    {"id": "kept-plus", "input": [2], "expected_output": 3},
                ],
                "official_split": "test",
            }
        ]
    )
    assert [(case.test_id, case.hidden) for case in result.evaluator_tasks[0].tests] == [
        ("base", False),
        ("kept-plus", True),
    ]


def test_livecodebench_selects_exact_release_v5_to_v6_new_slice():
    rows = [
        {
            "task_id": "old",
            "release": "v6",
            "prompt": "old prompt",
            "candidate_code": "pass",
            "tests": _lcb_tests(),
        },
        {
            "task_id": "new",
            "release": "v6",
            "prompt": "new prompt",
            "candidate_code": "pass",
            "tests": _lcb_tests(),
        },
        {
            "task_id": "v7",
            "release": "v7",
            "prompt": "future prompt",
            "candidate_code": "pass",
            "tests": _lcb_tests(),
        },
    ]
    result = adapt_livecodebench(rows, release_v5_task_ids={"old"})
    assert [task.task_id for task in result.public_tasks] == ["new"]
    assert result.public_tasks[0].protocol == "LiveCodeBench"
    assert result.exclusion_reasons == {"not_new_in_v6": 2}
    expected_v5_hash = hashlib.sha256(b'["old"]').hexdigest()
    assert result.resolved_options["release_v5_task_ids_hash"] == expected_v5_hash


def test_livecodebench_accepts_rows_from_a_revision_verified_v6_snapshot():
    result = adapt_livecodebench(
        [
            {
                "task_id": "new-without-label",
                "prompt": "snapshot identifies the release",
                "candidate_code": "pass",
                "tests": _lcb_tests(),
            }
        ],
        release_v5_task_ids=set(),
    )
    assert [task.task_id for task in result.public_tasks] == ["new-without-label"]


def test_livecodebench_rejects_tasks_without_a_hidden_case():
    result = adapt_livecodebench(
        [
            {
                "task_id": "public-only",
                "prompt": "no locked case",
                "candidate_code": "pass",
                "tests": _lcb_tests(4),
            }
        ],
        release_v5_task_ids=set(),
    )
    assert result.public_tasks == ()
    assert result.exclusion_reasons == {"no_hidden_cases": 1}


def test_livecodebench_preserves_explicit_upstream_public_private_partition():
    result = adapt_livecodebench(
        [
            {
                "task_id": "partitioned",
                "prompt": "locked partition",
                "candidate_code": "pass",
                "public_tests": [
                    {"id": "public-b", "source": "assert solve(2)"},
                    {"id": "public-a", "source": "assert solve(1)"},
                ],
                "hidden_tests": [
                    {"id": "hidden", "source": "assert solve(3)"},
                ],
            }
        ],
        release_v5_task_ids=set(),
    )
    task = result.evaluator_tasks[0]
    assert sum(not case.hidden for case in task.tests) == 2
    assert sum(case.hidden for case in task.tests) == 1


def test_livecodebench_unpartitioned_fallback_orders_before_selecting_visibility():
    cases = [
        {"id": test_id, "input": [test_id], "expected_output": test_id}
        for test_id in ("z", "a", "m", "q", "b", "y")
    ]
    forward = adapt_livecodebench(
        [
            {
                "task_id": "unpartitioned",
                "prompt": "solve",
                "candidate_code": "pass",
                "tests": cases,
            }
        ],
        release_v5_task_ids=set(),
    )
    reverse = adapt_livecodebench(
        [
            {
                "task_id": "unpartitioned",
                "prompt": "solve",
                "candidate_code": "pass",
                "tests": list(reversed(cases)),
            }
        ],
        release_v5_task_ids=set(),
    )
    expected = sorted(
        (case["id"] for case in cases),
        key=lambda test_id: (hashlib.sha256(test_id.encode()).hexdigest(), test_id),
    )
    assert forward == reverse
    assert [case.test_id for case in forward.public_tasks[0].visible_tests] == expected[:4]
    assert [
        case.test_id for case in forward.evaluator_tasks[0].tests if case.hidden
    ] == expected[4:]


def test_adapter_outputs_are_deterministic_under_input_permutation():
    rows = [_rbr_row("b"), _rbr_row("a")]
    forward = adapt_runbugrun(rows)
    reverse = adapt_runbugrun(list(reversed(rows)))
    assert forward == reverse
    assert [task.task_id for task in forward.public_tasks] == ["a", "b"]


def _duplicate_adapter_case(name):
    if name == "runbugrun":
        row = _rbr_row("duplicate")
        return lambda rows: adapt_runbugrun(rows), row
    if name == "codearc":
        row = {
            "task_id": "duplicate",
            "anonymous": True,
            "task_text": "infer",
            "candidate_code": "pass",
            "target_code": "return 1",
            "invocations": [
                {"id": f"i-{index}", "input": index, "expected_output": index}
                for index in range(10)
            ],
        }
        return lambda rows: adapt_codearc(rows), row
    if name == "evalplus":
        row = {
            "task_id": "duplicate",
            "prompt": "def f(): pass",
            "candidate_code": "def f(): return 0",
            "base_tests": [{"id": "base", "source": "assert f() == 0"}],
            "plus_tests": [{"id": "plus", "source": "assert f() == 1"}],
        }
        return lambda rows: adapt_evalplus(rows), row
    row = {
        "task_id": "duplicate",
        "prompt": "solve",
        "candidate_code": "pass",
        "tests": _lcb_tests(),
    }
    return lambda rows: adapt_livecodebench(rows, release_v5_task_ids=set()), row


@pytest.mark.parametrize("name", ("runbugrun", "codearc", "evalplus", "livecodebench"))
def test_all_adapters_resolve_exact_and_conflicting_duplicates_order_invariant(name):
    adapt, row = _duplicate_adapter_case(name)
    exact_forward = adapt([row, dict(row)])
    exact_reverse = adapt([dict(row), row])
    assert exact_forward == exact_reverse
    assert len(exact_forward.public_tasks) == 1
    assert exact_forward.exclusion_reasons == {"exact_duplicate_task": 1}

    conflicting = {**row, "task_text": "different task text", "prompt": "different prompt"}
    conflict_forward = adapt([row, conflicting])
    conflict_reverse = adapt([conflicting, row])
    assert conflict_forward == conflict_reverse
    assert conflict_forward.public_tasks == ()
    assert conflict_forward.exclusion_reasons == {"conflicting_duplicate_task_id": 2}


@pytest.mark.parametrize("name", ("runbugrun", "codearc", "evalplus", "livecodebench"))
def test_all_adapters_turn_non_mapping_rows_into_deterministic_exclusions(name):
    adapt, row = _duplicate_adapter_case(name)
    result = adapt([object(), row])
    assert [task.task_id for task in result.public_tasks] == ["duplicate"]
    assert result.exclusion_reasons == {"malformed_record": 1}
