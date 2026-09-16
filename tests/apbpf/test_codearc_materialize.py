import copy
import json

import pytest

from pbpf.apbpf.codearc_materialize import build_records


def fixtures():
    problems = [{"type": "anonymous", "problem_id": i,
                 "code": f'def solution(x):\n    "PRIVATE_DOC_{i}"\n    return x + {i}\n'}
                for i in range(5)]
    calls = [{"type": "anonymous", "problem_id": i, "index": j,
              "input": f"print(solution({j}))", "output": f"value-{i}-{j}", "errored": False}
             for i in range(5) for j in range(10)]
    return problems, calls


def test_public_records_exclude_reference_and_future_examples():
    problems, calls = fixtures()
    public, private, inventory = build_records(problems, calls, seed=1701,
        development_components=1, primary_components=2)
    text = json.dumps(public)
    assert "PRIVATE_DOC" not in text and "return x" not in text
    for i in range(5):
        assert f"value-{i}-0" in text
        for j in range(4, 10):
            assert f"value-{i}-{j}" not in text
    assert all(len(row["tests"]) == 10 for row in private)
    assert inventory["component_counts"] == {"primary": 2, "development": 1, "train": 2}


def test_hidden_labels_cannot_change_population_or_source_split():
    problems, calls = fixtures()
    public, _, inventory = build_records(problems, calls, seed=1,
        development_components=1, primary_components=2)
    changed = copy.deepcopy(calls)
    for row in changed:
        if row["index"] >= 4:
            row.update(output="CHANGED-HIDDEN-OUTCOME", errored=True)
    new_public, _, new_inventory = build_records(problems, changed, seed=1,
        development_components=1, primary_components=2)
    assert public == new_public and inventory == new_inventory


def test_duplicate_reference_components_cannot_cross_splits():
    problems, calls = fixtures()
    problems[4]["code"] = problems[0]["code"].replace("PRIVATE_DOC_0", "different docstring")
    public, _, inventory = build_records(problems, calls, seed=2,
        development_components=1, primary_components=2)
    by_id = {row["task_id"]: row for row in public}
    assert by_id["CodeARC/0"]["source_component_id"] == by_id["CodeARC/4"]["source_component_id"]
    assert by_id["CodeARC/0"]["split"] == by_id["CodeARC/4"]["split"]
    assert inventory["source_components"] == 4


def test_incomplete_or_duplicate_invocations_fail_closed():
    problems, calls = fixtures()
    for invalid in (calls[:-1], calls + calls[:1]):
        with pytest.raises(ValueError):
            build_records(problems, invalid, seed=1, development_components=1, primary_components=2)
