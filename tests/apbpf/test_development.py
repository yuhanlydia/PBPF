import copy
import json

from pbpf.apbpf.development import development_cache


def test_development_iterations_exclude_original_test_and_are_source_disjoint():
    payload = {"schema": "example", "test_secret_metadata": "SECRET",
               "execution_protocol": {"stdin_policy": "official-terminal-newline-if-missing"},
               "records": [{"problem_id": f"train{i}", "split": "train", "candidate": "x"}
                           for i in range(10)]
                          + [{"problem_id": "dev0", "split": "development", "candidate": "dev"},
                             {"problem_id": "heldout", "split": "test", "candidate": "TEST_CANARY"}]}
    original = copy.deepcopy(payload)
    result = development_cache(payload)
    assert payload == original
    assert "TEST_CANARY" not in json.dumps(result) and "SECRET" not in json.dumps(result)
    groups = {name: {r["problem_id"] for r in result["records"] if r["split"] == name}
              for name in ("train", "development", "test")}
    assert groups["test"] == {"dev0"}
    assert not groups["train"] & groups["development"]
    assert result["evaluation_role"] == "development_assessment_only"
    assert result["execution_protocol"] == payload["execution_protocol"]
    altered_test = copy.deepcopy(payload)
    altered_test["records"][-1]["candidate"] = "DIFFERENT_TEST_RESULT"
    assert development_cache(altered_test) == result
