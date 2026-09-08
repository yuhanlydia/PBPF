import json
import subprocess
import sys

import pytest

from pbpf.config import canonical_config_hash, load_experiment, validate_experiment
from pbpf.registry import BASELINE_PROVENANCE_MODES, DATASETS, MODELS, OUTCOMES


def test_registry_contains_preregistered_models_datasets_outcomes_and_provenance():
    assert {spec.model_id for spec in MODELS.values()} == {
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen3-8B",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
    }
    assert {spec.role for spec in DATASETS.values()} == {
        "exact_identification",
        "real_repair",
        "interactive_transfer",
        "functional_smoke",
        "coding_transfer",
        "repository_development",
        "repository_final",
    }
    assert OUTCOMES == (
        "PASS",
        "WRONG_OUTPUT",
        "RUNTIME_EXCEPTION",
        "TIMEOUT",
        "COMPILE_ERROR",
    )
    assert BASELINE_PROVENANCE_MODES == {
        "official_adapter",
        "paper_spec_reimplementation",
        "controlled_ablation",
    }


def test_formal_runs_require_formal_profile_and_full_revisions(tmp_path):
    config = {
        "name": "bad-formal",
        "formal": True,
        "profile": "24gb",
        "model": {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "revision": "main"},
        "data": {"id": "runbugrun", "revision": "v1"},
        "container": {"digest": "latest"},
        "protocol": {"fixed_test_order": True, "rounds": 4, "early_stop": False},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_experiment(path)
    with pytest.raises(ValueError, match="formal profile"):
        validate_experiment(loaded)


def test_formal_revision_validation_fails_closed():
    config = {
        "name": "formal",
        "formal": True,
        "profile": "h200_formal",
        "model": {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "revision": "main"},
        "data": {"id": "runbugrun", "revision": "abc"},
        "container": {"digest": "sha256:abc"},
        "protocol": {"fixed_test_order": True, "rounds": 4, "early_stop": False},
    }
    with pytest.raises(ValueError, match="40-character"):
        validate_experiment(config)


def test_canonical_hash_ignores_mapping_order():
    assert canonical_config_hash({"b": [2, 1], "a": 1}) == canonical_config_hash(
        {"a": 1, "b": [2, 1]}
    )


def test_protocol_rejects_unmatched_candidate_and_token_budgets():
    config = {
        "name": "pilot",
        "formal": False,
        "profile": "16gb",
        "model": {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "revision": "main"},
        "data": {"id": "runbugrun", "revision": "v1"},
        "container": {"digest": "latest"},
        "hardware": {"context": 2048},
        "protocol": {
            "fixed_test_order": True,
            "rounds": 4,
            "early_stop": False,
            "candidates": 7,
            "visible_token_budget": 1024,
        },
    }
    with pytest.raises(ValueError, match="G=8"):
        validate_experiment(config)


def test_formal_protocol_requires_three_seeds_and_ten_thousand_bootstraps():
    config = {
        "name": "formal",
        "formal": True,
        "profile": "h200_formal",
        "model": {
            "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "revision": "a" * 40,
        },
        "data": {"id": "runbugrun", "revision": "b" * 40},
        "container": {"digest": "sha256:" + "c" * 64},
        "hardware": {"context": 4096},
        "protocol": {
            "fixed_test_order": True,
            "rounds": 4,
            "early_stop": False,
            "candidates": 8,
            "visible_token_budget": 4096,
        },
        "statistics": {"seeds": [1, 2], "bootstrap_replicates": 9999},
    }
    with pytest.raises(ValueError, match="three seeds"):
        validate_experiment(config)


def test_import_has_no_torch_or_transformers_side_effect():
    code = "import json,sys,pbpf; print(json.dumps([x for x in ('torch','transformers') if x in sys.modules]))"
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == []


def test_formal_flag_has_strict_boolean_type_and_execution_requires_resolved_inputs():
    config = {
        "name": "quoted-formal",
        "formal": "true",
        "profile": "h200_formal",
        "model": {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "revision": "a" * 40},
        "data": {"id": "runbugrun", "revision": "b" * 40},
        "container": {"digest": "sha256:" + "c" * 64},
        "hardware": {"context": 4096},
        "protocol": {
            "fixed_test_order": True,
            "rounds": 4,
            "early_stop": False,
            "candidates": 8,
            "visible_token_budget": 4096,
        },
    }
    with pytest.raises(ValueError, match="boolean"):
        validate_experiment(config)
    config["formal"] = False
    config["model"]["revision"] = "main"
    with pytest.raises(ValueError, match="execution.*40-character"):
        validate_experiment(config, for_execution=True)


def test_execution_rejects_unknown_and_registered_unavailable_arms():
    from pathlib import Path

    root = Path(__file__).parents[1]
    config = load_experiment(root / "configs/experiments/exact_smoke.yaml")
    config["arms"] = ["pbpf_low_rank_kv", "made_up_arm"]
    with pytest.raises(ValueError, match="unknown experiment method"):
        validate_experiment(config, for_execution=True)
    config["arms"] = ["pbpf_low_rank_kv", "ladi_rl_audit_required"]
    with pytest.raises(ValueError, match="unavailable.*independent code audit"):
        validate_experiment(config, for_execution=True)
