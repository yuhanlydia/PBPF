from pathlib import Path

import yaml
import pytest

from pbpf.config import load_experiment, validate_experiment
from pbpf.registry import BASELINE_PROVENANCE_MODES, DATASETS, MODELS
from pbpf.baselines import require_available_baseline


ROOT = Path(__file__).parents[1]


def load_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_model_and_benchmark_config_matrices_match_registries():
    model_configs = [load_yaml(path) for path in sorted((ROOT / "configs/models").glob("*.yaml"))]
    benchmark_configs = [
        load_yaml(path) for path in sorted((ROOT / "configs/benchmarks").glob("*.yaml"))
    ]
    assert {item["model_id"] for item in model_configs} == {
        spec.model_id for spec in MODELS.values()
    }
    assert {item["dataset_id"] for item in benchmark_configs} == set(DATASETS)
    assert all(len(item["revision"]) == 40 for item in model_configs + benchmark_configs)
    assert all(item["fixed_test_order"] is True for item in benchmark_configs)


def test_baseline_catalog_has_provenance_and_distinguishes_particle_neighbor():
    catalog = load_yaml(ROOT / "configs/baselines.yaml")
    required = {"rollout_roulette", "rex", "rlef", "ldb", "rsp", "self_debug"}
    assert required <= set(catalog)
    for name, entry in catalog.items():
        assert entry["provenance_mode"] in BASELINE_PROVENANCE_MODES
        assert "paper_url" in entry and "repository_url" in entry
        assert len(entry["license_hash"]) == 64
    assert catalog["rollout_roulette"]["particle_semantics"] == "language_trajectories"
    assert catalog["rlef"]["provenance_mode"] == "paper_spec_reimplementation"
    assert catalog["rlef"]["official"] is False
    for name in ("rex", "ldb"):
        assert len(catalog[name]["upstream_revision"]) == 40
        assert catalog[name]["execution_enabled"] is False
        assert "adapter_revision" not in catalog[name]
        with pytest.raises(ValueError, match="unavailable"):
            require_available_baseline(name, ROOT / "configs/baselines.yaml")


def test_experiment_configs_are_no_claim_plans_with_fixed_equal_budget_protocol():
    for path in sorted((ROOT / "configs/experiments").glob("*.yaml")):
        raw = load_yaml(path)
        if raw.get("schema") == "apbpf-iclr-v1":
            from pbpf.apbpf.config import resolve_config as resolve_apbpf_config
            resolved = resolve_apbpf_config(path, "local_smoke")
            assert resolved.config["execution"]["claim_status"] == "smoke-only-no-claim"
            continue
        if path.name == "iclr_pbpf.yaml":
            from pbpf.iclr_config import resolve_config
            resolved = resolve_config(path, "local_cpu")
            assert resolved.claim_status == "smoke-only-no-claim"
            assert resolved.science["protocol"]["initial_actor_samples"] == 8
            assert resolved.science["protocol"]["initial_origin"] == "actor_sample"
            continue
        config = validate_experiment(load_experiment(path))
        assert config["claim_status"] == "preregistered_configuration_only"
        assert config["protocol"] == {
            "fixed_test_order": True,
            "rounds": 4,
            "early_stop": False,
            "candidates": 8,
            "visible_token_budget": config["hardware"]["context"],
        }
        assert config["feedback_mode"] == "status_only"
        assert set(config["budget"]) == {
            "max_visible_tokens",
            "max_generated_tokens",
            "max_model_calls",
            "max_executions",
            "max_cpu_seconds",
            "max_wall_seconds",
            "max_gpu_hours",
        }
        assert config["budget"]["max_model_calls"] == 50
        assert config["candidate_bank"] == {
            "model_samples": 6,
            "deterministic_mutants": 2,
            "temperature": 0.8,
            "top_p": 0.95,
            "mutation_registry_required": True,
        }
        assert "tokenwise_remixture_fault" in config["arms"]


def test_required_documentation_and_ci_files_exist():
    required = {
        ROOT / "docs/EXPERIMENTS.md",
        ROOT / "docs/BASELINES.md",
        ROOT / "docs/DATA.md",
        ROOT / "docs/ARTIFACTS.md",
        ROOT / ".github/workflows/ci.yml",
    }
    assert all(path.is_file() and path.stat().st_size > 200 for path in required)


def test_cpu_ci_keeps_neural_tests_optional():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "-e '.[test]'" in workflow
    prediction_tests = (ROOT / "tests/arms/test_prediction_arms.py").read_text()
    assert 'pytest.importorskip("torch")' in prediction_tests


def test_real_evalplus_launchers_use_executable_dataset_matched_configs():
    root = Path(__file__).parents[1]
    from pbpf.config import load_experiment, validate_experiment

    for launcher, config_name in (
        ("run_frozen_7b.sh", "evalplus_frozen_7b_16gb.yaml"),
        ("run_repair.sh", "evalplus_repair_7b_24gb.yaml"),
    ):
        script = (root / "scripts" / launcher).read_text()
        assert f"configs/diagnostics/{config_name}" in script
        config = load_experiment(root / "configs" / "diagnostics" / config_name)
        validated = validate_experiment(config, for_execution=True)
        assert validated["data"]["id"] == "evalplus"
        assert validated["baseline"]["name"] == "pbpf_soft_prompt"
        assert validated["arms"] == ["pbpf_soft_prompt"]

    for runner in ("run_evalplus_screen.py", "run_real_pilot.py"):
        source = (root / "scripts" / runner).read_text()
        assert 'python_executable="/usr/bin/python3"' in source

    pilot = load_experiment(root / "configs" / "pilots" / "local_real_7b.yaml")
    benchmark = load_yaml(root / "configs" / "benchmarks" / "runbugrun.yaml")
    assert pilot["data"]["revision"] == benchmark["revision"]
    assert 'parser.add_argument("--task-id", type=int, default=6581)' in (
        root / "scripts" / "run_real_pilot.py"
    ).read_text()


def test_smoke_cli_executes_protocol_instead_of_only_planning(tmp_path, capsys):
    from pbpf.cli import main

    assert main(["smoke", "--output", str(tmp_path / "run")]) == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["status"] == "smoke-only-no-claim"
    assert payload["rounds_completed"] == 4
