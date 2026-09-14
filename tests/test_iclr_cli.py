"""Public CLI contracts; subprocesses exercise the installed command boundary."""
import builtins
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/iclr_pbpf.yaml"


def cli(*args, root=None):
    command = [sys.executable, "-m", "pbpf.iclr_cli", *args, "--config", str(CONFIG),
               "--profile", "local_cpu"]
    if root is not None:
        command += ["--output-root", str(root)]
    return subprocess.run(command, text=True, capture_output=True, cwd=ROOT)


def test_doctor_emits_one_json_with_complete_synthetic_s0_s4_resources():
    result = cli("doctor", "--dry-run")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["claim_status"] == "smoke-only-no-claim"
    assert {cell["stage"] for cell in report["cells"]} == {"S0", "S1", "S2", "S3", "S4"}
    for cell in report["cells"]:
        assert cell["tasks"] > 0
        assert cell["projected_cost"] == cell["gpu_hours"] == 0
        assert {"actor_decodes", "input_token_bound", "output_token_bound", "sandbox_suites",
                "disk_bytes_bound", "memory_bytes_bound"} <= cell.keys()


def test_local_all_commands_share_run_id_resume_and_verify_package(tmp_path):
    outputs = {}
    for command in ("doctor", "prepare", "run", "aggregate", "verify", "package"):
        result = cli(command, "--resume", root=tmp_path)
        assert result.returncode == 0, (command, result.stdout, result.stderr)
        outputs[command] = json.loads(result.stdout)
        if command == "prepare":
            assert not list(tmp_path.rglob("repair/*.jsonl"))
    assert len({out["run_id"] for out in outputs.values()}) == 1
    directory = Path(outputs["run"]["run_directory"])
    before = {p: p.read_bytes() for p in directory.glob("stages/**/*.jsonl")}
    resumed = cli("run", "--resume", root=tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["cached_stages"] == 15
    assert before == {p: p.read_bytes() for p in before}
    assert Path(outputs["package"]["archive"]).is_file()
    next(iter(before)).write_bytes(b"broken")
    for command in ("aggregate", "verify", "package"):
        assert cli(command, root=tmp_path).returncode != 0


def test_fresh_local_runs_have_identical_scientific_leaves_and_packages(tmp_path):
    reports = []
    leaves = []
    for name in ("first", "second"):
        root = tmp_path / name
        run = cli("run", "--resume", root=root)
        assert run.returncode == 0, run.stderr
        package = cli("package", root=root)
        assert package.returncode == 0, package.stderr
        report = json.loads(package.stdout)
        directory = root / report["run_id"]
        reports.append(report)
        leaves.append({p.relative_to(directory): p.read_bytes()
                       for p in directory.glob("stages/**/*.jsonl")})
    assert reports[0]["run_id"] == reports[1]["run_id"]
    assert leaves[0] == leaves[1]
    assert reports[0]["sha256"] == reports[1]["sha256"]


def test_local_dag_has_no_optional_import_or_socket(monkeypatch, tmp_path, capsys):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "datasets", "socket", "requests", "huggingface_hub"}:
            raise AssertionError("forbidden offline dependency: " + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    main = importlib.import_module("pbpf.iclr_cli").main
    assert main(["run", "--config", str(CONFIG), "--profile", "local_cpu",
                 "--output-root", str(tmp_path), "--resume"]) == 0
    assert json.loads(capsys.readouterr().out)["claim_status"] == "smoke-only-no-claim"


def test_config_recursive_resolution_binds_science_not_site_paths(tmp_path):
    module = importlib.import_module("pbpf.iclr_config")
    first = module.resolve_config(CONFIG, "local_cpu")
    assert '"$ref"' not in json.dumps(first.science)
    second = module.resolve_config(CONFIG, "local_cpu", site={"partition": "another", "output_root": "/different"})
    assert first.fingerprint.digest == second.fingerprint.digest
    assert first.science["protocol"]["initial_actor_samples"] == 8
    assert first.science["protocol"]["repair_decodes"] == 4
    assert "deterministic_mutants" not in first.science["protocol"]
    changed = copy.deepcopy(first.science)
    changed["development"]["brier_margin"] = .02
    assert module.science_digest(changed) != module.science_digest(first.science)
    directory = tmp_path / "resolved"
    first.persist(directory)
    assert yaml.safe_load((directory / "resolved.yaml").read_text()) == first.science
    (directory / "resolved.yaml").write_text("changed: true\n")
    with pytest.raises(ValueError, match="resolved.*conflict"):
        first.persist(directory)


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c.update(schema="wrong"), "schema"),
    (lambda c: c["models"]["models"]["qwen25_7b"].update(revision="main"), "immutable"),
    (lambda c: c["matrix"][0].update(model="absent"), "model"),
    (lambda c: c["matrix"][0].update(arms=["imaginary"]), "arm"),
    (lambda c: c["ablations"].update(design="cartesian"), "one.factor"),
    (lambda c: c["protocol"].update(initial_actor_samples=6), "eight"),
])
def test_invalid_config_precedes_optional_imports(mutation, match, monkeypatch):
    module = importlib.import_module("pbpf.iclr_config")
    science = module.read_config(CONFIG)
    mutation(science)
    calls = []
    monkeypatch.setattr(module, "validate_dependencies", lambda: calls.append("imports"))
    with pytest.raises(ValueError, match=match):
        module.validate_config(science, profile="slurm_h200x16", site={}, execution=True)
    assert calls == []


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda c: c["matrix"].pop(), id="missing-cell"),
    pytest.param(lambda c: c["matrix"].append({**copy.deepcopy(c["matrix"][0]), "id": "substitute"}),
                 id="extra-cell"),
    pytest.param(lambda c: c["matrix"][0].update(stage="S2"), id="stage"),
    pytest.param(lambda c: c["matrix"][0].update(phase="repair"), id="phase"),
    pytest.param(lambda c: c["matrix"][0].update(model="qwen3_8b"), id="model"),
    pytest.param(lambda c: c["matrix"][0].update(dataset="CodeARC-Replay"), id="dataset"),
    pytest.param(lambda c: c["matrix"][0].update(tasks=15), id="tasks"),
    pytest.param(lambda c: c["matrix"][0]["arms"].pop(), id="arms"),
])
def test_compatibility_rejects_substitute_matrix(mutation):
    from pbpf.iclr_config import read_config, validate_compatibility
    science = read_config(CONFIG)
    mutation(science)
    with pytest.raises(ValueError, match="matrix"):
        validate_compatibility(science)


@pytest.mark.parametrize("mutation", [
    pytest.param(lambda c: c["gates"]["B"].update(relative_nll_gain=.01), id="changed-gate"),
    pytest.param(lambda c: c["ablations"]["patches"].pop(), id="reduced-patches"),
    pytest.param(lambda c: c["ablations"]["patches"].__setitem__(0, {"particles": 2}),
                 id="changed-patch"),
    pytest.param(lambda c: c["ablations"]["factors"]["particles"].pop(), id="reduced-factors"),
    pytest.param(lambda c: (c["ablations"]["confirmatory_controls"].pop(),
                            c["development"]["causal_controls"].pop()), id="reduced-controls"),
])
def test_compatibility_rejects_changed_gates_and_ablation_inventory(mutation):
    from pbpf.iclr_config import read_config, validate_compatibility
    science = read_config(CONFIG)
    mutation(science)
    with pytest.raises(ValueError, match="gate|ablation"):
        validate_compatibility(science)


def _complete_formal_site(science):
    return dict(
        partition="provisioned", account="provisioned", usable_h200_bytes=1000,
        gpu_price_per_hour=1, cpu_price_per_hour=1, approved_projection=100000,
        container_digest="sha256:" + "a"*64, identity={}, factory="provisioned",
        factory_sha256="provisioned", public_root="provisioned", private_root="provisioned",
        trust_anchor="provisioned", sandbox_spec="provisioned", authority_receipt="provisioned",
        cells={cell["id"]: {
            "tasks": cell["tasks"] if type(cell["tasks"]) is int else 1,
            "hidden_cases": 6, "memory_bytes_bound": 100, "disk_bytes_bound": 1000,
            "gpu_hours": 1, "cpu_hours": 1,
        } for cell in science["matrix"]},
    )


@pytest.mark.parametrize("cell_id", ["s0_rbr", "s1_finite", "s3_evalplus"])
def test_formal_site_rejects_underdeclared_required_task_counts(cell_id):
    from pbpf.iclr_config import read_config, validate_static
    science = read_config(CONFIG)
    site = _complete_formal_site(science)
    site["cells"][cell_id]["tasks"] = 1
    with pytest.raises(ValueError, match="task inventory"):
        validate_static(science, "slurm_h200x16", site)


def test_formal_no_site_is_actionable_and_never_uses_smoke(monkeypatch, tmp_path, capsys):
    module = importlib.import_module("pbpf.iclr_cli")
    monkeypatch.delenv("PBPF_ICLR_SITE", raising=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("formal silently selected offline backend")
    monkeypatch.setattr(module, "offline_backend", forbidden)
    code = module.main(["run", "--config", str(CONFIG), "--profile", "slurm_h200x16",
                        "--output-root", str(tmp_path), "--resume"])
    assert code == 2
    assert "PBPF_ICLR_SITE" in capsys.readouterr().err
    assert not list(tmp_path.rglob("*.sbatch"))


def test_duplicate_unknown_and_invalid_shards_are_rejected():
    for extra in (["--profile", "local_cpu"], ["--unknown"], ["--shard-count", "0"],
                  ["--shard-index", "2", "--shard-count", "2"]):
        assert cli("doctor", *extra).returncode == 2


def test_task6_fingerprint_accepts_complete_contract_digest_but_not_operational_fields():
    from pbpf.runner.fingerprint import scientific_config
    assert scientific_config({"mode": "formal", "experiment_hash": "a"*64})["experiment_hash"] == "a"*64
    with pytest.raises(ValueError):
        scientific_config({"experiment_hash": "unfrozen"})
    with pytest.raises(ValueError):
        scientific_config({"slurm_job_id": "123"})


def test_formal_model_registry_pins_tokenizer_id_and_revision():
    from pbpf import iclr_config as module
    science = module.read_config(CONFIG)
    module.validate_pins(science)
    for row in science["models"]["models"].values():
        assert row["tokenizer_id"] == row["id"]
        assert row["tokenizer_revision"] == row["revision"]
    for key, value in (("tokenizer_id", "attacker/re-tokenized"), ("tokenizer_revision", "f" * 40)):
        changed = copy.deepcopy(science)
        changed["models"]["models"]["qwen3_8b"][key] = value
        with pytest.raises(ValueError, match="tokenizer"):
            module.validate_pins(changed)


def test_development_identity_requires_no_future_checkpoint_and_freezes_real_output_hashes():
    module = importlib.import_module("pbpf.runner.fingerprint")
    assert hasattr(module, "DevelopmentIdentity"), "typed development phase is missing"
    resolved = importlib.import_module("pbpf.iclr_config").resolve_config(CONFIG, "local_cpu")
    values = resolved.fingerprint.to_dict()
    values.pop("locks")
    dev = module.DevelopmentIdentity(**values)
    assert dev.to_dict()["phase"] == "development_inputs"
    assert "locks" not in dev.to_dict()
    locked = dev.freeze(checkpoint_hash=hashlib.sha256(b"trained parameters").hexdigest(),
                        selection_hash=hashlib.sha256(b"selected comparator").hexdigest(),
                        calibration_hash=hashlib.sha256(b"fitted temperature").hexdigest(), brier_margin=.01)
    assert locked.digest != dev.digest
    assert locked.to_dict()["locks"]["checkpoint_hash"] == hashlib.sha256(b"trained parameters").hexdigest()


def test_shell_preserves_literal_argv_and_rejects_bad_options(tmp_path):
    interpreter = tmp_path / "python shim"
    interpreter.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    interpreter.chmod(0o700)
    env = dict(os.environ, PBPF_ICLR_PYTHON=str(interpreter))
    weird = str(tmp_path / "config ;$(touch HACKED).yaml")
    result = subprocess.run(["bash", str(ROOT / "scripts/run_iclr.sh"), "--config", weird,
                             "--profile", "local_cpu", "--resume"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)
    assert argv == ["-m", "pbpf.iclr_cli", "run", "--config", weird, "--profile", "local_cpu", "--resume"]
    assert not (tmp_path / "HACKED").exists()
    for arguments in (["--config"], ["--wat"], ["--resume", "--resume"],
                      ["--shard-count", "0"], ["--shard-index", "2", "--shard-count", "2"]):
        bad = subprocess.run(["bash", str(ROOT / "scripts/run_iclr.sh"), *arguments],
                             cwd=tmp_path, env=env, capture_output=True, text=True)
        assert bad.returncode == 2


def test_sbatch_workers_reject_private_gpu_paths_and_preserve_spaces(tmp_path):
    interpreter = tmp_path / "python shim"
    interpreter.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    interpreter.chmod(0o700)
    directory = str(tmp_path / "run ;$literal")
    for script, stage in (("pbpf_gpu_array.sbatch", "bank"), ("pbpf_cpu_eval_array.sbatch", "hidden_eval"),
                          ("pbpf_control.sbatch", "gate_b")):
        args = ["bash", str(ROOT / "scripts/slurm" / script), str(interpreter), directory, stage, "1", "--resume"]
        if stage == "hidden_eval":
            args += ["--evaluator-site", "/private with spaces/site.yaml"]
        result = subprocess.run(args, env=dict(os.environ, SLURM_ARRAY_TASK_ID="0"), capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)[:5] == ["-m", "pbpf.runner.formal", directory, stage, "1"]
        if stage != "hidden_eval":
            rejected = subprocess.run(args+["--evaluator-site", "/secret"], capture_output=True, text=True)
            assert rejected.returncode == 2


def test_static_overrun_and_runtime_model_mismatch_fail_before_factory_or_imports(monkeypatch):
    from pbpf import iclr_config as module
    science = module.read_config(CONFIG)
    site = {key: "provisioned" for key in ("partition", "account", "factory", "factory_sha256", "public_root",
            "private_root", "trust_anchor", "sandbox_spec", "authority_receipt")}
    site.update(usable_h200_bytes=1000, gpu_price_per_hour=2, cpu_price_per_hour=1,
                approved_projection=1, container_digest="sha256:"+"a"*64, identity={},
                cells={cell["id"]: {"tasks": cell["tasks"] if type(cell["tasks"]) is int else 1,
                       "hidden_cases": 6, "memory_bytes_bound": 100,
                       "disk_bytes_bound": 1000, "gpu_hours": 1, "cpu_hours": 1} for cell in science["matrix"]})
    calls = []
    monkeypatch.setattr(module, "validate_dependencies", lambda: calls.append("imports"))
    with pytest.raises(ValueError, match="projected cost"):
        module.validate_config(science, profile="slurm_h200x16", site=site, execution=True)
    assert not calls
    site["approved_projection"] = 100000
    site["cells"]["s0_rbr"]["memory_bytes_bound"] = 901
    with pytest.raises(ValueError, match="90 percent"):
        module.validate_config(science, profile="slurm_h200x16", site=site, execution=True)
    assert not calls
    site["identity"] = {"models": {"qwen25_7b": {"model_id": "wrong", "revision": "main"}}, "datasets": {}}
    with pytest.raises(ValueError, match="runtime model"):
        module.validate_config(science, profile="slurm_h200x16", site=site, execution=True)


def _formal_site_with_runtime_identities(module, science):
    models = {
        name: {
            "model_id": model_id,
            "revision": revision,
            "tokenizer_id": model_id,
            "tokenizer_revision": revision,
            "chat_template_hash": "1" * 64,
            "weights_hash": "2" * 64,
        }
        for name, (model_id, revision) in module.MODEL_PINS.items()
        if name != "qwen25_1p5b"
    }
    datasets = {
        protocol: {
            "dataset_id": configured["dataset_id"],
            "revision": configured["revisions"]["repository"],
            "adapter_hash": "3" * 64,
            "split_hash": "4" * 64,
            "raw_hash": "5" * 64,
        }
        for protocol, configured in science["datasets"]["protocols"].items()
    }
    return {
        "partition": "h200",
        "account": "science",
        "usable_h200_bytes": 1_000_000,
        "gpu_price_per_hour": 1,
        "cpu_price_per_hour": 1,
        "approved_projection": 1_000_000,
        "container_digest": "sha256:" + "6" * 64,
        "cells": {
            cell["id"]: {
                "tasks": 1,
                "hidden_cases": 1,
                "memory_bytes_bound": 1,
                "disk_bytes_bound": 1,
                "gpu_hours": 0,
                "cpu_hours": 0,
            }
            for cell in science["matrix"]
        },
        "identity": {"models": models, "datasets": datasets},
        "factory": "/operator/factory.py",
        "factory_sha256": "7" * 64,
        "public_root": "/operator/public",
        "private_root": "/operator/private",
        "trust_anchor": "/operator/key",
        "sandbox_spec": "/operator/sandbox.yaml",
        "authority_receipt": "/operator/receipt.json",
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda datasets: datasets["PBPF-RBR"].update(dataset_id="operator/renamed-dataset"),
            "runtime dataset.*id",
        ),
        (
            lambda datasets: datasets["PBPF-RBR"].update(revision="8" * 40),
            "runtime dataset.*revision",
        ),
        (
            lambda datasets: datasets.update(
                {"operator-renamed-protocol": datasets.pop("PBPF-RBR")}
            ),
            "runtime dataset.*protocol",
        ),
        *[
            (
                lambda datasets, field=field: datasets["PBPF-RBR"].pop(field),
                "runtime dataset.*snapshot hashes",
            )
            for field in ("adapter_hash", "split_hash", "raw_hash")
        ],
    ],
)
def test_formal_runtime_dataset_identity_fails_closed_before_factory_or_model_imports(
    mutation, match, monkeypatch
):
    module = importlib.import_module("pbpf.iclr_config")
    formal = importlib.import_module("pbpf.runner.formal")
    science = module.read_config(CONFIG)
    site = _formal_site_with_runtime_identities(module, science)
    mutation(site["identity"]["datasets"])
    calls = []

    class Factory:
        def preflight(self, **kwargs):
            calls.append("factory")

    monkeypatch.setattr(formal, "validate_deployment", lambda site: Factory())
    monkeypatch.setattr(module, "validate_dependencies", lambda: calls.append("imports"))

    with pytest.raises(ValueError, match=match):
        module.resolve_config(CONFIG, "slurm_h200x16", site=site, execution=True)
    assert calls == []


def test_formal_fingerprint_rechecks_dataset_identity_after_preflight():
    module = importlib.import_module("pbpf.iclr_config")
    science = module.read_config(CONFIG)
    site = _formal_site_with_runtime_identities(module, science)
    assert module._fingerprint(science, "slurm_h200x16", site).to_dict()["datasets"]
    site["identity"]["datasets"]["PBPF-RBR"]["dataset_id"] = "operator/mutated-after-preflight"
    with pytest.raises(ValueError, match="runtime dataset.*id"):
        module._fingerprint(science, "slurm_h200x16", site)


def test_formal_verify_checks_gate_eligibility_not_just_claim_label():
    from pbpf.iclr_cli import gates_eligible
    from pbpf.runner.shard import digest
    def gate(name, passed):
        row = {"gate": name, "passed": passed}
        return dict(row, decision_hash=digest(row))
    assert not gates_eligible(gate("B", False), gate("C", True), {"passed": True})
    assert not gates_eligible(gate("B", True), gate("C", False), {"passed": True})
    assert not gates_eligible(gate("B", True), gate("C", True), {"passed": False})


def test_formal_gate_eligibility_requires_closed_replication_provenance():
    from pbpf.iclr_cli import gates_eligible
    from pbpf.iclr_config import resolve_config
    from pbpf.runner.shard import digest

    identity = resolve_config(CONFIG, "local_cpu").fingerprint
    selection_hash = identity.to_dict()["locks"]["selection_hash"]
    stage_b_body = {"gate": "B", "passed": True, "selection_hash": selection_hash}
    stage_b = dict(stage_b_body, decision_hash=digest(stage_b_body))
    stage_c_body = {"gate": "C", "passed": True, "stage_b_hash": stage_b["decision_hash"]}
    stage_c = dict(stage_c_body, decision_hash=digest(stage_c_body))
    evidence_hashes = {"replication_hash": "d"*64, "replication_eval_hash": "e"*64}
    gate_body = {"schema": "pbpf-replication-gate-v1", "gate": "replication", "passed": True,
                 "scientific_identity": identity.digest, "selection_hash": selection_hash,
                 "gate_c_hash": stage_c["decision_hash"], **evidence_hashes}
    replication_gate = dict(gate_body, decision_hash=digest(gate_body))

    assert gates_eligible(stage_b, stage_c, replication_gate, scientific_identity=identity,
                          **evidence_hashes)
    for key, value in {
        "schema": "wrong", "decision_hash": "0"*64, "scientific_identity": "1"*64,
        "selection_hash": "2"*64, "gate_c_hash": "3"*64,
        "replication_hash": "4"*64, "replication_eval_hash": "5"*64,
    }.items():
        tampered = dict(replication_gate, **{key: value})
        with pytest.raises(ValueError, match="replication gate"):
            gates_eligible(stage_b, stage_c, tampered, scientific_identity=identity,
                           **evidence_hashes)


def test_doctor_counts_future_verification():
    from pbpf.iclr_config import matrix_report, resolve_config
    resolved = resolve_config(CONFIG, "local_cpu")
    # 60 visible suites, 24 future prediction suites, 9 selected hidden suites.
    assert sum(cell["sandbox_suites"] for cell in matrix_report(resolved)) == 93


def test_doctor_counts_roulette_partial_input_requests():
    from dataclasses import replace
    from pbpf.iclr_config import matrix_report, resolve_config
    resolved = resolve_config(CONFIG, "local_cpu")
    site = {"gpu_price_per_hour": 1, "cpu_price_per_hour": 1,
            "cells": {cell["id"]: {"tasks": 1, "hidden_cases": 6, "memory_bytes_bound": 100,
                       "disk_bytes_bound": 1000, "gpu_hours": 1, "cpu_hours": 1} for cell in resolved.science["matrix"]}}
    formal = replace(resolved, profile="slurm_h200x16", site=site)
    cell = next(row for row in matrix_report(formal) if row["id"] == "s3_rbr")
    assert cell["actor_decodes"] == 96  # 3 seeds * (8 roots + 6 matched arms * 4)
    assert cell["actor_requests_bound"] == 108  # 4 additional Roulette prefix calls per seed.
    assert cell["input_token_bound"] == 884736
    assert cell["sandbox_suites"] >= 2 * 96 + 3 * 7
    assert cell["protocol_accounting_scope"] == "single-three-seed-sweep-excludes-training-grid-search-retries"
    replication = next(row for row in matrix_report(formal) if row["id"] == "s4_qwen3_evalplus")
    assert replication["actor_decodes"] == 60
    # Either locked comparator might be Roulette before development selection.
    assert replication["actor_requests_bound"] == 84
