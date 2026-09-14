"""Formal dispatch tests use explicit test-owned factories, never formal results."""
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pbpf.runner.fingerprint import DevelopmentIdentity
from pbpf.runner.shard import canonical_bytes, digest

ROOT = Path(__file__).resolve().parents[2]


def development():
    from pbpf.iclr_config import resolve_config
    config = resolve_config(ROOT / "configs/experiments/iclr_pbpf.yaml", "local_cpu")
    values = config.fingerprint.to_dict()
    values.pop("locks")
    return config, DevelopmentIdentity(**values)


def test_development_identity_cannot_enter_any_sealed_confirmatory_worker():
    module = importlib.import_module("pbpf.runner.formal")
    _, identity = development()
    for name in ("predict", "prediction_eval", "repair", "hidden_eval", "replication", "replication_eval"):
        with pytest.raises(ValueError, match="confirmatory.*identity"):
            module.require_phase(identity, name)


def test_freeze_checks_actual_artifact_bytes_and_refuses_changed_outputs(tmp_path):
    module = importlib.import_module("pbpf.runner.formal")
    _, identity = development()
    artifacts = {}
    for name, content in (("checkpoint", b"trained tensor bytes"), ("selection", b"selected comparator"), ("calibration", b"fitted temperature")):
        path = tmp_path / (name + ".bin")
        path.write_bytes(content)
        artifacts[name] = {"file": path.name, "sha256": hashlib.sha256(content).hexdigest()}
    frozen = module.freeze_artifacts(identity, tmp_path, artifacts, brier_margin=.01)
    module.require_phase(frozen, "predict")
    assert frozen.to_dict()["locks"]["checkpoint_hash"] == hashlib.sha256(b"trained tensor bytes").hexdigest()
    (tmp_path / "checkpoint.bin").write_bytes(b"altered")
    with pytest.raises(ValueError, match="artifact.*checksum"):
        module.freeze_artifacts(identity, tmp_path, artifacts, brier_margin=.01)


def test_formal_factory_is_explicit_hash_checked_and_never_offline(tmp_path):
    module = importlib.import_module("pbpf.runner.formal")
    with pytest.raises(ValueError, match="production.*factory"):
        module.load_factory({})
    path = tmp_path / "factory.py"
    path.write_text("from pbpf.runner.formal import FormalFactory\nclass Adapter(FormalFactory):\n    def preflight(self, **kwargs): return None\n    def work_keys(self, stage): return [[stage]]\n    def execute(self, context, key): return {'backend': 'test-owned-dispatch-only'}\ndef create_factory(): return Adapter()\n")
    site = {"factory": str(path), "factory_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    factory = module.load_factory(site)
    assert factory.execute(None, []) == {"backend": "test-owned-dispatch-only"}
    path.write_text("raise RuntimeError('must not import changed module')\n")
    with pytest.raises(ValueError, match="factory.*checksum"):
        module.load_factory(site)


def test_slurm_dag_one_gpu_percent16_singleton_gates_and_private_evaluator_only(tmp_path):
    module = importlib.import_module("pbpf.runner.formal")
    calls = []
    def submit(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=str(100+len(calls))+"\n", stderr="")
    site = {"partition": "h200", "account": "science", "evaluator_site": str(tmp_path / "secret ;$x.yaml")}
    result = module.submit_dag(directory=tmp_path / "run ;$(touch bad)", site=site, shard_count=32,
        python="/interpreter with spaces/python", resume=True, force=False, submit=submit)
    assert set(result) == set(module.FORMAL_STAGES)
    for name, (argv, kwargs) in zip(module.FORMAL_STAGES, calls):
        assert isinstance(argv, list) and kwargs.get("shell", False) is False
        assert "--kill-on-invalid-dep=yes" in argv
        assert "--export=NONE" in argv
        assert not any(x in argv for x in ("--wrap", "bash -c", "eval"))
        if name in module.GPU_STAGES:
            assert "--gres=gpu:h200:1" in argv
            assert ("--array=0-0%16" if name == "train_belief" else "--array=0-31%16") in argv
        if name in {"gate_b", "gate_c"}:
            assert not any(x.startswith("--array=") for x in argv)
        private = site["evaluator_site"] in argv
        assert private is (name in module.EVALUATOR_STAGES)
        if module.FORMAL_DEPENDENCIES[name]:
            dependency = next(x for x in argv if x.startswith("--dependency="))
            assert dependency.startswith("--dependency=afterok:")
    assert not (tmp_path / "bad").exists()


def test_slurm_rejects_non_numeric_job_id_before_further_submission(tmp_path):
    module = importlib.import_module("pbpf.runner.formal")
    calls = []
    def submit(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="123;touch bad", stderr="")
    with pytest.raises(ValueError, match="numeric.*job"):
        module.submit_dag(directory=tmp_path, site={"partition": "h200", "account": "science", "evaluator_site": "/secret"},
                          shard_count=16, python="/python", resume=True, force=False, submit=submit)
    assert len(calls) == 1


def test_calibration_required_before_full_scaling_and_checks_exact_64():
    module = importlib.import_module("pbpf.runner.formal")
    with pytest.raises(ValueError, match="64-task"):
        module.check_calibration({}, run_id="a"*64)
    report = dict(run_id="a"*64, expected_work=64, failed_work=0, projected_cost=100., measured_cost=100.,
                  allocated_memory=80, available_memory=100, source_disjoint=True)
    assert module.check_calibration(report, run_id="a"*64)
    report["allocated_memory"] = 91
    with pytest.raises(ValueError, match="90 percent"):
        module.check_calibration(report, run_id="a"*64)


def test_gate_c_blocks_replication_unless_already_nonconfirmatory_from_failed_b():
    module = importlib.import_module("pbpf.runner.formal")
    row = {"gate": "C", "passed": False}
    decision = dict(row, decision_hash=digest(row))
    with pytest.raises(ValueError, match="Gate C failed"):
        module.check_gate_c(decision, failed_gate_b_hash=None)
    assert module.check_gate_c(decision, failed_gate_b_hash="a"*64) is False


def test_fresh_worker_initializes_factory_before_deriving_work_inventory(tmp_path, monkeypatch):
    from dataclasses import replace
    module = importlib.import_module("pbpf.runner.formal")
    resolved, identity = development()
    resolved = replace(resolved, profile="slurm_h200x16", fingerprint=identity)
    class Factory(module.FormalFactory):
        def preflight(self, *, science, site): self.name = science["name"]
        def work_keys(self, stage): return [[stage, self.name]]
        def execute(self, context, key):
            return dict(run_id=identity.digest, expected_work=64, failed_work=0, projected_cost=100., measured_cost=100.,
                        allocated_memory=80, available_memory=100, source_disjoint=True)
    monkeypatch.setattr(module, "load_factory", lambda site: Factory())
    module.run_worker(resolved, tmp_path, stage="calibrate", shard_index=0, shard_count=1)
    assert json.loads((tmp_path / "calibration.json").read_bytes())["expected_work"] == 64


def test_submit_normalizes_relative_private_overlay_before_jobs(tmp_path, monkeypatch):
    from dataclasses import replace
    module = importlib.import_module("pbpf.runner.formal")
    resolved, identity = development()
    resolved = replace(resolved, profile="slurm_h200x16", fingerprint=identity,
                       site={"partition": "h200", "account": "science"})
    monkeypatch.chdir(tmp_path)
    (tmp_path / "site.yaml").write_text("site: fixture\n")
    monkeypatch.setattr(module, "load_factory", lambda site: SimpleNamespace(preflight=lambda **kwargs: None))
    captured = []
    def submit(argv, function):
        captured.append(argv)
        return str(100+len(captured))
    monkeypatch.setattr(module, "_submit", submit)
    args = SimpleNamespace(site="site.yaml", resume=True, force_after_failed_gate=False,
                           calibration_only=False, command="run", shard_count=16)
    module.execute_formal(resolved, tmp_path / "run", args)
    evaluator = next(argv for argv in captured if "--evaluator-site" in argv)
    assert evaluator[evaluator.index("--evaluator-site")+1] == str(tmp_path / "site.yaml")


def test_preflight_accepts_explicit_evaluator_owner_not_generator_owner(tmp_path, monkeypatch):
    module = importlib.import_module("pbpf.runner.formal")
    target = tmp_path / "authority.key"
    target.write_bytes(b"k"*32)
    target.chmod(0o400)
    original_stat, original_lstat = Path.stat, Path.lstat
    owner = [4242]
    def replace_owner(info, uid):
        values = list(info)
        values[4] = uid
        return os.stat_result(values)
    def metadata(path, function, **kwargs):
        info = function(path, **kwargs)
        if path == target:
            return replace_owner(info, owner[0])
        if path in target.parents:
            return replace_owner(info, 0)  # Explicit root-owned ancestry fixture, portable to non-root CI.
        return info
    monkeypatch.setattr(Path, "stat", lambda path, **kwargs: metadata(path, original_stat, **kwargs))
    monkeypatch.setattr(Path, "lstat", lambda path: metadata(path, original_lstat))
    module.validate_deployment_path(target, evaluator_uid=4242)
    owner[0] = 1001
    with pytest.raises(ValueError, match="owner"):
        module.validate_deployment_path(target, evaluator_uid=4242)


def test_private_overlay_must_match_frozen_public_scientific_deployment():
    module = importlib.import_module("pbpf.runner.formal")
    public = {"container_digest": "sha256:"+"a"*64, "identity": {"dataset": "original"}}
    private = dict(public, trust_anchor="/evaluator/key")
    module.validate_evaluator_overlay(public, private)
    private["identity"] = {"dataset": "different"}
    with pytest.raises(ValueError, match="evaluator.*deployment"):
        module.validate_evaluator_overlay(public, private)


def test_supported_deployment_requires_root_owned_public_executable_inputs(tmp_path, monkeypatch):
    module = importlib.import_module("pbpf.runner.formal")
    target = tmp_path / "factory.py"
    target.write_text("# public audited code\n")
    original_lstat = Path.lstat
    def owner(path):
        info = original_lstat(path)
        if path == target:
            values = list(info)
            values[4] = 4242
            return os.stat_result(values)
        return info
    monkeypatch.setattr(Path, "lstat", owner)
    with pytest.raises(ValueError, match="root-owned public executable"):
        module.validate_public_executables([target])
