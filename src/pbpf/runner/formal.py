"""Fail-closed production integration and argv-only Slurm dispatch.

There is intentionally no built-in production scientific factory: the operator
must provision and audit one alongside immutable snapshots and the independent
Task-6 evaluator authority. A synthetic backend is never selected here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

import yaml

from .aggregate import validate_decision
from .doctor import validate_calibration
from .fingerprint import DevelopmentIdentity, ScientificIdentity
from .shard import atomic_write, canonical_bytes, digest
from .stage import DEPENDENCIES, STAGES, GateFailure, run_stage

ROOT = Path(__file__).resolve().parents[3]
FORMAL_STAGES = (*STAGES[:8], "prediction_eval", *STAGES[8:14], "replication_eval", "tables")
FORMAL_DEPENDENCIES = dict(DEPENDENCIES, prediction_eval=("predict",), gate_b=("predict", "prediction_eval"),
                           replication_eval=("replication",), tables=("replication", "replication_eval", "gate_c", "gate_b"))
GPU_STAGES = frozenset({"bank", "train_belief", "predict", "repair", "replication"})
EVALUATOR_STAGES = frozenset({"prepare", "visible_execute", "prediction_eval", "hidden_eval", "replication_eval"})
PRIVATE_KEYS = frozenset({"private_root", "trust_anchor", "sandbox_spec", "authority_receipt", "evaluator_site"})


class FormalFactory(ABC):
    """Audited, lazy production integration, loaded from an exact file checksum.

    preflight must validate availability without importing ML or loading models.
    execute receives public config, declared immutable dependencies and a work
    key. Only evaluator-class stages receive the private operational overlay.
    Actor dispatch must use MeteredActor; evaluator operations must use the
    separately provisioned TrustedEvaluatorService and signed capabilities.
    Training publishes actual checkpoint/selection/calibration lock artifacts.
    No handler may treat a smoke fixture or a declaration as measured evidence.
    """
    @abstractmethod
    def preflight(self, *, science, site):
        raise NotImplementedError

    @abstractmethod
    def work_keys(self, stage):
        raise NotImplementedError

    @abstractmethod
    def execute(self, context, key):
        raise NotImplementedError


def load_factory(site):
    path, expected = site.get("factory"), site.get("factory_sha256")
    if not path or not expected:
        raise ValueError("production handler factory missing; provision an audited FormalFactory and immutable file checksum")
    source = Path(path)
    if not source.is_absolute() or not source.is_file() or source.suffix != ".py":
        raise ValueError("production handler factory must be a directly named absolute Python file")
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("production factory checksum mismatch; refusing import")
    from .evaluator import _trusted_dependency_path
    _trusted_dependency_path(source)
    spec = importlib.util.spec_from_file_location("pbpf_site_factory_"+expected, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    create = getattr(module, "create_factory", None)
    if not callable(create):
        raise ValueError("production factory must export create_factory()")
    factory = create()
    if not isinstance(factory, FormalFactory):
        raise ValueError("production factory must implement FormalFactory; offline/local backends are prohibited")
    return factory


def validate_deployment_path(path, *, evaluator_uid):
    """Launcher-side check uses the designated evaluator, not the caller UID."""
    path = Path(path)
    resolved = path.resolve(strict=True)
    for entry in (path, *path.parents, resolved, *resolved.parents):
        info = entry.lstat()
        if info.st_uid not in {0, evaluator_uid}:
            raise ValueError("deployment path owner is neither root nor the designated evaluator")
        if not stat.S_ISLNK(info.st_mode) and stat.S_IMODE(info.st_mode) & 0o022:
            if not (entry not in {path, resolved} and stat.S_ISDIR(info.st_mode) and info.st_mode & stat.S_ISVTX):
                raise ValueError("deployment path or ancestor is generator-writable")


def validate_evaluator_overlay(public, private):
    if canonical_bytes({key: value for key, value in private.items() if key not in PRIVATE_KEYS}) != canonical_bytes(public):
        raise ValueError("evaluator overlay differs from frozen public deployment identity")


def validate_public_executables(paths):
    """Supported deployment: root installs public code; evaluator owns secrets."""
    for path in paths:
        try:
            validate_deployment_path(Path(path), evaluator_uid=0)
        except (ValueError, OSError):
            raise ValueError("root-owned public executable files and protected root-owned ancestors required for factory/sandbox dependencies") from None


def validate_deployment(site):
    """Check pre-provisioned authority metadata; never create/read secret keys."""
    from .evaluator import ExternalSandboxSpec
    for name in PRIVATE_KEYS - {"evaluator_site"}:
        if not site.get(name):
            raise ValueError(f"formal deployment missing {name}; provision evaluator authority before launch")
        path = Path(site[name])
        if not path.is_absolute() or not path.exists():
            raise ValueError(f"formal deployment {name} must be an existing absolute provisioned location")
    receipt = json.loads(Path(site["authority_receipt"]).read_bytes())
    if (receipt.get("schema") != "pbpf-external-deployment-v1" or receipt.get("generator_uid") == receipt.get("evaluator_uid")
            or any(type(receipt.get(k)) is not int for k in ("generator_uid", "evaluator_uid"))
            or receipt.get("container_digest") != site["container_digest"]
            or receipt.get("factory_sha256") != site["factory_sha256"]):
        raise ValueError("external deployment attestation must bind distinct UIDs, container, and production factory")
    for name in PRIVATE_KEYS - {"evaluator_site"}:
        validate_deployment_path(Path(site[name]), evaluator_uid=receipt["evaluator_uid"])
    anchor = Path(site["trust_anchor"])
    info = anchor.stat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o400 or info.st_size < 32:
        raise ValueError("pre-provisioned 0400 trust anchor with at least 32 secret bytes required")
    specification = yaml.safe_load(Path(site["sandbox_spec"]).read_text())
    validate_public_executables([site["factory"], *specification["dependency_files"]])
    spec = ExternalSandboxSpec(**specification)
    if spec.container_digest != site["container_digest"]:
        raise ValueError("sandbox and experiment immutable container digest differ")
    # This receipt is an external operator attestation, not a self-issued token.
    # Its trusted filesystem ownership is checked above; live capabilities and
    # measured UID/backend checks remain mandatory at every evaluator request.
    return load_factory(site)


def require_phase(identity, stage):
    if stage in FORMAL_STAGES[FORMAL_STAGES.index("predict"):] and type(identity) is not ScientificIdentity:
        raise ValueError("confirmatory identity (ScientificIdentity) must be frozen from real training outputs before sealed work")


def freeze_artifacts(identity, root, artifacts, *, brier_margin):
    if type(identity) is not DevelopmentIdentity or set(artifacts) != {"checkpoint", "selection", "calibration"}:
        raise ValueError("development identity and complete real learned lock artifacts required")
    root, locks = Path(root).resolve(), {}
    for name, record in artifacts.items():
        if type(record) is not dict or set(record) != {"file", "sha256"}:
            raise ValueError("real lock artifact file and checksum required")
        path = (root / record["file"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("lock artifact must exist inside the immutable public run")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != record["sha256"]:
            raise ValueError("learned artifact checksum mismatch")
        locks[name+"_hash"] = checksum
    return identity.freeze(**locks, brier_margin=brier_margin)


def confirmatory_identity(resolved, directory, *, publish=False):
    from ..iclr_cli import load_stage
    _, rows = load_stage(directory, "train_belief", resolved.fingerprint.digest)
    # The trainer publishes a single canonical manifest covering every model,
    # grid selection, comparator and temperature; it is not a future placeholder.
    manifests = [row["value"]["lock_artifacts"] for row in rows if "lock_artifacts" in row["value"]]
    if len(manifests) != 1:
        raise ValueError("one complete checkpoint/selection/calibration manifest required before confirmatory identity")
    frozen = freeze_artifacts(resolved.fingerprint, directory, manifests[0], brier_margin=resolved.science["development"]["brier_margin"])
    path = Path(directory) / "confirmatory-fingerprint.json"
    payload = canonical_bytes(frozen.to_dict())
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("confirmatory identity conflicts with current verified training artifacts")
    elif publish:
        try:
            atomic_write(path, payload, create_once=True)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("confirmatory identity publication conflict")
    else:
        raise ValueError("confirmatory identity has not been materialized")
    return frozen


def check_calibration(report, *, run_id):
    if (not isinstance(report, dict) or report.get("run_id") != run_id or report.get("expected_work") != 64
            or report.get("source_disjoint") is not True):
        raise ValueError("verified source-disjoint 64-task calibration is required before full scaling")
    return validate_calibration(**{key: report[key] for key in ("expected_work", "failed_work", "projected_cost", "measured_cost",
                                                               "allocated_memory", "available_memory")})


def check_gate_c(decision, *, failed_gate_b_hash):
    validate_decision(decision)
    if decision.get("gate") != "C":
        raise ValueError("Gate C controller returned the wrong decision")
    if not decision.get("passed") and failed_gate_b_hash is None:
        raise ValueError("Gate C failed: locked confirmatory replication stopped")
    return decision.get("passed") is True


def _submit(argv, submit):
    result = submit(argv, text=True, capture_output=True, check=False,
                    env={key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL", "USER", "LOGNAME"}})
    if result.returncode:
        raise ValueError("sbatch rejected submission: " + result.stderr.strip())
    job = result.stdout.strip()
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("numeric Slurm job ID required; remaining jobs were not submitted")
    return job


def _argv(stage, *, directory, site, shard_count, python, resume, force):
    evaluator, gpu = stage in EVALUATOR_STAGES, stage in GPU_STAGES or stage == "calibrate"
    script = "pbpf_gpu_array.sbatch" if gpu else "pbpf_cpu_eval_array.sbatch" if evaluator else "pbpf_control.sbatch"
    argv = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes", "--export=NONE",
            "--partition="+site["partition"], "--account="+site["account"], "--job-name=pbpf-"+stage]
    count = shard_count if stage in GPU_STAGES | EVALUATOR_STAGES else 1
    # Training publishes one complete lock manifest; one array task owns that
    # inventory, while arbitrary model-specific work remains inside that task.
    if stage == "train_belief":
        count = 1
    if gpu:
        argv += ["--gres=gpu:h200:1", f"--array=0-{count-1}%16"]
    elif evaluator:
        argv += [f"--array=0-{count-1}%16"]
    argv += [str(ROOT / "scripts/slurm" / script), python, str(directory), stage, str(count)]
    if resume:
        argv.append("--resume")
    if force:
        argv.append("--force-after-failed-gate")
    if evaluator:
        argv += ["--evaluator-site", site["evaluator_site"]]
    return argv


def submit_dag(*, directory, site, shard_count, python, resume, force, submit=subprocess.run, calibration_job=None,
               stop_after=None):
    if type(shard_count) is not int or shard_count < 1:
        raise ValueError("positive shard count required")
    jobs = {}
    for stage in FORMAL_STAGES:
        argv = _argv(stage, directory=directory, site=site, shard_count=shard_count, python=python, resume=resume, force=force)
        dependencies = [jobs[name] for name in FORMAL_DEPENDENCIES[stage]]
        if calibration_job and stage == "doctor":
            dependencies.append(calibration_job)
        if dependencies:
            argv.insert(1, "--dependency=afterok:"+":".join(dependencies))
        jobs[stage] = _submit(argv, submit)
        if stage == stop_after:
            break
    return jobs


@dataclass
class FormalContext:
    name: str
    science: dict
    identity: object
    root: Path
    public_site: dict
    evaluator_site: dict | None
    dependencies: dict
    confirmatory: bool
    failed_gate_hash: str | None

    def __post_init__(self):
        self._consumed = set()

    def input(self, name):
        self._consumed.add(name)
        return [row["value"] for row in self.dependencies[name]]


def run_worker(resolved, directory, *, stage, shard_index, shard_count, force=False, evaluator_site=None):
    from ..iclr_cli import load_stage
    if stage not in (*FORMAL_STAGES, "calibrate"):
        raise ValueError("unknown formal worker stage")
    if (evaluator_site is not None) != (stage in EVALUATOR_STAGES):
        raise ValueError("private evaluator overlay is required only for evaluator stages")
    identity = resolved.fingerprint
    if stage in FORMAL_STAGES[FORMAL_STAGES.index("predict"):]:
        identity = confirmatory_identity(resolved, directory, publish=True)
    require_phase(identity, stage)
    rows, checksums = {}, {}
    for dep in FORMAL_DEPENDENCIES.get(stage, ()):
        expected = identity.digest if FORMAL_STAGES.index(dep) >= FORMAL_STAGES.index("predict") else resolved.fingerprint.digest
        results, rows[dep] = load_stage(directory, dep, expected)
        checksums[dep] = digest([r.checksum for r in results])
    failed_hash = None
    if stage in FORMAL_STAGES[FORMAL_STAGES.index("repair"):]:
        _, gate_rows = load_stage(directory, "gate_b", identity.digest)
        gate = gate_rows[0]["value"]
        validate_decision(gate)
        if not gate["passed"]:
            if not force:
                raise GateFailure(gate)
            failed_hash = gate["decision_hash"]
    if stage == "doctor":
        check_calibration(json.loads((Path(directory) / "calibration.json").read_bytes()), run_id=resolved.fingerprint.digest)
    factory = load_factory(resolved.site)
    factory.preflight(science=resolved.science, site=resolved.site)
    keys = factory.work_keys(stage)
    if stage in {"gate_b", "gate_c", "tables", "calibrate", "train_belief"} and (len(keys) != 1 or shard_count != 1):
        raise ValueError("controller/calibration/lock-producing stage requires exactly one work key and shard")
    def execute(key, dependencies):
        context = FormalContext(stage, resolved.science, identity, Path(directory), resolved.site, evaluator_site,
                                rows, failed_hash is None, failed_hash)
        result = factory.execute(context, key)
        if context._consumed != set(rows):
            raise ValueError("formal handler must consume every declared dependency")
        return result
    result = run_stage(root=Path(directory) / "stages", name=stage, fingerprint=identity,
        keys=keys, dependencies=checksums, config=identity.to_dict()["config"], handler=execute,
        shard_index=shard_index, shard_count=shard_count, confirmatory=failed_hash is None,
        reason="forced_after_failed_gate_b" if failed_hash else None, failed_gate_hash=failed_hash)
    if stage == "gate_b":
        gate = result.rows()[0]["value"]
        validate_decision(gate)
        if gate.get("gate") != "B":
            raise ValueError("Gate B controller returned wrong gate")
        if not gate["passed"] and not force:
            raise GateFailure(gate)
    if stage == "gate_c":
        check_gate_c(result.rows()[0]["value"], failed_gate_b_hash=failed_hash)
    if stage == "calibrate":
        report = result.rows()[0]["value"]
        check_calibration(report, run_id=resolved.fingerprint.digest)
        from ..iclr_cli import _write_once_json
        _write_once_json(Path(directory) / "calibration.json", report)
    return {"stage": stage, "cached": result.cached, "confirmatory": failed_hash is None}


def execute_formal(resolved, directory, args):
    from ..iclr_cli import _write_once_json
    factory = load_factory(resolved.site)
    factory.preflight(science=resolved.science, site={k: v for k, v in resolved.site.items() if k not in PRIVATE_KEYS})
    public = {key: value for key, value in resolved.site.items() if key not in PRIVATE_KEYS}
    _write_once_json(Path(directory) / "public-deployment.json", public)
    # The private overlay is operator-owned, not generated in a public run.
    site = dict(resolved.site, evaluator_site=str(Path(args.site or os.environ["PBPF_ICLR_SITE"]).resolve()))
    calibration_job = None
    calibration = Path(directory) / "calibration.json"
    if args.calibration_only or not calibration.exists():
        calibration_job = _submit(_argv("calibrate", directory=directory, site=site, shard_count=1,
            python=sys.executable, resume=args.resume, force=False), subprocess.run)
    else:
        check_calibration(json.loads(calibration.read_bytes()), run_id=resolved.fingerprint.digest)
    if args.calibration_only:
        return {"submitted": True, "calibration_job": calibration_job}
    jobs = submit_dag(directory=directory, site=site, shard_count=args.shard_count, python=sys.executable,
        resume=args.resume, force=args.force_after_failed_gate, calibration_job=calibration_job,
        stop_after="prepare" if args.command == "prepare" else None)
    return {"submitted": True, "jobs": jobs, "calibration_job": calibration_job}


def worker_main(argv=None):
    """Internal sbatch-only argv contract; no secret-bearing site on GPU workers."""
    import argparse
    from ..iclr_config import ResolvedConfig, _fingerprint, read_config, validate_compatibility, validate_pins, validate_schema
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("directory")
    parser.add_argument("stage", choices=(*FORMAL_STAGES, "calibrate"))
    parser.add_argument("shard_count", type=int)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-after-failed-gate", action="store_true")
    parser.add_argument("--evaluator-site")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        directory = Path(args.directory).resolve()
        science = read_config(directory / "resolved.yaml")
        validate_schema(science, "slurm_h200x16")
        validate_pins(science)
        validate_compatibility(science)
        values = json.loads((directory / "fingerprint.json").read_bytes())
        if values.pop("phase", None) != "development_inputs":
            raise ValueError("formal worker requires typed development run identity")
        identity = DevelopmentIdentity(**values)
        if identity.digest != directory.name or identity.to_dict()["config"]["experiment_hash"] != digest(science):
            raise ValueError("worker scientific config/run identity mismatch")
        site = json.loads((directory / "public-deployment.json").read_bytes())
        if set(site) & PRIVATE_KEYS:
            raise ValueError("public deployment illegally contains evaluator-private material")
        if _fingerprint(science, "slurm_h200x16", site).digest != identity.digest:
            raise ValueError("worker code/model/data/factory immutable identity changed")
        validate_public_executables([site["factory"]])
        private = None
        if args.evaluator_site:
            if args.stage not in EVALUATOR_STAGES:
                raise ValueError("GPU/controller worker cannot receive private evaluator overlay")
            private = yaml.safe_load(Path(args.evaluator_site).read_text())
            validate_evaluator_overlay(site, private)
            validate_deployment(private)
        resolved = ResolvedConfig(science, "slurm_h200x16", site, identity)
        result = run_worker(resolved, directory, stage=args.stage, shard_index=args.shard_index,
                            shard_count=args.shard_count, force=args.force_after_failed_gate, evaluator_site=private)
        print(json.dumps(result, sort_keys=True))
        return 0
    except GateFailure as error:
        print(str(error), file=sys.stderr)
        return 20
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print("formal worker: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(worker_main())
