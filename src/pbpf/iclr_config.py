"""Canonical ICLR contract resolution; no legacy diagnostic configuration path.

Site overlays are operational. Immutable snapshot and container identities are
copied into the scientific fingerprint, never paths, Slurm IDs, or prices.
Validation deliberately precedes all optional dependency discovery/imports.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
from importlib.util import find_spec
import json
import math
from pathlib import Path
import re

import yaml

from .arms.registry import ABLATION_FACTORS, FORMAL_BUDGET, load_arm_config, load_ablation_config
from .runner.fingerprint import DevelopmentIdentity, RunFingerprint, code_identity
from .runner.shard import atomic_write, canonical_bytes, digest

ROOT = Path(__file__).resolve().parents[2]
PIN = re.compile(r"[0-9a-f]{40}")
HASH = re.compile(r"[0-9a-f]{64}")
MODEL_PINS = {
    "qwen25_1p5b": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "2e1fd397ee46e1388853d2af2c993145b0f1098a"),
    "qwen25_7b": ("Qwen/Qwen2.5-Coder-7B-Instruct", "c03e6d358207e414f1eca0bb1891e29f1db0e242"),
    "qwen3_8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "deepseek_6p7b": ("deepseek-ai/deepseek-coder-6.7b-instruct", "e5d64addd26a6a1db0f9b863abf6ee3141936807"),
}
CANONICAL_SECTION_DIGESTS = {
    "matrix": "e0a3e027d0412116e8e3751c46c0abc9650040391a28639a43b5449b3521c945",
    "gates": "c243e4ad448cedafa6086b3b1a04be73a7956dcdbf13e2d00845511f0bbf8cc6",
    "ablations": "14e90e9f51db39cb516bde17433dd28f5752a7494a85a1d8dba06e91988d597d",
}
DATASET_IDENTITY_FIELDS = {"dataset_id", "revision", "adapter_hash", "split_hash", "raw_hash"}


def science_digest(value):
    return digest(value)


def read_config(path, _stack=()):
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError("recursive config reference cycle")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config schema requires a mapping")
    def expand(item):
        if isinstance(item, dict):
            if "$ref" in item:
                if set(item) != {"$ref"} or type(item["$ref"]) is not str:
                    raise ValueError("registry references cannot mix overrides")
                return read_config(path.parent / item["$ref"], (*_stack, path))
            return {key: expand(child) for key, child in item.items()}
        if isinstance(item, list):
            return [expand(child) for child in item]
        return item
    return expand(value)


def validate_schema(config, profile):
    required = {"schema", "name", "claim_status", "models", "datasets", "arms", "ablations", "profiles",
                "finite", "prompts", "protocol", "statistics", "development", "calibration", "gates", "matrix", "excluded_from_core"}
    if (type(config) is not dict or set(config) != required
            or config.get("schema") != "pbpf-iclr-matrix-v1"):
        raise ValueError("ICLR schema requires exactly the versioned scientific contract fields")
    if profile not in {"local_cpu", "slurm_h200x16"}:
        raise ValueError("profile must be local_cpu or slurm_h200x16")
    for name in required - {"schema", "name", "claim_status", "matrix", "excluded_from_core"}:
        if type(config[name]) is not dict:
            raise ValueError(f"schema field {name} requires a mapping")
    if not isinstance(config["matrix"], list) or not config["matrix"]:
        raise ValueError("schema requires explicit nonempty matrix cells")
    for cell in config["matrix"]:
        if type(cell) is not dict or set(cell) != {"id", "stage", "phase", "model", "dataset", "tasks", "arms"}:
            raise ValueError("schema requires complete matrix cell fields")
        if not isinstance(cell["arms"], list) or not cell["arms"]:
            raise ValueError("schema requires nonempty arm lists")


def validate_pins(config):
    models = config["models"].get("models", {})
    if set(models) != set(MODEL_PINS):
        raise ValueError("immutable model inventory mismatch")
    for name, (identifier, revision) in MODEL_PINS.items():
        row = models[name]
        if row.get("id") != identifier or row.get("revision") != revision:
            raise ValueError(f"immutable model revision/ID mismatch for {name}")
        if row.get("tokenizer_id") != identifier or row.get("tokenizer_revision") != revision:
            raise ValueError(f"immutable tokenizer revision/ID mismatch for {name}")
        if row.get("native_chat_template") is not True or row.get("thinking") is not False:
            raise ValueError("native pinned chat templates and disabled Qwen3 thinking required")
    from .data.firewall import checked_protocol_provenance
    for name, row in config["datasets"].get("protocols", {}).items():
        checked_protocol_provenance(name, row["dataset_id"], row["dataset_id"], row["revisions"], row["revisions"])
    if set(config["datasets"].get("protocols", {})) != {"PBPF-RBR", "CodeARC-Replay", "PBPF-EvalPlus", "LiveCodeBench"}:
        raise ValueError("immutable core dataset inventory mismatch")
    if config["prompts"].get("schema") != "pbpf-prompts-v1" or config["finite"].get("schema") != "pbpf-finite-generator-v1":
        raise ValueError("versioned prompt bundle and finite generator required")
    for name in ("root_system", "root_user", "repair_system", "repair_user"):
        if not config["prompts"].get(name):
            raise ValueError("actual immutable prompt text required")


def _validate_runtime_datasets(config, identity):
    """Bind formal runtime snapshots to the configured adapted protocols."""
    configured = config["datasets"]["protocols"]
    runtime = identity.get("datasets") if type(identity) is dict else None
    if type(runtime) is not dict or set(runtime) != set(configured):
        raise ValueError("runtime dataset protocol inventory differs from immutable configuration")
    for protocol, expected in configured.items():
        row = runtime[protocol]
        if type(row) is not dict or set(row) != DATASET_IDENTITY_FIELDS:
            raise ValueError("runtime dataset identity requires exact ID, revision, and snapshot hashes")
        if row["dataset_id"] != expected["dataset_id"]:
            raise ValueError(f"runtime dataset id differs from immutable configuration for {protocol}")
        if row["revision"] != expected["revisions"]["repository"]:
            raise ValueError(f"runtime dataset revision differs from immutable configuration for {protocol}")
        if any(type(row[name]) is not str or not HASH.fullmatch(row[name])
               for name in ("adapter_hash", "split_hash", "raw_hash")):
            raise ValueError("runtime dataset identity requires immutable materialized snapshot hashes")


def validate_compatibility(config):
    arms = load_arm_config(config["arms"])
    ablations = load_ablation_config(config["ablations"])
    patches = ablations.get("patches")
    if not patches or any(type(p) is not dict or len(p) != 1 for p in patches):
        raise ValueError("explicit one-factor patch list required; Cartesian expansion prohibited")
    for patch in patches:
        name, value = next(iter(patch.items()))
        if name not in ABLATION_FACTORS or value not in ABLATION_FACTORS[name]:
            raise ValueError("unregistered one-factor patch")
    if science_digest(ablations) != CANONICAL_SECTION_DIGESTS["ablations"]:
        raise ValueError("exact canonical ablation inventory required")
    protocol = config["protocol"]
    for key, expected in FORMAL_BUDGET.items():
        if type(protocol.get(key)) is not type(expected) or protocol.get(key) != expected:
            raise ValueError("eight genuine actor roots and four matched repair decodes required")
    fixed = {"visible_tests": 4, "context_length": 8192, "dtype": "bfloat16", "particles": 8,
             "latent_dimension": 32, "prefix_tokens": 8, "condition": "sample_once_per_continuation"}
    if any(protocol.get(key) != value for key, value in fixed.items()):
        raise ValueError("fixed formal protocol mismatch")
    stats = config["statistics"]
    if stats.get("seeds") != [1701, 1702, 1703] or stats.get("bootstrap_replicates") != 10000:
        raise ValueError("fixed seeds and 10000 bootstrap replicates required")
    if science_digest(config["gates"]) != CANONICAL_SECTION_DIGESTS["gates"]:
        raise ValueError("exact canonical gates required")
    if config["development"].get("causal_controls") != ablations.get("confirmatory_controls"):
        raise ValueError("all five causal controls must be frozen")
    margin = config["development"].get("brier_margin")
    if type(margin) not in {int, float} or not math.isfinite(margin) or not 0 <= margin <= 2:
        raise ValueError("explicit preregistered absolute Brier margin required")
    known = set(arms["prediction"]) | set(arms["repair"]) | {"locked_baseline_1", "locked_baseline_2"}
    ids = []
    for cell in config["matrix"]:
        ids.append(cell["id"])
        if cell["stage"] == "S1":
            if cell["model"] != "exact_numpy" or cell["dataset"] != "finite":
                raise ValueError("finite model/dataset compatibility mismatch")
            continue
        if cell["model"] not in set(MODEL_PINS) - {"qwen25_1p5b"}:
            raise ValueError("unknown or opt-in-only model in core matrix")
        if cell["dataset"] not in config["datasets"]["protocols"]:
            raise ValueError("unknown or excluded core dataset")
        if not set(cell["arms"]) <= known:
            raise ValueError("unknown/unavailable formal arm")
        if not (type(cell["tasks"]) is int and cell["tasks"] > 0 or cell["tasks"] == "snapshot_inventory"):
            raise ValueError("positive task cap or immutable snapshot inventory required")
    if len(set(ids)) != len(ids) or {c["stage"] for c in config["matrix"]} != {"S0", "S1", "S2", "S3", "S4"}:
        raise ValueError("unique complete S0-S4 matrix required")
    if science_digest(config["matrix"]) != CANONICAL_SECTION_DIGESTS["matrix"]:
        raise ValueError("exact canonical 19-cell matrix required")


def _number(value, name, *, positive=False, integer=False):
    if (type(value) not in ({int} if integer else {int, float}) or not math.isfinite(value)
            or value < 0 or positive and value <= 0):
        raise ValueError(f"site {name} requires a finite {'positive' if positive else 'nonnegative'} value")
    return value


def validate_static(config, profile, site):
    if profile == "local_cpu":
        return
    required = {"partition", "account", "usable_h200_bytes", "gpu_price_per_hour", "cpu_price_per_hour",
                "approved_projection", "container_digest", "cells", "identity", "factory", "factory_sha256",
                "public_root", "private_root", "trust_anchor", "sandbox_spec", "authority_receipt"}
    missing = sorted(required - site.keys())
    if missing:
        raise ValueError("formal preflight: set PBPF_ICLR_SITE to an operator-provisioned YAML; missing "
                         + ", ".join(missing) + ". See docs/EXPERIMENTS.md; no jobs or models started")
    for key in ("partition", "account"):
        if type(site[key]) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]+", site[key]):
            raise ValueError(f"site {key} requires an actual scheduler identifier")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(site["container_digest"])):
        raise ValueError("immutable site container digest required")
    for key in ("usable_h200_bytes", "approved_projection"):
        _number(site[key], key, positive=True)
    for key in ("gpu_price_per_hour", "cpu_price_per_hour"):
        _number(site[key], key)
    cost = 0.
    if set(site["cells"]) != {cell["id"] for cell in config["matrix"]}:
        raise ValueError("site resource inventory must cover every explicit matrix cell exactly")
    for cell in config["matrix"]:
        resource = site["cells"][cell["id"]]
        for key in ("tasks", "hidden_cases", "memory_bytes_bound", "disk_bytes_bound"):
            _number(resource.get(key), f"{cell['id']}.{key}", positive=True, integer=True)
        for key in ("gpu_hours", "cpu_hours"):
            _number(resource.get(key), f"{cell['id']}.{key}")
        if type(cell["tasks"]) is int and resource["tasks"] != cell["tasks"]:
            raise ValueError("site task inventory must equal the preregistered cell count")
        if resource["memory_bytes_bound"] > .9 * site["usable_h200_bytes"]:
            raise ValueError("static memory bound exceeds 90 percent before model loading")
        cost += resource["gpu_hours"] * site["gpu_price_per_hour"] + resource["cpu_hours"] * site["cpu_price_per_hour"]
    if cost > site["approved_projection"]:
        raise ValueError("static projected cost exceeds approved projection before model loading")


def validate_dependencies():
    missing = [name for name in ("torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes",
                                 "evalplus", "pyarrow", "pandas", "scipy") if find_spec(name) is None]
    if missing:
        raise ValueError("install '.[ml,experiment]' on the provisioned host; missing " + ", ".join(missing))


def validate_config(config, *, profile, site, execution=False):
    validate_schema(config, profile)
    validate_pins(config)
    if profile != "local_cpu" and site.get("identity"):
        for name, row in site["identity"].get("models", {}).items():
            if (name not in MODEL_PINS or (row.get("model_id"), row.get("revision")) != MODEL_PINS[name]
                    or (row.get("tokenizer_id"), row.get("tokenizer_revision")) != MODEL_PINS[name]):
                raise ValueError("runtime model identity differs from immutable configuration")
        _validate_runtime_datasets(config, site["identity"])
    validate_compatibility(config)
    validate_static(config, profile, site)
    if profile != "local_cpu":
        # The factory loader performs trust/integration checks before ML discovery.
        from .runner.formal import validate_deployment
        factory = validate_deployment(site)
        factory.preflight(science=config, site={key: value for key, value in site.items()
                           if key not in {"private_root", "trust_anchor", "sandbox_spec", "authority_receipt", "evaluator_site"}})
        if execution:
            validate_dependencies()


def _fingerprint(science, profile, site):
    code = code_identity(ROOT, [p.relative_to(ROOT).as_posix() for parent in (ROOT / "src", ROOT / "scripts")
                               for p in parent.rglob("*") if p.is_file() and p.suffix in {".py", ".sh", ".sbatch"}])
    active = {"mode": "smoke" if profile == "local_cpu" else "formal", "experiment_hash": science_digest(science)}
    if profile == "local_cpu":
        fixture = digest({"fixture": "OfflineExperiment-v1", "source": code["content_hash"]})
        model = dict(model_id="synthetic-offline-no-model", revision=fixture[:40], tokenizer_id="synthetic-token-ids",
                     tokenizer_revision=fixture[:40], tokenizer_hash=digest("no-tokenizer-artifacts"),
                     chat_template_hash=digest("no-chat-model"), weights_hash=digest("no-weights"))
        data = dict(dataset_id="synthetic-offline", revision=fixture[:40], adapter_hash=code["content_hash"],
                    split_hash=digest("synthetic-disjoint-fixture"), raw_hash=fixture)
        identities = dict(models={"synthetic": model}, datasets={"synthetic": data},
            container_digest="sha256:" + digest("local-cpu-no-container-no-formal-claim"),
            locks=dict(checkpoint_hash=digest("synthetic-no-trained-checkpoint"), selection_hash=digest("synthetic-selection"),
                       calibration_hash=digest("synthetic-calibration"), brier_margin=science["development"]["brier_margin"]))
    else:
        _validate_runtime_datasets(science, site["identity"])
        code = {name: digest([value, site["factory_sha256"]]) for name, value in code.items()}
        identities = copy.deepcopy(site["identity"])
        if set(identities) != {"models", "datasets"}:
            raise ValueError("development identity requires immutable base-model and dataset snapshot hashes; learned locks are stage outputs")
        if set(identities["models"]) != set(MODEL_PINS)-{"qwen25_1p5b"}:
            raise ValueError("complete three-model runtime inventory required")
        if set(identities["datasets"]) != set(science["datasets"]["protocols"]):
            raise ValueError("complete four-dataset runtime inventory required")
        for name, row in identities["models"].items():
            if (name not in MODEL_PINS or (row["model_id"], row["revision"]) != MODEL_PINS[name]
                    or (row["tokenizer_id"], row["tokenizer_revision"]) != MODEL_PINS[name]):
                raise ValueError("site runtime model identity differs from configured immutable pin")
        identities["container_digest"] = site["container_digest"]
    kind = RunFingerprint if profile == "local_cpu" else DevelopmentIdentity
    return kind(code=code, prompt={"revision": digest(science["prompts"]), "template_hash": digest(science["prompts"])},
                          config=active, seeds=(1701, 1702, 1703), schema="pbpf-iclr-v1", **identities)


@dataclass(frozen=True)
class ResolvedConfig:
    science: dict
    profile: str
    site: dict
    fingerprint: RunFingerprint

    @property
    def claim_status(self):
        return "smoke-only-no-claim" if self.profile == "local_cpu" else "preregistered-no-results"

    def persist(self, directory):
        directory = Path(directory)
        payload = yaml.safe_dump(self.science, sort_keys=True, allow_unicode=True).encode()
        values = {"resolved.yaml": payload, "resolved.sha256": (hashlib.sha256(payload).hexdigest()+"\n").encode(),
                  "fingerprint.json": canonical_bytes(self.fingerprint.to_dict())}
        for name, data in values.items():
            target = directory / name
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError("immutable resolved config conflict; refusing overwrite/resume")
            else:
                try:
                    atomic_write(target, data, create_once=True)
                except FileExistsError:
                    if target.read_bytes() != data:
                        raise ValueError("immutable resolved config conflict")


def resolve_config(path, profile, *, site=None, execution=False):
    science, site = read_config(path), copy.deepcopy(site or {})
    validate_config(science, profile=profile, site=site, execution=execution)
    return ResolvedConfig(science, profile, site, _fingerprint(science, profile, site))


def matrix_report(resolved):
    if resolved.profile == "local_cpu":
        return [dict(id=f"smoke_{stage.lower()}", stage=stage, model="synthetic", dataset="synthetic", tasks=1,
                     actor_decodes=60 if stage == "S3" else 0, actor_requests_bound=60 if stage == "S3" else 0,
                     input_token_bound=491520 if stage == "S3" else 0,
                     output_token_bound=61440 if stage == "S3" else 0, sandbox_suites=93 if stage == "S3" else 0,
                     disk_bytes_bound=1048576, memory_bytes_bound=268435456, gpu_hours=0, projected_cost=0,
                     protocol_accounting_scope="whole-synthetic-dag-no-retries-work-attributed-to-s3",
                     scope="synthetic-control-flow-only-not-an-empirical-stage") for stage in ("S0", "S1", "S2", "S3", "S4")]
    rows = []
    for cell in resolved.science["matrix"]:
        resource = resolved.site["cells"][cell["id"]]
        tasks, arms = resource["tasks"], cell["arms"]
        repair = cell["phase"] in {"repair", "replication", "integrity"}
        decodes = tasks*3*(8+4*sum(arm != "no_repair" for arm in arms)) if repair else tasks*3*8 if cell["phase"] != "finite" else 0
        # Roulette splits each 1024-token output budget into two model requests.
        # Before selection, either locked comparator may use that implementation.
        partial_arms = sum(arm in {"rollout_roulette", "locked_baseline_1", "locked_baseline_2"} for arm in arms)
        requests = decodes + (tasks*3*4*partial_arms if repair else 0)
        rows.append(dict(cell, **{k: v for k, v in resource.items() if k != "tasks"}, tasks=tasks,
            actor_decodes=decodes, actor_requests_bound=requests,
            input_token_bound=requests*8192, output_token_bound=decodes*1024,
            # Conservatively allow visible plus future-label verification for
            # every full candidate, then one selected hidden suite per arm.
            sandbox_suites=2*decodes+tasks*3*len(arms),
            projected_cost=resource["gpu_hours"]*resolved.site["gpu_price_per_hour"]+resource["cpu_hours"]*resolved.site["cpu_price_per_hour"],
            protocol_accounting_scope="single-three-seed-sweep-excludes-training-grid-search-retries",
            estimate_scope="operator-approved-static-bound-calibration-required"))
    return rows
