"""Independent, content-addressed configuration for A-PBPF (not legacy ICLR)."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import yaml

SCHEMA = "apbpf-iclr-v1"
SECTIONS = ("models", "datasets", "protocol", "gates", "ablations", "profiles")
REQUIRED_PATHS = ("dataset_root", "model_cache", "artifact_cache")
ROOT = Path(__file__).resolve().parents[3]


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _mapping(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


@dataclass(frozen=True)
class ResolvedConfig:
    """Store canonical bytes so callers cannot mutate a fingerprinted mapping."""

    payload: bytes
    source: Path
    site: Path | None

    @property
    def config(self):
        return json.loads(self.payload)

    @property
    def fingerprint(self):
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def backend(self):
        return self.config["execution"]["backend"]

    @property
    def profile(self):
        return self.config["execution"]["profile"]


def resolve_config(path, profile="local_24gb", *, site=None):
    """Resolve references relative to their declaring experiment YAML.

    Real commands are explicit argv arrays, never shell snippets. Site command
    changes, scientific config changes, and checked-in source changes all create
    a different identity. Missing provisioning is reported by doctor/run, not
    replaced with the fake backend.
    """
    from .stages import STAGES

    path = Path(path).resolve()
    experiment = _mapping(path)
    if set(experiment) != {"schema", "name", "includes"} or experiment["schema"] != SCHEMA:
        raise ValueError("expected independent apbpf-iclr-v1 experiment schema")
    if not isinstance(experiment["includes"], dict) or set(experiment["includes"]) != set(SECTIONS):
        raise ValueError("experiment must include models/datasets/protocol/gates/ablations/profiles")
    sections = {}
    for name, relative in experiment["includes"].items():
        if not isinstance(relative, str):
            raise ValueError(f"include path must be a string: {name}")
        sections[name] = _mapping(path.parent / relative)
    if profile not in sections["profiles"]:
        raise ValueError(f"unknown A-PBPF profile: {profile}")
    execution = dict(sections["profiles"][profile], profile=profile)
    if execution.get("backend") not in {"real", "fake"}:
        raise ValueError("profile backend must be real or fake")
    if execution["backend"] == "fake" and execution.get("claim_status") != "smoke-only-no-claim":
        raise ValueError("fake backend must be explicitly smoke-only-no-claim")
    if execution["backend"] == "real" and execution.get("claim_status") not in {"prospective-gated", "exploratory-predeclared"}:
        raise ValueError("real backend must use prospective-gated or exploratory-predeclared claim status")
    protocol = sections["protocol"]
    locked = {"seeds": [1701, 1702, 1703], "visible_tests": 4, "future_tests": 6,
              "particles": 8, "bootstrap_draws": 10000,
              "split_unit": "source_problem_component", "bootstrap_unit": "source_problem_component",
              "association_population": "full_locked_population", "fixed_budgets": [1, 2, 4],
              "diagnostic_information": "mutual_information", "repair_component": "sample_once"}
    for key, value in locked.items():
        if protocol.get(key) != value:
            raise ValueError(f"{SCHEMA} requires protocol.{key}={value!r}")
    locked_gates = {
        "baseline_fairness": {"minimum_nll_advantage": 0.02, "clustered_lower_bound_positive": True,
                              "baselines": ["pair_aware_deterministic", "exchangeable_deep_sets", "tuned_dirichlet_rate", "no_particle_bottleneck"]},
        "hard_bank": {"minimum_mixed_pilot_groups": 300, "minimum_confirmatory_groups": 500,
                      "maximum_confirmatory_groups": 1000, "visible_selection_below_hidden_oracle": True,
                      "requires_pre_hidden_lock": True, "requires_source_disjointness": True,
                      "requires_candidate_and_source_inventory": True},
        "association": {"minimum_gap_nats_per_test": 0.03, "clustered_lower_bound_positive": True, "every_confirmatory_domain": True},
        "pair_invariance": {
            "maximum_joint_reversal_fraction_of_shuffle_gap": 0.25,
            "required_controls": ["joint_reversal", "presentation_permutation"],
        },
        "oracle_headroom": {"minimum_nll_advantage_over_fixed_and_random": 0.03},
        "active_testing": {"minimum_test_reduction_at_matched_hidden_quality": 0.25,
                           "minimum_nll_advantage_at_four_tests": 0.03, "decision": "either"},
        "selection": {"minimum_absolute_selected_pass1_advantage": 0.03, "clustered_lower_bound_positive": True,
                      "comparator": "strongest_cross_fitted_deterministic"},
        "repair": {"requires": ["association_gate", "selection_gate"], "role": "supportive_not_primary"},
        "replication": {"minimum_data_protocols": 2, "minimum_model_families": 2, "direction_agreement": True},
    }
    if sections["gates"] != locked_gates:
        raise ValueError("apbpf-iclr-v1 gate thresholds/controls are locked; change the schema for a new protocol")
    for name, model in sections["models"].items():
        if not isinstance(model, dict) or not re.fullmatch(r"[a-f0-9]{40}", str(model.get("revision", ""))):
            raise ValueError(f"model {name} requires an immutable 40-character revision")
    if len({m.get("family") for m in sections["models"].values()}) < 2:
        raise ValueError("replication requires two model families")
    confirmatory = [item for item in sections["datasets"].values() if item.get("role") == "confirmatory"]
    if len(confirmatory) < 2:
        raise ValueError("replication requires two confirmatory data protocols")
    for item in sections["datasets"].values():
        if not re.fullmatch(r"[a-f0-9]{40}", str(item.get("revision", ""))):
            raise ValueError("datasets require immutable revisions")
    site_path = Path(site).resolve() if site else None
    provisioning = _mapping(site_path) if site_path else {"commands": {}}
    allowed_site_fields = {"commands", "worker_revisions", "worker_files",
                           "working_directory", "timeout_seconds", "paths"}
    if set(provisioning) - allowed_site_fields:
        raise ValueError("unknown site fields; expected commands/worker_revisions/worker_files/working_directory/timeout_seconds/paths")
    commands = provisioning.get("commands", {})
    if not isinstance(commands, dict) or set(commands) - set(STAGES):
        raise ValueError("site commands must map known stage names to argv arrays")
    for name, argv in commands.items():
        if argv is None:
            continue  # An explicit, unprovisioned template entry; doctor reports it.
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise ValueError(f"{name}: command must be a nonempty argv string array")
    worker_revisions = provisioning.get("worker_revisions", {})
    if not isinstance(worker_revisions, dict) or set(worker_revisions) - set(STAGES):
        raise ValueError("site worker_revisions must map known stage names to immutable SHA-256 identities")
    for name, revision in worker_revisions.items():
        if revision is None:
            continue
        if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{64}", revision):
            raise ValueError(f"{name}: worker revision must be a lowercase SHA-256 identity or null")
        if not commands.get(name):
            raise ValueError(f"{name}: a worker revision requires a provisioned command")
    worker_files = provisioning.get("worker_files", {})
    if not isinstance(worker_files, dict) or set(worker_files) - set(STAGES):
        raise ValueError("site worker_files must map known stages to local adapter files")
    normalized_worker_files = {}
    for name, value in worker_files.items():
        if value is None:
            normalized_worker_files[name] = None
            continue
        if not isinstance(value, str) or not value or not commands.get(name) or not worker_revisions.get(name):
            raise ValueError(f"{name}: worker file requires a command and immutable worker revision")
        target = Path(value)
        if not target.is_absolute():
            target = (site_path.parent if site_path else ROOT) / target
        normalized_worker_files[name] = str(target.resolve())
    if execution["backend"] == "fake" and site_path:
        raise ValueError("site commands are forbidden with the local smoke backend")
    timeout = provisioning.get("timeout_seconds", 86400)
    if type(timeout) not in {int, float} or not 0 < timeout <= 604800:
        raise ValueError("timeout_seconds must be positive and at most seven days")
    cwd = Path(provisioning.get("working_directory", ROOT))
    if not cwd.is_absolute():
        cwd = (site_path.parent if site_path else ROOT) / cwd
    paths = provisioning.get("paths", {})
    if not isinstance(paths, dict) or set(paths) - set(REQUIRED_PATHS):
        raise ValueError(f"site.paths must contain only {REQUIRED_PATHS}")
    normalized_paths = {}
    for name in REQUIRED_PATHS:
        value = paths.get(name)
        if value is None:
            normalized_paths[name] = None
        elif not isinstance(value, str) or not value:
            raise ValueError(f"site.paths.{name} must be a directory string or null")
        else:
            target = Path(value)
            if not target.is_absolute():
                target = (site_path.parent if site_path else ROOT) / target
            normalized_paths[name] = str(target.resolve())
    sources = {}
    for folder in (ROOT / "src" / "pbpf", ROOT / "scripts"):
        for file in sorted(folder.rglob("*")):
            if file.is_file() and file.suffix in {".py", ".sh", ".sbatch"}:
                sources[file.relative_to(ROOT).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    sources["pyproject.toml"] = hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()
    resolved = {"schema": SCHEMA, "name": experiment["name"], **sections,
                "execution": execution, "source_hashes": sources,
                "site": {"commands": commands, "worker_revisions": worker_revisions,
                         "worker_files": normalized_worker_files,
                         "working_directory": str(cwd.resolve()),
                         "timeout_seconds": timeout, "paths": normalized_paths}}
    return ResolvedConfig(canonical_bytes(resolved), path, site_path)
