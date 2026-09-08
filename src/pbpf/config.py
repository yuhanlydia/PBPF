from __future__ import annotations

import hashlib
import json
import math
import re
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Mapping

import yaml

from .registry import (
    BASELINE_PROVENANCE_MODES,
    DATASETS,
    FORMAL_PROFILES,
    HARDWARE_PROFILES,
    EXPERIMENT_METHODS,
    MODELS,
)

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_FEEDBACK_MODES = {
    "status_only",
    "status_output_diff",
    "exception_trace",
    "full_test_source",
}
_BUDGET_FIELDS = {
    "max_visible_tokens",
    "max_generated_tokens",
    "max_model_calls",
    "max_executions",
    "max_cpu_seconds",
    "max_wall_seconds",
    "max_gpu_hours",
}
_CANDIDATE_BANK_PROTOCOL = {
    "model_samples": 6,
    "deterministic_mutants": 2,
    "temperature": 0.8,
    "top_p": 0.95,
    "mutation_registry_required": True,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def load_experiment(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("experiment config must be a mapping")
    return loaded


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def validate_experiment(
    config: Mapping[str, Any], *, for_execution: bool = False
) -> dict[str, Any]:
    required = {
        "name",
        "profile",
        "formal",
        "model",
        "data",
        "container",
        "protocol",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing experiment fields: {', '.join(missing)}")
    if type(config["formal"]) is not bool:
        raise ValueError("formal must have boolean type")
    execution_mode = config.get("execution_mode")
    if execution_mode is not None:
        if execution_mode != "deterministic_smoke":
            raise ValueError("unknown execution_mode")
        if (
            config["formal"]
            or config.get("name") != "deterministic-smoke"
            or config.get("claim_status") != "smoke-only-no-claim"
        ):
            raise ValueError(
                "deterministic_smoke execution must be non-formal, explicitly named, and no-claim"
            )
    profile = config["profile"]
    if not isinstance(profile, str):
        raise ValueError("profile must be a string")
    if profile not in HARDWARE_PROFILES:
        raise ValueError(f"unknown hardware profile: {profile}")

    model = _require_mapping(config, "model")
    data = _require_mapping(config, "data")
    container = _require_mapping(config, "container")
    protocol = _require_mapping(config, "protocol")
    known_model_ids = {item.model_id for item in MODELS.values()}
    if model.get("id") not in known_model_ids:
        raise ValueError(f"unknown model id: {model.get('id')}")
    if data.get("id") not in DATASETS:
        raise ValueError(f"unknown dataset id: {data.get('id')}")

    if protocol.get("fixed_test_order") is not True:
        raise ValueError("fixed_test_order must be true")
    if protocol.get("early_stop") is not False:
        raise ValueError("early_stop must be false")
    if protocol.get("rounds") != 4:
        raise ValueError("repair protocol requires exactly four rounds")

    formal = config["formal"]
    if formal and profile not in FORMAL_PROFILES:
        raise ValueError("formal profile must be 4x24gb or h200_formal")
    if formal or for_execution:
        for label, revision in (
            ("model revision", model.get("revision")),
            ("data revision", data.get("revision")),
        ):
            if not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision):
                scope = "formal" if formal else "execution"
                raise ValueError(f"{scope} {label} must be a full 40-character commit SHA")
        digest = container.get("digest")
        if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            scope = "formal" if formal else "execution"
            raise ValueError(
                f"{scope} container digest must be sha256 plus 64 hexadecimal digits"
            )

    if protocol.get("candidates") != 8:
        raise ValueError("repair protocol requires a shared G=8 candidate bank")
    hardware = _require_mapping(config, "hardware")
    context = hardware.get("context")
    visible_budget = protocol.get("visible_token_budget")
    if not isinstance(context, int) or context <= 0:
        raise ValueError("hardware context must be a positive integer")
    if visible_budget != context:
        raise ValueError("visible token budget must match the configured context across arms")
    if formal:
        statistics = _require_mapping(config, "statistics")
        seeds = statistics.get("seeds")
        if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError("formal runs require exactly three seeds")
        if statistics.get("bootstrap_replicates") != 10_000:
            raise ValueError("formal runs require 10,000 bootstrap replicates")

    feedback_mode = config.get("feedback_mode")
    if feedback_mode is not None and feedback_mode not in _FEEDBACK_MODES:
        raise ValueError("unknown feedback_mode")
    budget = config.get("budget")
    if for_execution and budget is None:
        raise ValueError("execution requires a complete central budget")
    if budget is not None:
        if not isinstance(budget, Mapping) or set(budget) != _BUDGET_FIELDS:
            raise ValueError("budget must define every central accounting limit exactly once")
        for key, value in budget.items():
            if type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"budget {key} must be a finite non-negative number")
        for key in (
            "max_visible_tokens",
            "max_generated_tokens",
            "max_model_calls",
            "max_executions",
        ):
            if type(budget[key]) is not int:
                raise ValueError(f"budget {key} must be an integer")
        if budget["max_model_calls"] != 50:
            raise ValueError("fixed G=8/four-round protocol requires exactly 50 model calls")
    candidate_bank = config.get("candidate_bank")
    if for_execution and candidate_bank is None:
        raise ValueError("execution requires complete initial candidate provenance")
    if candidate_bank is not None and candidate_bank != _CANDIDATE_BANK_PROTOCOL:
        raise ValueError(
            "candidate_bank must fix six samples, two mutants, temperature=0.8, top_p=0.95, and a mutation registry"
        )
    filtering = config.get("filtering")
    if for_execution and filtering is None:
        raise ValueError("execution requires an ESS resampling threshold")
    if filtering is not None and (
        not isinstance(filtering, Mapping)
        or set(filtering) != {"ess_fraction"}
        or type(filtering["ess_fraction"]) not in {int, float}
        or not 0.0 <= filtering["ess_fraction"] <= 1.0
    ):
        raise ValueError("filtering must define ess_fraction in [0, 1]")

    baseline = config.get("baseline")
    requested_methods: list[str] = []
    if baseline is not None:
        if not isinstance(baseline, Mapping):
            raise ValueError("baseline must be a mapping")
        mode = baseline.get("provenance_mode")
        if mode not in BASELINE_PROVENANCE_MODES:
            raise ValueError(f"invalid baseline provenance_mode: {mode}")
        if mode == "paper_spec_reimplementation" and baseline.get("official") is True:
            raise ValueError("paper-spec reimplementations cannot be marked official")
        baseline_name = baseline.get("name")
        if not isinstance(baseline_name, str):
            raise ValueError("baseline name must be a registered experiment method")
        requested_methods.append(baseline_name)
    arms = config.get("arms", [])
    if not isinstance(arms, list) or not all(isinstance(arm, str) for arm in arms):
        raise ValueError("arms must be a list of registered names")
    requested_methods.extend(arms)
    method = config.get("method")
    if method is not None:
        if not isinstance(method, str):
            raise ValueError("method must be a registered name")
        requested_methods.append(method)
    methods = config.get("methods")
    if methods is not None:
        if not isinstance(methods, list) or not all(isinstance(item, str) for item in methods):
            raise ValueError("methods must be a list of registered names")
        requested_methods.extend(methods)
    unknown_methods = sorted(set(requested_methods) - EXPERIMENT_METHODS.keys())
    if unknown_methods:
        raise ValueError(f"unknown experiment method: {', '.join(unknown_methods)}")
    if for_execution:
        unavailable = [
            name for name in dict.fromkeys(requested_methods)
            if EXPERIMENT_METHODS[name] is not None
        ]
        if unavailable:
            reasons = "; ".join(
                f"{name}: {EXPERIMENT_METHODS[name]}" for name in unavailable
            )
            raise ValueError(f"unavailable experiment method(s): {reasons}")
    return dict(config)


def dependency_status() -> dict[str, bool]:
    """Inspect availability without importing optional ML packages."""
    return {
        name: find_spec(name) is not None
        for name in ("numpy", "yaml", "torch", "transformers", "peft", "bitsandbytes", "datasets")
    }
