"""Formal registry is independent of the legacy diagnostic arm registry."""
from pathlib import Path
import copy
import math
import re
from types import MappingProxyType

import yaml


PREDICTION_ARMS = (
    "prior", "last", "full_transcript", "window_two", "window_four", "deepsets",
    "matched_gru", "scalar_correctness", "map", "posterior_mean", "pbpf",
    "shuffled_evidence", "wrong_candidate", "masked_outcomes", "random_latent", "faulty_shared_belief",
)
REPAIR_ARMS = ("independent", "self_debug", "rex", "strongest_deterministic", "rollout_roulette", "pbpf", "no_repair")
PROVENANCE_MODES = {"controlled", "official_adapter", "paper_spec_reimplementation"}
FORMAL_BUDGET = MappingProxyType({"initial_actor_samples": 8, "initial_origin": "actor_sample",
    "repair_decodes": 4, "decode_multiplicity": 1, "max_new_tokens": 1024,
    "temperature": .8, "top_p": .95, "early_stop": False})
PARAMETER_TOLERANCE = .05
REPAIR_CAPABILITIES = MappingProxyType({name: frozenset(
    {"belief"} if name == "pbpf" else {"deterministic_belief", "selected_deterministic"}
    if name == "strongest_deterministic" else {"partial_scorer"} if name == "rollout_roulette" else ())
    for name in REPAIR_ARMS})
ABLATION_FACTORS = {
    "particles": {1, 2, 4, 8, 16, 32},
    "memory": {"last", "window_two", "window_four", "full_transcript", "deepsets"},
    "conditioning": {"sample_once", "map", "posterior_mean", "token_remix_fault", "full_ensemble"},
    "prefix_tokens": {2, 4, 8, 16}, "injection": {"noop", "random", "text", "soft_prefix", "k", "v", "kv"},
    "proposal": {"uniform", "learned"}, "likelihood": {"uniform", "learned"}, "transition": {"uniform", "learned"},
    "proposal_correction": {"on", "off_biased"}, "resampling": {"never", "every_step", "ess"},
    "rejuvenation": {"mala", "off", "jitter_biased"},
}


def validate_provenance(record, *, upstream_invoker=None):
    if (not isinstance(record, dict) or record.get("mode") not in PROVENANCE_MODES
            or any(not isinstance(record.get(key), str) or not record[key].strip()
                   for key in ("url", "revision", "license"))
            or not record["url"].startswith("https://")):
        raise ValueError("complete provenance mode/url/revision/license is required")
    if record["mode"] != "controlled" and not re.fullmatch(r"[0-9a-f]{40}", record["revision"]):
        raise ValueError("external provenance revision must be an immutable commit")
    if record["mode"] == "official_adapter":
        # This package has no upstream execution adapter. A label never certifies one.
        raise ValueError("official_adapter requires actual audited pinned upstream invocation; no local arm is official")
    return copy.deepcopy(record)


def load_arm_config(config=None):
    if config is None:
        config = Path(__file__).resolve().parents[3] / "configs/iclr/arms.yaml"
    if isinstance(config, (str, Path)):
        config = yaml.safe_load(Path(config).read_text())
    if (not isinstance(config, dict) or config.get("schema") != "pbpf-formal-arms-v1"
            or not config.get("prediction") or not config.get("repair")):
        raise ValueError("formal arm config requires schema, prediction and repair inventories")
    config = copy.deepcopy(config)
    if not set(config.get("prediction", ())) <= set(PREDICTION_ARMS):
        raise ValueError("unknown prediction arm")
    for name, provenance in config.get("repair", {}).items():
        if name not in REPAIR_ARMS:
            raise ValueError("unknown repair arm")
        validate_provenance(provenance)
    budget = config.get("budget")
    if (not isinstance(budget, dict) or set(budget) != set(FORMAL_BUDGET)
            or any(type(budget[key]) is not type(value) or budget[key] != value for key, value in FORMAL_BUDGET.items())):
        raise ValueError("formal budget contradicts the eight-real/four-decode fixed runtime contract")
    tolerance = config.get("parameter_matching", {}).get("maximum_relative_error")
    if (type(tolerance) not in (float, int) or not math.isfinite(tolerance)
            or not 0 < tolerance <= PARAMETER_TOLERANCE):
        raise ValueError("parameter matching tolerance must be positive and at most .05")
    return config


def load_ablation_config(config=None):
    if config is None:
        config = Path(__file__).resolve().parents[3] / "configs/iclr/ablations.yaml"
    if isinstance(config, (str, Path)):
        config = yaml.safe_load(Path(config).read_text())
    if (not isinstance(config, dict) or config.get("schema") != "pbpf-development-ablations-v1"
            or config.get("design") != "one_factor_at_a_time"):
        raise ValueError("ablation config must use the development one-factor registry")
    controls = {"shuffled_evidence", "wrong_candidate", "masked_outcomes", "random_latent", "faulty_shared_belief"}
    if (not config.get("confirmatory_controls") or not set(config["confirmatory_controls"]) <= controls
            or not set(config.get("controls", ())) <= controls):
        raise ValueError("unregistered causal control")
    for name, values in config.get("factors", {}).items():
        if name not in ABLATION_FACTORS or not values or not set(values) <= ABLATION_FACTORS[name]:
            raise ValueError("unregistered development ablation factor or value")
    return copy.deepcopy(config)
