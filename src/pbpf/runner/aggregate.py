"""Machine-readable confirmatory gates with bound selection/seal provenance."""
import math

import numpy as np

from .shard import digest
from .stats import formal_interval


def _at_least(value, boundary):
    """Inclusive point gates allow at most 1e-12 absolute roundoff; CIs do not."""
    return value >= boundary or math.isclose(value, boundary, rel_tol=0., abs_tol=1e-12)


def _decision(row):
    def finite_json(value):
        if isinstance(value, dict):
            return {key: finite_json(child) for key, child in value.items()}
        if type(value) is float and math.isinf(value):
            return "+inf" if value > 0 else "-inf"
        return value
    row = finite_json(row)
    return dict(row, decision_hash=digest(row))


def validate_decision(row):
    if row.get("decision_hash") != digest({k: v for k, v in row.items() if k != "decision_hash"}):
        raise ValueError("gate decision checksum mismatch")


def aggregate_stage_b(*, metrics, plan, comparator, brier_margin, selection_hash, prediction_seal_hash):
    if isinstance(brier_margin, bool) or not math.isfinite(brier_margin) or brier_margin < 0:
        raise ValueError("explicit finite development-locked Brier margin required")
    if any(len(value) != 64 for value in (selection_hash, prediction_seal_hash)) or comparator == "pbpf":
        raise ValueError("frozen comparator and prediction seal hashes required")
    pbpf, baseline = metrics["pbpf"], metrics[comparator]
    interval = formal_interval(pbpf["nll"], baseline["nll"], plan, statistic="log_nll_ratio")
    brier = formal_interval(pbpf["brier"], baseline["brier"], plan, statistic="difference", one_sided=True)
    gain = float(np.mean(baseline["nll"]) - np.mean(pbpf["nll"]))
    baseline_mean = float(np.mean(baseline["nll"]))
    if baseline_mean <= 0:
        raise ValueError("relative NLL improvement requires positive comparator aggregate mean")
    relative = gain / baseline_mean
    removed = {}
    for control in ("shuffled_evidence", "wrong_candidate", "random_latent"):
        values = np.asarray(metrics[control]["nll"], dtype=float)
        if values.shape != (len(plan.task_keys),) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("paired causal-control inventory required")
        removed[control] = float((values.mean()-np.mean(pbpf["nll"])) / gain) if gain > 0 else 0.
    return _decision({"gate": "B", "passed": bool(_at_least(relative, .05) and interval.lower > 0
        and brier.upper < brier_margin and all(_at_least(value, .8) for value in removed.values())),
        "relative_gain": relative, "log_ratio_lower": interval.lower, "log_ratio_upper": interval.upper,
        "brier_upper": brier.upper, "brier_margin": brier_margin, "removed_gain": removed,
        "comparator": comparator, "selection_hash": selection_hash, "prediction_seal_hash": prediction_seal_hash,
        "plan_hash": plan.digest, "endpoint": "task_macro_prefix4_future_nll"})


def aggregate_stage_c(*, metrics, plan, comparator, stage_b, expected_stage_b_hash, endpoint):
    validate_decision(stage_b)
    if (stage_b["decision_hash"] != expected_stage_b_hash or stage_b["gate"] != "B"
            or stage_b["comparator"] != comparator):
        raise ValueError("retained Stage-B provenance mismatch")
    if endpoint != "final_selected_pass1":
        raise ValueError("final selected Pass@1 required; oracle Pass@8 is prohibited")
    comparisons = {}
    for name in ("self_debug", comparator):
        for arm in ("pbpf", name):
            values = np.asarray(metrics[arm], dtype=float)
            if np.any(values < 0) or np.any(values > 1):
                raise ValueError("Pass@1 must lie within [0,1]")
        result = formal_interval(metrics["pbpf"], metrics[name], plan, statistic="difference")
        comparisons[name] = {"delta": result.estimate, "lower": result.lower, "upper": result.upper,
            "passed": _at_least(result.estimate, .03) and result.lower > 0}
    return _decision({"gate": "C", "passed": bool(stage_b["passed"] and all(c["passed"] for c in comparisons.values())),
        "comparisons": comparisons, "stage_b_hash": expected_stage_b_hash, "plan_hash": plan.digest, "endpoint": endpoint})


def infrastructure_valid(*, expected_ids, failed_ids):
    expected, failed = list(expected_ids), list(failed_ids)
    if (not expected or len(expected) != len(set(expected)) or len(failed) != len(set(failed))
            or not set(failed) <= set(expected)):
        raise ValueError("invalid fixed infrastructure work inventory")
    return len(failed) / len(expected) <= .01


def require_main_tables(artifacts):
    rows = list(artifacts)
    if not rows or any(row.get("confirmatory") is not True or row.get("failed_gate_hash") for row in rows):
        raise ValueError("nonconfirmatory/forced artifacts cannot populate main tables")
    return True
