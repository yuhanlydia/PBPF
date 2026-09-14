"""Offline matrix accounting and fail-closed calibration limits."""
import math


def doctor(cells, *, max_new_tokens=1024):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("positive exact-integer token cap required")
    rows = []
    for cell in cells:
        cost = cell.get("projected_cost")
        if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
            raise ValueError("projected cost must be finite nonnegative resource measurement")
        for key in ("tasks", "arms", "visible_cases", "hidden_cases", "seeds"):
            if type(cell[key]) is not int or cell[key] < 1:
                raise ValueError("doctor cell counts must be positive integers")
        task_seeds = cell["tasks"] * cell["seeds"]
        decodes = task_seeds * (8 + 4 * cell["arms"])
        rows.append(dict(cell, actor_decodes=decodes, output_token_bound=decodes*max_new_tokens,
            visible_suites=decodes, visible_cases_bound=decodes*cell["visible_cases"],
            hidden_suites=task_seeds*cell["arms"],
            hidden_cases_bound=task_seeds*cell["arms"]*cell["hidden_cases"],
            disk_code_bytes_bound=decodes*max_new_tokens*16,
            memory_profile=cell.get("memory_profile", "unmeasured-requires-calibration"),
            projected_cost=cell.get("projected_cost", None)))
    return rows


def validate_calibration(*, expected_work, failed_work, projected_cost, measured_cost,
                         allocated_memory, available_memory):
    values = (expected_work, failed_work, projected_cost, measured_cost, allocated_memory, available_memory)
    if type(expected_work) is not int or type(failed_work) is not int or failed_work > expected_work:
        raise ValueError("exact integer expected/failed work inventory required")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("finite nonnegative calibration measurements required")
    if expected_work <= 0 or projected_cost <= 0 or available_memory <= 0:
        raise ValueError("positive calibration denominators required")
    if failed_work / expected_work > .01:
        raise ValueError("infrastructure failures exceed one percent of fixed expected inventory")
    if abs(measured_cost/projected_cost - 1) > .25:
        raise ValueError("projected cost deviation exceeds 25 percent")
    if allocated_memory / available_memory > .9:
        raise ValueError("memory exceeds 90 percent")
    return True
