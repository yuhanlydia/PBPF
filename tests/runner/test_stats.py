import math

import numpy as np
import pytest

from pbpf.runner.stats import ClusterPlan, PlanLock, formal_interval, reduce_metrics, holm_locked
from pbpf.runner.aggregate import aggregate_stage_b, aggregate_stage_c, infrastructure_valid


def test_common_plan_preserves_clusters_strata_and_has_no_seed_task_resample(tmp_path):
    plan = ClusterPlan.create(task_keys=["t0", "t1", "t2", "t3"],
        source_clusters=["a", "a", "b", "c"], strata=["r", "r", "r", "c"], seed=71, upstream_identity="a"*64)
    assert len(plan.draws) == 10000
    for draw in plan.draws[:40]:
        assert sum(c in {"a", "b"} for c in draw) == 2
        assert draw.count("c") == 1
        indices = plan.task_indices(draw)
        assert indices.count(0) == indices.count(1)
    plan.save(tmp_path / "plan.json")
    loaded = ClusterPlan.load(tmp_path / "plan.json", lock=PlanLock.from_plan(plan))
    assert loaded.digest == plan.digest
    assert loaded.draws == plan.draws
    with pytest.raises(ValueError):
        ClusterPlan.create(task_keys=["t0", "t1"], source_clusters=["a", "a"], strata=["x", "y"], seed=1, upstream_identity="a"*64)


def test_log_ratio_is_ratio_of_task_means_not_mean_log_ratio():
    plan = ClusterPlan.create(task_keys=["t0", "t1"], source_clusters=["a", "a"], strata=["x", "x"], seed=4, upstream_identity="a"*64)
    interval = formal_interval([1., 9.], [2., 9.], plan, statistic="log_nll_ratio")
    assert interval.estimate == pytest.approx(math.log(11/10))
    assert interval.lower == pytest.approx(math.log(11/10))
    assert interval.estimate != pytest.approx(math.log(2)/2)


def test_reduction_averages_tests_then_candidates_then_fixed_seeds():
    records = []
    for seed in (1701, 1702, 1703):
        for candidate, tests in (("c0", [0., 0., 0.]), ("c1", [1.])):
            for test, value in enumerate(tests):
                records.append(dict(task="t", seed=seed, arm="pbpf", candidate=candidate,
                    test=str(test), source_cluster="s", stratum="rbr", nll=value, brier=value))
    reduced = reduce_metrics(records, expected_keys=[(r["task"], r["seed"], r["arm"], r["candidate"], r["test"]) for r in records])
    assert reduced["pbpf"]["t"]["nll"] == .5
    with pytest.raises(ValueError, match="seed"):
        reduce_metrics([r for r in records if r["seed"] != 1703])
    with pytest.raises(ValueError):
        reduce_metrics(records + records[:1])


def passing_metrics():
    return {"pbpf": {"nll": [0.9]*3, "brier": [.11]*3},
        "det": {"nll": [1.]*3, "brier": [.1]*3},
        "shuffled_evidence": {"nll": [.99]*3, "brier": [.1]*3},
        "wrong_candidate": {"nll": [.99]*3, "brier": [.1]*3},
        "random_latent": {"nll": [.99]*3, "brier": [.1]*3}}


def test_gate_b_explicit_one_sided_margin_all_controls_and_bound():
    plan = ClusterPlan.create(task_keys=["a", "b", "c"], source_clusters=["a", "b", "c"], strata=["r"]*3, seed=3, upstream_identity="a"*64)
    args = dict(metrics=passing_metrics(), plan=plan, comparator="det", brier_margin=.011,
        selection_hash="a"*64, prediction_seal_hash="b"*64)
    gate = aggregate_stage_b(**args)
    assert gate["passed"]
    assert gate["relative_gain"] == pytest.approx(.1)
    assert gate["brier_upper"] == pytest.approx(.01)
    assert not aggregate_stage_b(**dict(args, brier_margin=.009))["passed"]
    metrics = passing_metrics()
    metrics["wrong_candidate"]["nll"] = [.91]*3
    assert not aggregate_stage_b(**dict(args, metrics=metrics))["passed"]
    with pytest.raises(TypeError):
        aggregate_stage_b(metrics=metrics, plan=plan, comparator="det", selection_hash="a"*64,
                          prediction_seal_hash="b"*64)


def test_gate_c_requires_both_comparators_selected_pass1_and_bound_stage_b():
    plan = ClusterPlan.create(task_keys=["a", "b", "c"], source_clusters=["a", "b", "c"], strata=["r"]*3, seed=3, upstream_identity="a"*64)
    b = aggregate_stage_b(metrics=passing_metrics(), plan=plan, comparator="det", brier_margin=.011,
        selection_hash="a"*64, prediction_seal_hash="b"*64)
    args = dict(metrics={"pbpf": [.8]*3, "self_debug": [.7]*3, "det": [.76]*3},
        plan=plan, comparator="det", stage_b=b, expected_stage_b_hash=b["decision_hash"],
        endpoint="final_selected_pass1")
    assert aggregate_stage_c(**args)["passed"]
    assert not aggregate_stage_c(**dict(args, metrics={"pbpf": [.8]*3, "self_debug": [.7]*3, "det": [.79]*3}))["passed"]
    with pytest.raises(ValueError):
        aggregate_stage_c(**dict(args, endpoint="initial_hidden_pass8_oracle"))
    with pytest.raises(ValueError):
        aggregate_stage_c(**dict(args, expected_stage_b_hash="c"*64))


def test_holm_monotone_full_locked_family_and_fixed_infra_denominator():
    assert holm_locked({"b": .04, "a": .01, "c": .03}, family=["a", "b", "c"]) == pytest.approx({"a": .03, "b": .06, "c": .06})
    with pytest.raises(ValueError):
        holm_locked({"a": .01}, family=["a", "b"])
    assert infrastructure_valid(expected_ids=list(range(100)), failed_ids=[0])
    assert not infrastructure_valid(expected_ids=list(range(100)), failed_ids=[0, 1])
    with pytest.raises(ValueError):
        infrastructure_valid(expected_ids=list(range(100)), failed_ids=[101])
