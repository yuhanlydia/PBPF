import math

import pytest

from pbpf.runner.budget import BudgetLedger, Usage, validate_matched
from pbpf.runner.doctor import doctor, validate_calibration


def test_effective_decodes_native_tokens_failures_and_cached_acceptance(tmp_path):
    ledger = BudgetLedger(tmp_path)
    usage = Usage(actor_requests=1, full_decodes=8, input_token_ids=(11, 22),
        output_token_ids=(33, 44, 55), cpu_seconds=.2, peak_allocated_bytes=128)
    ledger.charge("bank", "attempt-1", usage, accepted=False)
    ledger.charge("bank", "attempt-2", usage, accepted=True)
    ledger.charge("bank", "attempt-2", usage, accepted=True)
    totals = BudgetLedger(tmp_path).totals()
    assert totals["actor_requests"] == 2
    assert totals["full_decodes"] == 16
    assert totals["input_tokens"] == 4
    assert totals["output_tokens"] == 6
    assert totals["peak_allocated_bytes"] == 128
    with pytest.raises(ValueError, match="accepted"):
        ledger.charge("bank", "attempt-3", usage, accepted=True)
    with pytest.raises(ValueError):
        ledger.charge("bank", "attempt-2", Usage(full_decodes=3), accepted=True)


@pytest.mark.parametrize("bad", [-1, True, float("nan"), float("inf")])
def test_budget_rejects_invalid_or_boolean_measurements(bad):
    with pytest.raises(ValueError):
        Usage(cpu_seconds=bad)
    with pytest.raises(ValueError):
        Usage(full_decodes=bad)


def test_matched_contract_counts_sequences_and_order_not_only_requests():
    rows = [dict(bank_hash="a"*64, roots=8, genuine=True, arm="pbpf",
        actor_requests=4, full_decodes=4, multiplicities=[1]*4,
        visible_opportunities=[["v0", "v1"]]*4, max_new_tokens=1024,
        label="matched_four_decodes")]
    assert validate_matched(rows)
    for change in ({"full_decodes": 8}, {"multiplicities": [2]*4}, {"roots": 7},
                   {"arm": "no_repair"}, {"arm": "full_ensemble"}):
        with pytest.raises(ValueError):
            validate_matched([dict(rows[0], **change)])
    with pytest.raises(ValueError):
        validate_matched(rows + [dict(rows[0], visible_opportunities=[["v1", "v0"]]*4)])


def test_reexecution_is_charged_even_after_prior_accepted_attempt(tmp_path):
    ledger = BudgetLedger(tmp_path)
    usage = Usage(actor_requests=1, full_decodes=1)
    ledger.record_attempt("work", usage, accepted=True)
    ledger.record_attempt("work", usage, accepted=True)
    assert ledger.totals()["full_decodes"] == 2
    assert sum(row["accepted"] for row in ledger._rows()) == 1


def test_bank_builder_consumes_eight_actual_root_decode_results(tmp_path):
    from pbpf.runner.repair import build_bank
    from pbpf.arms.repair import DecodeResult
    from pbpf.data.schema import PublicTask, PublicTest
    from pbpf.runner.budget import MeteredActor
    class Actor:
        def generate_root(self, request):
            return DecodeResult(f"print({request.slot})", 2)
        def native_usage(self):
            return Usage(output_token_ids=(1, 2))
    task = PublicTask("task", "fake", "repair", "", (PublicTest("v"),), "test", ("s",))
    bank = build_bank(task, MeteredActor(Actor(), ledger=BudgetLedger(tmp_path), work_prefix="bank"), seed=1701,
        model_id="fake", model_revision="a"*40, prompt_revision="smoke")
    assert [c.source for c in bank.candidates] == [f"print({i})" for i in range(8)]
    class BadActor(Actor):
        def generate_root(self, request):
            return DecodeResult("print(1)", 1025)
    with pytest.raises(ValueError, match="cap"):
        build_bank(task, MeteredActor(BadActor(), ledger=BudgetLedger(tmp_path), work_prefix="bad"),
            seed=1701, model_id="fake", model_revision="a"*40, prompt_revision="smoke")


def test_doctor_cost_and_infrastructure_boundary():
    row = doctor([{"dataset": "rbr", "model": "qwen", "tasks": 10, "arms": 3,
                   "visible_cases": 4, "hidden_cases": 6, "seeds": 3}], max_new_tokens=1024)[0]
    assert row["actor_decodes"] == 600  # 30 task-seeds * (8 shared roots + 3*4)
    assert row["output_token_bound"] == 614400
    assert validate_calibration(expected_work=100, failed_work=1, projected_cost=100,
        measured_cost=125, allocated_memory=90, available_memory=100)
    for change in ({"failed_work": 2}, {"measured_cost": 126}, {"allocated_memory": 91}):
        with pytest.raises(ValueError):
            validate_calibration(**dict(dict(expected_work=100, failed_work=1, projected_cost=100,
                measured_cost=125, allocated_memory=90, available_memory=100), **change))
