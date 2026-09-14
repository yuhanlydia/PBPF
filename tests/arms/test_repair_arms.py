"""Breaks caught: hidden selection, extra decode, token remix, fake named algorithms."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from pbpf.arms.base import ArmCandidate, VisibleEvent
from pbpf.arms.repair import (
    DecodeResult, FormalBank, FrozenSelection, PartialResult, RepairArm,
    SelectionMetric, require_selected_pass1, selection_manifest,
)
from pbpf.arms.registry import REPAIR_ARMS
from test_prediction_arms import arm, candidate, event, task


def bank():
    return FormalBank(tuple(candidate(slot=i) for i in range(8)))


def feedback(c, passes=2):
    return tuple(event(c, i, "PASS" if i < passes else "WRONG_OUTPUT") for i in range(4))


class Actor:
    """Small external-boundary fake: output visibly depends on selected condition."""
    def __init__(self):
        self.calls, self.partial_calls = [], []

    def generate(self, request):
        self.calls.append(request)
        marker = request.condition.particle_index if request.condition else -1
        return DecodeResult(f"def f(x):\n    return x # round {request.round} latent {marker}\n", 25)

    def generate_partial(self, request, *, prefix, token_budget, complete):
        self.partial_calls.append((request, tuple(prefix), token_budget, complete))
        if not complete:
            return PartialResult((f"parent_{request.parent.slot}",), None, 1)
        # Preserve token order, making clone/resample effects observable in source.
        return PartialResult(tuple(prefix) + ("return x",),
                             f"def f(x): return x # {' '.join(prefix)}", 2)


def session(name, *, actor=None, passes=2):
    b = bank()
    belief = arm("full_transcript" if name == "strongest_deterministic" else "pbpf") if name in {"pbpf", "strongest_deterministic"} else None
    lock = FrozenSelection.freeze(belief, source_split_hash="d" * 64) if name == "strongest_deterministic" else None
    return RepairArm(name, task=task(), bank=b, initial_events={c.version_id: feedback(c, passes) for c in b.candidates},
        actor=actor or Actor(), seed=1701, belief=belief if name == "pbpf" else None,
        deterministic_belief=belief if name == "strongest_deterministic" else None,
        selected_deterministic=lock,
        partial_scorer=(lambda prefix: 1000. if prefix == ("parent_3",) else -1000.) if name == "rollout_roulette" else None)


@pytest.mark.parametrize("name", REPAIR_ARMS)
def test_exact_bank_budget_generation_profile_no_early_stop_and_one_selection(name):
    x = session(name, passes=4)
    before = x.initial_hashes
    if name == "no_repair":
        result = x.finish()
        assert x.accounting["actor_decodes"] == 0
        assert result.compute_class == "lower_compute_reference"
    else:
        for _ in range(4):
            child = x.propose()
            x.observe(child.version_id, feedback(child, 4))
        with pytest.raises(ValueError, match="four"):
            x.propose()
        result = x.finish()
        assert x.accounting["actor_decodes"] == 4
        assert x.accounting["decode_multiplicity"] == 1
        assert result.compute_class == "matched_four_decodes"
    assert x.initial_hashes == before == tuple(c.source_hash for c in bank().candidates)
    assert result.initial_selected_source_hash in before
    assert result.final_source_hash in [c.source_hash for c in x.candidates]
    assert result.key == ("PBPF-RBR", "m", "t", 1701, name)
    requests = x.actor.calls + [r for r, _, _, _ in x.actor.partial_calls]
    for request in requests:
        assert request.max_new_tokens == 1024 and request.temperature == .8 and request.top_p == .95
        assert request.num_return_sequences == 1
        assert [e.test_id for e in request.visible_events] == ["v0", "v1", "v2", "v3"]
        assert request.parent.source_hash in [c.source_hash for c in x.candidates]
    assert x.accounting["belief_forwards"] >= 0
    assert "HIDDEN_CANARY" not in json.dumps(asdict(result))


def test_bank_rejects_legacy_mutants_wrong_counts_and_preserves_duplicate_sources():
    original = bank().candidates
    with pytest.raises(ValueError, match="eight"):
        FormalBank(original[:6])
    with pytest.raises(ValueError, match="actor"):
        FormalBank(original[:6] + tuple(replace(c, origin="mutant") for c in original[6:]))
    duplicate_source = tuple(replace(c, source="def f(x): return x") for c in original)
    b = FormalBank(duplicate_source)
    assert len(b.candidates) == 8
    assert len(set(c.source_hash for c in b.candidates)) == 1
    assert len(set(c.version_id for c in b.candidates)) == 8
    with pytest.raises(ValueError, match="slot"):
        FormalBank(original[:7] + (original[0],))


def test_feedback_is_public_ordered_owner_bound_and_finish_requires_observed_repairs():
    x = session("independent")
    child = x.propose()
    with pytest.raises(ValueError, match="feedback"):
        x.propose()
    with pytest.raises(ValueError):
        x.observe(child.version_id, feedback(candidate(slot=1)))
    with pytest.raises(ValueError):
        x.observe(child.version_id, tuple(reversed(feedback(child))))
    with pytest.raises(ValueError):
        x.observe(child.version_id, tuple(replace(e, test_id="HIDDEN_CANARY") for e in feedback(child)))
    with pytest.raises(ValueError):
        x.finish()
    x.observe(child.version_id, feedback(child))
    with pytest.raises(ValueError):
        x.observe(child.version_id, feedback(child))


@pytest.mark.parametrize("name", REPAIR_ARMS)
def test_fixed_rng_replays_sources_parents_transcripts_and_decisions(name):
    a, b = session(name), session(name)
    for x in (a, b):
        for i in range(0 if name == "no_repair" else 4):
            child = x.propose()
            x.observe(child.version_id, feedback(child, i % 4))
    assert a.finish() == b.finish()
    assert a.decisions == b.decisions


def test_independent_and_full_transcript_are_behaviorally_distinct_without_extra_calls():
    independent, debug = session("independent"), session("self_debug")
    for x in (independent, debug):
        for _ in range(4):
            child = x.propose()
            x.observe(child.version_id, feedback(child, 4))
    assert [r.parent.round for r in independent.actor.calls] == [0, 0, 0, 0]
    assert [r.parent.round for r in debug.actor.calls] == [0, 1, 2, 3]
    assert len(debug.actor.calls[-1].transcript) == 4
    assert all(len(r.transcript) == 1 for r in independent.actor.calls)
    assert "explain" in debug.actor.calls[0].instruction.lower()


def test_rex_beta_heuristic_and_failed_parent_attempts_drive_thompson_allocation():
    x = session("rex", passes=2)
    first = x.propose()
    decision = x.decisions[-1]
    assert decision["algorithm"] == "rex_thompson"
    assert len(decision["draws"]) == 8
    assert all(row["alpha"] == 3 and row["beta"] == 3 for row in decision["draws"])
    assert decision["selected_parent"] == max(decision["draws"], key=lambda row: row["draw"])["version_id"]
    chosen = decision["selected_parent"]
    x.observe(first.version_id, feedback(first, 0))
    x.propose()
    changed = next(row for row in x.decisions[-1]["draws"] if row["version_id"] == chosen)
    assert changed["alpha"] == 3 and changed["beta"] == 4 and changed["failed_attempts"] == 1
    assert any(row["version_id"] == first.version_id and row["alpha"] == 1 and row["beta"] == 5 for row in x.decisions[-1]["draws"])


def test_roulette_resamples_ordered_partial_prefixes_before_any_completion():
    x = session("rollout_roulette")
    children = []
    for _ in range(4):
        child = x.propose()
        children.append(child)
        x.observe(child.version_id, feedback(child))
    calls = x.actor.partial_calls
    assert [complete for _, _, _, complete in calls] == [False] * 4 + [True] * 4
    assert [prefix for _, prefix, _, complete in calls if complete] == [("parent_3",)] * 4
    assert all("parent_3" in child.source for child in children)
    decision = x.decisions[0]
    assert decision["ancestors"] == [3, 3, 3, 3]
    assert decision["score_transform"] == "exp(score - max_score)"
    assert x.accounting["partial_actor_calls"] == 4
    assert x.accounting["actor_decodes"] == 4
    assert x.accounting["discarded_prefix_tokens"] == 3


def test_pbpf_samples_exactly_one_particle_and_prefix_for_whole_continuation():
    x = session("pbpf")
    # The external decoder receives one immutable condition, not particles/weights.
    for _ in range(4):
        child = x.propose()
        request = x.actor.calls[-1]
        condition = request.condition
        assert isinstance(condition.latent, tuple) and len(condition.latent) == 2
        posterior = x.belief.snapshot(request.parent.version_id)
        assert list(condition.latent) == posterior["particles"][condition.particle_index]
        assert str(condition.particle_index) in child.source
        assert not hasattr(request, "particles") and not hasattr(request, "log_weights")
        x.observe(child.version_id, feedback(child))
    assert len(x.actor.calls) == 4
    assert x.accounting["belief_forwards"] > 4


def test_roulette_total_sampled_token_cap_includes_discarded_prefixes():
    class UnequalPrefixActor(Actor):
        def generate_partial(self, request, *, prefix, token_budget, complete):
            if not complete:
                return PartialResult((f"parent_{request.parent.slot}",), None,
                                     1 if request.parent.slot == 3 else 512)
            return PartialResult(tuple(prefix) + ("done",), "def f(x): return x", token_budget)
    x = session("rollout_roulette", actor=UnequalPrefixActor())
    for _ in range(4):
        child = x.propose()
        x.observe(child.version_id, feedback(child))
    assert x.accounting["output_tokens"] <= 4096


def test_frozen_dev_selection_rejects_wrong_identity_and_changed_weights():
    x = session("strongest_deterministic")
    assert x.selected_deterministic.name == "full_transcript"
    assert all(not p.requires_grad for p in x.belief.model.parameters())
    with pytest.raises(ValueError, match="development"):
        FrozenSelection.freeze(arm("full_transcript"), source_split_hash="d" * 64, split="test")
    with pytest.raises(ValueError, match="deterministic"):
        FrozenSelection.freeze(arm("pbpf"), source_split_hash="d" * 64)
    next(x.belief.model.parameters()).data.add_(1)
    with pytest.raises(ValueError, match="frozen"):
        x.propose()


def test_selected_pass1_gate_rejects_oracle_and_seals_preserve_arm_mapping_on_duplicate_hash():
    with pytest.raises(ValueError, match="Pass@1"):
        require_selected_pass1(SelectionMetric("initial_hidden_pass8_oracle", .9), phase="final")
    with pytest.raises(ValueError, match="Pass@1"):
        require_selected_pass1(SelectionMetric("initial_selected_pass1", .4), phase="final")
    assert require_selected_pass1(SelectionMetric("final_selected_pass1", .5), phase="final") == .5
    first = session("no_repair").finish()
    second = replace(first, key=(*first.key[:-1], "self_debug"))
    mapping = selection_manifest([first, second])
    assert len(mapping["selections"]) == 2
    assert len(mapping["candidate_hashes"]) == 1
    assert [r["key"][-1] for r in mapping["selections"]] == ["no_repair", "self_debug"]
    with pytest.raises(ValueError, match="duplicate"):
        selection_manifest([first, first])


@pytest.mark.parametrize("field,value", [("split", "test"), ("source_split_hash", ""),
    ("source_split_hash", "z" * 64), ("metric", "hidden_pass1"), ("checkpoint_hash", "bad")])
def test_frozen_selection_rejects_invalid_replacement_or_loaded_fields(field, value):
    x = session("strongest_deterministic")
    with pytest.raises(ValueError, match="selection|frozen|development"):
        replace(x.selected_deterministic, **{field: value})


@pytest.mark.parametrize("mutation", ["kind", "forward", "features"])
def test_frozen_selection_binds_runtime_semantics_beyond_tensor_values(mutation):
    x = session("strongest_deterministic")
    if mutation == "kind":
        x.belief.model.kind = "last"
    elif mutation == "forward":
        x.belief.model.forward = lambda *args: None
    else:
        x.belief.feature_encoder = lambda text: [0., 0., 0.]
    with pytest.raises(ValueError, match="frozen"):
        x.propose()


@pytest.mark.parametrize("name", ["independent", "self_debug", "rex", "strongest_deterministic", "rollout_roulette", "no_repair"])
def test_non_pbpf_repair_labels_reject_generic_belief_injection(name):
    b = bank()
    with pytest.raises(ValueError, match="capability"):
        RepairArm(name, task=task(), bank=b, initial_events={c.version_id: feedback(c) for c in b.candidates},
                  actor=Actor(), seed=1701, belief=arm("pbpf"), partial_scorer=lambda p: 0.)


@pytest.mark.parametrize("name", ["independent", "self_debug", "rex", "rollout_roulette", "no_repair", "pbpf"])
def test_other_labels_reject_deterministic_capability(name):
    b = bank()
    deterministic = arm("full_transcript")
    lock = FrozenSelection.freeze(deterministic, source_split_hash="d" * 64)
    with pytest.raises(ValueError, match="capability"):
        RepairArm(name, task=task(), bank=b, initial_events={c.version_id: feedback(c) for c in b.candidates},
                  actor=Actor(), seed=1701, deterministic_belief=deterministic, selected_deterministic=lock)


def test_frozen_selection_revalidates_metadata_before_decode_after_unsafe_object_mutation():
    x = session("strongest_deterministic")
    object.__setattr__(x.selected_deterministic, "split", "test")
    with pytest.raises(ValueError, match="development"):
        x.propose()
    assert not x.actor.calls
