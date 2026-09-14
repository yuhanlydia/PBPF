"""Breaks caught: evidence leaks, collapsed mixtures, dead/miscounted parameters."""
from dataclasses import replace
import json
import hashlib
import random
import secrets

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from pbpf.data.schema import PublicTask, PublicTest
from pbpf.registry import OUTCOMES
from pbpf.belief.model import NeuralBeliefModel
from pbpf.arms.base import ArmCandidate, VisibleEvent, source_hash
from pbpf.arms.prediction import (
    MemoryNetwork, make_belief_arm, learned_parameter_report, resolve_memory_width,
)
from pbpf.arms.registry import PREDICTION_ARMS, load_arm_config


def task(name="t"):
    return PublicTask(name, "PBPF-RBR", "return the sum", "", tuple(
        PublicTest(f"v{i}", f"assert f({i}) == {i}") for i in range(4)), "dev-select", (name,))


def candidate(t="t", slot=0, **kwargs):
    return ArmCandidate(t, slot, 0, f"def f(x): return x + {slot}\n", "actor_sample",
                        "m", "a" * 40, "prompt-v1", **kwargs)


def event(c, i, outcome="PASS", feedback=""):
    return VisibleEvent(c.task_id, c.version_id, f"e-{c.slot}-{i}", f"v{i}", i, outcome, feedback)


def features(text):
    # Real semantic-content fixture, never candidate/test identity hashes.
    return torch.tensor([len(text) / 100, text.count("return") / 10, text.count("+") / 10])


def model():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        return NeuralBeliefModel(3, latent_dim=2, hidden_dim=16)


def arm(name):
    return make_belief_arm(name, model=model(), feature_encoder=features, seed=1701, particles=8)


@pytest.mark.parametrize("name", PREDICTION_ARMS)
def test_uniform_interface_is_five_way_owner_isolated_alias_safe_and_replayable(name):
    a, b = candidate(), candidate(slot=1)
    x, y = arm(name), arm(name)
    for instance in (x, y):
        instance.initialize(task(), a)
        instance.initialize(task(), b)
        instance.observe(b.version_id, event(b, 0, "WRONG_OUTPUT"))
        before = instance.snapshot(b.version_id)
        for i, outcome in enumerate(OUTCOMES[:4]):
            instance.observe(a.version_id, event(a, i, outcome))
        if name != "faulty_shared_belief":
            assert instance.snapshot(b.version_id) == before
        prediction = instance.predict(a.version_id, PublicTest("future", "assert f(10)==10"))
        assert tuple(prediction) == OUTCOMES
        assert sum(prediction.values()) == pytest.approx(1)
        assert all(np.isfinite(p) and p >= 0 for p in prediction.values())
        snapshot = instance.snapshot(a.version_id)
        assert snapshot["event_ids"] == [f"e-0-{i}" for i in range(4)]
        snapshot["event_ids"].append("mutated")
        assert "mutated" not in instance.snapshot(a.version_id)["event_ids"]
        condition = instance.condition(a.version_id)
        assert condition.version_id == a.version_id
        assert instance.accounting["actor_decodes"] == 0
        assert instance.accounting["belief_forwards"] > 0
    assert x.snapshot(a.version_id) == y.snapshot(a.version_id)


@pytest.mark.parametrize("name", PREDICTION_ARMS)
def test_invalid_events_and_cross_task_children_fail_closed(name):
    a, b = candidate(), candidate("other")
    x = arm(name)
    x.initialize(task(), a)
    for bad in [event(b, 0), replace(event(a, 0), test_id="HIDDEN_CANARY"), event(a, 1)]:
        with pytest.raises(ValueError):
            x.observe(a.version_id, bad)
    x.observe(a.version_id, event(a, 0))
    with pytest.raises(ValueError):
        x.observe(a.version_id, event(a, 0))
    child = replace(b, round=1, parent_version_id=a.version_id, origin="repair")
    with pytest.raises(ValueError, match="task"):
        x.spawn_child(a.version_id, child, event(child, 0))
    same_task_child = replace(a, round=1, parent_version_id=a.version_id, origin="repair")
    with pytest.raises(ValueError, match="spawn_child"):
        x.initialize(task(), same_task_child)


def test_source_identity_excludes_telemetry_but_version_records_lineage_and_provenance():
    a = candidate()
    b = replace(a, wall_seconds=123, output_tokens=56)
    assert a.source_hash == b.source_hash == source_hash(a.source)
    assert a.version_id == b.version_id
    assert replace(a, slot=1).source_hash == a.source_hash
    assert replace(a, slot=1).version_id != a.version_id
    assert replace(a, prompt_revision="p2").version_id != a.version_id
    assert replace(a, source=a.source + "\n").source_hash != a.source_hash


@pytest.mark.parametrize("kind", ["prior", "last", "full_transcript", "window_two", "window_four", "matched_gru", "deepsets", "scalar_correctness"])
def test_width_solver_counts_all_unique_learned_parameters_and_every_component_affects_prediction(kind):
    reference = model()
    target = sum(p.numel() for p in reference.parameters())
    report = resolve_memory_width(kind, 3, 2, target)
    net = MemoryNetwork(kind, 3, 2, report["width"])
    assert learned_parameter_report(net)["total"] == report["total"]
    assert abs(report["total"] / target - 1) <= .05
    assert report["total"] == sum(report["components"].values())
    task_feature = torch.tensor([.3, .2, .1])
    evidence = [(torch.tensor([.4, .1, .2]), i % 5) for i in range(5)]
    baseline = net(task_feature, task_feature, evidence, task_feature)
    (-baseline[0]).backward()
    for component, parameter in net.named_parameters():
        assert parameter.grad is not None, component
        assert torch.isfinite(parameter.grad).all(), component
        assert parameter.grad.abs().sum() > 0, component
        # A real perturbation in an active coordinate must change prediction,
        # rather than merely attaching a formally non-None gradient tensor.
        with torch.no_grad():
            flat = parameter.reshape(-1)
            index = int(parameter.grad.abs().reshape(-1).argmax())
            saved = flat[index].clone()
            flat[index] += .1
            changed = net(task_feature, task_feature, evidence, task_feature)
            flat[index] = saved
        assert not torch.allclose(baseline, changed, atol=1e-9, rtol=1e-9), component
    for p in net.parameters():
        p.requires_grad_(False)
    assert learned_parameter_report(net)["total"] == report["total"]
    choices = [(abs(MemoryNetwork.parameter_count(kind, 3, 2, h) / target - 1), h) for h in range(1, report["width"] + 10)]
    assert min(choices)[1] == report["width"]
    with pytest.raises(ValueError, match="five percent"):
        resolve_memory_width(kind, 3, 2, 1)


def test_parameter_count_deduplicates_tied_weights_not_buffers():
    net = torch.nn.Module()
    net.a = torch.nn.Linear(3, 2)
    net.b = net.a
    net.register_buffer("particles", torch.ones(100))
    assert learned_parameter_report(net)["total"] == 8


def test_memory_semantics_orderless_prior_last_and_windows():
    evidence = [(torch.tensor([float(i), .1, .2]), i % 5) for i in range(6)]
    context = torch.tensor([.2, .4, .6])
    for kind in ("deepsets", "full_transcript", "last", "window_two", "window_four", "prior", "scalar_correctness"):
        torch.manual_seed(7)
        net = MemoryNetwork(kind, 3, 2, 12)
        p = net(context, context, evidence, context)
        reverse = net(context, context, list(reversed(evidence)), context)
        if kind in ("deepsets", "prior", "scalar_correctness"):
            assert torch.allclose(p, reverse, atol=1e-7)
        else:
            assert not torch.allclose(p, reverse)
        window = {"last": 1, "window_two": 2, "window_four": 4}.get(kind)
        if window:
            assert torch.equal(p, net(context, context, evidence[-window:], context))
        if kind == "prior":
            assert torch.equal(p, net(context, context, [], context))


def test_controls_transform_only_already_seen_evidence_and_mask_every_outcome_channel():
    c, donor = candidate(), candidate(slot=1)
    for name in ("shuffled_evidence", "wrong_candidate", "masked_outcomes"):
        x = arm(name)
        x.initialize(task(), c)
        x.initialize(task(), donor)
        for i in range(4):
            x.observe(donor.version_id, event(donor, i, "TIMEOUT", "DONOR_FAILURE"))
            x.observe(c.version_id, event(c, i, OUTCOMES[i], "OUTCOME_TEXT_CANARY"))
        transformed = x.evidence(c.version_id)
        assert [row["test_id"] for row in transformed] == [f"v{i}" for i in range(4)]
        if name == "shuffled_evidence":
            assert sorted(row["outcome"] for row in transformed) == sorted(OUTCOMES[:4])
            assert [row["outcome"] for row in transformed] != list(OUTCOMES[:4])
            assert x.snapshot(c.version_id)["control_seed"] is not None
        elif name == "wrong_candidate":
            assert all(row["outcome"] == "TIMEOUT" for row in transformed)
            assert x.snapshot(c.version_id)["donor_version_id"] == donor.version_id
        else:
            assert all(row["outcome"] is None and row["feedback"] == "" for row in transformed)
            assert "OUTCOME_TEXT_CANARY" not in json.dumps(x.snapshot(c.version_id))
            assert "OUTCOME_TEXT_CANARY" not in x.condition(c.version_id).text


def test_particle_prediction_uses_exp_of_log_probs_and_mixture_not_mean_latent():
    x = arm("pbpf")
    c = candidate()
    x.initialize(task(), c)
    for i in range(4):
        x.observe(c.version_id, event(c, i, OUTCOMES[i]))
    state = x.snapshot(c.version_id)
    z = torch.tensor(state["particles"], dtype=torch.float32)
    weights = torch.tensor(state["log_weights"], dtype=torch.float32).exp()
    t, code, future = features(task().task_text), features(c.source), features("assert f(10)==10")
    with torch.no_grad():
        expected = (weights[:, None] * x.model.likelihood(z, t.expand(8, -1), code.expand(8, -1), future.expand(8, -1)).exp()).sum(0)
    observed = x.predict(c.version_id, PublicTest("future", "assert f(10)==10"))
    np.testing.assert_allclose(list(observed.values()), expected.numpy(), atol=1e-6)


def test_random_latent_radius_is_weighted_rms_and_candidate_children_are_isolated():
    x = arm("random_latent")
    a, b = candidate(), candidate(slot=1)
    x.initialize(task(), a)
    x.initialize(task(), b)
    for i in range(4):
        x.observe(a.version_id, event(a, i, OUTCOMES[i]))
    state = x.snapshot(a.version_id)
    condition = x.condition(a.version_id)
    expected = np.sqrt(np.sum(np.exp(state["log_weights"]) * np.square(state["particles"]).sum(1)))
    assert np.linalg.norm(condition.latent) == pytest.approx(expected)
    before_parent, before_b = x.snapshot(a.version_id), x.snapshot(b.version_id)
    children = [replace(a, round=1, source=f"def f(x): return x + {i + 10}", parent_version_id=a.version_id, origin="repair") for i in range(2)]
    for child in children:
        x.spawn_child(a.version_id, child, event(child, 0, "WRONG_OUTPUT"))
    before_child = x.snapshot(children[1].version_id)
    x.observe(children[0].version_id, event(children[0], 1, "TIMEOUT"))
    assert x.snapshot(children[1].version_id) == before_child
    assert x.snapshot(a.version_id) == before_parent
    assert x.snapshot(b.version_id) == before_b


def test_config_registry_rejects_missing_or_false_official_provenance():
    config = load_arm_config()
    assert set(config["prediction"]) == set(PREDICTION_ARMS)
    bad = dict(config)
    bad["repair"] = {"rex": {"mode": "official_adapter", "url": "https://github.com/haotang1995/REx", "revision": "a" * 40, "license": "MIT"}}
    with pytest.raises(ValueError, match="official"):
        load_arm_config(bad)
    bad["repair"] = {"rex": {"mode": "paper_spec_reimplementation"}}
    with pytest.raises(ValueError, match="provenance"):
        load_arm_config(bad)


def test_descendant_evidence_keeps_full_parent_cumulative_mass_for_three_generations():
    x = arm("pbpf")
    c = candidate()
    x.initialize(task(), c)
    for i in range(4):
        x.observe(c.version_id, event(c, i, OUTCOMES[i]))
    for generation in range(1, 4):
        parent = x.snapshot(c.version_id)
        child = replace(c, round=generation, source=f"def f(x): return x + {generation + 30}",
                        origin="repair", parent_version_id=c.version_id)
        x.spawn_child(c.version_id, child, event(child, 0, "TIMEOUT"))
        state = x.snapshot(child.version_id)
        local = sum(step["log_normalizer"] for step in state["filter_steps"])
        assert state["log_evidence"] == pytest.approx(parent["log_evidence"] + local)
        assert state["parent_hash"] == c.version_id
        c = child


def test_systematic_condition_blocks_resume_without_repeats_or_lost_choices():
    x = arm("pbpf")
    c = candidate()
    x.initialize(task(), c)  # prior samples have exactly uniform weights
    first = [x.condition(c.version_id) for _ in range(2)]
    before = x.snapshot(c.version_id)
    assert before["condition_cursor"] == 2
    assert len(before["generation_records"]) == 2
    encoded = x.serialize()
    y = arm("pbpf")
    y.restore(encoded)
    tail = [x.condition(c.version_id) for _ in range(6)]
    assert [y.condition(c.version_id) for _ in range(6)] == tail
    choices = [condition.particle_index for condition in first + tail]
    assert len(set(choices[:4])) == 4
    assert len(set(choices[4:])) == 4
    assert x.serialize() == y.serialize()
    assert x.snapshot(c.version_id) == y.snapshot(c.version_id)
    previous_fingerprint = x.snapshot(c.version_id)["condition_schedule"]["posterior_fingerprint"]
    x.observe(c.version_id, event(c, 0, "TIMEOUT"))
    x.condition(c.version_id)
    assert x.snapshot(c.version_id)["condition_schedule"]["posterior_fingerprint"] != previous_fingerprint
    assert x.snapshot(c.version_id)["condition_schedule"]["cursor"] == 1


def test_snapshot_preserves_filter_ancestry_resampling_and_entropy_diagnostics():
    x = arm("pbpf")
    c = candidate()
    x.initialize(task(), c)
    for i in range(4):
        x.observe(c.version_id, event(c, i, OUTCOMES[i]))
    state = x.snapshot(c.version_id)
    assert len(state["ancestors"]) == 8
    assert state["step"] == 4
    for step in state["filter_steps"]:
        assert len(step["resampling_indices"]) == 8
        assert 0 <= step["normalized_entropy"] <= 1
        assert 1 <= step["unique_ancestor_count"] <= 8


def test_public_structured_inputs_are_canonical_immutable_and_resume_safe():
    sources = [{"args": [i], "metadata": {"hint": "return +"}} for i in range(4)]
    public = replace(task(), visible_tests=tuple(PublicTest(f"v{i}", source) for i, source in enumerate(sources)))
    a, b = arm("full_transcript"), arm("full_transcript")
    c = candidate()
    a.initialize(public, c)
    b.initialize(public, c)
    for i in range(4):
        a.observe(c.version_id, event(c, i, OUTCOMES[i]))
        b.observe(c.version_id, event(c, i, OUTCOMES[i]))
    future = PublicTest("future", "assert f(7) == 7")
    expected = a.predict(c.version_id, future)
    sources[0]["args"].append("LEAKED_MUTATION" * 100)
    sources[3]["metadata"]["hint"] = "changed" * 100
    assert a.predict(c.version_id, future) == expected
    assert b.predict(c.version_id, future) == expected
    assert "LEAKED_MUTATION" not in a.serialize()
    for invalid in (object(), {1: "non-string-key"}, {"bad": float("nan")}, {"bad": {1, 2}}):
        invalid_task = replace(task(), visible_tests=(PublicTest("v0", invalid),))
        with pytest.raises((ValueError, TypeError), match="public|JSON|finite"):
            arm("prior").initialize(invalid_task, c)


@pytest.mark.parametrize("field,value", [("initial_actor_samples", 6), ("repair_decodes", 3),
    ("decode_multiplicity", 8), ("early_stop", True), ("max_new_tokens", 512),
    ("temperature", .5), ("top_p", .9)])
def test_formal_config_rejects_every_contradictory_budget(field, value):
    config = load_arm_config()
    config["budget"][field] = value
    with pytest.raises(ValueError, match="budget"):
        load_arm_config(config)


def test_formal_config_rejects_empty_configs_invalid_matching_and_ablation_mislabelling():
    with pytest.raises(ValueError, match="config"):
        load_arm_config({})
    for value in (.06, 0, float("nan")):
        config = load_arm_config()
        config["parameter_matching"]["maximum_relative_error"] = value
        with pytest.raises(ValueError, match="parameter"):
            load_arm_config(config)
    from pbpf.arms.registry import load_ablation_config
    config = load_ablation_config()
    assert config["design"] == "one_factor_at_a_time"
    config["confirmatory_controls"].append("shared_belief")
    with pytest.raises(ValueError, match="control"):
        load_ablation_config(config)


def test_runtime_honors_stricter_resolved_parameter_matching_tolerance():
    config = load_arm_config()
    config["parameter_matching"]["maximum_relative_error"] = .00001
    with pytest.raises(ValueError, match="parameter|percent|tolerance"):
        make_belief_arm("matched_gru", model=model(), feature_encoder=features,
                        seed=1701, arm_config=config)


def test_checkpoint_restore_then_child_and_future_observation_matches_uninterrupted():
    x = arm("pbpf")
    c = candidate()
    x.initialize(task(), c)
    for i in range(2):
        x.observe(c.version_id, event(c, i, OUTCOMES[i]))
    x.condition(c.version_id)
    encoded = x.serialize()
    y = arm("pbpf")
    y.restore(encoded)
    child = replace(c, round=1, origin="repair", source="def f(x): return x * 2", parent_version_id=c.version_id)
    for instance in (x, y):
        for i in range(2, 4):
            instance.observe(c.version_id, event(c, i, OUTCOMES[i]))
        instance.spawn_child(c.version_id, child, event(child, 0, "TIMEOUT"))
        instance.observe(child.version_id, event(child, 1, "PASS"))
        instance.condition(child.version_id)
    assert x.serialize() == y.serialize()
    changed = arm("pbpf")
    next(changed.model.parameters()).data.add_(.01)
    with pytest.raises(ValueError, match="identity"):
        changed.restore(encoded)
    payload = json.loads(encoded)
    payload["payload"]["condition_calls"][c.version_id] += 1
    with pytest.raises(ValueError, match="checksum"):
        y.restore(json.dumps(payload))


class ConfiguredEncoder:
    def __init__(self, scale=1.):
        self.scale = scale

    def semantic_identity(self):
        return {"implementation": "tests.ConfiguredEncoder", "revision": "v1", "config": {"scale": self.scale}}

    def __call__(self, text):
        return features(text) * self.scale

    def encode(self, text):
        return self(text)


class NeuralEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(3))

    def semantic_identity(self):
        return {"implementation": "tests.NeuralEncoder", "revision": "v1", "config": {}}

    def forward(self, text):
        return features(text) * self.scale


@pytest.mark.parametrize("kind", ["object", "bound", "neural"])
def test_semantic_encoder_state_is_bound_for_freeze_predict_and_restore(kind):
    from pbpf.arms.repair import FrozenSelection
    encoder = NeuralEncoder().eval() if kind == "neural" else ConfiguredEncoder()
    callback = encoder.encode if kind == "bound" else encoder
    x = make_belief_arm("full_transcript", model=model(), feature_encoder=callback)
    lock = FrozenSelection.freeze(x, source_split_hash="d" * 64)
    c = candidate()
    x.initialize(task(), c)
    checkpoint = x.serialize()
    if kind == "neural":
        encoder.scale.data.add_(1)
    else:
        encoder.scale = 2.
    with pytest.raises(ValueError, match="frozen|encoder"):
        lock.validate(x)
    with pytest.raises(ValueError, match="encoder"):
        x.predict(c.version_id, PublicTest("future", "assert f(7) == 7"))
    with pytest.raises(ValueError, match="identity|encoder"):
        x.restore(checkpoint)


def test_unsupported_or_incompletely_declared_mutable_encoder_state_fails_closed():
    class Undeclared:
        def __init__(self):
            self.scale = 2.
        def __call__(self, text):
            return features(text) * self.scale
    class Incomplete(ConfiguredEncoder):
        def semantic_identity(self):
            return {"implementation": "tests.Incomplete", "revision": "v1", "config": {}}
    for encoder in (Undeclared(), Incomplete()):
        with pytest.raises(ValueError, match="encoder"):
            make_belief_arm("prior", model=model(), feature_encoder=encoder)


def _rechecksum(payload):
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return json.dumps({"payload": payload, "sha256": hashlib.sha256(data.encode()).hexdigest()})


@pytest.mark.parametrize("attack", ["wrong_owner", "hidden_test", "extra_event", "extra_payload",
    "missing_inventory", "invalid_outcome", "duplicate_event", "foreign_donor", "false_parent",
    "negative_cursor", "schedule_index", "schedule_fingerprint", "extra_generation", "invalid_accounting"])
def test_checksum_consistent_malicious_restore_is_rejected_atomically_without_canary_leak(attack):
    x = arm("pbpf")
    c = candidate()
    x.initialize(task(), c)
    for i in range(2):
        x.observe(c.version_id, event(c, i, OUTCOMES[i]))
    x.condition(c.version_id)
    original = x.serialize()
    payload = json.loads(original)["payload"]
    key = c.version_id
    if attack == "wrong_owner":
        payload["events"][key][0]["version_id"] = "f" * 64
    elif attack == "hidden_test":
        payload["events"][key][0]["test_id"] = "HIDDEN_CANARY"
    elif attack == "extra_event":
        payload["events"][key][0]["hidden_outcomes"] = "HIDDEN_CANARY"
    elif attack == "extra_payload":
        payload["hidden_outcomes"] = "HIDDEN_CANARY"
    elif attack == "missing_inventory":
        payload["events"].pop(key)
    elif attack == "invalid_outcome":
        payload["events"][key][0]["outcome"] = "INFRASTRUCTURE_FAILURE"
    elif attack == "duplicate_event":
        payload["events"][key][1]["event_id"] = payload["events"][key][0]["event_id"]
    elif attack == "foreign_donor":
        payload["donors"][key] = "e" * 64
    elif attack == "false_parent":
        payload["parents"][key] = [payload["states"][key], payload["candidates"][key]]
    elif attack == "negative_cursor":
        payload["condition_calls"][key] = -1
    elif attack == "schedule_index":
        payload["condition_schedules"][key]["indices"][0] = 99
    elif attack == "schedule_fingerprint":
        payload["condition_schedules"][key]["posterior_fingerprint"] = "b" * 64
    elif attack == "extra_generation":
        payload["generation_records"][key][0]["hidden_outcomes"] = "HIDDEN_CANARY"
    else:
        payload["accounting"]["actor_decodes"] = 3
    with pytest.raises((ValueError, TypeError), match="restore|checkpoint|schema|owner|visible|event|schedule|donor|parent|cursor|account"):
        x.restore(_rechecksum(payload))
    assert x.serialize() == original
    assert "HIDDEN_CANARY" not in x.condition(key).text


@pytest.mark.parametrize("name", PREDICTION_ARMS)
def test_valid_restore_replays_all_controls_without_future_donor_or_outcome_text_leak(name):
    x, y = arm(name), arm(name)
    a, b = candidate(), candidate(slot=1)
    x.initialize(task(), a)
    x.initialize(task(), b)
    for i in range(2):
        x.observe(b.version_id, event(b, i, "TIMEOUT", "visible diagnostic"))
        x.observe(a.version_id, event(a, i, "PASS", "VISIBLE_OUTCOME_TEXT"))
    x.condition(a.version_id)
    x.predict(a.version_id, PublicTest("future", "f(7)"))
    encoded = x.serialize()
    if name == "masked_outcomes":
        assert "VISIBLE_OUTCOME_TEXT" not in encoded
    y.restore(encoded)
    assert y.serialize() == encoded
    assert y.condition(a.version_id) == x.condition(a.version_id)


def test_encoder_checkpoint_binds_nested_neural_module_configuration():
    class ScaledLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.factor = 1.
        def forward(self, value):
            return value * self.factor
    class Nested(NeuralEncoder):
        def __init__(self):
            super().__init__()
            self.layer = ScaledLayer()
        def forward(self, text):
            return self.layer(features(text) * self.scale)
    encoder = Nested().eval()
    x = make_belief_arm("prior", model=model(), feature_encoder=encoder)
    c = candidate()
    x.initialize(task(), c)
    encoder.layer.factor = 2.
    with pytest.raises(ValueError, match="encoder"):
        x.predict(c.version_id, PublicTest("future", "f(7)"))


def test_encoder_rejects_uncheckpointed_mutable_neural_forward_hooks():
    encoder = NeuralEncoder().eval()
    x = make_belief_arm("prior", model=model(), feature_encoder=encoder)
    c = candidate()
    x.initialize(task(), c)
    encoder.register_forward_hook(lambda module, inputs, output: output * 2)
    with pytest.raises(ValueError, match="encoder"):
        x.predict(c.version_id, PublicTest("future", "f(7)"))


def test_keyword_only_encoder_defaults_are_bound_and_mutable_defaults_rejected():
    from pbpf.arms.repair import FrozenSelection
    def kw_encoder(text, *, scale=1.):
        return features(text) * scale
    x = make_belief_arm("prior", model=model(), feature_encoder=kw_encoder)
    lock = FrozenSelection.freeze(x, source_split_hash="d" * 64)
    c = candidate()
    x.initialize(task(), c)
    encoded = x.serialize()
    kw_encoder.__kwdefaults__["scale"] = 2.
    with pytest.raises(ValueError, match="frozen|encoder"):
        lock.validate(x)
    with pytest.raises(ValueError, match="encoder"):
        x.predict(c.version_id, PublicTest("future", "f(7)"))
    with pytest.raises(ValueError, match="encoder|identity"):
        x.restore(encoded)
    def mutable_kw(text, *, scale=[1.]):
        return features(text) * scale[0]
    with pytest.raises(ValueError, match="encoder|callable|default"):
        make_belief_arm("prior", model=model(), feature_encoder=mutable_kw)


def test_callable_identity_rejects_mutable_positional_and_closure_state():
    from pbpf.arms.base import callable_identity
    def mutable_default(text, scale=[1.]):
        return features(text) * scale[0]
    scale = [1.]
    def mutable_closure(text):
        return features(text) * scale[0]
    for callback in (mutable_default, mutable_closure):
        with pytest.raises(ValueError, match="callable|default|closure"):
            callable_identity(callback)


def test_registered_encoder_parameter_cannot_hide_behind_custom_state_dict():
    class HiddenParameter(NeuralEncoder):
        def state_dict(self, *args, **kwargs):
            return {}
    encoder = HiddenParameter().eval()
    x = make_belief_arm("prior", model=model(), feature_encoder=encoder)
    c = candidate()
    x.initialize(task(), c)
    encoded = x.serialize()
    encoder.scale.data.add_(1.)
    with pytest.raises(ValueError, match="encoder"):
        x.predict(c.version_id, PublicTest("future", "f(7)"))
    with pytest.raises(ValueError, match="encoder|identity"):
        x.restore(encoded)


class DropoutEncoder(NeuralEncoder):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.5)

    def forward(self, text):
        return self.dropout(features(text) * self.scale)


def test_semantic_encoders_require_eval_for_root_and_every_submodule_without_switching_mode():
    encoder = DropoutEncoder()
    with pytest.raises(ValueError, match="encoder|eval|training"):
        make_belief_arm("prior", model=model(), feature_encoder=encoder)
    assert encoder.training and encoder.dropout.training
    encoder.eval()
    encoder.dropout.train()
    with pytest.raises(ValueError, match="encoder|eval|training"):
        make_belief_arm("prior", model=model(), feature_encoder=encoder)
    assert not encoder.training and encoder.dropout.training


def test_eval_dropout_prediction_and_restored_execution_are_identical():
    encoder = DropoutEncoder().eval()
    x = make_belief_arm("prior", model=model(), feature_encoder=encoder)
    c = candidate()
    x.initialize(task(), c)
    query = PublicTest("future", "f(7)")
    expected = x.predict(c.version_id, query)
    assert x.predict(c.version_id, query) == expected
    checkpoint = x.serialize()
    y = make_belief_arm("prior", model=model(), feature_encoder=DropoutEncoder().eval())
    y.restore(checkpoint)
    assert y.predict(c.version_id, query) == x.predict(c.version_id, query)
    assert y.serialize() == x.serialize()


def test_explicit_rng_encoder_is_rejected_even_in_eval_mode():
    class Noisy(NeuralEncoder):
        def forward(self, text):
            return features(text) + torch.randn(3)
    with pytest.raises(ValueError, match="encoder|stochastic|random"):
        make_belief_arm("prior", model=model(), feature_encoder=Noisy().eval())


def test_restore_reports_actual_validation_work_on_success_and_late_failure():
    x = arm("pbpf")
    c = candidate()
    x.initialize(task(), c)
    x.observe(c.version_id, event(c, 0, "PASS"))
    x.condition(c.version_id)
    original = x.serialize()
    expected_forwards = x.accounting["belief_forwards"]
    x.restore(original)
    assert x.last_restore_work == {"belief_forwards": expected_forwards, "actor_decodes": 0}
    payload = json.loads(original)["payload"]
    payload["condition_schedules"][c.version_id]["posterior_fingerprint"] = "b" * 64
    x.last_restore_work = {"belief_forwards": 0, "actor_decodes": 0}
    with pytest.raises(ValueError, match="restore"):
        x.restore(_rechecksum(payload))
    assert x.last_restore_work == {"belief_forwards": expected_forwards, "actor_decodes": 0}
    assert x.serialize() == original
    payload["hidden_outcomes"] = "HIDDEN_CANARY"
    with pytest.raises(ValueError, match="restore"):
        x.restore(_rechecksum(payload))
    assert x.last_restore_work == {"belief_forwards": 0, "actor_decodes": 0}
    assert x.serialize() == original


def test_neural_forward_keyword_defaults_are_part_of_encoder_identity():
    class Defaulted(NeuralEncoder):
        def forward(self, text, *, gain=1.):
            return features(text) * self.scale * gain
    encoder = Defaulted().eval()
    x = make_belief_arm("prior", model=model(), feature_encoder=encoder)
    c = candidate()
    x.initialize(task(), c)
    Defaulted.forward.__kwdefaults__["gain"] = 2.
    with pytest.raises(ValueError, match="encoder"):
        x.predict(c.version_id, PublicTest("future", "f(7)"))


def _direct_randn(self, text):
    return torch.randn(3)


def _direct_randint(self, text):
    return torch.randint(0, 10, (3,)).float()


def _direct_randperm(self, text):
    return torch.randperm(3).float()


def _direct_rand(self, text):
    return torch.rand(3)


def _direct_normal(self, text):
    return torch.normal(0., 1., size=(3,))


def _direct_bernoulli(self, text):
    return torch.bernoulli(torch.ones(3))


def _direct_multinomial(self, text):
    return torch.multinomial(torch.ones(3), 3, replacement=True).float()


def _direct_dropout(self, text):
    return torch.nn.functional.dropout(features(text), training=True)


def _direct_python_random(self, text):
    return features(text) * random.uniform(0., 1.)


def _direct_secrets(self, text):
    return features(text) * secrets.randbelow(10)


def _direct_numpy_random(self, text):
    return torch.tensor(np.random.default_rng().standard_normal(3)).float()


@pytest.mark.parametrize("kind", ["object", "bound", "neural"])
@pytest.mark.parametrize("execution", [_direct_randn, _direct_randint, _direct_randperm,
    _direct_rand, _direct_normal, _direct_bernoulli, _direct_multinomial,
    _direct_dropout, _direct_python_random, _direct_secrets, _direct_numpy_random])
def test_direct_rng_rejected_in_every_declared_execution_form_without_probing(kind, execution):
    base = NeuralEncoder if kind == "neural" else ConfiguredEncoder
    method = "forward" if kind == "neural" else "encode" if kind == "bound" else "__call__"
    instance = type("StochasticEncoder", (base,), {method: execution})()
    if kind == "neural":
        instance.eval()
    callback = instance.encode if kind == "bound" else instance
    before_torch, before_python, before_numpy = torch.get_rng_state(), random.getstate(), np.random.get_state()
    with pytest.raises(ValueError, match="encoder|stochastic|random"):
        make_belief_arm("prior", model=model(), feature_encoder=callback)
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert before_numpy[0] == after_numpy[0] and np.array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]


def test_rng_guard_rejects_unsupported_dynamic_indirection_without_execution():
    class Dynamic(ConfiguredEncoder):
        def __call__(self, text):
            self.scale += 1.
            return getattr(torch, "randn")(3)
    encoder = Dynamic()
    with pytest.raises(ValueError, match="encoder|indirection|stochastic"):
        make_belief_arm("prior", model=model(), feature_encoder=encoder)
    assert encoder.scale == 1.  # validation must never execute the callback


def test_rng_guard_checks_aliased_and_helper_calls():
    # Global helpers are inspectable; an RNG helper must not become a bypass.
    def through_helper(text):
        return _direct_randint(None, text)
    with pytest.raises(ValueError, match="encoder|indirection|stochastic"):
        make_belief_arm("prior", model=model(), feature_encoder=through_helper)


def test_eval_child_dropout_cannot_hide_stochastic_functional_dropout():
    class MixedDropout(DropoutEncoder):
        def forward(self, text):
            return self.dropout(features(text)) + torch.nn.functional.dropout(features(text), training=True)
    with pytest.raises(ValueError, match="encoder|stochastic"):
        make_belief_arm("prior", model=model(), feature_encoder=MixedDropout().eval())


@pytest.mark.parametrize("kind", ["object", "bound", "neural"])
def test_deterministic_declared_execution_forms_keep_predicting(kind):
    encoder = NeuralEncoder().eval() if kind == "neural" else ConfiguredEncoder()
    callback = encoder.encode if kind == "bound" else encoder
    x = make_belief_arm("prior", model=model(), feature_encoder=callback)
    c = candidate()
    x.initialize(task(), c)
    query = PublicTest("query", "f(7)")
    assert x.predict(c.version_id, query) == x.predict(c.version_id, query)
