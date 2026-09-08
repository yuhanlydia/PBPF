import json
import subprocess
import sys

import numpy as np
import pytest

from pbpf.benchmarks import load_benchmark_jsonl
from pbpf.history import feedback_view, history_view
from pbpf.kv import build_kv_delta
from pbpf.models import ParticleConditioner, TransformersRepairBackend
from pbpf.manifest import CandidateVersion
from pbpf.repair import CandidateProposal
from pbpf.registry import UNAVAILABLE_EXECUTION_ARMS
from pbpf.state import StateEncoder


HISTORY = (
    {"task_id": "a", "test_id": "t1", "outcome": "PASS"},
    {"task_id": "a", "test_id": "t2", "outcome": "TIMEOUT"},
    {"task_id": "a", "test_id": "t3", "outcome": "WRONG_OUTPUT"},
)


def test_history_views_apply_declared_order_and_mask_controls():
    rng = np.random.default_rng(3)
    assert [item["test_id"] for item in history_view(HISTORY, "last")] == ["t3"]
    assert [item["test_id"] for item in history_view(HISTORY, "window", k=2)] == ["t2", "t3"]
    assert history_view(HISTORY, "full") == HISTORY
    assert [item["test_id"] for item in history_view(HISTORY, "orderless")] == ["t1", "t2", "t3"]
    shuffled = history_view(HISTORY, "shuffled", rng=rng)
    assert {item["test_id"] for item in shuffled} == {"t1", "t2", "t3"}
    assert [item["outcome"] for item in history_view(HISTORY, "outcome_masked")] == [
        "MASKED",
        "MASKED",
        "MASKED",
    ]
    other = ({"task_id": "b", "test_id": "u1", "outcome": "PASS"},)
    assert history_view(HISTORY, "wrong_task", wrong_task_history=other) == other


def test_outcome_mask_and_feedback_modes_remove_derived_information():
    detailed = (
        {
            "task_id": "a",
            "test_id": "t1",
            "outcome": "WRONG_OUTPUT",
            "visible_feedback": "secret raw output",
            "output_diff": "-1 +2",
            "exception_trace": "trace",
            "test_source": "assert f() == 2",
        },
    )
    masked = history_view(detailed, "outcome_masked")
    assert masked == ({"task_id": "a", "test_id": "t1", "outcome": "MASKED"},)
    assert feedback_view(detailed, "status_only") == (
        {"task_id": "a", "test_id": "t1", "outcome": "WRONG_OUTPUT"},
    )
    assert feedback_view(detailed, "status_output_diff")[0]["output_diff"] == "-1 +2"
    assert "exception_trace" not in feedback_view(detailed, "status_output_diff")[0]


def test_matched_capacity_encoders_have_equal_shapes_and_exchangeable_is_equivariant():
    encodings = {
        name: StateEncoder(name, dimension=12, seed=7).encode(HISTORY)
        for name in ("gru", "exchangeable", "pass_rate", "random")
    }
    assert {value.shape for value in encodings.values()} == {(12,)}
    exchangeable = StateEncoder("exchangeable", dimension=12, seed=7)
    assert exchangeable.encode(HISTORY).tolist() == pytest.approx(
        exchangeable.encode(tuple(reversed(HISTORY))).tolist()
    )
    gru = StateEncoder("gru", dimension=12, seed=7)
    assert not np.allclose(gru.encode(HISTORY), gru.encode(tuple(reversed(HISTORY))))
    assert gru.trainable_parameter_count == exchangeable.trainable_parameter_count


def test_equal_parameter_controls_are_not_advertised_as_matched_compute():
    assert "matched_gru" in UNAVAILABLE_EXECUTION_ARMS
    assert "matched_exchangeable" in UNAVAILABLE_EXECUTION_ARMS
    assert "compute" in UNAVAILABLE_EXECUTION_ARMS["matched_gru"]


def test_matched_state_encoder_parameters_are_trainable():
    histories = (HISTORY[:1], HISTORY[:2], HISTORY)
    targets = np.asarray(
        [[1.0] + [0.0] * 11, [0.0, 1.0] + [0.0] * 10, [0.0, 0.0, 1.0] + [0.0] * 9]
    )
    encoder = StateEncoder("gru", dimension=12, seed=11)
    gates_before = encoder.gru_parameter_vector.copy()
    before = np.mean(
        [np.square(encoder.encode(history) - target).mean() for history, target in zip(histories, targets)]
    )
    encoder.fit(histories, targets, epochs=250, learning_rate=0.08)
    after = np.mean(
        [np.square(encoder.encode(history) - target).mean() for history, target in zip(histories, targets)]
    )
    assert after < before * 0.5
    assert not np.allclose(gates_before, encoder.gru_parameter_vector)
    assert encoder.trainable_parameter_count == 7 * 12 * 12 + 4 * 12
    exchangeable = StateEncoder("exchangeable", dimension=12, seed=11)
    parameters_before = exchangeable.trainable_parameter_vector.copy()
    exchangeable.fit(histories, targets, epochs=50, learning_rate=0.02)
    assert not np.allclose(parameters_before, exchangeable.trainable_parameter_vector)
    assert exchangeable.trainable_parameter_count == encoder.trainable_parameter_count


def test_kv_delta_enforces_layer_rank_mode_and_frobenius_norm():
    delta = build_kv_delta(
        np.array([0.2, -0.4, 0.7]),
        layers=6,
        heads=2,
        head_dim=4,
        rank=2,
        mode="k_only",
        active_layers=2,
        target_norm=3.0,
        rng=np.random.default_rng(8),
    )
    assert delta.key.shape == (6, 2, 4, 4)
    assert delta.value.shape == delta.key.shape
    assert np.count_nonzero(delta.key[:4]) == 0
    assert np.count_nonzero(delta.value) == 0
    ranks = np.linalg.matrix_rank(delta.key[-2:].reshape(-1, 4, 4))
    assert int(ranks.max()) <= 2
    assert np.linalg.norm(delta.key) == pytest.approx(3.0)
    assert delta.physical_bytes == delta.key.nbytes + delta.value.nbytes


def test_equal_norm_latents_change_cache_conditioned_logits():
    kwargs = dict(
        layers=2,
        heads=1,
        head_dim=3,
        rank=2,
        mode="kv",
        active_layers=2,
        target_norm=1.0,
    )
    first = build_kv_delta(np.array([1.0, 0.0]), rng=np.random.default_rng(19), **kwargs)
    second = build_kv_delta(np.array([0.0, 1.0]), rng=np.random.default_rng(19), **kwargs)
    assert np.linalg.norm(first.key) ** 2 + np.linalg.norm(first.value) ** 2 == pytest.approx(1.0)
    assert np.linalg.norm(second.key) ** 2 + np.linalg.norm(second.value) ** 2 == pytest.approx(1.0)
    assert not np.allclose(first.key, second.key)
    cache = tuple((np.ones((1, 1, 2, 3)), np.ones((1, 1, 2, 3))) for _ in range(2))
    ramp = np.arange(12, dtype=float)
    first_logits = np.dot(np.concatenate([part.ravel() for layer in first.apply_to_cache(cache) for part in layer]), np.tile(ramp, 2))
    second_logits = np.dot(np.concatenate([part.ravel() for layer in second.apply_to_cache(cache) for part in layer]), np.tile(ramp, 2))
    assert first_logits != pytest.approx(second_logits)


def test_transformers_backend_applies_kv_delta_before_generation():
    class Tokenizer:
        def __call__(self, prompt, return_tensors):
            return {"input_ids": np.array([[1, 2, 3]]), "attention_mask": np.ones((1, 3), dtype=int)}

        def decode(self, tokens, skip_special_tokens):
            return str(int(np.asarray(tokens)[-1]))

    class Model:
        device = None

        def __init__(self):
            self.seen_cache = None

        def __call__(self, **inputs):
            from types import SimpleNamespace

            cache = tuple((np.ones((1, 1, 2, 3)), np.ones((1, 1, 2, 3))) for _ in range(2))
            return SimpleNamespace(past_key_values=cache)

        def generate(self, **inputs):
            self.seen_cache = inputs["past_key_values"]
            token = 9 if not np.allclose(self.seen_cache[0][0], 1.0) else 4
            return np.array([[3, token]])

    model = Model()
    backend = TransformersRepairBackend("local/fake", "a" * 40, model=model, tokenizer=Tokenizer())
    delta = build_kv_delta(
        np.array([1.0, -0.5]),
        layers=2,
        heads=1,
        head_dim=3,
        rank=1,
        mode="k_only",
        active_layers=2,
        target_norm=1.0,
        rng=np.random.default_rng(7),
    )
    assert backend.generate("p", particle_state={"kv_delta": delta}, max_new_tokens=1) == "9"
    assert not np.allclose(model.seen_cache[0][0], 1.0)


def test_transformers_backend_implements_measured_repair_protocol():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def __call__(self, prompt, return_tensors):
            return {"input_ids": np.array([[1, 2]])}

        def decode(self, tokens, skip_special_tokens):
            return "fixed program"

    class Model:
        device = None

        def generate(self, **inputs):
            from types import SimpleNamespace

            assert inputs["return_dict_in_generate"] is True
            assert inputs["output_scores"] is True
            first = np.array([[0.0, 0.0, 0.0, 0.0, 3.0, 0.0]])
            second = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
            return SimpleNamespace(
                sequences=np.array([[1, 2, 4, 5]]), scores=(first, second)
            )

    backend = TransformersRepairBackend(
        "local/fake",
        "a" * 40,
        model=Model(),
        tokenizer=Tokenizer(),
        success_scorer=lambda task, candidate, history, request_text: 0.75,
        generation={"max_new_tokens": 2},
    )
    candidate = CandidateVersion.create("c", "t", "broken")
    assert backend.count_tokens("one two three") == 3
    assert backend.predict_success({}, candidate, (), request_text="score") == pytest.approx(0.75)
    proposal = backend.repair(
        {},
        candidate,
        (),
        1,
        request_text="repair request",
        max_generated_tokens=2,
    )
    assert isinstance(proposal, CandidateProposal)
    assert proposal.code == "fixed program"
    assert proposal.generation_logprob < 0
    assert np.isfinite(proposal.generation_logprob)


def test_kv_delta_supports_dynamic_cache_and_fails_closed_on_gqa_mismatch():
    class DynamicCache:
        def __init__(self, legacy):
            self.legacy = legacy

        def to_legacy_cache(self):
            return self.legacy

        @classmethod
        def from_legacy_cache(cls, legacy):
            return cls(legacy)

    delta = build_kv_delta(
        np.array([0.3, 0.8]),
        layers=2,
        heads=1,
        head_dim=3,
        rank=1,
        mode="kv",
        active_layers=2,
        target_norm=1.0,
        rng=np.random.default_rng(4),
    )
    legacy = tuple((np.ones((1, 1, 2, 3)), np.ones((1, 1, 2, 3))) for _ in range(2))
    transformed = delta.apply_to_cache(DynamicCache(legacy))
    assert isinstance(transformed, DynamicCache)
    assert not np.allclose(transformed.legacy[0][0], legacy[0][0])
    bad_gqa = tuple((np.ones((1, 2, 2, 3)), np.ones((1, 2, 2, 3))) for _ in range(2))
    with pytest.raises(ValueError, match="num_key_value_heads"):
        delta.apply_to_cache(bad_gqa)


def test_particle_conditioner_samples_exactly_one_particle_per_sequence():
    class Backend:
        def __init__(self):
            self.states = []

        def generate(self, prompt, *, particle_state, **generation):
            self.states.append(particle_state["id"])
            return prompt + ":" + particle_state["id"]

    backend = Backend()
    conditioner = ParticleConditioner(backend)
    result = conditioner.generate(
        "repair",
        particles=({"id": "p0"}, {"id": "p1"}),
        log_weights=np.log([0.0 + 1e-12, 1.0]),
        rng=np.random.default_rng(2),
    )
    assert result.text == "repair:p1"
    assert result.particle_index == 1
    assert backend.states == ["p1"]


def test_benchmark_loader_rejects_evaluator_fields_and_keeps_test_order(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps({"task_id": "a", "prompt": "p", "tests": ["t2", "t1"]}) + "\n",
        encoding="utf-8",
    )
    records = load_benchmark_jsonl(path, dataset="codearc")
    assert records[0].test_order == ("t2", "t1")
    path.write_text(
        json.dumps({"task_id": "x", "prompt": "p", "tests": [], "test_patch": "secret"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evaluator-only"):
        load_benchmark_jsonl(path, dataset="swebench_verified")


def test_benchmark_loader_rejects_nested_and_alias_gold_fields(tmp_path):
    path = tmp_path / "leaky.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "x",
                "prompt": "p",
                "tests": ["t1"],
                "metadata": {"fixed_code": "secret"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evaluator-only"):
        load_benchmark_jsonl(path, dataset="runbugrun")


def test_model_adapter_import_is_lazy():
    code = "import json,sys,pbpf.models; print(json.dumps([x for x in ('torch','transformers','peft') if x in sys.modules]))"
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == []
