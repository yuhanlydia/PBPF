"""Adapter tests use a tiny native-token fake, not pretrained weights."""
import builtins
import hashlib
import importlib
import json
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest


def test_optional_hf_adapter_module_is_lazy_and_bad_pin_precedes_import(monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "datasets"}:
            raise AssertionError("optional import before validation")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    module = importlib.import_module("pbpf.runner.hf")
    with pytest.raises(ValueError, match="immutable"):
        module.load_actor({"id": "Qwen/Qwen3-8B", "revision": "main"},
                          model_identity={}, protocol={}, prompts={}, ledger=None, work_prefix="x")


def test_native_chat_request_disables_thinking_and_rejects_hidden_prompt_fields():
    module = importlib.import_module("pbpf.runner.hf")
    tokenizer = SimpleNamespace(chat_template="native-template")
    captured = []
    def apply(messages, **kwargs):
        captured.append((messages, kwargs))
        return [11, 12]
    tokenizer.apply_chat_template = apply
    tokens = module.render_native_prompt(tokenizer, model_id="Qwen/Qwen3-8B", messages=[{"role": "user", "content": "public task"}])
    assert tokens == [11, 12]
    assert captured[0][1]["enable_thinking"] is False
    assert captured[0][1]["add_generation_prompt"] is True
    with pytest.raises(ValueError, match="public"):
        module.render_native_prompt(tokenizer, model_id="Qwen/Qwen3-8B", messages=[{"role": "user", "content": "task", "hidden_tests": "canary"}])


def test_dataset_loader_requires_exact_revision_before_optional_import(monkeypatch):
    module = importlib.import_module("pbpf.runner.hf")
    with pytest.raises(ValueError, match="immutable"):
        module.load_dataset_snapshot("anjiangwei/CodeARC-Problems", revision="latest", split="test")


def test_native_actor_decodes_are_metered_and_partial_tokens_not_double_counted(tmp_path):
    torch = pytest.importorskip("torch")
    module = importlib.import_module("pbpf.runner.hf")
    from pbpf.runner.budget import BudgetLedger, MeteredActor
    class Tokenizer:
        chat_template = "native"
        eos_token_id = 2
        def apply_chat_template(self, messages, **kwargs): return [11, 12]
        def decode(self, tokens, **kwargs): return "print('ok')"
    class Model:
        device = torch.device("cpu")
        def generate(self, *, input_ids, **kwargs):
            assert kwargs["num_return_sequences"] == 1
            assert kwargs["temperature"] == .8
            return torch.cat((input_ids, torch.tensor([[21, 22]])), dim=1)
    protocol = {"context_length": 8192, "max_new_tokens": 1024, "temperature": .8, "top_p": .95}
    backend = module.HFActor(Model(), Tokenizer(), model_id="Qwen/Qwen3-8B", protocol=protocol,
                           prompts={"root_system": "program", "root_user": "{task_text}"})
    ledger = BudgetLedger(tmp_path)
    actor = MeteredActor(backend, ledger=ledger, work_prefix="test")
    request = SimpleNamespace(task=SimpleNamespace(task_id="task-a", task_text="public"), seed=1701, slot=0, num_return_sequences=1, max_new_tokens=1024)
    assert actor.generate_root(request).source == "print('ok')"
    assert ledger.totals()["full_decodes"] == 1
    assert ledger.totals()["input_tokens"] == ledger.totals()["output_tokens"] == 2
    request = SimpleNamespace(task_text="public", parent=SimpleNamespace(source="bad"), visible_events=(), transcript=(),
                              instruction="repair", rng_seed=1701, condition=None, num_return_sequences=1, max_new_tokens=1024)
    backend.prompts.update(repair_system="repair", repair_user="{task_text} {candidate_source} {visible_feedback} {arm_instruction}")
    partial = actor.generate_partial(request, prefix=(), token_budget=512, complete=False)
    assert partial.source is None and partial.prefix == ("21", "22")
    final = actor.generate_partial(request, prefix=partial.prefix, token_budget=512, complete=True)
    assert final.source == "print('ok')" and final.prefix == ("21", "22", "21", "22")
    assert ledger.totals()["full_decodes"] == 2
    assert ledger.totals()["actor_requests"] == 3
    assert ledger.totals()["output_tokens"] == 6


def test_root_stream_binds_task_model_and_prompt_and_prefix_decode_counts_only_new_ids():
    torch = pytest.importorskip("torch")
    module = importlib.import_module("pbpf.runner.hf")
    seeds = []
    class Tokenizer:
        chat_template = "native"
        eos_token_id = 2
        def apply_chat_template(self, messages, **kwargs): return [11, 12]
        def decode(self, tokens, **kwargs):
            assert tokens == [31, 32]
            return "program"
    class Model:
        device = torch.device("cpu")
        def get_input_embeddings(self): return lambda ids: torch.zeros((1, ids.shape[1], 4))
        def generate(self, **kwargs):
            seeds.append(torch.initial_seed())
            if "inputs_embeds" in kwargs:
                assert kwargs["inputs_embeds"].shape == (1, 10, 4)
                return torch.tensor([[31, 32]])
            return torch.cat((kwargs["input_ids"], torch.tensor([[31, 32]])), dim=1)
    backend = module.HFActor(Model(), Tokenizer(), model_id="Qwen/Qwen3-8B", model_revision="b"*40,
        protocol={"context_length": 8192, "max_new_tokens": 1024, "temperature": .8, "top_p": .95},
        prompts={"root_system": "program", "root_user": "{task_text}", "repair_system": "repair", "repair_user": "{task_text}"},
        conditioner=lambda condition: torch.ones((1, 8, 4)))
    request = SimpleNamespace(task=SimpleNamespace(task_id="task-a", task_text="public"), seed=1701, slot=0,
                              num_return_sequences=1, max_new_tokens=1024)
    backend.generate_root(request)
    request.task.task_id = "task-b"
    backend.generate_root(request)
    assert seeds[0] != seeds[1]
    request.task.task_id = "task-a"
    backend.generate_root(request)
    assert seeds[0] == seeds[2]
    request = SimpleNamespace(task_text="public", parent=SimpleNamespace(source="bad"), visible_events=(), transcript=(),
        instruction="repair", rng_seed=1701, condition=object(), num_return_sequences=1, max_new_tokens=1024)
    result = backend.generate_partial(request, prefix=(), token_budget=512, complete=True)
    assert result.output_tokens == 2 and result.prefix == ("31", "32")
    assert backend.native_usage().output_token_ids == (31, 32)


def test_weight_snapshot_rejects_mutated_local_cache_before_loading(tmp_path):
    module = importlib.import_module("pbpf.runner.hf")
    (tmp_path / "config.json").write_bytes(b"config")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    manifest = [["config.json", hashlib.sha256(b"config").hexdigest()],
                ["model.safetensors", hashlib.sha256(b"weights").hexdigest()]]
    expected = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode()).hexdigest()
    assert module.verify_weight_snapshot(tmp_path, expected) == expected
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weight.*content"):
        module.verify_weight_snapshot(tmp_path, expected)


def test_tokenizer_snapshot_binds_inventory_and_content_before_loading(tmp_path):
    module = importlib.import_module("pbpf.runner.hf")
    files = {
        "special_tokens_map.json": b'{"eos_token":"<eos>"}',
        "tokenizer.json": b'{"model":{"type":"WordLevel"}}',
        "tokenizer_config.json": b'{"tokenizer_class":"TinyTokenizer"}',
    }
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    records = [[name, hashlib.sha256(files[name]).hexdigest()] for name in sorted(files)]
    expected = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()

    assert module.verify_tokenizer_snapshot(tmp_path, expected) == expected
    (tmp_path / "tokenizer.json").write_bytes(b'{"model":{"type":"changed"}}')
    with pytest.raises(ValueError, match="tokenizer.*content"):
        module.verify_tokenizer_snapshot(tmp_path, expected)
    (tmp_path / "tokenizer.json").write_bytes(files["tokenizer.json"])
    (tmp_path / "added_tokens.json").write_bytes(b'{"<new>":42}')
    with pytest.raises(ValueError, match="tokenizer.*content"):
        module.verify_tokenizer_snapshot(tmp_path, expected)


def test_tokenizer_snapshot_rejects_symlinked_artifacts(tmp_path):
    module = importlib.import_module("pbpf.runner.hf")
    outside = tmp_path.parent / "outside-tokenizer.json"
    outside.write_bytes(b'{"model":"mutable"}')
    (tmp_path / "tokenizer_config.json").write_bytes(b'{"tokenizer_class":"TinyTokenizer"}')
    (tmp_path / "tokenizer.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        module.verify_tokenizer_snapshot(tmp_path, "0" * 64)


def test_load_actor_rejects_tokenizer_id_mismatch_before_optional_import(monkeypatch):
    module = importlib.import_module("pbpf.runner.hf")
    revision = "b968826d9c46dd6066d109eabc6255188de91218"
    model_config = {
        "id": "Qwen/Qwen3-8B",
        "revision": revision,
        "tokenizer_id": "Qwen/Qwen3-8B",
        "tokenizer_revision": revision,
    }
    model_identity = {
        "model_id": model_config["id"],
        "revision": revision,
        "tokenizer_id": "attacker/re-tokenized-qwen3",
        "tokenizer_revision": revision,
        "tokenizer_hash": "a" * 64,
        "chat_template_hash": "b" * 64,
        "weights_hash": "c" * 64,
    }
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"huggingface_hub", "torch", "transformers"}:
            raise AssertionError("optional import before tokenizer identity validation")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(ValueError, match="tokenizer.*identity"):
        module.load_actor(model_config, model_identity=model_identity, protocol={}, prompts={}, ledger=None,
                          work_prefix="x")


def test_load_actor_verifies_shared_snapshot_once_and_builds_pinned_native_actor(tmp_path, monkeypatch):
    module = importlib.import_module("pbpf.runner.hf")
    from pbpf.runner.budget import BudgetLedger
    revision = "b968826d9c46dd6066d109eabc6255188de91218"
    files = {
        "config.json": b'{"model_type":"qwen3"}',
        "model.safetensors": b"tiny-weights",
        "special_tokens_map.json": b'{"eos_token":"<eos>"}',
        "tokenizer.json": b'{"model":{"type":"WordLevel"}}',
        "tokenizer_config.json": b'{"tokenizer_class":"TinyTokenizer"}',
    }
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    weight_names = ["config.json", "model.safetensors"]
    tokenizer_names = ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
    checksum = lambda names: hashlib.sha256(json.dumps(
        [[name, hashlib.sha256(files[name]).hexdigest()] for name in names], separators=(",", ":")).encode()).hexdigest()
    model_config = {"id": "Qwen/Qwen3-8B", "revision": revision,
                    "tokenizer_id": "Qwen/Qwen3-8B", "tokenizer_revision": revision}
    identity = {"model_id": model_config["id"], "revision": revision,
                "tokenizer_id": model_config["tokenizer_id"], "tokenizer_revision": revision,
                "tokenizer_hash": checksum(tokenizer_names), "weights_hash": checksum(weight_names),
                "chat_template_hash": hashlib.sha256(b"native-template").hexdigest()}
    snapshot_calls = []
    hub = ModuleType("huggingface_hub")
    def snapshot_download(identifier, **kwargs):
        snapshot_calls.append((identifier, kwargs))
        return str(tmp_path)
    hub.snapshot_download = snapshot_download
    torch = ModuleType("torch")
    torch.bfloat16 = object()
    transformers = ModuleType("transformers")
    class Tokenizer:
        chat_template = "native-template"
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(tmp_path)
            assert kwargs == {"revision": revision, "local_files_only": True, "trust_remote_code": False}
            return Tokenizer()
    class Model:
        def eval(self): self.evaluated = True
    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(tmp_path)
            assert kwargs["revision"] == revision and kwargs["local_files_only"] is True
            return Model()
    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    actor = module.load_actor(model_config, model_identity=identity, protocol={"context_length": 8192},
                              prompts={"root_system": "system"}, ledger=BudgetLedger(tmp_path / "ledger"),
                              work_prefix="test")
    assert snapshot_calls == [("Qwen/Qwen3-8B", {"revision": revision, "local_files_only": True})]
    assert actor.backend.model.evaluated is True
    assert actor.backend.tokenizer.chat_template == "native-template"
    assert actor.backend.model_id == "Qwen/Qwen3-8B" and actor.backend.model_revision == revision
