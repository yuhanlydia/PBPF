"""Lazy native-chat HF primitives for an externally integrated FormalFactory.

These are actor/data adapters, not a complete training/evaluator deployment.
Loads use exact revisions and the already-provisioned local cache by default.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time

from .budget import MeteredActor, Usage
from ..arms.repair import DecodeResult, PartialResult
from .shard import digest


def _revision(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("immutable full revision required before optional loading")


def render_native_prompt(tokenizer, *, model_id, messages):
    if not tokenizer.chat_template:
        raise ValueError("pinned tokenizer must supply a native chat template")
    if (not messages or any(type(row) is not dict or set(row) != {"role", "content"}
            or row["role"] not in {"system", "user", "assistant"} or type(row["content"]) is not str for row in messages)):
        raise ValueError("only public role/content prompt records are permitted")
    options = {"tokenize": True, "add_generation_prompt": True}
    if model_id == "Qwen/Qwen3-8B":
        options["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **options)


class HFActor:
    def __init__(self, model, tokenizer, *, model_id, protocol, prompts, conditioner=None, model_revision=None):
        self.model, self.tokenizer, self.model_id = model, tokenizer, model_id
        self.protocol, self.prompts, self.conditioner = dict(protocol), dict(prompts), conditioner
        self.model_revision = model_revision
        self._usage = Usage()

    def native_usage(self):
        return self._usage

    def _messages(self, request, root):
        if root:
            return [{"role": "system", "content": self.prompts["root_system"]},
                    {"role": "user", "content": self.prompts["root_user"].format(task_text=request.task.task_text)}]
        feedback = [{"test_id": event.test_id, "outcome": event.outcome, "bounded_feedback": event.feedback}
                    for event in request.visible_events]
        transcript = [{"source": code, "events": [asdict(event) for event in events]}
                      for code, events in request.transcript]
        text = self.prompts["repair_user"].format(task_text=request.task_text, candidate_source=request.parent.source,
            visible_feedback=json.dumps({"ordered_events": feedback, "transcript": transcript}, sort_keys=True),
            arm_instruction=request.instruction)
        return [{"role": "system", "content": self.prompts["repair_system"]}, {"role": "user", "content": text}]

    def _decode(self, request, *, root=False, prefix=(), token_budget=None):
        self._usage = Usage()
        import torch
        started, cpu = time.monotonic(), time.process_time()
        native_input, generated = (), ()
        device = self.model.device
        gpu = device.type == "cuda"
        if gpu:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        try:
            if request.num_return_sequences != 1:
                raise ValueError("matched HF adapter requires decode multiplicity one")
            tokens = render_native_prompt(self.tokenizer, model_id=self.model_id, messages=self._messages(request, root))
            previous = [int(token) for token in prefix]
            if any(token < 0 for token in previous):
                raise ValueError("partial prefix requires native nonnegative token IDs")
            tokens = list(tokens) + previous
            cap = token_budget if token_budget is not None else request.max_new_tokens
            if len(tokens) + cap + (8 if getattr(request, "condition", None) is not None else 0) > self.protocol["context_length"]:
                raise ValueError("native prompt plus output exceeds context; silent truncation is prohibited")
            native_input = tuple(tokens)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            kwargs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
            condition = getattr(request, "condition", None)
            if condition is not None:
                if self.conditioner is None:
                    raise ValueError("conditioned arm requires an actual frozen soft-prefix projector")
                embeddings = self.model.get_input_embeddings()(input_ids)
                soft = self.conditioner(condition).to(device=device, dtype=embeddings.dtype)
                if tuple(soft.shape) != (1, 8, embeddings.shape[-1]):
                    raise ValueError("primary soft-prefix shape must be [1,8,d_model]")
                kwargs = {"inputs_embeds": torch.cat((soft, embeddings), dim=1),
                          "attention_mask": torch.ones((1, len(tokens)+8), device=device, dtype=torch.long)}
            stream = [request.task.task_id, self.model_id, self.model_revision, digest(self.prompts), request.seed, request.slot] if root else None
            seed = int(digest(stream)[:16], 16) if root else request.rng_seed
            with torch.random.fork_rng(devices=[device.index or 0] if gpu else []):
                torch.manual_seed(seed)
                with torch.inference_mode():
                    output = self.model.generate(**kwargs, do_sample=True, temperature=self.protocol["temperature"],
                        top_p=self.protocol["top_p"], max_new_tokens=cap, num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id)
            ids = output[0].detach().cpu().tolist()
            generated = tuple(ids[len(tokens):] if "input_ids" in kwargs else ids)
            source = self.tokenizer.decode(previous + list(generated), skip_special_tokens=True)
            return source, generated
        finally:
            if gpu:
                torch.cuda.synchronize(device)
            elapsed = time.monotonic()-started
            self._usage = Usage(input_token_ids=native_input, output_token_ids=generated,
                cpu_seconds=time.process_time()-cpu, wall_seconds=elapsed, gpu_seconds=elapsed if gpu else 0.,
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if gpu else 0)

    def generate_root(self, request):
        source, tokens = self._decode(request, root=True)
        return DecodeResult(source, len(tokens))

    def generate(self, request):
        source, tokens = self._decode(request)
        return DecodeResult(source, len(tokens))

    def generate_partial(self, request, *, prefix, token_budget, complete):
        source, tokens = self._decode(request, prefix=prefix, token_budget=token_budget)
        return PartialResult(tuple(prefix)+tuple(str(token) for token in tokens), source if complete else None, len(tokens))


def _snapshot_inventory(directory, selected, kind):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"{kind} snapshot must be a concrete local directory")
    paths = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not selected(path.name):
            continue
        if path.is_symlink():
            raise ValueError(f"{kind} snapshot artifact cannot be a symbolic link")
        if not path.is_file():
            raise ValueError(f"{kind} snapshot artifact must be a regular file")
        paths.append(path)
    records = []
    for path in paths:
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append([path.name, checksum])
    return paths, records


def _weight_file(name):
    path = Path(name)
    return (path.suffix in {".safetensors", ".bin"} or name in {"config.json", "generation_config.json"}
            or name.endswith(".index.json"))


def verify_weight_snapshot(directory, expected):
    directory = Path(directory)
    paths, records = _snapshot_inventory(directory, _weight_file, "model weight/config")
    if not any(path.suffix in {".safetensors", ".bin"} for path in paths) or not (directory / "config.json").is_file():
        raise ValueError("complete local model weight/config content inventory required")
    checksum = digest(records)
    if checksum != expected:
        raise ValueError("immutable model weight/config content hash mismatch before model loading")
    return checksum


def _tokenizer_file(name):
    return (name in {"added_tokens.json", "chat_template.jinja", "config.json", "special_tokens_map.json",
                     "tokenizer.json", "tokenizer_config.json"}
            or name.startswith(("merges.", "tokenizer.", "vocab.")) or name.endswith(".model"))


def verify_tokenizer_snapshot(directory, expected):
    paths, records = _snapshot_inventory(directory, _tokenizer_file, "tokenizer")
    names = {path.name for path in paths}
    vocabulary = names - {"chat_template.jinja", "config.json", "special_tokens_map.json", "tokenizer_config.json"}
    if "tokenizer_config.json" not in names or not vocabulary:
        raise ValueError("complete local tokenizer vocab/config content inventory required")
    checksum = digest(records)
    if checksum != expected:
        raise ValueError("immutable tokenizer vocab/config content hash mismatch before tokenizer loading")
    return checksum


def load_actor(model_config, *, model_identity, protocol, prompts, ledger, work_prefix, conditioner=None):
    _revision(model_config.get("revision"))
    _revision(model_config.get("tokenizer_revision"))
    from ..iclr_config import MODEL_PINS
    if (model_config.get("id"), model_config["revision"]) not in MODEL_PINS.values():
        raise ValueError("immutable model/revision is not in the formal model registry")
    if (model_identity.get("model_id"), model_identity.get("revision")) != (model_config["id"], model_config["revision"]):
        raise ValueError("runtime identity differs from immutable model configuration")
    tokenizer_pin = (model_config.get("tokenizer_id"), model_config["tokenizer_revision"])
    if (model_identity.get("tokenizer_id"), model_identity.get("tokenizer_revision")) != tokenizer_pin:
        raise ValueError("runtime tokenizer identity differs from immutable tokenizer configuration")
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(model_config["id"], revision=model_config["revision"], local_files_only=True)
    verify_weight_snapshot(snapshot, model_identity.get("weights_hash"))
    tokenizer_snapshot = (snapshot if tokenizer_pin == (model_config["id"], model_config["revision"])
                          else snapshot_download(tokenizer_pin[0], revision=tokenizer_pin[1], local_files_only=True))
    verify_tokenizer_snapshot(tokenizer_snapshot, model_identity.get("tokenizer_hash"))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot, revision=tokenizer_pin[1],
                                              local_files_only=True, trust_remote_code=False)
    template_hash = hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest()
    if template_hash != model_identity.get("chat_template_hash"):
        raise ValueError("native chat-template content hash mismatch before model loading")
    model = AutoModelForCausalLM.from_pretrained(snapshot, revision=model_config["revision"],
        local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval()
    backend = HFActor(model, tokenizer, model_id=model_config["id"], model_revision=model_config["revision"],
                      protocol=protocol, prompts=prompts, conditioner=conditioner)
    return MeteredActor(backend, ledger=ledger, work_prefix=work_prefix)


def load_dataset_snapshot(dataset_id, *, revision, split, allow_network=False):
    _revision(revision)
    from datasets import DownloadConfig, load_dataset
    return load_dataset(dataset_id, revision=revision, split=split,
                        download_config=DownloadConfig(local_files_only=not allow_network))
