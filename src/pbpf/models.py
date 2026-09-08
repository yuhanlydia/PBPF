from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .particles import normalize_log_weights


@dataclass(frozen=True)
class ConditionedGeneration:
    text: str
    particle_index: int


class ParticleConditioner:
    """Samples one posterior particle once and holds it for a full continuation."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def generate(
        self,
        prompt: str,
        *,
        particles: Sequence[Any],
        log_weights: Sequence[float] | np.ndarray,
        rng: np.random.Generator,
        **generation: Any,
    ) -> ConditionedGeneration:
        if not particles:
            raise ValueError("at least one particle is required")
        probabilities = np.exp(normalize_log_weights(log_weights))
        if len(probabilities) != len(particles):
            raise ValueError("particles and weights must have equal length")
        index = int(rng.choice(len(particles), p=probabilities))
        text = self.backend.generate(
            prompt, particle_state=particles[index], **generation
        )
        return ConditionedGeneration(str(text), index)


class TransformersRepairBackend:
    """Lazy Hugging Face causal-LM backend; construction never imports ML stacks."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        quantization: Mapping[str, Any] | None = None,
        device_map: str | Mapping[str, Any] | None = "auto",
        success_scorer: Any | None = None,
        generation: Mapping[str, Any] | None = None,
        smoke_only: bool = False,
    ) -> None:
        if not model_id or not revision:
            raise ValueError("model_id and resolved revision are required")
        self.model_id = model_id
        self.revision = revision
        self.model = model
        self.tokenizer = tokenizer
        self.quantization = dict(quantization or {})
        self.device_map = device_map
        self.success_scorer = success_scorer
        self.generation = dict(generation or {"max_new_tokens": 256})
        self.smoke_only = bool(smoke_only)
        self._last_gpu_hours = 0.0
        self._last_generated_tokens: int | None = None

    def load(self, *, local_files_only: bool = False) -> "TransformersRepairBackend":
        if self.model is not None and self.tokenizer is not None:
            return self
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_kwargs: dict[str, Any] = {
            "revision": self.revision,
            "local_files_only": local_files_only,
        }
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        if self.quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(**self.quantization)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, local_files_only=local_files_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        return self

    def count_tokens(self, text: str) -> int:
        self.load()
        encode = getattr(self.tokenizer, "encode", None)
        if not callable(encode):
            raise TypeError("Transformers tokenizer must expose encode for token accounting")
        return len(encode(text, add_special_tokens=False))

    def predict_success(
        self,
        task: Any,
        candidate: Any,
        history: Any,
        *,
        request_text: str,
    ) -> float:
        scorer = self.success_scorer
        if scorer is None:
            raise RuntimeError(
                "Transformers repair execution requires an explicit success_scorer"
            )
        method = getattr(scorer, "predict_success", None)
        value = method(
            task, candidate, history, request_text=request_text
        ) if callable(method) else scorer(
            task, candidate, history, request_text=request_text
        )
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("success_scorer must return a finite value")
        meter = getattr(scorer, "gpu_hours_for_last_call", None)
        self._last_gpu_hours = float(meter()) if callable(meter) else 0.0
        return result

    def _gpu_count(self) -> int:
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, Mapping):
            devices = {
                str(device)
                for device in device_map.values()
                if str(device).startswith("cuda") or isinstance(device, int)
            }
            return len(devices)
        return int(str(getattr(self.model, "device", "")).startswith("cuda"))

    def gpu_hours_for_last_call(self) -> float:
        return self._last_gpu_hours

    def generated_tokens_for_last_call(self) -> int:
        if self._last_generated_tokens is None:
            raise RuntimeError("no completed generation token telemetry is available")
        return self._last_generated_tokens

    def _prepare_generation(
        self,
        prompt: str,
        *,
        particle_state: Mapping[str, Any] | None = None,
        **generation: Any,
    ) -> tuple[dict[str, Any], int, dict[str, Any]]:
        options = dict(generation)
        self.load(local_files_only=bool(options.pop("local_files_only", False)))
        encoded = self.tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            encoded = {name: tensor.to(model_device) for name, tensor in encoded.items()}
        state = dict(particle_state or {})
        if "soft_prompt" in state and "kv_delta" in state:
            raise ValueError("soft-prompt and KV-delta arms must execute separately")
        if "soft_prompt" in state:
            import torch

            token_embeddings = self.model.get_input_embeddings()(encoded.pop("input_ids"))
            soft_prompt = torch.as_tensor(
                state["soft_prompt"], device=token_embeddings.device, dtype=token_embeddings.dtype
            )
            if soft_prompt.ndim == 2:
                soft_prompt = soft_prompt.unsqueeze(0)
            soft_prompt = soft_prompt.expand(token_embeddings.shape[0], -1, -1)
            encoded["inputs_embeds"] = torch.cat((soft_prompt, token_embeddings), dim=1)
            if "attention_mask" in encoded:
                prefix_mask = torch.ones(
                    (encoded["attention_mask"].shape[0], soft_prompt.shape[1]),
                    device=encoded["attention_mask"].device,
                    dtype=encoded["attention_mask"].dtype,
                )
                encoded["attention_mask"] = torch.cat((prefix_mask, encoded["attention_mask"]), dim=1)
        decode_offset = int(encoded["input_ids"].shape[1]) if "input_ids" in encoded else 0
        if "kv_delta" in state:
            from .kv import KVDelta

            delta = state["kv_delta"]
            if not isinstance(delta, KVDelta):
                raise TypeError("particle_state kv_delta must be a KVDelta")
            past = state.get("past_key_values")
            if past is None:
                if "input_ids" not in encoded or encoded["input_ids"].shape[1] < 2:
                    raise ValueError("KV cache injection requires at least two prompt tokens")
                full_input_ids = encoded["input_ids"]
                prefill = dict(encoded)
                prefill["input_ids"] = full_input_ids[:, :-1]
                if "attention_mask" in prefill:
                    prefill["attention_mask"] = prefill["attention_mask"][:, :-1]
                outputs = self.model(**prefill, use_cache=True)
                past = outputs.past_key_values
                encoded["input_ids"] = full_input_ids[:, -1:]
                decode_offset = 1
            encoded["past_key_values"] = delta.apply_to_cache(past)
        elif "past_key_values" in state:
            encoded["past_key_values"] = state["past_key_values"]
        return encoded, decode_offset, options

    def generate(
        self,
        prompt: str,
        *,
        particle_state: Mapping[str, Any] | None = None,
        **generation: Any,
    ) -> str:
        options = dict(self.generation)
        options.update(generation)
        encoded, decode_offset, options = self._prepare_generation(
            prompt, particle_state=particle_state, **options
        )
        started = time.perf_counter()
        outputs = self.model.generate(**encoded, **options)
        self._last_gpu_hours = (
            max(time.perf_counter() - started, 0.0) * self._gpu_count() / 3600.0
        )
        continuation = outputs[0, decode_offset:]
        self._last_generated_tokens = int(len(continuation))
        return self.tokenizer.decode(continuation, skip_special_tokens=True)

    @staticmethod
    def _selected_logprob(scores: Any, token_id: int) -> float:
        if type(scores).__module__.split(".", 1)[0] == "torch":
            import torch

            return float(torch.log_softmax(scores[0].float(), dim=-1)[token_id].item())
        values = np.asarray(scores[0], dtype=np.float64)
        maximum = float(values.max())
        return float(values[token_id] - maximum - np.log(np.exp(values - maximum).sum()))

    def generate_with_logprob(
        self,
        prompt: str,
        *,
        particle_state: Mapping[str, Any] | None = None,
        **generation: Any,
    ) -> tuple[str, float]:
        options = dict(self.generation)
        options.update(generation)
        encoded, _, options = self._prepare_generation(
            prompt, particle_state=particle_state, **options
        )
        options["return_dict_in_generate"] = True
        options["output_scores"] = True
        started = time.perf_counter()
        outputs = self.model.generate(**encoded, **options)
        self._last_gpu_hours = (
            max(time.perf_counter() - started, 0.0) * self._gpu_count() / 3600.0
        )
        scores = tuple(outputs.scores)
        if not scores:
            raise RuntimeError("Transformers generation returned no token scores")
        continuation = outputs.sequences[0, -len(scores):]
        token_ids = [int(token.item() if hasattr(token, "item") else token) for token in continuation]
        self._last_generated_tokens = len(token_ids)
        logprob = sum(
            self._selected_logprob(step_scores, token_id)
            for step_scores, token_id in zip(scores, token_ids, strict=True)
        )
        text = self.tokenizer.decode(continuation, skip_special_tokens=True)
        if not math.isfinite(logprob):
            raise ValueError("Transformers generation produced a non-finite sequence log-probability")
        return str(text), float(logprob)

    def repair(
        self,
        task: Any,
        candidate: Any,
        history: Any,
        round_index: int,
        *,
        request_text: str,
        max_generated_tokens: int,
    ) -> Any:
        from .repair import CandidateProposal

        if max_generated_tokens <= 0:
            raise ValueError("max_generated_tokens must be positive")
        options = dict(self.generation)
        options["max_new_tokens"] = min(
            int(options.get("max_new_tokens", max_generated_tokens)),
            max_generated_tokens,
        )
        text, logprob = self.generate_with_logprob(request_text, **options)
        return CandidateProposal(text, logprob, self.generated_tokens_for_last_call())
