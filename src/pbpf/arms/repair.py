"""Bounded repair sessions, not experiment orchestration or hidden evaluation.

Caller supplies exactly the public visible feedback after each proposed source.
All matched sessions complete four repairs even after visible success. Partial
Rollout Roulette requests are explicitly additional *partial* actor work: four
prefixes and four completions, with generated/discarded tokens accounted. The
budget label matches complete continuations, not equal FLOPs or request count.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import re

import numpy as np

from pbpf.data.schema import PublicTask
from pbpf.particles import systematic_resample
from .base import (ArmCandidate, RepairCondition, VisibleEvent, belief_digest, canonical,
                   derived_seed, freeze_public_task, source_hash, validate_event)
from .registry import FORMAL_BUDGET, REPAIR_ARMS, REPAIR_CAPABILITIES, load_arm_config, validate_provenance


@dataclass(frozen=True)
class FormalBank:
    candidates: tuple[ArmCandidate, ...]

    def __post_init__(self):
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if len(self.candidates) != FORMAL_BUDGET["initial_actor_samples"]:
            raise ValueError("formal bank requires exactly eight genuine actor samples")
        if any(type(c) is not ArmCandidate or c.origin != FORMAL_BUDGET["initial_origin"] or c.round != 0 for c in self.candidates):
            raise ValueError("all eight roots must be genuine actor samples, never mutants")
        if sorted(c.slot for c in self.candidates) != list(range(FORMAL_BUDGET["initial_actor_samples"])):
            raise ValueError("formal actor slots must be exactly 0 through 7")
        context = {(c.task_id, c.model_id, c.model_revision, c.prompt_revision, c.sample_seed) for c in self.candidates}
        if len(context) != 1:
            raise ValueError("bank task/model/prompt/seed provenance must agree")
        object.__setattr__(self, "candidates", tuple(sorted(self.candidates, key=lambda c: c.slot)))

    @property
    def content_hash(self):
        return source_hash(canonical([c.version_id for c in self.candidates]))


@dataclass(frozen=True)
class DecodeResult:
    source: str
    output_tokens: int

    def __post_init__(self):
        if not isinstance(self.source, str) or not self.source or not isinstance(self.output_tokens, int) or self.output_tokens < 0:
            raise ValueError("decoder must return source and nonnegative token telemetry")


@dataclass(frozen=True)
class PartialResult:
    prefix: tuple[str, ...]
    source: str | None
    output_tokens: int

    def __post_init__(self):
        object.__setattr__(self, "prefix", tuple(self.prefix))
        if (not self.prefix or not all(isinstance(token, str) for token in self.prefix)
                or not isinstance(self.output_tokens, int) or self.output_tokens < 0):
            raise ValueError("partial trajectory requires ordered tokens and telemetry")


@dataclass(frozen=True)
class ActorRequest:
    task_text: str
    parent: ArmCandidate
    visible_events: tuple[VisibleEvent, ...]
    transcript: tuple[tuple[str, tuple[VisibleEvent, ...]], ...]
    instruction: str
    round: int
    rng_seed: int
    condition: RepairCondition | None = None
    max_new_tokens: int = FORMAL_BUDGET["max_new_tokens"]
    temperature: float = FORMAL_BUDGET["temperature"]
    top_p: float = FORMAL_BUDGET["top_p"]
    num_return_sequences: int = FORMAL_BUDGET["decode_multiplicity"]


def _belief_digest(belief):
    return belief_digest(belief)


@dataclass(frozen=True)
class FrozenSelection:
    name: str
    checkpoint_hash: str
    source_split_hash: str
    behavior_hash: str
    split: str = "dev-select"
    metric: str = "task_macro_prefix4_future_nll"

    def __post_init__(self):
        from .prediction import MEMORIES
        if self.name not in MEMORIES or self.split != "dev-select" or self.metric != "task_macro_prefix4_future_nll":
            raise ValueError("frozen selection requires a deterministic development-select identity and allowed metric")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (self.checkpoint_hash, self.source_split_hash, self.behavior_hash)):
            raise ValueError("frozen selection requires valid checkpoint, source split and behavior hashes")

    @classmethod
    def freeze(cls, belief, *, source_split_hash, split="dev-select"):
        from .prediction import MEMORIES, MemoryNetwork
        if belief.name not in MEMORIES or type(belief.model) is not MemoryNetwork or belief.model.kind != belief.name:
            raise ValueError("development-selected comparator must be deterministic")
        if split != "dev-select" or len(source_split_hash) != 64 or any(c not in "0123456789abcdef" for c in source_split_hash):
            raise ValueError("selection requires a source-disjoint development-select split hash")
        for p in belief.model.parameters():
            p.requires_grad_(False)
        belief.model.eval()
        return cls(belief.name, _belief_digest(belief), source_split_hash,
                   source_hash(canonical(belief.behavior_identity())))

    def validate(self, belief):
        self.__post_init__()
        if (belief.name != self.name or source_hash(canonical(belief.behavior_identity())) != self.behavior_hash
                or _belief_digest(belief) != self.checkpoint_hash
                or any(p.requires_grad for p in belief.model.parameters()) or belief.model.training):
            raise ValueError("frozen development-selected belief identity or weights changed")


@dataclass(frozen=True)
class SelectionMetric:
    name: str
    value: float

    def __post_init__(self):
        if self.name not in {"initial_selected_pass1", "initial_hidden_pass8_oracle", "final_selected_pass1"}:
            raise ValueError("unknown selection metric")
        if not math.isfinite(self.value) or not 0 <= self.value <= 1:
            raise ValueError("selection metric must be in [0,1]")


def require_selected_pass1(metric: SelectionMetric, *, phase="final"):
    if phase not in {"initial", "final"} or metric.name != f"{phase}_selected_pass1":
        raise ValueError("selected Pass@1 is required; initial oracle Pass@8 is not a gate metric")
    return metric.value


@dataclass(frozen=True)
class RepairSelection:
    key: tuple[str, str, str, int, str]
    initial_selected_source_hash: str
    final_source_hash: str
    final_version_id: str
    bank_hash: str
    compute_class: str
    provenance: dict
    budget: dict


def selection_manifest(selections):
    """Retain every experimental cell even when several cells select equal code.

    Task-4 seal_candidates may consume the unique hash inventory; the runner
    must retain this full cell-to-hash mapping alongside that authorization seal.
    """
    selections = tuple(selections)
    if len({selection.key for selection in selections}) != len(selections):
        raise ValueError("duplicate final selection key")
    return {"schema": "pbpf-final-selections-v1",
            "selections": [{"key": list(s.key), "source_hash": s.final_source_hash,
                            "version_id": s.final_version_id} for s in sorted(selections, key=lambda s: s.key)],
            "candidate_hashes": sorted({s.final_source_hash for s in selections})}


class RepairArm:
    """One task/seed/arm session. propose -> visible observe, repeated four times.

    The injected actor implements generate(ActorRequest)->DecodeResult. Roulette
    additionally implements generate_partial(request,prefix,token_budget,complete)
    and receives no mutable particle population. That adapter must preserve the
    native actor chat template and fixed decoding controls in ActorRequest.
    """
    def __init__(self, name, *, task, bank, initial_events, actor, seed,
                 belief=None, deterministic_belief=None, selected_deterministic=None, partial_scorer=None, rex_concentration=4.):
        if name not in REPAIR_ARMS or type(task) is not PublicTask or type(bank) is not FormalBank:
            raise ValueError("registered arm, public task and formal bank are required")
        task = freeze_public_task(task)
        supplied = {"belief": belief, "deterministic_belief": deterministic_belief,
                    "selected_deterministic": selected_deterministic, "partial_scorer": partial_scorer}
        if any(value is not None and capability not in REPAIR_CAPABILITIES[name] for capability, value in supplied.items()):
            raise ValueError(f"{name} rejects unsupported belief/conditioning capability")
        if name == "strongest_deterministic":
            belief = deterministic_belief
        if bank.candidates[0].task_id != task.task_id or seed != bank.candidates[0].sample_seed:
            raise ValueError("task/seed does not match immutable bank")
        if not math.isfinite(rex_concentration) or rex_concentration <= 0:
            raise ValueError("REx concentration must be finite and positive")
        if name in {"pbpf", "strongest_deterministic"} and belief is None:
            raise ValueError("repair arm requires a candidate-specific belief")
        if name == "pbpf" and belief.name != "pbpf":
            raise ValueError("PBPF repair requires sample-once PBPF belief, not a collapse/control")
        if name == "strongest_deterministic":
            if not isinstance(selected_deterministic, FrozenSelection):
                raise ValueError("frozen development-selected comparator is required")
            selected_deterministic.validate(belief)
        if name == "rollout_roulette" and (not callable(partial_scorer) or not callable(getattr(actor, "generate_partial", None))):
            raise ValueError("Rollout Roulette requires ordered partial-generation actor and scorer")
        self.name, self.task, self.bank, self.actor, self.seed = name, task, bank, actor, seed
        self.belief, self.selected_deterministic = belief, selected_deterministic
        self.partial_scorer, self.rex_concentration = partial_scorer, rex_concentration
        self.provenance = validate_provenance(load_arm_config()["repair"][name])
        self.candidates = list(bank.candidates)
        self.initial_hashes = tuple(c.source_hash for c in bank.candidates)
        self._events, self._failed_attempts, self.decisions = {}, {}, []
        self._pending, self._count, self._roulette = None, 0, None
        self.rng = np.random.default_rng(derived_seed(seed, task.task_id, name, "repair"))
        self.accounting = {"actor_decodes": 0, "decode_multiplicity": FORMAL_BUDGET["decode_multiplicity"], "belief_forwards": 0,
                           "partial_actor_calls": 0, "output_tokens": 0, "discarded_prefix_tokens": 0,
                           "verifier_cases": 0, "verifier_suites": 0}
        if set(initial_events) != {c.version_id for c in bank.candidates}:
            raise ValueError("initial visible feedback inventory must equal all eight bank versions")
        for candidate in bank.candidates:
            self._events[candidate.version_id] = self._validate_feedback(candidate, initial_events[candidate.version_id])
            self._failed_attempts[candidate.version_id] = 0
            if belief is not None:
                belief.initialize(task, candidate)
                for event in self._events[candidate.version_id]:
                    belief.observe(candidate.version_id, event)
        self._initial_selected = self._visible_best(bank.candidates)
        self._debug_parent = self._initial_selected
        self._sync_belief()

    def _validate_feedback(self, candidate, events):
        events = tuple(events)
        if len(events) != len(self.task.visible_tests) or len({event.event_id for event in events}) != len(events):
            raise ValueError("feedback must contain exactly the ordered visible suite")
        for i, event in enumerate(events):
            validate_event(self.task, candidate, event, i)
        return events

    def _sync_belief(self):
        if self.belief is not None:
            self.accounting["belief_forwards"] = self.belief.accounting["belief_forwards"]

    def _score(self, candidate):
        events = self._events[candidate.version_id]
        return sum(event.outcome == "PASS" for event in events) / len(events) if events else 0.

    def _visible_best(self, candidates):
        return min(candidates, key=lambda c: (-self._score(c), c.round, c.slot, c.source_hash, c.version_id))

    def _parent(self):
        if self.name == "independent":
            return self.bank.candidates[self._count]
        if self.name == "self_debug":
            return self._debug_parent
        if self.name == "rex":
            draws = []
            for candidate in self.candidates:
                h = self._score(candidate)
                failed = self._failed_attempts[candidate.version_id]
                alpha, beta = 1 + self.rex_concentration*h, 1 + self.rex_concentration*(1 - h) + failed
                draws.append({"version_id": candidate.version_id, "alpha": alpha, "beta": beta,
                              "draw": float(self.rng.beta(alpha, beta)), "failed_attempts": failed,
                              "visible_pass_fraction": h})
            selected = max(draws, key=lambda row: row["draw"])["version_id"]
            self.decisions.append({"algorithm": "rex_thompson", "draws": draws, "selected_parent": selected})
            return next(c for c in self.candidates if c.version_id == selected)
        return self._visible_best(self.candidates)

    def _request(self, parent, *, round_index=None):
        condition = self.belief.condition(parent.version_id) if self.belief is not None else None
        transcript = []
        cursor = parent
        while True:
            transcript.append((cursor.source, self._events[cursor.version_id]))
            if self.name != "self_debug" or cursor.parent_version_id is None:
                break
            cursor = next(c for c in self.candidates if c.version_id == cursor.parent_version_id)
        transcript.reverse()
        instruction = ("Explain the observed failure and repair the complete program in this single response."
                       if self.name == "self_debug" else "Repair the complete program using only the supplied visible evidence.")
        self._sync_belief()
        return ActorRequest(self.task.task_text, parent, self._events[parent.version_id], tuple(transcript),
            instruction, self._count + 1 if round_index is None else round_index,
            derived_seed(self.seed, self.task.task_id, self.name, "decode", str(self._count if round_index is None else round_index)), condition)

    def _prepare_roulette(self):
        partials = []
        for i, parent in enumerate(self.bank.candidates[:FORMAL_BUDGET["repair_decodes"]]):
            request = self._request(parent, round_index=i + 1)
            partial = self.actor.generate_partial(request, prefix=(), token_budget=FORMAL_BUDGET["max_new_tokens"] // 2, complete=False)
            self.accounting["partial_actor_calls"] += 1
            if type(partial) is not PartialResult or partial.source is not None or partial.output_tokens > FORMAL_BUDGET["max_new_tokens"] // 2:
                raise ValueError("roulette prefix must be unfinished and within its token budget")
            self.accounting["output_tokens"] += partial.output_tokens
            score = float(self.partial_scorer(tuple(partial.prefix)))
            self.accounting["belief_forwards"] += 1
            if not math.isfinite(score):
                raise ValueError("partial trajectory score must be finite")
            partials.append((request, partial, score))
        scores = np.array([score for _, _, score in partials])
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        indices = systematic_resample(weights, rng=self.rng)
        self._roulette = [(replace(partials[index][0], round=i + 1,
                           rng_seed=derived_seed(self.seed, self.task.task_id, "roulette-completion", str(i))),
                           tuple(partials[index][1].prefix), partials[index][1].output_tokens)
                          for i, index in enumerate(indices)]
        self.accounting["discarded_prefix_tokens"] = sum(partial.output_tokens for i, (_, partial, _) in enumerate(partials) if i not in indices)
        self.decisions.append({"algorithm": "rollout_roulette_partial_pf", "ancestors": indices.tolist(),
            "scores": scores.tolist(), "weights": weights.tolist(), "score_transform": "exp(score - max_score)",
            "profile": "four_trajectories_two_segments", "partial_prefixes": [list(p.prefix) for _, p, _ in partials]})

    def propose(self):
        if self.name == "no_repair":
            raise ValueError("no-repair is a lower-compute selector with no actor decode")
        if self._count >= FORMAL_BUDGET["repair_decodes"]:
            raise ValueError("matched arm permits exactly four complete repair decodes")
        if self._pending is not None:
            raise ValueError("ordered visible feedback is required before the next proposal")
        if self.selected_deterministic is not None:
            self.selected_deterministic.validate(self.belief)
        if self.name == "rollout_roulette":
            if self._roulette is None:
                self._prepare_roulette()
            request, prefix, prefix_tokens = self._roulette[self._count]
            # Reserve half the cap for each stage, including discarded prefixes.
            # Reclaiming a short selected prefix's slack would hide actor work
            # already spent on discarded long prefixes and exceed 4 * 1024.
            partial = self.actor.generate_partial(request, prefix=prefix, token_budget=FORMAL_BUDGET["max_new_tokens"] // 2, complete=True)
            if (type(partial) is not PartialResult or not partial.source
                    or partial.prefix[:len(prefix)] != prefix or partial.output_tokens > FORMAL_BUDGET["max_new_tokens"] // 2):
                raise ValueError("roulette completion must preserve ordered cloned prefix and token cap")
            result = DecodeResult(partial.source, partial.output_tokens)
        else:
            request = self._request(self._parent())
            result = self.actor.generate(request)
            if type(result) is not DecodeResult:
                raise TypeError("actor must return DecodeResult, not arbitrary metadata")
        if result.output_tokens > FORMAL_BUDGET["max_new_tokens"]:
            raise ValueError("repair output exceeded 1024-token cap")
        parent = request.parent
        child = replace(parent, source=result.source, round=parent.round + 1,
                        parent_version_id=parent.version_id, origin="repair", sample_seed=request.rng_seed,
                        output_tokens=result.output_tokens, wall_seconds=0.)
        if child.version_id in self._events:
            raise ValueError("duplicate repair version")
        self._count += 1
        self.accounting["actor_decodes"] += 1
        self.accounting["output_tokens"] += result.output_tokens
        self.candidates.append(child)
        self._pending = child
        completion = {"algorithm": self.name, "repair_index": self._count,
            "selected_parent": parent.version_id, "child_version_id": child.version_id,
            "particle_index": request.condition.particle_index if request.condition else None}
        if self.name == "rex":
            completion.pop("algorithm")
            self.decisions[-1].update(completion)
        else:
            self.decisions.append(completion)
        return child

    def observe(self, version_id, events):
        if self._pending is None or self._pending.version_id != version_id:
            raise ValueError("feedback owner must be the current pending repair")
        candidate = self._pending
        rows = self._validate_feedback(candidate, events)
        if self.belief is not None:
            self.belief.spawn_child(candidate.parent_version_id, candidate, rows[0])
            for row in rows[1:]:
                self.belief.observe(version_id, row)
        self._events[version_id] = rows
        self._failed_attempts[version_id] = 0
        if any(row.outcome != "PASS" for row in rows):
            self._failed_attempts[candidate.parent_version_id] += 1
        self._debug_parent = candidate
        self._pending = None
        self.accounting["verifier_cases"] += len(rows)
        self.accounting["verifier_suites"] += 1
        self._sync_belief()

    def finish(self):
        expected = 0 if self.name == "no_repair" else FORMAL_BUDGET["repair_decodes"]
        if self._count != expected or self._pending is not None:
            raise ValueError("finish requires all four completed repairs and their visible feedback")
        if self.selected_deterministic is not None:
            self.selected_deterministic.validate(self.belief)
        selected = self._visible_best(self.candidates)
        return RepairSelection((self.task.protocol, self.bank.candidates[0].model_id, self.task.task_id, self.seed, self.name),
            self._initial_selected.source_hash, selected.source_hash, selected.version_id, self.bank.content_hash,
            "lower_compute_reference" if self.name == "no_repair" else "matched_four_decodes",
            dict(self.provenance), dict(self.accounting))
