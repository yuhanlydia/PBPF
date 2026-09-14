"""Learned deterministic memories and candidate-specific neural particle arms.

All features are injected semantic content tensors. Identifiers seed random
streams but are never neural inputs. Every learned Stage-B parameter (including
frozen weights) is counted; shared frozen encoders/actors and Stage-C projectors
are not children of the counted module. There are no padding parameters.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pbpf.belief.filter import BeliefStore
from pbpf.belief.types import ParticleSet
from pbpf.data.schema import PublicTask, PublicTest
from pbpf.registry import OUTCOMES
from .base import (ArmCandidate, BeliefArm, RepairCondition, belief_digest, callable_identity,
                   canonical, derived_seed, encoder_identity, freeze_public_task, freeze_public_test, source_hash, validate_event)
from .registry import FORMAL_BUDGET, PARAMETER_TOLERANCE, load_arm_config


MEMORIES = {"prior", "last", "full_transcript", "window_two", "window_four", "matched_gru", "deepsets", "scalar_correctness"}
PARTICLE_ARMS = {"pbpf", "map", "posterior_mean", "shuffled_evidence", "wrong_candidate", "masked_outcomes", "random_latent", "faulty_shared_belief"}


def learned_parameter_report(module):
    components, seen = {}, set()
    for name, parameter in module.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        component = name.split(".")[0]
        components[component] = components.get(component, 0) + parameter.numel()
    return {"total": sum(components.values()), "components": components,
            "denominator": "all_unique_learned_stage_b_parameters_including_frozen"}


class MemoryNetwork(nn.Module):
    """Trainable content memory plus a five-way future-test likelihood head."""
    def __init__(self, kind, feature_dim, latent_dim, width):
        super().__init__()
        if kind not in MEMORIES or min(feature_dim, latent_dim, width) < 1:
            raise ValueError("invalid memory kind or dimensions")
        self.kind, self.width = kind, width
        self.context = nn.Linear(2 * feature_dim, width)
        if kind == "matched_gru":
            self.memory = nn.GRUCell(feature_dim + 5, width)
        elif kind == "deepsets":
            self.memory = nn.Sequential(nn.Linear(feature_dim + 5, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        elif kind == "scalar_correctness":
            self.memory = nn.Linear(1, width)
        elif kind != "prior":
            self.memory = nn.RNNCell(feature_dim + 5, width)
        self.projection = nn.Linear(width, latent_dim)
        self.outcome_head = nn.Sequential(nn.Linear(3 * feature_dim + latent_dim, width), nn.Tanh(), nn.Linear(width, 5))

    @staticmethod
    def parameter_count(kind, f, d, h):
        base = (2*f + 1)*h + (h + 1)*d + (3*f + d + 1)*h + 5*h + 5
        if kind == "prior":
            return base
        if kind == "scalar_correctness":
            return base + 2*h
        if kind == "deepsets":
            return base + (f + 6)*h + (h + 1)*h
        return base + (3 if kind == "matched_gru" else 1)*((f + 5)*h + h*h + 2*h)

    def encode(self, task, candidate, evidence):
        state = torch.tanh(self.context(torch.cat([task, candidate])))
        window = {"last": 1, "window_two": 2, "window_four": 4}.get(self.kind)
        rows = evidence[-window:] if window else evidence
        if self.kind == "prior":
            rows = []
        if self.kind == "scalar_correctness":
            # Beta(1,1) posterior mean: only correctness, not exception identity.
            rate = (1 + sum(outcome == 0 for _, outcome in rows)) / (2 + len(rows))
            state = torch.tanh(state + self.memory(task.new_tensor([rate])))
        elif self.kind == "deepsets":
            if rows:
                values = [self.memory(torch.cat([test, F.one_hot(task.new_tensor(outcome, dtype=torch.long), 5).to(task)])) for test, outcome in rows]
                state = torch.tanh(state + torch.stack(values).mean(0))
        else:
            for test, outcome in rows:
                value = torch.cat([test, F.one_hot(task.new_tensor(outcome, dtype=torch.long), 5).to(task)])
                state = self.memory(value, state)
        return torch.tanh(self.projection(state))

    def forward(self, task, candidate, evidence, test):
        state = self.encode(task, candidate, evidence)
        return F.log_softmax(self.outcome_head(torch.cat([task, candidate, test, state])), -1)


def resolve_memory_width(kind, feature_dim, latent_dim, target_count, *, max_width=16384,
                         maximum_relative_error=PARAMETER_TOLERANCE):
    if kind not in MEMORIES or not isinstance(target_count, int) or target_count <= 0:
        raise ValueError("valid memory and positive parameter denominator required")
    if not math.isfinite(maximum_relative_error) or not 0 < maximum_relative_error <= PARAMETER_TOLERANCE:
        raise ValueError("parameter tolerance must be positive and at most five percent")
    # Integer search is deterministic. Tie-break is the smaller width.
    error, width = min((abs(MemoryNetwork.parameter_count(kind, feature_dim, latent_dim, h) / target_count - 1), h)
                       for h in range(1, max_width + 1))
    if error > maximum_relative_error:
        raise ValueError("no integer memory width matches the parameter tolerance (at most five percent)")
    with torch.random.fork_rng():
        report = learned_parameter_report(MemoryNetwork(kind, feature_dim, latent_dim, width))
    return dict(report, width=width, target_count=target_count, relative_error=error,
                maximum_relative_error=maximum_relative_error)


class PredictionArm(BeliefArm):
    def __init__(self, name, *, model, feature_encoder, seed, particles=8, projector=None, ess_fraction=.5):
        if name not in MEMORIES | PARTICLE_ARMS or particles < 1:
            raise ValueError("unknown prediction arm or particle count")
        self.name, self.model, self.feature_encoder = name, model, feature_encoder
        self.seed, self.particles, self.projector, self.ess_fraction = seed, particles, projector, ess_fraction
        self._tasks, self._candidates, self._events, self._states, self._parents = {}, {}, {}, {}, {}
        self._condition_calls = {}
        self._condition_schedules, self._generation_records = {}, {}
        self._donors = {}
        self._donor_evidence = {}
        self._operations, self._record_suppressed = [], False
        self._encoder_identity = encoder_identity(feature_encoder)
        self.accounting = {"actor_decodes": 0, "belief_forwards": 0}
        self.parameter_report = learned_parameter_report(model)

    def behavior_identity(self):
        return {"schema": "pbpf-belief-behavior-v2", "name": self.name,
            "seed": self.seed, "particles": self.particles, "ess_fraction": self.ess_fraction,
            "outcomes": list(OUTCOMES), "parameter_report": self.parameter_report,
            "model_class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            "model_architecture": repr(self.model),
            "model_config": {key: getattr(self.model, key, None) for key in ("kind", "width", "feature_dim", "latent_dim")},
            "model_methods": {key: callable_identity(getattr(self.model, key)) for key in
                              ("forward", "encode", "root", "transition", "proposal", "likelihood") if hasattr(self.model, key)},
            "features": encoder_identity(self.feature_encoder),
            "arm_methods": {key: callable_identity(getattr(self, key)) for key in
                            ("_feature", "_semantic_evidence", "_refresh", "predict", "condition", "_latent")},
            "projector": None if self.projector is None else {"architecture": repr(self.projector),
                "forward": callable_identity(self.projector.forward)}}

    def _assert_encoder(self):
        if encoder_identity(self.feature_encoder) != self._encoder_identity:
            raise ValueError("frozen semantic encoder configuration/checkpoint changed")

    def _record(self, kind, key, **values):
        if not self._record_suppressed:
            self._operations.append({"kind": kind, "key": key, **copy.deepcopy(values)})

    def _event_record(self, event):
        row = asdict(event)
        if self.name == "masked_outcomes":
            row["outcome"], row["feedback"] = None, ""
        return row

    def _feature(self, text):
        feature = self.feature_encoder(str(text))
        parameter = next(self.model.parameters())
        feature = torch.as_tensor(feature, device=parameter.device, dtype=parameter.dtype).detach()
        if feature.ndim != 1 or not torch.isfinite(feature).all():
            raise ValueError("semantic encoder must return a finite vector")
        return feature

    def _forward(self, method, *args, **kwargs):
        self.accounting["belief_forwards"] += 1
        return method(*args, **kwargs)

    def initialize(self, task, candidate):
        self._assert_encoder()
        if type(task) is not PublicTask or type(candidate) is not ArmCandidate:
            raise TypeError("generator arms accept only public typed records")
        task = freeze_public_task(task)
        if task.task_id != candidate.task_id:
            raise ValueError("candidate task mismatch")
        key = candidate.version_id
        if candidate.round > 0 and key not in self._parents:
            raise ValueError("child versions require spawn_child with explicit selected parent")
        if key in self._candidates:
            raise ValueError("candidate version already initialized")
        self._tasks[key], self._candidates[key], self._events[key] = task, candidate, []
        self._donor_evidence[key] = []
        self._condition_calls[key] = 0
        self._generation_records[key] = []
        self._refresh(key)
        self._record("initialize", key)

    def observe(self, version_id, event):
        self._assert_encoder()
        rows = self._events[version_id]
        validate_event(self._tasks[version_id], self._candidates[version_id], event, len(rows))
        if any(row["event_id"] == event.event_id for row in rows):
            raise ValueError("duplicate event ID")
        row = asdict(event)
        if self.name == "wrong_candidate":
            self._donor_evidence[version_id].append(copy.deepcopy(row))
            donor = self._donors.setdefault(version_id, self._donor(version_id))
            available = self._donor_evidence[donor] if donor else []
            source = available[len(rows)] if len(rows) < len(available) else {"outcome": None, "feedback": ""}
            row["outcome"], row["feedback"] = source["outcome"], source["feedback"]
        if self.name == "masked_outcomes":
            row["outcome"], row["feedback"] = None, ""
        rows.append(row)
        self._refresh(version_id)
        if self.name == "faulty_shared_belief":
            for key in self._candidates:
                if key != version_id and self._candidates[key].task_id == event.task_id:
                    self._refresh(key)
        self._record("observe", version_id, event=self._event_record(event))

    def spawn_child(self, parent_version_id, candidate, first_event):
        self._assert_encoder()
        parent = self._candidates[parent_version_id]
        if candidate.task_id != parent.task_id:
            raise ValueError("child must remain within its parent task")
        if (candidate.parent_version_id != parent_version_id or candidate.round != parent.round + 1
                or candidate.origin != "repair"):
            raise ValueError("child must name the selected parent and next round")
        validate_event(self._tasks[parent_version_id], candidate, first_event, 0)
        key = candidate.version_id
        if key in self._candidates:
            raise ValueError("child already exists")
        self._parents[key] = (copy.deepcopy(self._states[parent_version_id]), parent)
        self._record_suppressed = True
        try:
            self.initialize(self._tasks[parent_version_id], candidate)
            self.observe(key, first_event)
        finally:
            self._record_suppressed = False
        self._record("spawn_child", key, parent=parent_version_id, event=self._event_record(first_event))

    def _donor(self, key):
        candidate = self._candidates[key]
        choices = sorted(k for k, other in self._candidates.items()
                         if k != key and other.task_id == candidate.task_id and other.round == candidate.round)
        return choices[0] if choices else None

    def evidence(self, key):
        rows = copy.deepcopy(self._events[key])
        if self.name == "shuffled_evidence" and rows:
            rng = np.random.default_rng(derived_seed(self.seed, self._candidates[key].task_id, "shuffle"))
            order = rng.permutation(len(rows))
            # Do not silently leave the causal control unchanged for n > 1.
            if len(rows) > 1 and np.array_equal(order, np.arange(len(rows))):
                order = np.roll(order, 1)
            original = copy.deepcopy(rows)
            for row, index in zip(rows, order):
                row["outcome"], row["feedback"] = original[index]["outcome"], original[index]["feedback"]
        elif self.name == "faulty_shared_belief":
            peers = [k for k, c in self._candidates.items() if c.task_id == self._candidates[key].task_id]
            for peer in peers:
                for i, row in enumerate(self._events[peer]):
                    if i >= len(rows):
                        rows.append(copy.deepcopy(row))
                    else:
                        rows[i]["outcome"], rows[i]["feedback"] = row["outcome"], row["feedback"]
        return rows

    def _semantic_evidence(self, key):
        task = self._tasks[key]
        return [(self._feature(task.visible_tests[row["ordinal"]].source or ""), OUTCOMES.index(row["outcome"]))
                for row in self.evidence(key) if row["outcome"] is not None]

    def _refresh(self, key):
        if self.name in MEMORIES:
            self._states[key] = {}
            return
        task, candidate = self._tasks[key], self._candidates[key]
        rng = np.random.default_rng(derived_seed(self.seed, task.task_id, key, "filter"))
        store = BeliefStore(rng=rng, ess_fraction=self.ess_fraction)
        t, c = self._feature(task.task_text)[None], self._feature(candidate.source)[None]
        rows = self._semantic_evidence(key)
        parent_entry = self._parents.get(key)
        with torch.no_grad():
            if parent_entry:
                parent_state, parent_candidate = parent_entry
                pz = t.new_tensor(parent_state["particles"])[None]
                diff = self._feature(candidate.source + "\nPREVIOUS\n" + parent_candidate.source)[None]
                prior = self._forward(self.model.transition, pz, t, c, diff)
            else:
                prior = self._forward(self.model.root, t, c)
            if rows:
                test, outcome = rows[0]
                kwargs = {"parent_z": pz, "diff": diff} if parent_entry else {}
                proposal = self._forward(self.model.proposal, t, c, test[None], t.new_tensor([outcome], dtype=torch.long), **kwargs)
            else:
                proposal = prior
            shape = (1, self.particles, self.model.latent_dim)
            noise = t.new_tensor(rng.normal(size=shape))
            if proposal.mean.ndim == 2:
                from pbpf.belief.model import GaussianParams
                proposal = GaussianParams(proposal.mean[:, None].expand(shape), proposal.log_std[:, None].expand(shape))
            z = proposal.rsample(noise=noise)
            lp, lq = prior.log_prob(z)[0].cpu().numpy(), proposal.log_prob(z)[0].cpu().numpy()
            likelihood = np.zeros(self.particles)
            if rows:
                likelihood = self._forward(self.model.likelihood, z, t, c, rows[0][0][None])[0, :, rows[0][1]].cpu().numpy()
            if parent_entry:
                parent_key = parent_candidate.version_id
                # The core has no public import method. Restore one fully
                # validated immutable ParticleSet, not a new zero-evidence root.
                store._states[parent_key] = ParticleSet(parent_key, parent_state["parent_hash"],
                    parent_state["particles"], parent_state["log_weights"], parent_state["ancestors"],
                    parent_state["step"], parent_state["log_evidence"])
                step = store.spawn_child(key, parent_key, ancestor_scheme="deterministic_enumeration",
                    sampled_z=z[0].cpu().numpy(), ancestors=np.arange(self.particles), log_transition=lp,
                    log_proposal=lq, log_likelihood=likelihood)
            else:
                step = store.initialize_root(key, z[0].cpu().numpy(), log_prior=lp, log_proposal=lq,
                                             log_likelihood=likelihood if rows else None)
        telemetry = []
        for index in range(max(1, len(rows))):
            if index:
                state = store.snapshot(key)
                with torch.no_grad():
                    ll = self._forward(self.model.likelihood, t.new_tensor(state.z)[None], t, c, rows[index][0][None])[0, :, rows[index][1]].cpu().numpy()
                step = store.observe(key, ll)
            if step.resampled:
                # Full current latent target; MALA uses an MH correction, not jitter.
                def target(values, gradient=False):
                    zz = t.new_tensor(values).requires_grad_(gradient)
                    if parent_entry:
                        # Child marginal prior sums the selected parent's mixture.
                        all_prior = prior.log_prob(zz[:, None, :])
                        density = torch.logsumexp(all_prior + t.new_tensor(parent_state["log_weights"])[None], -1)
                    else:
                        density = prior.log_prob(zz)
                    for test, outcome in rows[:index + 1]:
                        density = density + self._forward(self.model.likelihood, zz, t.expand(len(zz), -1), c.expand(len(zz), -1), test.expand(len(zz), -1))[:, outcome]
                    return (torch.autograd.grad(density.sum(), zz)[0] if gradient else density).detach().cpu().numpy()
                move = store.mala_move(key, log_density=target, gradient=lambda z: target(z, True), step_size=.05)
                acceptance = move.acceptance_rate
            else:
                acceptance = None
            telemetry.append({"log_normalizer": step.log_normalizer, "ess": step.ess,
                "resampled": step.resampled, "mala_acceptance": acceptance,
                "normalized_entropy": step.normalized_entropy, "unique_ancestor_count": step.unique_ancestor_count,
                "resampling_indices": step.resampling_indices.tolist(),
                "normalized_log_weights": step.normalized_log_weights.tolist()})
        state = store.snapshot(key)
        self._states[key] = {"particles": state.z.tolist(), "log_weights": state.log_weights.tolist(),
                            "log_evidence": state.log_evidence, "filter_steps": telemetry,
                            "proposal_noise": noise[0].tolist(), "parent_hash": state.parent_hash,
                            "ancestors": state.ancestors.tolist(), "step": state.step}

    def _latent(self, key, *, sample):
        state = self._states[key]
        z, weights = np.asarray(state["particles"]), np.exp(state["log_weights"])
        if self.name == "map":
            index = int(weights.argmax())
            return z[index], index
        if self.name == "posterior_mean":
            return weights @ z, None
        rng = np.random.default_rng(derived_seed(self.seed, key, "condition", str(self._condition_calls[key])))
        if self.name == "random_latent":
            direction = rng.normal(size=z.shape[1])
            radius = np.sqrt(np.sum(weights * np.square(z).sum(1)))
            return direction * radius / np.linalg.norm(direction), None
        fingerprint = source_hash(canonical({"particles": state["particles"], "log_weights": state["log_weights"]}))
        schedule = self._condition_schedules.get(key)
        block_size = FORMAL_BUDGET["repair_decodes"]
        if schedule is None or schedule["posterior_fingerprint"] != fingerprint:
            schedule = {"posterior_fingerprint": fingerprint, "block": 0, "cursor": block_size}
        elif schedule["cursor"] == block_size:
            schedule = dict(schedule, block=schedule["block"] + 1)
        if schedule["cursor"] == block_size:
            schedule_seed = derived_seed(self.seed, key, fingerprint, "systematic-condition", str(schedule["block"]))
            block_rng = np.random.default_rng(schedule_seed)
            offset = float(block_rng.random())
            cdf = (weights / weights.sum()).cumsum()
            cdf[-1] = 1.
            indices = np.searchsorted(cdf, (offset + np.arange(block_size)) / block_size, side="right")
            # Randomize the order of the systematic strata so each adaptive
            # call is marginally posterior-distributed, not always stratum 0.
            indices = block_rng.permutation(indices)
            schedule = dict(schedule, cursor=0, indices=indices.tolist(), seed=schedule_seed, offset=offset)
        index = int(schedule["indices"][schedule["cursor"]])
        if sample:
            self._condition_schedules[key] = dict(schedule, cursor=schedule["cursor"] + 1)
        return z[index], index

    def predict(self, version_id, test):
        self._assert_encoder()
        if type(test) is not PublicTest:
            raise TypeError("prediction requires an outcome-free public test descriptor")
        test = freeze_public_test(test)
        t = self._feature(self._tasks[version_id].task_text)
        c = self._feature(self._candidates[version_id].source)
        f = self._feature(test.source or "")
        with torch.no_grad():
            if self.name in MEMORIES:
                probabilities = self._forward(self.model, t, c, self._semantic_evidence(version_id), f).exp()
            elif self.name in {"map", "posterior_mean", "random_latent"}:
                z, _ = self._latent(version_id, sample=False)
                probabilities = self._forward(self.model.likelihood, t.new_tensor(z)[None], t[None], c[None], f[None])[0].exp()
            else:
                state = self._states[version_id]
                z = t.new_tensor(state["particles"])
                weights = t.new_tensor(state["log_weights"]).exp()
                probabilities = (weights[:, None] * self._forward(self.model.likelihood, z, t.expand(len(z), -1), c.expand(len(z), -1), f.expand(len(z), -1)).exp()).sum(0)
        values = probabilities.double().cpu().numpy()
        if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-5):
            raise ValueError("model did not produce a normalized five-way distribution")
        result = dict(zip(OUTCOMES, (values / values.sum()).tolist()))
        self._record("predict", version_id, test=asdict(test))
        return result

    def condition(self, version_id):
        self._assert_encoder()
        t = self._feature(self._tasks[version_id].task_text)
        c = self._feature(self._candidates[version_id].source)
        with torch.no_grad():
            if self.name in MEMORIES:
                latent = self._forward(self.model.encode, t, c, self._semantic_evidence(version_id)).cpu().numpy()
                index = None
            else:
                latent, index = self._latent(version_id, sample=True)
            prefix = None
            if self.projector is not None:
                self.accounting["belief_forwards"] += 1
                value = self.projector(t.new_tensor(latent)[None])[0].detach().cpu().tolist()
                prefix = tuple(tuple(float(x) for x in row) for row in value)
        result = RepairCondition(version_id, canonical(self.evidence(version_id)), tuple(float(x) for x in latent), index, prefix)
        self._condition_calls[version_id] += 1
        self._generation_records[version_id].append({"call": self._condition_calls[version_id],
            "particle_index": index, "latent": list(result.latent),
            "soft_prefix": None if prefix is None else [list(row) for row in prefix],
            "posterior_fingerprint": self._condition_schedules.get(version_id, {}).get("posterior_fingerprint")})
        self._record("condition", version_id)
        return result

    def snapshot(self, version_id):
        return copy.deepcopy({"arm": self.name, "version_id": version_id,
            "source_hash": self._candidates[version_id].source_hash,
            "event_ids": [row["event_id"] for row in self._events[version_id]],
            "evidence": self.evidence(version_id),
            "control_seed": derived_seed(self.seed, self._candidates[version_id].task_id, "shuffle") if self.name == "shuffled_evidence" else None,
            "donor_version_id": self._donors.get(version_id) if self.name == "wrong_candidate" else None,
            "condition_cursor": self._condition_calls[version_id],
            "condition_schedule": self._condition_schedules.get(version_id),
            "generation_records": self._generation_records[version_id],
            **self._states[version_id]})

    def serialize(self):
        """Alias-free JSON checkpoint, with modules supplied separately on restore."""
        self._assert_encoder()
        payload = {"schema": "pbpf-belief-checkpoint-v3", "identity": belief_digest(self),
            "tasks": {key: asdict(task) for key, task in self._tasks.items()},
            "candidates": {key: asdict(candidate) for key, candidate in self._candidates.items()},
            "events": self._events, "states": self._states,
            "parents": {key: [state, asdict(candidate)] for key, (state, candidate) in self._parents.items()},
            "condition_calls": self._condition_calls, "condition_schedules": self._condition_schedules,
            "generation_records": self._generation_records, "donors": self._donors,
            "donor_evidence": self._donor_evidence, "accounting": self.accounting, "operations": self._operations}
        return canonical({"payload": payload, "sha256": source_hash(canonical(payload))})

    def restore(self, serialized):
        """Validate an untrusted JSON checkpoint, then atomically publish replay.

        Checksums authenticate no semantics. Strict schemas are checked before
        an isolated belief-only operation replay verifies lineage, donor timing,
        posterior fingerprints, component schedules and accounting. No actor is
        invoked. Validation forwards are exposed separately as last_restore_work.
        """
        from .checkpoint import checked_payload, replay_checkpoint
        self.last_restore_work = {"belief_forwards": 0, "actor_decodes": 0}
        self._assert_encoder()
        try:
            payload, candidates, tasks = checked_payload(serialized, self)
            shadow = replay_checkpoint(payload, candidates, tasks, self, work=self.last_restore_work)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            reason = str(error) if str(error).startswith("checkpoint ") else type(error).__name__
            raise ValueError(f"checkpoint restore validation failed: {reason}") from error
        for field in ("_tasks", "_candidates", "_states", "_parents", "_events", "_condition_calls",
                      "_condition_schedules", "_generation_records", "_donors", "_donor_evidence", "_operations", "accounting"):
            setattr(self, field, getattr(shadow, field))


def make_belief_arm(name, *, model, feature_encoder, seed=1701, particles=8, projector=None, ess_fraction=.5,
                   arm_config=None):
    config = load_arm_config(arm_config)
    if name not in config["prediction"]:
        raise ValueError("prediction arm absent from resolved configuration")
    if name in MEMORIES:
        target = learned_parameter_report(model)["total"]
        report = resolve_memory_width(name, model.feature_dim, model.latent_dim, target,
            maximum_relative_error=config["parameter_matching"]["maximum_relative_error"])
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            model = MemoryNetwork(name, model.feature_dim, model.latent_dim, report["width"])
    else:
        report = learned_parameter_report(model)
    arm = PredictionArm(name, model=model, feature_encoder=feature_encoder, seed=seed,
                        particles=particles, projector=projector, ess_fraction=ess_fraction)
    arm.parameter_report = report
    return arm
