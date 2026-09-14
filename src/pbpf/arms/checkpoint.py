"""Untrusted arm checkpoint boundary: exact schemas, typed ownership, replay.

The content checksum proves only bytes. Replaying the public operation journal
in a private arm proves that the claimed state, donor timing, lineage and
conditioning schedule can arise through the same validated public interface.
"""
from dataclasses import asdict, fields
import copy
import json
import math
import re

from pbpf.belief.types import ParticleSet
from pbpf.data.schema import PublicTask, PublicTest
from pbpf.registry import OUTCOMES
from .base import (ArmCandidate, VisibleEvent, _public_json, belief_digest, canonical,
                   freeze_public_task, freeze_public_test, source_hash, validate_event)
from .registry import FORMAL_BUDGET


def exact(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("checkpoint schema has unknown or missing fields")


def integer(value, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        raise ValueError("checkpoint integer/cursor/index outside permitted range")


def number(value, minimum=None, maximum=None):
    if (type(value) not in (float, int) or not math.isfinite(value)
            or minimum is not None and value < minimum or maximum is not None and value > maximum):
        raise ValueError("checkpoint numeric field must be finite and in range")


def sha(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("checkpoint identity/fingerprint must be lowercase SHA-256")


def array(value, size=None):
    if type(value) is not list or size is not None and len(value) != size:
        raise ValueError("checkpoint array shape mismatch")


def vector(value, size):
    array(value, size)
    for item in value:
        number(item)


def matrix(value, rows, columns):
    array(value, rows)
    for row in value:
        vector(row, columns)


def test_record(value):
    exact(value, (field.name for field in fields(PublicTest)))
    if value["source"] is not None and type(value["source"]) is not str:
        raise ValueError("checkpoint public test source must be canonical immutable text")
    return freeze_public_test(PublicTest(**value))


def event_record(row, task, candidate, *, nullable=False):
    exact(row, (field.name for field in fields(VisibleEvent)))
    for key in ("task_id", "version_id", "event_id", "test_id", "feedback"):
        if type(row[key]) is not str:
            raise ValueError("checkpoint event text must be strings")
    integer(row["ordinal"])
    if row["outcome"] is None:
        if not nullable or row["feedback"] != "":
            raise ValueError("checkpoint null event is legal only for an outcome-masked control")
        values = dict(row, outcome="PASS")
    else:
        if row["outcome"] not in OUTCOMES:
            raise ValueError("checkpoint event has invalid outcome")
        values = row
    event = VisibleEvent(**values)
    validate_event(task, candidate, event, row["ordinal"])
    return event


def particle_record(state, key, candidate, arm):
    from .prediction import MEMORIES
    if arm.name in MEMORIES:
        exact(state, ())
        return
    exact(state, ("particles", "log_weights", "log_evidence", "filter_steps", "proposal_noise", "parent_hash", "ancestors", "step"))
    count, dimension = arm.particles, arm.model.latent_dim
    matrix(state["particles"], count, dimension)
    matrix(state["proposal_noise"], count, dimension)
    vector(state["log_weights"], count)
    array(state["ancestors"], count)
    for ancestor in state["ancestors"]:
        integer(ancestor, maximum=count - 1)
    integer(state["step"])
    number(state["log_evidence"])
    if state["parent_hash"] != candidate.parent_version_id:
        raise ValueError("checkpoint parent pointer differs from owned version lineage")
    ParticleSet(key, state["parent_hash"], state["particles"], state["log_weights"], state["ancestors"], state["step"], state["log_evidence"])
    array(state["filter_steps"])
    if not state["filter_steps"]:
        raise ValueError("checkpoint filter diagnostics are missing")
    for step in state["filter_steps"]:
        exact(step, ("log_normalizer", "ess", "resampled", "mala_acceptance", "normalized_entropy",
                     "unique_ancestor_count", "resampling_indices", "normalized_log_weights"))
        number(step["log_normalizer"])
        number(step["ess"], 1 - 1e-10, count + 1e-10)
        number(step["normalized_entropy"], -1e-10, 1 + 1e-10)
        integer(step["unique_ancestor_count"], 1, count)
        if type(step["resampled"]) is not bool:
            raise ValueError("checkpoint resampling flag must be boolean")
        if step["mala_acceptance"] is not None:
            number(step["mala_acceptance"], 0, 1)
        array(step["resampling_indices"], count)
        for index in step["resampling_indices"]:
            integer(index, maximum=count - 1)
        vector(step["normalized_log_weights"], count)


def checked_payload(serialized, arm):
    envelope = json.loads(serialized)
    _public_json(envelope)  # rejects nonfinite numbers and all unsupported types
    exact(envelope, ("payload", "sha256"))
    payload = envelope["payload"]
    exact(payload, ("schema", "identity", "tasks", "candidates", "events", "states", "parents", "condition_calls",
                    "condition_schedules", "generation_records", "donors", "donor_evidence", "accounting", "operations"))
    if (payload["schema"] != "pbpf-belief-checkpoint-v3" or payload["identity"] != belief_digest(arm)
            or envelope["sha256"] != source_hash(canonical(payload))):
        raise ValueError("checkpoint identity or checksum mismatch")
    for key in ("tasks", "candidates", "events", "states", "parents", "condition_calls", "condition_schedules",
                "generation_records", "donors", "donor_evidence", "accounting"):
        if type(payload[key]) is not dict:
            raise ValueError("checkpoint inventory must be a mapping")
    candidates, tasks = {}, {}
    keys = set(payload["candidates"])
    for inventory in ("tasks", "events", "states", "condition_calls", "generation_records", "donor_evidence"):
        if set(payload[inventory]) != keys:
            raise ValueError("checkpoint owner inventory mismatch")
    for key, value in payload["candidates"].items():
        sha(key)
        exact(value, (field.name for field in fields(ArmCandidate)))
        for field in ("task_id", "source", "origin", "model_id", "model_revision", "prompt_revision"):
            if type(value[field]) is not str:
                raise ValueError("checkpoint candidate identity/source must be strings")
        for field in ("slot", "round", "sample_seed", "output_tokens"):
            integer(value[field])
        number(value["wall_seconds"], 0)
        if value["parent_version_id"] is not None:
            sha(value["parent_version_id"])
        candidate = ArmCandidate(**value)
        if candidate.version_id != key:
            raise ValueError("checkpoint candidate/version owner mismatch")
        candidates[key] = candidate
        value = payload["tasks"][key]
        exact(value, (field.name for field in fields(PublicTask)))
        array(value["visible_tests"])
        task = freeze_public_task(PublicTask(**dict(value, visible_tests=tuple(test_record(test) for test in value["visible_tests"]))))
        if task.task_id != candidate.task_id:
            raise ValueError("checkpoint task/candidate owner mismatch")
        tasks[key] = task
    expected_parents = {key for key, candidate in candidates.items() if candidate.parent_version_id is not None}
    if set(payload["parents"]) != expected_parents:
        raise ValueError("checkpoint parent inventory mismatch")
    for key in keys:
        candidate, task = candidates[key], tasks[key]
        for inventory in ("events", "donor_evidence"):
            rows = payload[inventory][key]
            array(rows)
            if inventory == "donor_evidence" and arm.name != "wrong_candidate" and rows:
                raise ValueError("checkpoint donor evidence is forbidden for this arm")
            for index, row in enumerate(rows):
                event_record(row, task, candidate, nullable=inventory == "events" and arm.name in {"masked_outcomes", "wrong_candidate"})
                if row["ordinal"] != index:
                    raise ValueError("checkpoint events violate ordered visible evidence")
            if len({row["event_id"] for row in rows}) != len(rows):
                raise ValueError("checkpoint event IDs must be unique")
        particle_record(payload["states"][key], key, candidate, arm)
        integer(payload["condition_calls"][key])
        records = payload["generation_records"][key]
        array(records, payload["condition_calls"][key])
        dimension = arm.model.projection.out_features if hasattr(arm.model, "projection") else arm.model.latent_dim
        for index, record in enumerate(records):
            exact(record, ("call", "particle_index", "latent", "soft_prefix", "posterior_fingerprint"))
            integer(record["call"], 1)
            if record["call"] != index + 1:
                raise ValueError("checkpoint generation cursor is inconsistent")
            if record["particle_index"] is not None:
                integer(record["particle_index"], maximum=arm.particles - 1)
            vector(record["latent"], dimension)
            if record["soft_prefix"] is not None:
                array(record["soft_prefix"])
                for row in record["soft_prefix"]:
                    array(row)
                    vector(row, len(row))
            if record["posterior_fingerprint"] is not None:
                sha(record["posterior_fingerprint"])
        if key in expected_parents:
            parent_key = candidate.parent_version_id
            if parent_key not in keys or candidates[parent_key].task_id != candidate.task_id or candidates[parent_key].round + 1 != candidate.round:
                raise ValueError("checkpoint parent lineage is invalid")
            parent = payload["parents"][key]
            array(parent, 2)
            if parent[1] != asdict(candidates[parent_key]):
                raise ValueError("checkpoint parent version identity mismatch")
            particle_record(parent[0], parent_key, candidates[parent_key], arm)
    if arm.name != "wrong_candidate" and payload["donors"]:
        raise ValueError("checkpoint donor mapping is forbidden for this arm")
    for key, donor in payload["donors"].items():
        if key not in keys or donor is not None and (donor not in keys or donor == key
                or candidates[donor].task_id != candidates[key].task_id or candidates[donor].round != candidates[key].round):
            raise ValueError("checkpoint donor must be a distinct same-task/same-round owner")
    for key, schedule in payload["condition_schedules"].items():
        if key not in keys:
            raise ValueError("checkpoint schedule owner is unknown")
        exact(schedule, ("posterior_fingerprint", "block", "cursor", "indices", "seed", "offset"))
        sha(schedule["posterior_fingerprint"])
        integer(schedule["block"])
        integer(schedule["cursor"], 1, FORMAL_BUDGET["repair_decodes"])
        integer(schedule["seed"], 0, 2**64 - 1)
        number(schedule["offset"], 0)
        if schedule["offset"] >= 1:
            raise ValueError("checkpoint systematic offset out of range")
        array(schedule["indices"], FORMAL_BUDGET["repair_decodes"])
        for index in schedule["indices"]:
            integer(index, maximum=arm.particles - 1)
    exact(payload["accounting"], ("actor_decodes", "belief_forwards"))
    integer(payload["accounting"]["belief_forwards"])
    if payload["accounting"]["actor_decodes"] != 0 or type(payload["accounting"]["actor_decodes"]) is not int:
        raise ValueError("checkpoint belief accounting cannot contain actor decodes")
    array(payload["operations"])
    for operation in payload["operations"]:
        if type(operation) is not dict or operation.get("kind") not in {"initialize", "observe", "spawn_child", "predict", "condition"}:
            raise ValueError("checkpoint operation schema is invalid")
        kind = operation["kind"]
        names = {"kind", "key"} | ({"parent", "event"} if kind == "spawn_child" else {"event"} if kind == "observe" else {"test"} if kind == "predict" else set())
        exact(operation, names)
        key = operation["key"]
        if key not in keys:
            raise ValueError("checkpoint operation owner is unknown")
        if kind in {"observe", "spawn_child"}:
            event_record(operation["event"], tasks[key], candidates[key], nullable=arm.name == "masked_outcomes")
        if kind == "spawn_child" and operation["parent"] != candidates[key].parent_version_id:
            raise ValueError("checkpoint operation parent mismatch")
        if kind == "predict":
            test_record(operation["test"])
    return payload, candidates, tasks


def replay_checkpoint(payload, candidates, tasks, original, *, work):
    from .prediction import PredictionArm
    shadow = PredictionArm(original.name, model=original.model, feature_encoder=original.feature_encoder,
        seed=original.seed, particles=original.particles, projector=original.projector, ess_fraction=original.ess_fraction)
    shadow.parameter_report = copy.deepcopy(original.parameter_report)
    try:
        for operation in payload["operations"]:
            key, kind = operation["key"], operation["kind"]
            if kind == "initialize":
                shadow.initialize(tasks[key], candidates[key])
            elif kind in {"observe", "spawn_child"}:
                row = dict(operation["event"])
                if row["outcome"] is None:
                    row["outcome"] = "PASS"  # masked before any semantic use
                event = VisibleEvent(**row)
                if kind == "observe":
                    shadow.observe(key, event)
                else:
                    shadow.spawn_child(operation["parent"], candidates[key], event)
            elif kind == "condition":
                shadow.condition(key)
            else:
                shadow.predict(key, PublicTest(**operation["test"]))
        # This recomputes posterior states, parent snapshots, cross-owner donor
        # timing, all schedules and cursor history rather than trusting claims.
        if json.loads(shadow.serialize())["payload"] != payload:
            raise ValueError("checkpoint state/schedule/donor/accounting differs from validated replay")
    finally:
        # Separate validation telemetry is retained even when a late comparison
        # rejects the checkpoint. Invalid experiment state is never published.
        work.update(belief_forwards=shadow.accounting["belief_forwards"], actor_decodes=0)
    return shadow
