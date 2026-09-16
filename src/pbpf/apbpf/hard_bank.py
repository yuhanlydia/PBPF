"""Audits for diagnostic candidate banks with evaluator-hidden labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


def _binary(values: Any, *, name: str, dimensions: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != dimensions or array.size == 0 or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must be a nonempty binary array with {dimensions} dimensions")
    return array.astype(np.int8, copy=False)


def _ordered_group_ids(values: Sequence[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("group_ids must be an ordered sequence of nonempty strings")
    ids = tuple(values)
    if not ids or any(not isinstance(group_id, str) or not group_id for group_id in ids):
        raise ValueError("group_ids must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("group_ids must be unique")
    return ids


def _visible_outcomes_digest(outcomes: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": outcomes.shape, "dtype": "int8"}, sort_keys=True).encode("utf-8"))
    digest.update(outcomes.astype(np.int8, copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _candidate_inventory_digest(candidate_ids: Any, *, groups: int, candidates: int) -> str | None:
    if candidate_ids is None:
        return None
    if isinstance(candidate_ids, (str, bytes)):
        raise TypeError("candidate_ids must be an ordered group-by-candidate sequence")
    rows = tuple(tuple(row) for row in candidate_ids)
    if (len(rows) != groups or any(len(row) != candidates for row in rows)
            or any(any(not isinstance(value, str) or not value for value in row) for row in rows)
            or any(len(set(row)) != len(row) for row in rows)):
        raise ValueError("candidate_ids must bind unique ordered candidates for every group")
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def _population_digest(
    group_ids: tuple[str, ...], split: str, provenance: str, visible_outcomes_digest: str,
    candidate_inventory_digest: str | None = None,
    source_component_ids: tuple[str, ...] | None = None,
) -> str:
    payload = {
        "schema": "apbpf-hard-bank-population-v2",
        "group_ids": group_ids,
        "split": split,
        "provenance": provenance,
        "visible_outcomes_digest": visible_outcomes_digest,
        "candidate_inventory_digest": candidate_inventory_digest,
        "source_component_ids": source_component_ids,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HardBankPopulationLock:
    """Pre-hidden-label binding for the exact primary bank population."""

    group_ids: tuple[str, ...]
    split: str
    provenance: str
    visible_outcomes_digest: str
    digest: str
    candidate_inventory_digest: str | None = None
    source_component_ids: tuple[str, ...] | None = None


def lock_hard_bank_population(
    group_ids: Sequence[Any], visible_outcomes: Any, *, split: str, provenance: str,
    candidate_ids: Any = None, source_component_ids: Sequence[Any] | None = None,
) -> HardBankPopulationLock:
    """Seal ordered primary-group IDs and visible outcomes before hidden labels mount."""
    ids = _ordered_group_ids(group_ids)
    outcomes = _binary(visible_outcomes, name="visible_outcomes", dimensions=3)
    if len(ids) != outcomes.shape[0]:
        raise ValueError("group_ids must cover the full visible-outcome population")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a nonempty string")
    if not isinstance(provenance, str) or not provenance:
        raise ValueError("provenance must be a nonempty string")
    outcomes_digest = _visible_outcomes_digest(outcomes)
    candidate_digest = _candidate_inventory_digest(
        candidate_ids, groups=len(ids), candidates=outcomes.shape[1])
    sources = None if source_component_ids is None else _ordered_group_ids(source_component_ids)
    if sources is not None and len(sources) != len(ids):
        raise ValueError("source_component_ids must cover the full locked population")
    return HardBankPopulationLock(
        group_ids=ids,
        split=split,
        provenance=provenance,
        visible_outcomes_digest=outcomes_digest,
        digest=_population_digest(ids, split, provenance, outcomes_digest, candidate_digest, sources),
        candidate_inventory_digest=candidate_digest,
        source_component_ids=sources,
    )


@dataclass(frozen=True)
class HardBankAudit:
    """Full-population visibility-versus-oracle audit for a candidate bank."""

    groups: int
    candidates_per_group: int
    visible_tests: int
    all_fail_groups: int
    all_pass_groups: int
    mixed_groups: int
    visible_score_tie_groups: int
    visible_selected_pass_at_1: float
    hidden_oracle_pass_at_k: float
    headroom: float
    saturated: bool
    population_digest: str
    split: str
    provenance: str

    @property
    def has_headroom(self) -> bool:
        return not self.saturated


def audit_hard_bank(
    visible_outcomes: Any,
    hidden_labels: Any,
    *,
    group_ids: Sequence[Any],
    population_lock: HardBankPopulationLock,
    candidate_ids: Any = None,
    source_component_ids: Sequence[Any] | None = None,
) -> HardBankAudit:
    """Report full-bank visible-baseline headroom against hidden oracle labels.

    The lock must have been made before hidden labels are available. It binds the
    exact ordered group IDs and visible-outcome bytes, preventing a caller from
    presenting an already hidden-label-filtered subset as the primary bank.
    Visible pass-rate selection is deterministic (the first maximum wins) and
    is computed before indexing ``hidden_labels``.  Hidden labels are used only
    to score that already-selected candidate and to calculate the oracle
    ceiling; no group is filtered from the primary population.
    """
    outcomes = _binary(visible_outcomes, name="visible_outcomes", dimensions=3)
    labels = _binary(hidden_labels, name="hidden_labels", dimensions=2)
    if not isinstance(population_lock, HardBankPopulationLock):
        raise TypeError("audit requires a HardBankPopulationLock")
    if population_lock.digest != _population_digest(
        population_lock.group_ids,
        population_lock.split,
        population_lock.provenance,
        population_lock.visible_outcomes_digest,
        population_lock.candidate_inventory_digest,
        population_lock.source_component_ids,
    ):
        raise ValueError("population lock digest is invalid")
    ids = _ordered_group_ids(group_ids)
    if len(ids) != len(population_lock.group_ids):
        raise ValueError("audit population does not match the locked population")
    if ids != population_lock.group_ids:
        raise ValueError("audit group IDs do not match the locked ordered population")
    if _visible_outcomes_digest(outcomes) != population_lock.visible_outcomes_digest:
        raise ValueError("audit visible outcomes do not match the locked population")
    candidate_digest = _candidate_inventory_digest(
        candidate_ids, groups=len(ids), candidates=outcomes.shape[1])
    if candidate_digest != population_lock.candidate_inventory_digest:
        raise ValueError("audit candidate inventory does not match the pre-hidden population lock")
    sources = None if source_component_ids is None else _ordered_group_ids(source_component_ids)
    if sources != population_lock.source_component_ids:
        raise ValueError("audit source-component inventory does not match the pre-hidden population lock")
    if population_lock.split not in {"primary", "test"}:
        raise ValueError("primary audit requires a pre-hidden primary or test population lock")
    if outcomes.shape[:2] != labels.shape or outcomes.shape[1] < 2 or outcomes.shape[2] < 1:
        raise ValueError("outcomes and labels require [groups, candidates, visible_tests] shapes")

    visible_scores = outcomes.mean(axis=2)
    selected_indices = visible_scores.argmax(axis=1)
    selected_labels = labels[np.arange(labels.shape[0]), selected_indices]
    oracle_labels = labels.max(axis=1)
    visible_selected = float(selected_labels.mean())
    oracle = float(oracle_labels.mean())
    headroom = float(oracle - visible_selected)
    positives = labels.sum(axis=1)
    return HardBankAudit(
        groups=int(labels.shape[0]),
        candidates_per_group=int(labels.shape[1]),
        visible_tests=int(outcomes.shape[2]),
        all_fail_groups=int((positives == 0).sum()),
        all_pass_groups=int((positives == labels.shape[1]).sum()),
        mixed_groups=int(((positives > 0) & (positives < labels.shape[1])).sum()),
        visible_score_tie_groups=int(
            (np.isclose(visible_scores, visible_scores.max(axis=1, keepdims=True), atol=0.0, rtol=0.0).sum(axis=1) > 1).sum()
        ),
        visible_selected_pass_at_1=visible_selected,
        hidden_oracle_pass_at_k=oracle,
        headroom=headroom,
        saturated=bool(np.isclose(headroom, 0.0, atol=1e-12, rtol=0.0)),
        population_digest=population_lock.digest,
        split=population_lock.split,
        provenance=population_lock.provenance,
    )


def select_development_eligible_groups(
    records: Sequence[Mapping[str, Any]], minimum_groups: int
) -> tuple[Mapping[str, Any], ...]:
    """Return mixed development groups; reject primary/test hidden-label filtering.

    The evaluator label may be used to build or tune a *development* hard bank,
    where its use is explicit.  Passing a test/primary record fails rather than
    silently deriving a favorable evaluation subset from hidden labels.
    """
    if isinstance(minimum_groups, bool) or not isinstance(minimum_groups, (int, np.integer)):
        raise TypeError("minimum_groups must be an integer")
    if int(minimum_groups) < 1:
        raise ValueError("minimum_groups must be positive")

    eligible: list[Mapping[str, Any]] = []
    seen_ids: set[Any] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("development records must be mappings")
        if record.get("split") != "development":
            raise ValueError("hidden-label eligibility filtering is allowed only for development groups")
        if "group_id" not in record or "hidden_labels" not in record:
            raise ValueError("development records require group_id and hidden_labels")
        group_id = record["group_id"]
        if group_id in seen_ids:
            raise ValueError("development group IDs must be unique")
        seen_ids.add(group_id)
        labels = _binary(record["hidden_labels"], name="record.hidden_labels", dimensions=1)
        if labels.size < 2:
            raise ValueError("development groups require at least two candidates")
        if 0 < int(labels.sum()) < labels.size:
            eligible.append(record)
    if len(eligible) < int(minimum_groups):
        raise ValueError("insufficient mixed development groups")
    return tuple(eligible)
