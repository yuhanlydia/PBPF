from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    role: str


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    role: str


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    provenance_mode: str
    paper_url: str | None = None
    repository_url: str | None = None
    commit: str | None = None
    license_hash: str | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    profile: str
    formal: bool
    model: Mapping[str, Any]
    data: Mapping[str, Any]
    container: Mapping[str, Any]
    protocol: Mapping[str, Any]
    baseline: Mapping[str, Any] = field(default_factory=dict)
