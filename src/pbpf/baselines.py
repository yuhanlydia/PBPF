from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_baseline_catalog(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        catalog = yaml.safe_load(stream)
    if not isinstance(catalog, dict):
        raise ValueError("baseline catalog must be a mapping")
    return catalog


def require_available_baseline(name: str, path: str | Path) -> dict[str, Any]:
    catalog = load_baseline_catalog(path)
    if name not in catalog:
        raise ValueError(f"unknown baseline: {name}")
    entry = catalog[name]
    if entry.get("execution_enabled") is not True:
        reason = entry.get("unavailable_reason", "adapter is unavailable")
        raise ValueError(f"baseline {name} unavailable for execution: {reason}")
    return dict(entry)
