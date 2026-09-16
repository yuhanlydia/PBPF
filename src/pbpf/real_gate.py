"""Small, honest utilities for the real-data PBPF prediction gate.

The encoder hashes lexical *content* into a fixed frozen feature space.  Task
and candidate identifiers are intentionally never accepted by this interface.
"""
from __future__ import annotations

import hashlib
import json
from html.parser import HTMLParser
import math
import re
from collections.abc import Mapping

import numpy as np

from .registry import OUTCOMES


RBR_CACHE_SCHEMA = "apbpf-rbr-rich-cache-v3"
EXECUTION_TEXT_LIMIT = 4096
ASSOCIATION_ARMS = (
    "aligned", "outcome_shuffled", "joint_reversed", "presentation_permuted",
    "orderless", "semantics_masked", "wrong_candidate", "random_latent", "history_rate",
)


def bounded_execution_record(case, *, actual="", stderr="", returncode=None,
                             timed_out=False, outcome):
    """Cache evaluator-side execution evidence, never an actor-facing payload."""
    def bounded(value):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return str(value or "")[:EXECUTION_TEXT_LIMIT]
    if outcome not in OUTCOMES:
        raise ValueError("unregistered execution outcome")
    expected = case.get("expected", case.get("output"))
    if not isinstance(expected, str) or len(expected) > EXECUTION_TEXT_LIMIT:
        raise ValueError("expected output must be present and bounded")
    return {"expected": expected, "actual": bounded(actual), "stderr": bounded(stderr),
            "returncode": returncode, "timed_out": bool(timed_out), "outcome": outcome,
            "actual_truncated": len(actual or "") > EXECUTION_TEXT_LIMIT,
            "stderr_truncated": len(stderr or "") > EXECUTION_TEXT_LIMIT}


def validate_rbr_cache(payload):
    """Fail closed on legacy caches, missing evidence, and overlapping sources."""
    if payload.get("schema") != RBR_CACHE_SCHEMA:
        raise ValueError("rich execution cache required; rebuild with --prepare")
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("cache requires nonempty records")
    owners, problem_owners, seen = {}, {}, set()
    width = payload.get("tests_per_candidate")
    if type(width) is not int or width <= 4:
        raise ValueError("cache requires four visible and at least one future test")
    for row in rows:
        if not all(isinstance(row.get(key), str) and row[key]
                   for key in ("task_id", "problem_id", "task_text", "candidate", "split")):
            raise ValueError("cache is missing candidate/source metadata")
        if row["split"] not in {"train", "development", "test"}:
            raise ValueError("invalid cache split")
        source = row.get("source_component_id", row["problem_id"])
        if not isinstance(source, str) or not source:
            raise ValueError("source components must be nonempty strings")
        if ((source in owners and owners[source] != row["split"])
                or (row["problem_id"] in problem_owners
                    and problem_owners[row["problem_id"]] != row["split"])):
            raise ValueError("source component crosses training/evaluation splits")
        owners[source] = row["split"]
        problem_owners[row["problem_id"]] = row["split"]
        if row["task_id"] in seen:
            raise ValueError("duplicate candidate identity")
        seen.add(row["task_id"])
        if len(row.get("tests", ())) != width or len(row.get("outcomes", ())) != width:
            raise ValueError("cache histories must match the declared test count")
        for case, outcome in zip(row["tests"], row["outcomes"], strict=True):
            if (outcome not in OUTCOMES or case.get("outcome") != outcome
                    or not isinstance(case.get("input"), str)
                    or len(case["input"]) > 8192
                    or any(not isinstance(case.get(key), str) or len(case[key]) > EXECUTION_TEXT_LIMIT
                           for key in ("expected", "actual", "stderr"))
                    or type(case.get("timed_out")) is not bool
                    or "returncode" not in case
                    or (case["returncode"] is not None and type(case["returncode"]) is not int)):
                raise ValueError("missing or invalid bounded execution record")
    return payload


def public_test_text(case, *, expected_is_public=False):
    """Whitelist semantic features; observed/future execution evidence stays hidden."""
    if type(expected_is_public) is not bool:
        raise TypeError("expected-output visibility must be explicitly boolean")
    value = {"input": case["input"]}
    if expected_is_public:
        value["expected"] = case["expected"]
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def clustered_nll_gap(labels, aligned, comparator, clusters, *, seed=0, replicates=10000):
    """Example-weighted gap, bootstrapping complete source components."""
    labels = np.asarray(labels, dtype=np.int64)
    clusters = np.asarray(clusters)
    if labels.ndim != 1 or not len(labels) or clusters.shape != labels.shape:
        raise ValueError("nonempty labels and one source cluster per example required")
    # Reuse strict normalization checks before indexing probability matrices.
    compare_predictions(labels, {"baseline": comparator, "aligned": aligned})
    index = np.arange(len(labels))
    gap = (-np.log(np.clip(np.asarray(comparator)[index, labels], 1e-12, 1.0))
           + np.log(np.clip(np.asarray(aligned)[index, labels], 1e-12, 1.0)))
    keys = sorted(set(clusters.tolist()))
    sums = np.asarray([gap[clusters == key].sum() for key in keys])
    counts = np.asarray([(clusters == key).sum() for key in keys])
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, replicates, 256):
        indices = rng.integers(0, len(keys), size=(min(256, replicates - start), len(keys)))
        draws.extend((sums[indices].sum(1) / counts[indices].sum(1)).tolist())
    return {"unit": "source_component", "clusters": len(keys), "replicates": replicates,
            "mean_nll_gap": float(gap.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "estimand": "future-example-weighted; whole-source-component resampling"}


def association_strata(outcomes, *, visible_steps=4):
    """Locked full population plus prespecified descriptive ambiguity strata."""
    from .apbpf.counterfactual import association_eligibility
    values = np.asarray(outcomes)
    if values.ndim != 2 or values.shape[1] <= visible_steps:
        raise ValueError("outcomes require visible and future histories")
    eligible = association_eligibility(values, visible_steps)
    future = values[:, visible_steps:]
    return {"full": np.ones(len(values), dtype=bool), "eligible": eligible,
            "constant_visible": ~eligible,
            "future_variable": np.any(future != future[:, :1], axis=1)}


class FrozenTextEncoder:
    """Deterministic signed feature hashing over source text tokens/ngrams."""

    def __init__(self, dimension: int = 256) -> None:
        if not isinstance(dimension, int) or dimension < 16:
            raise ValueError("feature dimension must be an integer of at least 16")
        self.dimension = dimension

    def __call__(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError("semantic encoder accepts source text only")
        tokens = re.findall(r"[A-Za-z_]+|\d+(?:\.\d+)?|[^\w\s]", text.lower())
        features = np.zeros(self.dimension, dtype=np.float32)
        items = [("token", token) for token in tokens]
        items += [("bigram", left + "\0" + right) for left, right in zip(tokens, tokens[1:])]
        for kind, value in items:
            digest = hashlib.sha256((kind + "\0" + value).encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % (self.dimension - 4)
            sign = 1.0 if digest[8] & 1 else -1.0
            features[index] += sign
        scale = max(1.0, math.sqrt(len(items)))
        features[:-4] /= scale
        features[-4:] = np.asarray(
            [math.log1p(len(text)), math.log1p(len(tokens)), "\n" in text, bool(re.search(r"\d", text))],
            dtype=np.float32,
        )
        features[-4:-2] /= 16.0
        return features


def classify_execution(
    returncode: int | None,
    stdout: str,
    stderr: str,
    expected_stdout: str,
    *,
    timed_out: bool = False,
) -> str:
    """Map one isolated stdin/stdout execution to the registered outcome set."""
    if timed_out:
        return "TIMEOUT"
    if returncode is None or returncode != 0:
        return "RUNTIME_EXCEPTION"
    return "PASS" if stdout.rstrip() == expected_stdout.rstrip() else "WRONG_OUTPUT"


def select_disjoint_problem_ids(
    inventories: Mapping[str, set[str]],
    caps: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, tuple[str, ...]]:
    """Select reproducible problem-disjoint subsets from official partitions."""
    if set(inventories) != set(caps) or any(value < 0 for value in caps.values()):
        raise ValueError("inventories and nonnegative caps must have identical partitions")
    ownership: dict[str, int] = {}
    for values in inventories.values():
        for value in values:
            ownership[value] = ownership.get(value, 0) + 1
    result = {}
    for split, values in inventories.items():
        eligible = [value for value in values if ownership[value] == 1]
        ranked = sorted(
            eligible,
            key=lambda value: hashlib.sha256(f"{seed}\0{split}\0{value}".encode()).digest(),
        )
        result[split] = tuple(ranked[: caps[split]])
    return result


def compare_predictions(labels: np.ndarray, predictions: Mapping[str, np.ndarray]) -> dict:
    """Compute proper metrics and paired gains against the named baseline."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or not len(labels) or "baseline" not in predictions:
        raise ValueError("nonempty labels and a baseline prediction matrix are required")
    reports: dict[str, dict[str, float]] = {}
    losses: dict[str, np.ndarray] = {}
    for name, raw in predictions.items():
        probabilities = np.asarray(raw, dtype=np.float64)
        if (
            probabilities.ndim != 2
            or probabilities.shape[0] != len(labels)
            or np.any(labels < 0)
            or np.any(labels >= probabilities.shape[1])
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(1), 1.0, atol=1e-6)
        ):
            raise ValueError("predictions must be normalized [examples,classes]")
        clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
        losses[name] = -np.log(clipped)
        one_hot = np.eye(probabilities.shape[1])[labels]
        reports[name] = {
            "nll": float(losses[name].mean()),
            "brier": float(np.square(probabilities - one_hot).sum(1).mean()),
            "accuracy": float((probabilities.argmax(1) == labels).mean()),
        }
    baseline = losses["baseline"]
    for name, row in reports.items():
        row["nll_gain_vs_baseline"] = float((baseline - losses[name]).mean())
    return reports


def selector_bank_audit(visible_outcomes: np.ndarray, trusted_labels: np.ndarray,
                        *, prefixes=(1, 2, 4, 8)) -> dict:
    """Measure whether a grouped candidate bank leaves room beyond pass rate."""
    outcomes = np.asarray(visible_outcomes)
    labels = np.asarray(trusted_labels)
    if (outcomes.ndim != 3 or labels.shape != outcomes.shape[:2]
            or outcomes.shape[0] == 0 or outcomes.shape[1] < 2 or outcomes.shape[2] == 0
            or not np.isin(outcomes, (0, 1)).all() or not np.isin(labels, (0, 1)).all()):
        raise ValueError("selector audit requires binary [tasks,candidates,tests] outcomes and labels")
    prefixes = tuple(prefixes)
    if (not prefixes or any(type(value) is not int or not 1 <= value <= outcomes.shape[2]
                            for value in prefixes) or len(set(prefixes)) != len(prefixes)):
        raise ValueError("selector prefixes must be unique positive integers within the test count")
    positives = labels.astype(np.int64).sum(1)
    rankable = (positives > 0) & (positives < labels.shape[1])
    selected, selected_rankable = {}, {}
    for prefix in prefixes:
        indices = outcomes[:, :, :prefix].mean(2).argmax(1)
        values = labels[np.arange(len(labels)), indices]
        selected[str(prefix)] = float(values.mean())
        selected_rankable[str(prefix)] = float(values[rankable].mean()) if rankable.any() else None
    return {
        "tasks": int(len(labels)), "candidates": int(labels.size),
        "candidates_per_task": int(labels.shape[1]), "visible_tests": int(outcomes.shape[2]),
        "all_fail_tasks": int((positives == 0).sum()),
        "all_pass_tasks": int((positives == labels.shape[1]).sum()),
        "rankable_tasks": int(rankable.sum()),
        "passes_per_task": {str(value): int((positives == value).sum())
                            for value in sorted(set(positives.tolist()))},
        "random_selected_pass_at_1": float(labels.mean()),
        "oracle_pass_at_k": float(labels.max(1).mean()),
        "visible_pass_rate_selected_pass_at_1": selected,
        "rankable_visible_pass_rate_selected_pass_at_1": selected_rankable,
    }


def _histogram_features(histories) -> np.ndarray:
    rows = []
    for history in histories:
        if not history or any(value not in OUTCOMES for value in history):
            raise ValueError("histogram control requires nonempty registered visible outcomes")
        counts = np.asarray([history.count(value) for value in OUTCOMES], dtype=np.float64)
        rows.append(np.r_[counts / len(history), 1.0])
    return np.asarray(rows)


def fit_histogram_latent(histories, latents: np.ndarray, *, ridge: float = 1e-3) -> np.ndarray:
    """Fit the fixed low-capacity outcome-histogram control on training data."""
    features = _histogram_features(histories)
    targets = np.asarray(latents, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[0] != len(features) or not np.isfinite(targets).all():
        raise ValueError("latents must have finite shape [histories,latent_dim]")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    gram = features.T @ features + ridge * np.eye(features.shape[1])
    return np.linalg.pinv(gram) @ features.T @ targets


def histogram_latent(histories, coefficients: np.ndarray) -> np.ndarray:
    features = _histogram_features(histories)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 2 or coefficients.shape[0] != features.shape[1] or not np.isfinite(coefficients).all():
        raise ValueError("histogram coefficients have the wrong finite shape")
    return features @ coefficients


class _VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif not self.hidden and tag in {"h1", "h2", "h3", "p", "pre", "br", "li"}:
            self.rows.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        elif not self.hidden and tag in {"h1", "h2", "h3", "p", "pre", "li"}:
            self.rows.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.rows.append(data)


def html_to_text(source: str) -> str:
    if not isinstance(source, str):
        raise TypeError("HTML description must be text")
    parser = _VisibleHTML()
    parser.feed(source)
    lines = [" ".join(line.split()) for line in "".join(parser.rows).splitlines()]
    return "\n".join(line for line in lines if line)


def render_repair_prompt(candidate: str, tests, outcomes, *, visible: int = 4,
                         task_text: str = "") -> str:
    """Render only generator-visible code, inputs and ordered outcome categories."""
    if not isinstance(candidate, str) or not 1 <= visible <= len(tests) or len(tests) != len(outcomes):
        raise ValueError("candidate and aligned visible test history are required")
    evidence = []
    for index in range(visible):
        case = tests[index]
        if not isinstance(case, Mapping) or "input" not in case or outcomes[index] not in OUTCOMES:
            raise ValueError("visible evidence must contain input text and registered outcomes")
        evidence.append({"ordinal": index, "input": str(case["input"]), "outcome": outcomes[index]})
    payload = {"problem": str(task_text), "buggy_python": candidate, "visible_executions": evidence}
    return (
        "Repair this Python stdin/stdout program using the visible executions. "
        "Return only complete executable Python source with no Markdown or explanation.\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\nCorrected Python source:\n"
    )
