"""Prepare CodeARC replay inputs without exposing reference bodies to generators.

This is data preparation, not evidence that the A-PBPF hard-bank gate passes.
Reference implementations and invocations 4--9 are written only to the separate
evaluator directory. Splits are locked by source component before generation.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


SOURCE_PINS = {
    "problems.jsonl": "5f32ea675043740f277ac9c397166ada17cc3f26065ed0bab55f8d452468bc92",
    "invocations.jsonl": "3f08820594d4825c78bbdcc3b8eb2e6c8c0285fb3de966e900463b53f1723962",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _body_identity(code):
    tree = ast.parse(code)
    # Ignore docstrings, formatting and comments when grouping exact source
    # duplicates. Keep constants and executable syntax; this is not a semantic
    # equivalence detector. The digest remains evaluator-side.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def build_records(problems, invocations, *, seed, development_components, primary_components):
    """Return disjoint public/evaluator records; never filter on future outputs."""
    sources = {}
    for row in problems:
        if row["type"] != "anonymous":
            continue
        key = str(row["problem_id"])
        if key in sources:
            raise ValueError("duplicate anonymous problem ID")
        sources[key] = row
    calls = defaultdict(dict)
    for row in invocations:
        if row["type"] != "anonymous":
            continue
        key, index = str(row["problem_id"]), row["index"]
        if key not in sources or type(index) is not int or not 0 <= index < 10:
            raise ValueError("invalid invocation identity")
        if index in calls[key]:
            raise ValueError("duplicate invocation index")
        if not isinstance(row["input"], str) or not isinstance(row["output"], str):
            raise ValueError("invocation input/output must be strings")
        if type(row["errored"]) is not bool:
            raise ValueError("invocation errored must be boolean")
        calls[key][index] = row
    if not sources or any(set(calls[k]) != set(range(10)) for k in sources):
        raise ValueError("each anonymous source needs exactly ten invocations")
    components = defaultdict(list)
    for key, row in sources.items():
        components[_body_identity(row["code"])].append(key)
    # Public component IDs derive only from problem IDs, not reference hashes.
    groups = {"codearc-component-" + _digest(sorted(ids))[:24]: sorted(ids)
              for ids in components.values()}
    ordered = sorted(groups, key=lambda k: _digest([seed, "source-split", k]))
    if min(development_components, primary_components) < 1:
        raise ValueError("development and primary counts must be positive")
    if development_components + primary_components >= len(ordered):
        raise ValueError("split counts must leave at least one training component")
    splits = {key: ("primary" if i < primary_components else "development"
                    if i < primary_components + development_components else "train")
              for i, key in enumerate(ordered)}
    public, private = [], []
    for component in ordered:
        for key in groups[component]:
            tests = [calls[key][index] for index in range(10)]
            identity = {"task_id": "CodeARC/" + key, "source_component_id": component,
                        "split": splits[component], "protocol": "CodeARC-Replay"}
            visible = [{"id": str(i), "input": r["input"], "expected": r["output"],
                        "expected_error": r["errored"]} for i, r in enumerate(tests[:4])]
            prompt = ("Infer a Python function named solution from the observed calls below. "
                      "Return a complete Python implementation, including any imports. "
                      "Do not print the examples as a substitute for implementing the function.\n\n"
                      + "\n\n".join(json.dumps(t, ensure_ascii=False) for t in visible))
            public.append({**identity, "task_text": prompt, "visible_tests": visible})
            private.append({**identity, "reference_code": sources[key]["code"],
                            "tests": [{"id": str(i), "input": r["input"], "expected": r["output"],
                                       "expected_error": r["errored"], "hidden": i >= 4}
                                      for i, r in enumerate(tests)]})
    inventory = {"seed": seed, "source_components": len(groups), "tasks": len(public),
                 "component_counts": dict(Counter(splits.values())),
                 "task_counts": dict(Counter(r["split"] for r in public)),
                 "source_inventory": [{"source_component_id": k, "task_ids": groups[k],
                                       "split": splits[k]} for k in ordered],
                 "split_policy": "hashed source components; exact AST duplicates co-located",
                 "filter_policy": "anonymous only; no correctness/hidden-outcome filtering"}
    return public, private, inventory


def materialize(source_root, public_root, evaluator_root, *, seed=1701,
                development_components=400, primary_components=500):
    source_root, public_root, evaluator_root = map(Path, (source_root, public_root, evaluator_root))
    a, b = public_root.resolve(), evaluator_root.resolve()
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise ValueError("public and evaluator directories must be separate, nonnested paths")
    if public_root.exists() or evaluator_root.exists():
        raise FileExistsError("materializations are create-once; choose new directories")
    rows = {}
    for name, digest in SOURCE_PINS.items():
        path = source_root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"pinned source checksum mismatch: {name}")
        # File iteration handles literal Unicode line separators inside strings.
        with path.open(encoding="utf-8") as stream:
            rows[name] = [json.loads(line) for line in stream]
    public, private, inventory = build_records(rows["problems.jsonl"], rows["invocations.jsonl"],
        seed=seed, development_components=development_components, primary_components=primary_components)
    public_root.mkdir(parents=True, mode=0o700)
    evaluator_root.mkdir(parents=True, mode=0o700)
    for root, name, values in ((public_root, "tasks.jsonl", public),
                               (evaluator_root, "tasks.jsonl", private)):
        with (root / name).open("x", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {"schema": "apbpf-codearc-replay-materialization-v1", **inventory,
                "source_sha256": SOURCE_PINS,
                "public_tasks_sha256": hashlib.sha256((public_root / "tasks.jsonl").read_bytes()).hexdigest(),
                "status": "data-materialized-no-experiment-claim",
                "isolation": "separate files; generator execution must mount only public_root"}
    (public_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (evaluator_root / "manifest.json").write_text(json.dumps({**manifest,
        "evaluator_tasks_sha256": hashlib.sha256((evaluator_root / "tasks.jsonl").read_bytes()).hexdigest()}, indent=2) + "\n")
    return manifest
