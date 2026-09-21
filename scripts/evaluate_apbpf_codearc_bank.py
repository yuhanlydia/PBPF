#!/usr/bin/env python3
"""Evaluate CodeARC replay banks in separate public and locked hidden phases."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

from pbpf.apbpf.codearc_bank import file_sha, load_bank, select_tests, verify_hidden_lock
from pbpf.apbpf.codearc_execution import execute_call


def work(job):
    task, candidate_id, code, tests, timeout = job
    results = [dict(test_id=test["id"], **execute_call(code, test, timeout=timeout)) for test in tests]
    visible = [r for r in results if int(r["test_id"]) < 4]
    hidden = [r for r in results if int(r["test_id"]) >= 4]
    return {"task_id": task, "candidate_id": candidate_id, "tests": results,
            "visible_passes": sum(r["outcome"] == "PASS" for r in visible) if visible else None,
            "hidden_passes": sum(r["outcome"] == "PASS" for r in hidden) if hidden else None,
            "all_pass": all(row["outcome"] == "PASS" for row in results)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluator-root", type=Path)
    p.add_argument("--public-root", type=Path)
    p.add_argument("--phase", choices=["visible", "hidden", "all"], default="all")
    p.add_argument("--population-lock", type=Path)
    p.add_argument("--bank", type=Path)
    p.add_argument("--reference-control", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--split", choices=["train", "development", "primary"])
    p.add_argument("--timeout", type=float, default=6.)
    args = p.parse_args()
    if args.reference_control == (args.bank is not None):
        p.error("choose exactly one of --reference-control or --bank")
    if min(args.workers, args.timeout) <= 0 or args.limit < 0:
        p.error("invalid execution bounds")
    if args.phase == "visible":
        if args.public_root is None or args.evaluator_root is not None or args.reference_control:
            p.error("visible phase accepts only --public-root and a generated bank")
        root = args.public_root
    else:
        if args.evaluator_root is None or args.public_root is not None:
            p.error("hidden/all phase requires only --evaluator-root")
        root = args.evaluator_root
    if args.output.exists():
        raise FileExistsError("evaluations are create-once")
    bank_digest, lock_digest, bank = None, None, []
    if args.bank is not None:
        run, bank, bank_digest = load_bank(args.bank)
        if args.split and args.split != run["split"]:
            raise ValueError("split differs from immutable generation inventory")
        args.split = run["split"]
        if args.limit:
            p.error("generated banks must be evaluated as complete populations")
    if args.phase == "all" and args.split == "primary":
        p.error("primary banks require visible evaluation, population lock, then hidden evaluation")
    if args.phase == "hidden":
        if args.population_lock is None or not bank:
            p.error("hidden execution requires a pre-hidden population lock and complete bank")
        # This check precedes opening the evaluator manifest or task data.
        lock_digest = verify_hidden_lock(args.population_lock, bank, bank_digest)
    elif args.population_lock is not None:
        p.error("--population-lock is only valid for hidden execution")
    manifest = json.loads((root / "manifest.json").read_bytes())
    if bank and run["public_tasks_sha256"] != manifest["public_tasks_sha256"]:
        raise ValueError("generation bank and execution data refer to different public tasks")
    key = "public_tasks_sha256" if args.phase == "visible" else "evaluator_tasks_sha256"
    if file_sha(root / "tasks.jsonl") != manifest[key]:
        raise ValueError("task file checksum mismatch")
    with (root / "tasks.jsonl").open() as stream:
        tasks = {r["task_id"]: r for r in map(json.loads, stream)}
    if args.split:
        tasks = {key: row for key, row in tasks.items() if row["split"] == args.split}
    jobs = []
    if args.reference_control:
        for task in list(tasks.values())[:args.limit or None]:
            jobs.append((task["task_id"], "reference", task["reference_code"], select_tests(task, args.phase), args.timeout))
    else:
        for row in bank:
            task = tasks[row["task_id"]]
            if row["source_component_id"] != task["source_component_id"] or row["split"] != task["split"]:
                raise ValueError("bank and task source bindings differ")
            for candidate in row["candidates"]:
                jobs.append((row["task_id"], candidate["candidate_id"], candidate["code"], select_tests(task, args.phase), args.timeout))
    args.output.mkdir(parents=True)
    identity = {"schema": "apbpf-codearc-execution-identity-v1", "phase": args.phase,
                "bank_complete_sha256": bank_digest, "population_lock_sha256": lock_digest,
                "task_manifest_sha256": file_sha(root / "manifest.json"), "timeout": args.timeout,
                "candidate_count": len(jobs), "split": args.split,
                "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): file_sha(path)
                    for path in [Path(__file__).resolve(), *[Path(__file__).resolve().parents[1] / "src/pbpf/apbpf" / name
                    for name in ("codearc_bank.py", "codearc_execution.py")]]}}
    (args.output / "run.json").write_text(json.dumps(identity, indent=2) + "\n")
    rows = []
    with (args.output / "records.jsonl").open("x") as stream, concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(work, jobs):
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(rows) % 16 == 0:
                print(json.dumps({"completed": len(rows), "total": len(jobs)}), flush=True)
    groups = []
    by_candidate = {(r["task_id"], r["candidate_id"]): r for r in rows}
    for row in bank:
        records = [by_candidate[row["task_id"], c["candidate_id"]] for c in row["candidates"]]
        group = {"group_id": row["task_id"], "candidate_ids": [r["candidate_id"] for r in records]}
        if args.phase in {"visible", "all"}:
            group.update(source_component_id=row["source_component_id"], split=row["split"],
                         visible_outcomes=[[int(t["outcome"] == "PASS") for t in r["tests"]
                                            if int(t["test_id"]) < 4] for r in records])
        if args.phase in {"hidden", "all"}:
            group["hidden_labels"] = [int(r["hidden_passes"] == 6) for r in records]
        groups.append(group)
    (args.output / "bank_groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    result = {"schema": "apbpf-codearc-evaluation-v3", "reference_control": args.reference_control,
              **identity, "candidate_count": len(rows),
              "tests": sum(len(r["tests"]) for r in rows), "all_pass": sum(r["all_pass"] for r in rows),
              "test_passes": sum(t["outcome"] == "PASS" for r in rows for t in r["tests"]),
              "reference_behavior_matches": sum(t["reference_behavior_matches"] for r in rows for t in r["tests"]),
              "dictionary_order_only_mismatches": sum(t["dictionary_order_only_mismatch"] for r in rows for t in r["tests"]),
              "scoring_policy": "exact stripped stdout; version-normalized exception strings; timeouts remain TIMEOUT and never PASS",
              "scope": "exploratory replay diagnostic; not sealed stage evidence", "records": rows}
    result["schema"] = "apbpf-codearc-evaluation-v3"
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
