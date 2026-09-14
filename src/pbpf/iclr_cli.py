"""One public command family for the synthetic and provisioned formal DAGs."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import yaml

from .iclr_config import ROOT, matrix_report, resolve_config
from .runner.aggregate import require_main_tables, validate_decision
from .runner.shard import atomic_write, canonical_bytes, digest, merge_leaves
from .runner.stage import DEPENDENCIES, STAGES, GateFailure, StageContext, StageResult, run_pipeline, run_stage


def offline_backend(root, fingerprint):
    from .runner.repair import OfflineExperiment
    return OfflineExperiment(root, fingerprint=fingerprint)


def _parser(argv):
    parser = argparse.ArgumentParser(prog="pbpf-iclr", allow_abbrev=False)
    parser.add_argument("command", choices=("doctor", "prepare", "run", "aggregate", "verify", "package"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", required=True, choices=("local_cpu", "slurm_h200x16"))
    parser.add_argument("--site", help="operator-provisioned site YAML; defaults to PBPF_ICLR_SITE")
    parser.add_argument("--output-root", default="runs/iclr")
    for flag in ("resume", "dry-run", "force-after-failed-gate", "calibration-only"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int)
    seen = set()
    for token in argv:
        if token.startswith("--"):
            option = token.split("=", 1)[0]
            if option in seen:
                parser.error("duplicate option: " + option)
            seen.add(option)
    args = parser.parse_args(argv)
    count = args.shard_count if args.shard_count is not None else (1 if args.profile == "local_cpu" else 16)
    if count < 1 or not 0 <= args.shard_index < count:
        parser.error("shard index must be nonnegative and less than positive shard count")
    if args.profile == "local_cpu" and (count != 1 or args.shard_index != 0):
        parser.error("local_cpu uses one complete synthetic DAG, not distributed shards")
    args.shard_count = count
    return args


def _site(args):
    path = args.site or os.environ.get("PBPF_ICLR_SITE")
    if not path:
        return {}
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("PBPF_ICLR_SITE must contain an operator-provisioned YAML mapping")
    return value


def _prepare_local(resolved, directory, backend):
    results = {}
    handlers = backend.handlers()
    config = resolved.fingerprint.to_dict()["config"]
    for name in STAGES[:3]:
        dependencies = {dep: results[dep] for dep in DEPENDENCIES[name]}
        def execute(key, checksums, stage=name, deps=dependencies):
            context = StageContext(stage, deps, config, False, "smoke-only-no-claim", None)
            value = handlers[stage](context)
            if context._consumed != set(deps):
                raise ValueError("handler failed to consume declared dependencies")
            return value
        results[name] = run_stage(root=directory / "stages", name=name, fingerprint=resolved.fingerprint,
            config=config, keys=[["pipeline", name]], dependencies={k: v.checksum for k, v in dependencies.items()},
            handler=execute, confirmatory=False, reason="smoke-only-no-claim")
    return results


def load_stage(directory, name, fingerprint=None):
    directory = Path(directory) / "stages" / name
    paths = sorted(directory.glob("shard-*.identity.json"))
    if not paths:
        raise ValueError("missing stage inventory: " + name)
    results = []
    for path in paths:
        identity = json.loads(path.read_bytes())
        if identity.get("stage") != name or fingerprint is not None and identity.get("fingerprint") != fingerprint:
            raise ValueError("stage fingerprint identity mismatch: " + name)
        base = path.name.removesuffix(".identity.json")
        marker = directory / (base + ".complete.json")
        body = json.loads(marker.read_bytes())
        results.append(StageResult(name, directory / (base + ".jsonl"), marker, identity, body["checksum"], True))
    keys = []
    for result in results:
        keys.extend(row["key"] for row in result.rows())
    rows = merge_leaves(results, expected_keys=keys)
    allowed = {result.artifact.name for result in results} | {result.completion.name for result in results}
    allowed |= {p.name for p in paths} | {p.name.removesuffix(".identity.json")+".lock" for p in paths}
    if any(path.name not in allowed for path in directory.iterdir()):
        raise ValueError("unexpected stage artifact inventory")
    for row in rows:
        for key in ("confirmatory", "reason", "failed_gate_hash"):
            if row[key] != results[0].identity[key]:
                raise ValueError("artifact confirmatory envelope differs from bound leaf identity")
    return results, rows


def verify_run(resolved, directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("run does not exist; prepare and run first")
    # Verification is read-only; unlike persist it never repairs absent metadata.
    expected = {"fingerprint.json": canonical_bytes(resolved.fingerprint.to_dict())}
    config_bytes = (directory / "resolved.yaml").read_bytes()
    if yaml.safe_load(config_bytes) != resolved.science:
        raise ValueError("resolved scientific config mismatch")
    expected["resolved.sha256"] = (hashlib.sha256(config_bytes).hexdigest()+"\n").encode()
    for name, content in expected.items():
        if (directory / name).read_bytes() != content:
            raise ValueError("immutable metadata checksum mismatch: " + name)
    stages, dependency_graph = STAGES, DEPENDENCIES
    if resolved.profile != "local_cpu":
        from .runner.formal import FORMAL_STAGES, FORMAL_DEPENDENCIES
        stages, dependency_graph = FORMAL_STAGES, FORMAL_DEPENDENCIES
    all_results, all_rows = {}, {}
    for name in stages:
        fingerprint = resolved.fingerprint.digest
        if resolved.profile != "local_cpu" and stages.index(name) >= stages.index("predict"):
            from .runner.formal import confirmatory_identity
            fingerprint = confirmatory_identity(resolved, directory).digest
        results, rows = load_stage(directory, name, fingerprint)
        dependencies = {dep: digest([r.checksum for r in all_results[dep]]) if resolved.profile != "local_cpu"
                        else all_results[dep][0].checksum for dep in dependency_graph[name]}
        if results[0].identity["dependencies"] != dependencies:
            raise ValueError("stage upstream dependency checksum mismatch")
        all_results[name], all_rows[name] = results, rows
    for name in ("gate_b", "gate_c"):
        validate_decision(all_rows[name][0]["value"])
    forced = any(r.identity.get("failed_gate_hash") for values in all_results.values() for r in values)
    eligible = False
    if resolved.profile != "local_cpu":
        from .runner.formal import confirmatory_identity
        scientific_identity = confirmatory_identity(resolved, directory)
        eligible = gates_eligible(all_rows["gate_b"][0]["value"], all_rows["gate_c"][0]["value"],
                                 all_rows["tables"][0]["value"].get("replication_gate", {}),
                                 scientific_identity=scientific_identity,
                                 replication_hash=digest([r.checksum for r in all_results["replication"]]),
                                 replication_eval_hash=digest([r.checksum for r in all_results["replication_eval"]]))
        eligible = eligible and all(r.identity["confirmatory"] for values in all_results.values() for r in values)
    return {"verified_stages": len(all_results), "forced": forced,
            "main_table_eligible": eligible and not forced,
            "stage_hashes": {name: [r.checksum for r in results] for name, results in all_results.items()}}


def gates_eligible(stage_b, stage_c, replication, *, scientific_identity=None,
                   replication_hash=None, replication_eval_hash=None):
    validate_decision(stage_b)
    validate_decision(stage_c)
    if not (stage_b.get("passed") is True and stage_c.get("passed") is True
            and replication.get("passed") is True):
        return False
    from .runner.fingerprint import ScientificIdentity
    expected_keys = {"schema", "gate", "passed", "scientific_identity", "selection_hash",
                     "gate_c_hash", "replication_hash", "replication_eval_hash", "decision_hash"}
    hashes = (replication_hash, replication_eval_hash)
    valid_hashes = all(type(value) is str and len(value) == 64
                       and all(character in "0123456789abcdef" for character in value)
                       for value in hashes)
    if (type(scientific_identity) is not ScientificIdentity or set(replication) != expected_keys
            or replication.get("schema") != "pbpf-replication-gate-v1"
            or replication.get("gate") != "replication" or type(replication.get("passed")) is not bool
            or not valid_hashes
            or stage_b.get("gate") != "B" or stage_c.get("gate") != "C"
            or stage_c.get("stage_b_hash") != stage_b.get("decision_hash")
            or stage_b.get("selection_hash") != scientific_identity.to_dict()["locks"]["selection_hash"]
            or replication.get("scientific_identity") != scientific_identity.digest
            or replication.get("selection_hash") != scientific_identity.to_dict()["locks"]["selection_hash"]
            or replication.get("gate_c_hash") != stage_c.get("decision_hash")
            or replication.get("replication_hash") != replication_hash
            or replication.get("replication_eval_hash") != replication_eval_hash
            or replication.get("decision_hash") != digest({key: value for key, value in replication.items()
                                                            if key != "decision_hash"})):
        raise ValueError("replication gate schema, checksum, or scientific provenance mismatch")
    return True


def _write_once_json(path, value):
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("immutable aggregate/publication conflict")
    else:
        atomic_write(path, payload, create_once=True)


def aggregate_run(resolved, directory):
    report = verify_run(resolved, directory)
    if report["forced"]:
        raise ValueError("forced/nonconfirmatory gate-override artifacts cannot be aggregated")
    if resolved.profile != "local_cpu":
        if not report["main_table_eligible"]:
            raise ValueError("formal tables require verified passing B, C and locked replication gates")
        require_main_tables([{"confirmatory": True, "failed_gate_hash": None}])
    report.update(run_id=resolved.fingerprint.digest, claim_status=resolved.claim_status)
    _write_once_json(directory / "aggregate.json", report)
    return report


def package_run(resolved, directory):
    report = verify_run(resolved, directory)
    if report["forced"]:
        raise ValueError("forced artifacts cannot be packaged as a verified result bundle")
    names = ["resolved.yaml", "resolved.sha256", "fingerprint.json"]
    if (directory / "confirmatory-fingerprint.json").exists():
        names.append("confirmatory-fingerprint.json")
    names += [p.relative_to(directory).as_posix() for p in (directory / "stages").rglob("*")
              if p.is_file() and p.suffix in {".json", ".jsonl"}]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o600 << 16
            archive.writestr(info, (directory / name).read_bytes())
    target = directory / "verified-run.zip"
    payload = buffer.getvalue()
    if target.exists() and target.read_bytes() != payload:
        raise ValueError("immutable package conflict")
    if not target.exists():
        atomic_write(target, payload, create_once=True)
    return {"archive": str(target), "sha256": hashlib.sha256(payload).hexdigest(), **report}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _parser(argv)
        os.umask(0o077)
        site = _site(args)
        # All diagnostics, including site-factory output, go to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            resolved = resolve_config(args.config, args.profile, site=site,
                                      execution=args.command in {"prepare", "run"})
        directory = Path(args.output_root).resolve() / resolved.fingerprint.digest
        response = dict(run_id=resolved.fingerprint.digest, run_directory=str(directory),
                        claim_status=resolved.claim_status)
        if args.command == "doctor":
            response.update(valid=True, cells=matrix_report(resolved), profile=args.profile)
        elif args.command in {"prepare", "run"}:
            if args.dry_run:
                response.update(valid=True, cells=matrix_report(resolved), submitted=False)
            else:
                resolved.persist(directory)
                if args.profile == "local_cpu":
                    if args.calibration_only:
                        raise ValueError("64-task throughput calibration requires the provisioned formal backend")
                    if not args.resume and (directory / "stages" / "repair").exists():
                        raise ValueError("existing run requires --resume; immutable leaves are never overwritten")
                    backend = offline_backend(Path(args.output_root).resolve() / ".trusted" / resolved.fingerprint.digest,
                                              resolved.fingerprint)
                    with contextlib.redirect_stdout(sys.stderr):
                        if args.command == "prepare":
                            results = _prepare_local(resolved, directory, backend)
                        else:
                            results = run_pipeline(root=directory / "stages", fingerprint=resolved.fingerprint,
                                handlers=backend.handlers(), config=resolved.fingerprint.to_dict()["config"],
                                force_after_failed_gate=args.force_after_failed_gate, confirmatory=False, reason="smoke-only-no-claim")
                    response.update(stages=list(results), cached_stages=sum(r.cached for r in results.values()),
                                    actor_requests=backend.actor_requests)
                else:
                    from .runner.formal import execute_formal
                    with contextlib.redirect_stdout(sys.stderr):
                        response.update(execute_formal(resolved, directory, args))
        elif args.command == "verify":
            response.update(verify_run(resolved, directory))
        elif args.command == "aggregate":
            response.update(aggregate_run(resolved, directory))
        elif args.command == "package":
            response.update(package_run(resolved, directory))
        print(json.dumps(response, sort_keys=True, allow_nan=False))
        return 0
    except GateFailure as error:
        print(str(error), file=sys.stderr)
        return error.exit_code
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print("pbpf-iclr: " + str(error), file=sys.stderr)
        if argv and argv[0] == "doctor":
            print(json.dumps({"valid": False, "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
