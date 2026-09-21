"""Immutable, fail-closed A-PBPF DAG with an explicitly smoke-only backend.

Real workers are provisioned argv commands. They read APBPF_REQUEST and write
APBPF_RESULT; they must bind their results to every declared input checksum.
No real worker or missing result is ever replaced by a synthetic value.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

from .config import SCHEMA, canonical_bytes, digest
from .hard_bank import (
    _binary as _hard_bank_binary,
    _candidate_inventory_digest,
    _ordered_group_ids,
    _population_digest,
    _visible_outcomes_digest,
)

DEPENDENCIES = {
    "materialize": (),
    "hard_bank_lock": ("materialize",),
    "execution_cache": ("materialize", "hard_bank_lock"),
    "hard_bank": ("execution_cache", "hard_bank_lock"),
    "hard_bank_gate": ("hard_bank", "hard_bank_lock"),
    "train_baselines": ("execution_cache", "hard_bank_gate"),
    "train_belief": ("execution_cache", "hard_bank_gate"),
    "baseline_fairness_gate": ("train_baselines", "train_belief"),
    "association": ("execution_cache", "train_belief", "baseline_fairness_gate"),
    "association_gate": ("association",),
    "pair_invariance_gate": ("association", "association_gate"),
    "oracle_headroom": ("hard_bank", "association_gate", "pair_invariance_gate"),
    "oracle_headroom_gate": ("oracle_headroom",),
    "active_testing": ("oracle_headroom_gate", "train_belief", "hard_bank"),
    "active_testing_gate": ("active_testing",),
    "selection": ("hard_bank", "train_baselines", "train_belief", "association_gate", "active_testing_gate"),
    "selection_gate": ("selection",),
    "repair": ("association_gate", "selection_gate", "train_belief", "execution_cache"),
    "replication": ("association_gate", "selection_gate", "active_testing_gate"),
    "replication_gate": ("replication",),
    "paper_tables": ("baseline_fairness_gate", "hard_bank_gate", "association_gate",
                     "pair_invariance_gate", "active_testing_gate", "selection_gate",
                     "repair", "replication_gate"),
}
STAGES = tuple(DEPENDENCIES)
GATE_STAGES = tuple(name for name in STAGES if name.endswith("_gate"))
RUNNER_LINEAGE_SCHEMA = "apbpf-runner-lineage-v1"


class PipelineError(RuntimeError):
    exit_code = 1


class GateFailure(PipelineError):
    exit_code = 20


def write_once(path, value):
    """Atomically publish a create-once JSON record; never replace an artifact."""
    path = Path(path)
    payload = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact conflict: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"immutable artifact conflict: {path}") from None
    finally:
        os.unlink(temporary)


def _read(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return value


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _runner_lineage(identity, dependencies):
    """Bind a stage to the exact completion records of every DAG ancestor."""
    ancestors = {}
    for name, dependency in dependencies.items():
        inherited = dependency.value["runner_lineage"]["ancestor_completion_sha256"]
        for ancestor, checksum in inherited.items():
            if ancestor in ancestors and ancestors[ancestor] != checksum:
                raise ValueError(f"conflicting runner lineage for ancestor {ancestor}")
            ancestors[ancestor] = checksum
        if name in ancestors and ancestors[name] != dependency.checksum:
            raise ValueError(f"conflicting direct runner lineage for {name}")
        ancestors[name] = dependency.checksum
    return {"schema": RUNNER_LINEAGE_SCHEMA, "fingerprint": identity["fingerprint"],
            "ancestor_completion_sha256": dict(sorted(ancestors.items()))}


def _validate_transitive_lineage(result, identity, dependencies):
    if result["runner_lineage"] != _runner_lineage(identity, dependencies):
        raise ValueError("runner lineage is not the exact transitive dependency union")


@dataclass(frozen=True)
class StageResult:
    name: str
    directory: Path
    identity: dict
    value: dict
    checksum: str


def _attempts(root, stage):
    return sorted((Path(root) / "stages" / stage).glob("attempt-[0-9][0-9][0-9][0-9][0-9][0-9]"))


def _load(directory):
    identity = _read(directory / "identity.json")
    complete = _read(directory / "complete.json")
    if complete.get("identity_hash") != digest(identity):
        raise ValueError(f"stage identity checksum mismatch: {directory}")
    for name, checksum in complete["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or _sha(directory / name) != checksum:
            raise ValueError(f"stage artifact checksum mismatch: {directory}/{name}")
    required = {"identity.json", "request.json", "work.json", "summary.json", "result.json", "command.json"}
    if not required.issubset(complete["files"]):
        raise ValueError("stage completion inventory is incomplete")
    result = _read(directory / "result.json")
    request = _read(directory / "request.json")
    if digest(request["config"]) != identity["fingerprint"] or any(request.get(key) != value for key, value in identity.items()):
        raise ValueError("stage request is not bound to its configuration/identity")
    _validate_result(result, identity)
    _validate_gate_decision(result, identity, request["config"])
    for artifact in result["artifacts"]:
        path = directory / artifact["path"]
        if not path.resolve().is_relative_to((directory / "outputs").resolve()) or path.is_symlink():
            raise ValueError("worker artifact escaped its attempt directory")
        if artifact["path"] not in complete["files"] or _sha(path) != artifact["sha256"]:
            raise ValueError("worker artifact is absent or does not match its declared checksum")
    return StageResult(identity["stage"], directory, identity, result, digest(complete))


def load_stage_attempt(directory, *, _cache=None, _visiting=None):
    """Load one attempt and recursively prove its exact same-run DAG lineage."""
    directory = Path(directory).resolve()
    cache = {} if _cache is None else _cache
    visiting = set() if _visiting is None else _visiting
    if directory in cache:
        return cache[directory]
    if directory in visiting:
        raise ValueError("cycle in stage-attempt lineage")
    visiting.add(directory)
    try:
        result = _load(directory)
        request = _read(directory / "request.json")
        inputs = request.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != set(result.identity["dependencies"]):
            raise ValueError("stage request inputs do not match declared dependencies")
        run_root = directory.parents[2]
        dependencies = {}
        for name, item in inputs.items():
            if not isinstance(item, dict) or set(item) != {"result", "checksum"}:
                raise ValueError("stage request input requires exact result/checksum fields")
            result_path = Path(item["result"]).resolve()
            if (result_path.name != "result.json"
                    or not result_path.is_relative_to((run_root / "stages").resolve())):
                raise ValueError("dependency result is outside the current run")
            dependency = load_stage_attempt(result_path.parent, _cache=cache, _visiting=visiting)
            expected = result.identity["dependencies"][name]
            if dependency.name != name or dependency.checksum != item["checksum"] or item["checksum"] != expected:
                raise ValueError("dependency attempt does not match the declared completion checksum")
            dependencies[name] = dependency
        _validate_transitive_lineage(result.value, result.identity, dependencies)
        _validate_gate_decision(result.value, result.identity, request["config"], dependencies)
        cache[directory] = result
        return result
    finally:
        visiting.remove(directory)


def _validate_result(result, identity):
    expected = {"schema": "apbpf-stage-result-v1", "stage": identity["stage"],
                "fingerprint": identity["fingerprint"], "dependencies": identity["dependencies"]}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"worker result {key} does not match stage request")
    lineage = result.get("runner_lineage")
    if (not isinstance(lineage, dict)
            or set(lineage) != {"schema", "fingerprint", "ancestor_completion_sha256"}
            or lineage.get("schema") != RUNNER_LINEAGE_SCHEMA
            or lineage.get("fingerprint") != identity["fingerprint"]
            or not isinstance(lineage.get("ancestor_completion_sha256"), dict)
            or any(not isinstance(name, str) or not name or not _is_sha256(checksum)
                   for name, checksum in lineage.get("ancestor_completion_sha256", {}).items())):
        raise ValueError("stage result lacks valid runner-authored transitive lineage")
    ancestors = lineage["ancestor_completion_sha256"]
    if any(ancestors.get(name) != checksum for name, checksum in identity["dependencies"].items()):
        raise ValueError("runner lineage does not bind every direct dependency completion")
    if not isinstance(result.get("summary"), dict) or not result["summary"]:
        raise ValueError("worker must supply a nonempty summary object")
    if not isinstance(result.get("artifacts"), list):
        raise ValueError("worker must supply an artifacts inventory")
    if identity["backend"] == "real" and not result["artifacts"]:
        raise ValueError("real worker must supply evidence artifacts, not a prospective placeholder")
    for artifact in result["artifacts"]:
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise ValueError("each artifact requires exactly path and sha256")
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "outputs":
            raise ValueError("worker artifacts must be attempt-relative outputs/ paths")
    if identity["stage"] in GATE_STAGES:
        gate = result.get("gate")
        if not isinstance(gate, dict) or type(gate.get("passed")) is not bool or not gate.get("reason"):
            raise ValueError("gate worker requires a boolean passed decision and reason")
        if identity["backend"] == "real" and (not isinstance(gate.get("metrics"), dict) or not gate["metrics"]):
            raise ValueError("real gate worker must report metrics supporting the locked decision")


def _validate_gate_decision(result, identity, config, dependencies=None):
    """Recompute real decisions from the locked thresholds, never trust a flag."""
    if identity["backend"] != "real" or identity["stage"] not in GATE_STAGES:
        return
    metrics = result["gate"]["metrics"]
    gates = config["gates"]
    stage = identity["stage"]
    domains = {name for name, item in config["datasets"].items() if item["role"] == "confirmatory"}

    def number(mapping, key):
        value = mapping[key]
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError(f"gate metric {key} must be a finite number")
        return value

    def domain_rows():
        rows = metrics["domains"]
        if not isinstance(rows, dict) or set(rows) != domains:
            raise ValueError("gate must cover every locked confirmatory domain exactly")
        return list(rows.values())

    if stage == "hard_bank_gate":
        rule = gates["hard_bank"]
        for key in ("lock_artifact_sha256", "candidate_inventory_sha256", "source_inventory_sha256"):
            if (not isinstance(metrics.get(key), str) or len(metrics[key]) != 64
                    or any(character not in "0123456789abcdef" for character in metrics[key])):
                raise ValueError(f"hard-bank gate requires a lowercase SHA-256 {key}")
        if dependencies is not None:
            lock_result = dependencies.get("hard_bank_lock")
            matching = ([artifact for artifact in lock_result.value["artifacts"]
                         if artifact["sha256"] == metrics["lock_artifact_sha256"]]
                        if lock_result is not None else [])
            if not matching:
                raise ValueError("hard-bank gate lock hash is not an artifact of its exact lock dependency")
            lock_payloads = []
            for artifact in matching:
                payload = _read(lock_result.directory / artifact["path"])
                content = dict(payload)
                content_sha256 = content.pop("content_sha256", None)
                if payload.get("schema") != "apbpf-hard-bank-lock-v2" or content_sha256 != digest(content):
                    raise ValueError("matched hard-bank artifact is not a sealed v2 population lock")
                lock_payloads.append(payload)
            lock_payload = lock_payloads[0]
            if any(payload != lock_payload for payload in lock_payloads[1:]):
                raise ValueError("lock dependency contains conflicting artifacts with the declared SHA-256")
            population_lock = lock_payload.get("population_lock", {})
            groups = lock_payload.get("groups", [])
            outcomes = _hard_bank_binary([row["visible_outcomes"] for row in groups],
                                         name="visible_outcomes", dimensions=3)
            group_ids = _ordered_group_ids([row["group_id"] for row in groups])
            sources = _ordered_group_ids([row["source_component_id"] for row in groups])
            candidate_digest = _candidate_inventory_digest(
                [row["candidate_ids"] for row in groups], groups=len(groups),
                candidates=outcomes.shape[1])
            visible_digest = _visible_outcomes_digest(outcomes)
            population_digest = _population_digest(
                group_ids, population_lock.get("split"), population_lock.get("provenance"),
                visible_digest, candidate_digest, sources)
            source_inventory = [{"group_id": group_id, "source_component_id": source}
                                for group_id, source in zip(group_ids, sources, strict=True)]
            if (not groups
                    or list(group_ids) != population_lock.get("group_ids")
                    or list(sources) != population_lock.get("source_component_ids")
                    or visible_digest != population_lock.get("visible_outcomes_digest")
                    or population_digest != population_lock.get("digest")
                    or metrics["candidate_inventory_sha256"] != candidate_digest
                    or metrics["candidate_inventory_sha256"] != lock_payload.get("candidate_inventory_sha256")
                    or metrics["candidate_inventory_sha256"] != population_lock.get("candidate_inventory_digest")
                    or metrics["source_inventory_sha256"] != lock_payload.get("source_inventory_sha256")
                    or metrics["source_inventory_sha256"] != digest(source_inventory)):
                raise ValueError("hard-bank gate inventory hashes do not match the sealed lock content")
        passed = (metrics.get("lock_schema") == "apbpf-hard-bank-lock-v2"
                  and metrics.get("audit_schema") == "apbpf-hard-bank-audit-v2"
                  and metrics.get("lock_precedes_hidden_execution") is True
                  and metrics.get("candidate_inventory_bound") is True
                  and metrics.get("source_inventory_bound") is True
                  and metrics.get("pilot_primary_source_disjoint") is True
                  and number(metrics, "mixed_pilot_groups") >= rule["minimum_mixed_pilot_groups"]
                  and rule["minimum_confirmatory_groups"] <= number(metrics, "confirmatory_groups") <= rule["maximum_confirmatory_groups"]
                  and number(metrics, "visible_selection_pass1") < number(metrics, "hidden_oracle_pass1"))
    elif stage == "baseline_fairness_gate":
        rule = gates["baseline_fairness"]
        comparisons = metrics["baselines"]
        if set(comparisons) != set(rule["baselines"]):
            raise ValueError("fairness gate requires every locked baseline")
        passed = all(number(row, "nll_advantage") >= rule["minimum_nll_advantage"]
                     and number(row, "clustered_lower_bound") > 0 for row in comparisons.values())
    elif stage == "association_gate":
        if metrics.get("population") != "full_locked_population":
            raise ValueError("association gate cannot use a favorable subset")
        passed = all(number(row, "gap_nats_per_test") >= gates["association"]["minimum_gap_nats_per_test"]
                     and number(row, "clustered_lower_bound") > 0 for row in domain_rows())
    elif stage == "pair_invariance_gate":
        rule = gates["pair_invariance"]
        if rule.get("required_controls") != ["joint_reversal", "presentation_permutation"]:
            raise ValueError("pair-invariance gate requires both locked pair-preserving controls")
        fraction = rule["maximum_joint_reversal_fraction_of_shuffle_gap"]
        passed = all(number(row, "shuffle_gap") > 0 and
                     number(row, "joint_reversal_degradation") <= fraction * number(row, "shuffle_gap") and
                     number(row, "presentation_permutation_degradation") <= fraction * number(row, "shuffle_gap")
                     for row in domain_rows())
    elif stage == "oracle_headroom_gate":
        threshold = gates["oracle_headroom"]["minimum_nll_advantage_over_fixed_and_random"]
        passed = all(number(row, "nll_advantage_over_fixed") >= threshold and
                     number(row, "nll_advantage_over_random") >= threshold for row in domain_rows())
    elif stage == "active_testing_gate":
        rule = gates["active_testing"]
        passed = all((row.get("matched_hidden_quality") is True and
                      number(row, "test_reduction") >= rule["minimum_test_reduction_at_matched_hidden_quality"])
                     or (row.get("fixed_budget") == 4 and row.get("all_policies_execute_exact_budget") is True and
                         number(row, "nll_advantage_at_four_tests") >= rule["minimum_nll_advantage_at_four_tests"])
                     for row in domain_rows())
    elif stage == "selection_gate":
        if metrics.get("comparator") != gates["selection"]["comparator"]:
            raise ValueError("selection requires the strongest cross-fitted deterministic comparator")
        passed = all(number(row, "absolute_selected_pass1_advantage") >= gates["selection"]["minimum_absolute_selected_pass1_advantage"]
                     and number(row, "clustered_lower_bound") > 0 for row in domain_rows())
    else:  # replication_gate
        families = {item["family"] for item in config["models"].values()}
        cells = metrics["cells"]
        keys = [(row["domain"], row["family"]) for row in cells]
        if len(keys) != len(set(keys)) or set(keys) != {(domain, family) for domain in domains for family in families}:
            raise ValueError("replication must cover the complete locked domain/family cross product")
        passed = all(number(row, "association_gap") > 0 and number(row, "selection_advantage") > 0 for row in cells)
    if result["gate"]["passed"] is not passed:
        raise ValueError("worker gate decision disagrees with locked threshold evaluation")


def _identity(resolved, stage, dependencies, failed_gates):
    smoke = resolved.backend == "fake"
    exploratory = resolved.config['execution']['claim_status'] == 'exploratory-predeclared'
    return {"schema": SCHEMA, "stage": stage, "fingerprint": resolved.fingerprint,
            "backend": resolved.backend, "dependencies": {name: value.checksum for name, value in dependencies.items()},
            "confirmatory": not smoke and not exploratory and not failed_gates,
            "claim_status": "smoke-only-no-claim" if smoke else ('exploratory-predeclared' if exploratory else
                            ("exploratory-after-failed-gate" if failed_gates else "prospective-gated")),
            "failed_gates": list(failed_gates)}


def command_for(resolved, stage, directory):
    """Return actual argv without shell interpolation; only two path tokens exist."""
    configured = resolved.config["site"]["commands"].get(stage)
    if configured is None:
        return None
    replacements = {"{request}": str(directory / "request.json"), "{result}": str(directory / "worker-result.json")}
    return [replacements.get(token, token) for token in configured]


def retry_command(resolved, root, stage, *, continue_exploratory=False, through_stage=None):
    argv = [sys.executable, "-m", "pbpf.apbpf.cli", "rerun-stage", "--config", str(resolved.source),
            "--profile", resolved.profile, "--output-root", str(Path(root).parent), "--stage", stage]
    if resolved.site:
        argv += ["--site", str(resolved.site)]
    if continue_exploratory:
        argv.append("--continue-exploratory")
    if through_stage is not None:
        argv += ['--through-stage', through_stage]
    return argv


def run_after_changes_command(resolved, root, *, continue_exploratory=False, through_stage=None):
    """Start/resume the new fingerprint produced by changed code or provisioning."""
    argv = [sys.executable, "-m", "pbpf.apbpf.cli", "run", "--config", str(resolved.source),
            "--profile", resolved.profile, "--output-root", str(Path(root).parent), "--resume"]
    if resolved.site:
        argv += ["--site", str(resolved.site)]
    if continue_exploratory:
        argv.append("--continue-exploratory")
    if through_stage is not None:
        argv += ['--through-stage', through_stage]
    return argv


def doctor(resolved, *, through_stage=None):
    if through_stage is not None and through_stage not in STAGES:
        raise ValueError('unknown terminal stage')
    required = STAGES if through_stage is None else STAGES[:STAGES.index(through_stage)+1]
    commands = resolved.config["site"]["commands"]
    missing = [name for name in required if not commands.get(name)] if resolved.backend == "real" else []
    revisions = resolved.config["site"]["worker_revisions"]
    missing_revisions = ([name for name in required if commands.get(name) and not revisions.get(name)]
                         if resolved.backend == "real" else [])
    missing_paths = {}
    unavailable = {}
    revision_mismatches = {}
    if resolved.backend == "real":
        for name, value in resolved.config["site"]["paths"].items():
            if value is None or not Path(value).is_dir():
                missing_paths[name] = {"path": value, "reason": "unset" if value is None else "directory does not exist"}
        cwd = resolved.config["site"]["working_directory"]
        if not Path(cwd).is_dir():
            missing_paths["working_directory"] = {"path": cwd, "reason": "directory does not exist"}
        for name, argv in commands.items():
            if name not in required:
                continue
            if argv:
                executable = str(Path(cwd) / argv[0]) if "/" in argv[0] and not Path(argv[0]).is_absolute() else argv[0]
                if shutil.which(executable) is None:
                    unavailable[name] = argv[0]
        for name, value in resolved.config["site"]["worker_files"].items():
            if name not in required:
                continue
            if value is None:
                continue
            path = Path(value)
            if not path.is_file():
                revision_mismatches[name] = {"path": value, "reason": "worker file does not exist"}
            else:
                actual = _sha(path)
                expected = revisions.get(name)
                if actual != expected:
                    revision_mismatches[name] = {"path": value, "expected": expected, "actual": actual}
    return {"schema": SCHEMA, "fingerprint": resolved.fingerprint, "profile": resolved.profile,
            "backend": resolved.backend, "claim_status": resolved.config["execution"]["claim_status"],
            "configuration_valid": True,
            "ready": not missing and not missing_revisions and not missing_paths and not unavailable and not revision_mismatches,
            "missing_commands": missing, "missing_paths": missing_paths,
            "missing_worker_revisions": missing_revisions,
            "unavailable_executables": unavailable,
            "worker_revision_mismatches": revision_mismatches,
            "commands": commands, "stages": list(STAGES), "required_stages": list(required), "dependencies": DEPENDENCIES,
            "note": "Readiness checks provisioning only; no worker or experiment has run."}


def _execute_stage(resolved, root, identity, dependencies, *, fail_smoke_gate=None, through_stage=None):
    stage = identity["stage"]
    attempts = _attempts(root, stage)
    number = int(attempts[-1].name.split("-")[1]) + 1 if attempts else 1
    directory = root / "stages" / stage / f"attempt-{number:06d}"
    directory.mkdir(parents=True, exist_ok=False)
    write_once(directory / "identity.json", identity)
    request = {**identity, "config": resolved.config,
               "inputs": {name: {"result": str(result.directory / "result.json"), "checksum": result.checksum}
                          for name, result in dependencies.items()},
               "result_path": str(directory / "worker-result.json"), "outputs_directory": str(directory / "outputs")}
    write_once(directory / "request.json", request)
    argv = command_for(resolved, stage, directory)
    write_once(directory / "command.json", {"argv": argv, "shell": False,
               "display": shlex.join(argv) if argv else None,
               "working_directory": resolved.config["site"]["working_directory"], "backend": resolved.backend})
    try:
        if resolved.backend == "fake":
            result = {"schema": "apbpf-stage-result-v1", "stage": stage,
                      "fingerprint": resolved.fingerprint, "dependencies": identity["dependencies"],
                      "summary": {"label": "smoke-only-no-claim", "scientific_measurements": False}, "artifacts": []}
            if stage in GATE_STAGES:
                result["gate"] = {"passed": stage != fail_smoke_gate, "reason": "synthetic smoke control; not a scientific decision"}
        else:
            diagnostics = doctor(resolved, through_stage=through_stage)
            if not diagnostics["ready"]:
                details = {name: diagnostics[name] for name in
                           ("missing_commands", "missing_worker_revisions", "missing_paths",
                            "unavailable_executables", "worker_revision_mismatches") if diagnostics[name]}
                raise PipelineError(f"Real backend provisioning is incomplete: {json.dumps(details, sort_keys=True)}")
            if argv is None:
                raise PipelineError(f"No real command provisioned for {stage}; add site.commands.{stage} as an argv list")
            (directory / "outputs").mkdir()
            environment = dict(os.environ, APBPF_REQUEST=str(directory / "request.json"),
                               APBPF_RESULT=str(directory / "worker-result.json"),
                               APBPF_OUTPUTS=str(directory / "outputs"))
            with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
                process = subprocess.run(argv, cwd=resolved.config["site"]["working_directory"],
                                         env=environment, stdout=stdout, stderr=stderr,
                                         timeout=resolved.config["site"]["timeout_seconds"], check=False)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, argv)
            result = _read(directory / "worker-result.json")
        if _read(directory / "identity.json") != identity or _read(directory / "request.json") != request:
            raise ValueError("worker modified immutable identity/request records")
        if "runner_lineage" in result:
            raise ValueError("worker must not supply the runner-owned lineage field")
        result = {**result, "runner_lineage": _runner_lineage(identity, dependencies)}
        _validate_result(result, identity)
        _validate_transitive_lineage(result, identity, dependencies)
        _validate_gate_decision(result, identity, resolved.config, dependencies)
        for artifact in result["artifacts"]:
            target = directory / artifact["path"]
            if not target.resolve().is_relative_to((directory / "outputs").resolve()) or target.is_symlink():
                raise ValueError("worker output escaped the immutable attempt directory")
            if _sha(target) != artifact["sha256"]:
                raise ValueError(f"worker artifact checksum mismatch: {target}")
        write_once(directory / "result.json", result)
        write_once(directory / "work.json", {"identity_hash": digest(identity), "result_hash": digest(result),
                                             "input_checksums": identity["dependencies"]})
        write_once(directory / "summary.json", {**identity, "summary": result["summary"], "gate": result.get("gate")})
        if stage in GATE_STAGES and not result["gate"]["passed"]:
            write_once(directory / "gate-failure.json", {**identity, "decision": result["gate"],
                       "retry_command": retry_command(resolved, root, stage,
                                                      continue_exploratory=bool(identity["failed_gates"]), through_stage=through_stage),
                       "run_after_changes_command": run_after_changes_command(
                           resolved, root, continue_exploratory=bool(identity["failed_gates"]), through_stage=through_stage),
                       "remediation": "Stop confirmatory claims; --continue-exploratory permits only explicitly nonconfirmatory downstream artifacts."})
        files = {file.relative_to(directory).as_posix(): _sha(file) for file in directory.rglob("*") if file.is_file()}
        write_once(directory / "complete.json", {"identity_hash": digest(identity), "files": files})
        return _load(directory)
    except Exception as exc:
        failure = {**identity, "error_type": type(exc).__name__, "message": str(exc),
                   "command": argv, "command_display": shlex.join(argv) if argv else None,
                   "returncode": getattr(exc, "returncode", None),
                   "retry_command": retry_command(resolved, root, stage,
                                                  continue_exploratory=bool(identity["failed_gates"]), through_stage=through_stage),
                   "run_after_changes_command": run_after_changes_command(
                       resolved, root, continue_exploratory=bool(identity["failed_gates"]), through_stage=through_stage),
                   "remediation": "Inspect stderr.log, provision/fix the real worker, then rerun; changed config/source uses a new fingerprint.",
                   "synthetic_fallback": False}
        write_once(directory / "failure.json", failure)
        raise PipelineError(f"{stage} failed: {exc}; metadata: {directory / 'failure.json'}") from exc


def _descendants(stage):
    # Gate labels propagate globally, so replay the suffix, not only graph edges.
    return set(STAGES[STAGES.index(stage):])


def run_pipeline(resolved, root, *, resume=False, continue_exploratory=False,
                 rerun_stage=None, fail_smoke_gate=None, through_stage=None):
    """Resume matching immutable inputs; reruns append attempts for the suffix."""
    root = Path(root).resolve()
    if root.name != resolved.fingerprint:
        raise ValueError("run directory must be named by the exact configuration fingerprint")
    if rerun_stage is not None and rerun_stage not in STAGES:
        raise ValueError("unknown rerun stage")
    if through_stage is not None and (through_stage not in STAGES or
            (rerun_stage is not None and STAGES.index(through_stage) < STAGES.index(rerun_stage))):
        raise ValueError('terminal stage must be known and not precede rerun stage')
    if fail_smoke_gate is not None and (resolved.backend != "fake" or fail_smoke_gate not in GATE_STAGES):
        raise ValueError("failure injection is available only for smoke gate stages")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "identity.json").exists() and not (resume or rerun_stage):
            raise ValueError("run already exists; use --resume or rerun-stage")
        write_once(root / "identity.json", {"schema": SCHEMA, "fingerprint": resolved.fingerprint,
                                          "resolved": resolved.config})
        if fail_smoke_gate is None and (root / "smoke-control.json").exists():
            fail_smoke_gate = _read(root / "smoke-control.json")["fail_gate"]
        write_once(root / "smoke-control.json", {"fail_gate": fail_smoke_gate})
        if rerun_stage and not _attempts(root, rerun_stage):
            raise ValueError("rerun-stage requires an existing attempt; use run for a new DAG")
        forced = _descendants(rerun_stage) if rerun_stage else set()
        results, failed = {}, []
        for stage in STAGES:
            dependencies = {name: results[name] for name in DEPENDENCIES[stage]}
            identity = _identity(resolved, stage, dependencies, failed)
            attempts = _attempts(root, stage)
            latest = attempts[-1] if attempts else None
            if stage not in forced and latest is not None and (latest / "complete.json").exists():
                result = _load(latest)
                if result.identity != identity:
                    raise ValueError(f"{stage}: identity/dependency mismatch; use rerun-stage --stage {stage}")
                _validate_transitive_lineage(result.value, result.identity, dependencies)
                _validate_gate_decision(result.value, result.identity, resolved.config, dependencies)
            else:
                if latest is not None and (latest / "identity.json").exists() and stage not in forced:
                    if _read(latest / "identity.json") != identity:
                        raise ValueError(f"{stage}: partial attempt identity mismatch; explicit rerun required")
                result = _execute_stage(resolved, root, identity, dependencies,
                                        fail_smoke_gate=fail_smoke_gate, through_stage=through_stage)
            results[stage] = result
            if stage in GATE_STAGES and not result.value["gate"]["passed"]:
                failed.append({"stage": stage, "checksum": result.checksum})
                if not continue_exploratory:
                    raise GateFailure(f"{stage} failed; downstream stages stopped. --continue-exploratory labels all subsequent work nonconfirmatory.")
            if stage == through_stage:
                break
        return report_run(resolved, root)


def report_run(resolved, root, *, require_complete=False):
    """Read-only checksum/provenance validation, including incomplete DAG reports."""
    root = Path(root).resolve()
    metadata = _read(root / "identity.json")
    if metadata != {"schema": SCHEMA, "fingerprint": resolved.fingerprint, "resolved": resolved.config}:
        raise ValueError("run fingerprint/configuration mismatch")
    results, stages, failed = {}, {}, []
    for stage in STAGES:
        attempts = _attempts(root, stage)
        if not attempts:
            stages[stage] = {"status": "pending"}
            continue
        latest = attempts[-1]
        if not (latest / "complete.json").exists():
            failure_path = latest / "failure.json"
            stages[stage] = {"status": "failed" if failure_path.exists() else "interrupted",
                             "attempt": str(latest), "failure": _read(failure_path) if failure_path.exists() else None}
            continue
        result = _load(latest)
        missing_upstream = [name for name in DEPENDENCIES[stage] if name not in results]
        if missing_upstream:
            stages[stage] = {"status": "stale", "checksum": result.checksum,
                             "attempt": str(latest), "reason": "upstream incomplete or stale",
                             "missing_upstream": missing_upstream}
            continue
        expected = _identity(resolved, stage, {name: results[name] for name in DEPENDENCIES[stage]}, failed)
        if result.identity != expected:
            stages[stage] = {"status": "stale", "checksum": result.checksum,
                             "attempt": str(latest),
                             "reason": "identity or upstream provenance changed; rerun this stage"}
            continue
        dependencies = {name: results[name] for name in DEPENDENCIES[stage]}
        _validate_transitive_lineage(result.value, result.identity, dependencies)
        _validate_gate_decision(result.value, result.identity, resolved.config, dependencies)
        results[stage] = result
        stages[stage] = {"status": "complete", "checksum": result.checksum, "attempt": str(latest),
                         "confirmatory": result.identity["confirmatory"], "gate": result.value.get("gate")}
        if stage in GATE_STAGES and not result.value["gate"]["passed"]:
            failed.append({"stage": stage, "checksum": result.checksum})
    complete = len(results) == len(STAGES)
    if require_complete and not complete:
        raise ValueError("verification requires every DAG stage to be complete")
    eligible = complete and resolved.backend == "real" and not failed and all(item.identity["confirmatory"] for item in results.values())
    return {"schema": SCHEMA, "fingerprint": resolved.fingerprint, "run_directory": str(root),
            "complete": complete, "backend": resolved.backend, "failed_gates": failed,
            "claim_status": "smoke-only-no-claim" if resolved.backend == "fake" else
                ('exploratory-predeclared' if resolved.config['execution']['claim_status'] == 'exploratory-predeclared'
                 else ("exploratory-after-failed-gate" if failed else "prospective-gated")),
            "main_table_eligible": eligible, "stages": stages}
