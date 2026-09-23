#!/usr/bin/env python3
"""Run the closed-loop EESD recursive study for equal-weight SD vs full EESD.

The main shared_eesd_teacher mode forks both updates from one teacher and one
experience bank each round, continuing with the EESD checkpoint. The legacy
arm-specific mode is supplemental. Each round uses fresh greedy primary evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path
import subprocess
import sys

import yaml


RULES = ("equal_weight", "eesd_full")
# Reuse exactly the direct-training receipt checks, without importing a GPU model.
_TRAINING = runpy.run_path(str(Path(__file__).with_name("run_eesd_downstream.py")))



def call(command, *, root: Path):
    print(json.dumps({"command": command}), flush=True)
    subprocess.check_call(command, cwd=root)


def generation_script(domain: str) -> str:
    return (
        "scripts/generate_apbpf_rbr_bank.py"
        if domain == "rbr"
        else "scripts/generate_apbpf_codearc_bank.py"
    )


def generate_bank(
    *,
    root: Path,
    domain: str,
    public_root: Path,
    family: str,
    split: str,
    output: Path,
    seed: int,
    adapter: Path | None,
    greedy: bool,
):
    if (output / "complete.json").exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        generation_script(domain),
        "--public-root", str(public_root),
        "--output", str(output),
        "--family", family,
        "--split", split,
        "--candidates", "1",
        "--seed", str(seed),
    ]
    if greedy:
        command.append("--greedy")
    if adapter is not None:
        command += ["--adapter", str(adapter)]
    call(command, root=root)


def evaluate_primary(
    *,
    root: Path,
    domain: str,
    evaluator_root: Path,
    bank: Path,
    output: Path,
):
    if (output / "report.json").exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    call(
        [
            sys.executable,
            "scripts/evaluate_eesd_fresh_bank.py",
            "--domain", domain,
            "--evaluator-root", str(evaluator_root),
            "--bank", str(bank),
            "--output", str(output),
        ],
        root=root,
    )


def compare_reports(root: Path, baseline: Path, method: Path, output: Path):
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    call(
        [
            sys.executable,
            "scripts/compare_eesd_fresh_eval.py",
            "--baseline", str(baseline),
            "--method", str(method),
            "--output", str(output),
        ],
        root=root,
    )


def collect_experience(
    *,
    root: Path,
    domain: str,
    public_root: Path,
    model_config: Path,
    family: str,
    output: Path,
    seed: int,
    adapter: Path | None,
    config: Path,
    alpha: float,
    uncertainty_penalty: float,
    relevance_strength: float,
):
    """Generate current-policy originals, corrections, and all locked trust weights."""
    train_bank = output / "original-banks" / "train"
    development_bank = output / "original-banks" / "development"
    generate_bank(
        root=root, domain=domain, public_root=public_root, family=family,
        split="train", output=train_bank, seed=seed, adapter=adapter, greedy=False,
    )
    generate_bank(
        root=root, domain=domain, public_root=public_root, family=family,
        split="development", output=development_bank, seed=seed, adapter=adapter, greedy=False,
    )

    public_bank = output / "public-corrections"
    if not (public_bank / "audit.json").exists():
        public_bank.parent.mkdir(parents=True, exist_ok=True)
        call(
            [
                sys.executable,
                "scripts/prepare_eesd_recursive_corrections.py",
                "--domain", domain,
                "--public-root", str(public_root),
                "--bank", str(train_bank),
                "--bank", str(development_bank),
                "--output", str(public_bank),
            ],
            root=root,
        )

    generated = output / "generated-corrections"
    if not (generated / "report.json").exists():
        generated.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "scripts/generate_eesd_corrections.py",
            "--public-bank", str(public_bank / "public-corrections.jsonl"),
            "--model-config", str(model_config),
            "--domain", domain,
            "--output", str(generated),
            "--relevance-strength", str(relevance_strength),
        ]
        if adapter is not None:
            command += ["--adapter", str(adapter)]
        call(command, root=root)

    attributed = output / "shapley-relevance"
    if not (attributed / "report.json").exists():
        attributed.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "scripts/score_eesd_shapley_relevance.py",
            "--input", str(generated / "corrections.jsonl"),
            "--model-config", str(model_config),
            "--output", str(attributed),
            "--mode", "exact",
        ]
        if adapter is not None:
            command += ["--adapter", str(adapter)]
        call(command, root=root)

    scored = output / "scored"
    if not (scored / "summary.json").exists():
        scored.parent.mkdir(parents=True, exist_ok=True)
        call(
            [
                sys.executable,
                "scripts/score_eesd_corrections.py",
                "--input", str(attributed / "corrections-shapley.jsonl"),
                "--config", str(config),
                "--output", str(scored),
                "--alpha", str(alpha),
            ],
            root=root,
        )
    return scored / "scored-corrections.jsonl"


def train_next(
    *,
    root: Path,
    model_config: Path,
    scored: Path,
    rule: str,
    previous_adapter: Path | None,
    output: Path,
    seed: int,
    max_steps: int,
    anchor_beta: float,
    response_token_budget: int,
):
    task = _TRAINING['make_training_task'](
        root=root, scored=scored, model_config=model_config, rule=rule,
        previous_adapter=previous_adapter, output=output, seed=seed,
        budget=response_token_budget, max_steps=max_steps, anchor_beta=anchor_beta,
    )
    if task['status'] not in {'verified_complete', 'verified_non_estimable'}:
        output.parent.mkdir(parents=True, exist_ok=True)
        call(task['command'], root=root)
        return _TRAINING['check_training_binding'](task, seal=True)
    return _TRAINING['check_training_binding'](task)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/experiments/eesd_iclr2027.yaml"))
    p.add_argument("--domain", choices=["rbr", "codearc"], required=True)
    p.add_argument("--public-root", type=Path, required=True)
    p.add_argument("--evaluator-root", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--response-token-budget", type=int, required=True)
    p.add_argument("--experience-policy", choices=["shared_eesd_teacher", "arm-specific"], required=True,
                   help="Main shared teacher protocol, or explicit supplemental arm-specific protocol")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--uncertainty-penalty", type=float, default=0.5)
    p.add_argument("--anchor-beta", type=float, default=0.03)
    p.add_argument("--relevance-strength", type=float, default=16.0)
    args = p.parse_args()
    if args.rounds < 1 or args.max_steps < 1 or args.response_token_budget < 1:
        raise ValueError("round and training budgets must be positive")

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("schema") != "eesd-iclr2027-v1":
        raise ValueError("locked EESD config required")
    if args.seed not in [int(x) for x in cfg["seeds"]]:
        raise ValueError("recursive seed must be one of the locked seeds")
    public_root = args.public_root.resolve()
    evaluator_root = args.evaluator_root.resolve()
    model_config = args.model_config.resolve()
    if not public_root.is_dir() or not evaluator_root.is_dir() or not model_config.is_file():
        raise FileNotFoundError("recursive roots/model config are missing")
    args.output = args.output.resolve()
    digest = _TRAINING['file_digest']
    identity = {
        "schema": "eesd-recursive-run-v1", "domain": args.domain, "family": args.family,
        "zero_weight_protocol": "eesd-zero-positive-training-weight-v1",
        "zero_weight_adoption_sha256": digest(root / "docs/EESD_ZERO_WEIGHT_ADOPTION_20260920.md"),
        "seed": args.seed, "rounds": args.rounds, "response_token_budget": args.response_token_budget,
        "max_steps": args.max_steps, "experience_policy": args.experience_policy,
        "alpha": args.alpha,
        "trust_rule": "posterior_excess_benefit_confidence",
        "relevance_method": "exact_evidence_shapley_edit_logprob_contrast",
        "anchor_beta": args.anchor_beta,
        "legacy_uncertainty_penalty_argument_ignored": args.uncertainty_penalty,
        "legacy_generation_relevance_strength_ignored_by_eesd": args.relevance_strength,
        "public_root": str(public_root), "evaluator_root": str(evaluator_root),
        "config_sha256": digest(args.config), "model_config_sha256": digest(model_config),
        "public_manifest_sha256": digest(public_root / 'manifest.json'),
        "evaluator_manifest_sha256": digest(evaluator_root / 'manifest.json'),
        "source_sha256": {name: digest(root / name) for name in (
            'scripts/run_eesd_recursive.py', 'scripts/run_eesd_downstream.py',
            'scripts/run_eesd_weighted_sft.py', 'src/pbpf/eesd/training_outcomes.py')},
    }
    lock = args.output / "recursive-run.json"
    if lock.exists():
        if json.loads(lock.read_text()) != identity:
            raise ValueError("recursive request identity/budget/policy mismatch; preserve existing output")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("unbound legacy recursive output; preserve and use a new output root")
        args.output.mkdir(parents=True, exist_ok=True)
        with lock.open('x') as stream:
            json.dump(identity, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')

    # Round-0 fresh evaluation: one shared base-policy result.
    base_bank = args.output / "round0" / "base" / "primary-bank"
    generate_bank(
        root=root, domain=args.domain, public_root=public_root, family=args.family,
        split="primary", output=base_bank, seed=args.seed, adapter=None, greedy=True,
    )
    base_eval = args.output / "round0" / "base" / "fresh-eval"
    evaluate_primary(
        root=root, domain=args.domain, evaluator_root=evaluator_root,
        bank=base_bank, output=base_eval,
    )

    current = {rule: None for rule in RULES}
    previous_eval = {rule: base_eval / "report.json" for rule in RULES}
    base_report = base_eval / "report.json"
    summaries = []
    shared_mode = args.experience_policy == "shared_eesd_teacher"
    previous_outcomes = {}
    for round_index in range(1, args.rounds + 1):
        shared_round = shared_mode or round_index == 1
        teacher = None if round_index == 1 else current["eesd_full"]
        teacher_report = base_report if round_index == 1 else previous_eval["eesd_full"]
        plans = {}
        blocked = {}
        if round_index > 1:
            for rule in RULES:
                ancestor_rule = "eesd_full" if shared_mode else rule
                ancestor = previous_outcomes[ancestor_rule]
                if ancestor["status"] != "trained":
                    blocked[rule] = ancestor.get("dependency", {
                        "round": round_index - 1, "rule": ancestor_rule,
                        "status": ancestor["status"], "receipt": ancestor["receipt"],
                        "receipt_sha256": ancestor["receipt_sha256"],
                    }) if ancestor["status"] != "blocked_dependency" else ancestor["dependency"]
        # Freeze all parents before either arm updates. In the shared protocol,
        # equal_r is a matched fork of EESD_(r-1), not equal_(r-1)'s continuation.
        for rule in RULES:
            if rule in blocked:
                continue
            if shared_round and plans:
                plans[rule] = dict(plans[RULES[0]])
                continue
            parent = teacher if shared_round else current[rule]
            parent_report = teacher_report if shared_round else previous_eval[rule]
            experience = (args.output / f"round{round_index-1}" / "shared-experience" if shared_round
                          else args.output / f"round{round_index-1}" / rule / "experience")
            scored = collect_experience(
                root=root, domain=args.domain, public_root=public_root,
                model_config=model_config, family=args.family, output=experience,
                seed=args.seed, adapter=parent, config=args.config.resolve(),
                alpha=args.alpha, uncertainty_penalty=args.uncertainty_penalty,
                relevance_strength=args.relevance_strength,
            )
            plans[rule] = {"previous_adapter": str(parent) if parent else None,
                "previous_adapter_sha256": _TRAINING['adapter_tree_digest'](parent) if parent else None,
                "scored": str(scored), "scored_sha256": digest(scored),
                "parent_evaluation": str(parent_report), "parent_evaluation_sha256": digest(parent_report)}
        update_inputs = {
            "schema": "eesd-recursive-update-inputs-v1", "round": round_index,
            "experience_policy": args.experience_policy, "seed": args.seed,
            "response_token_budget": args.response_token_budget,
            "shared_experience": shared_round,
            "teacher_rule": ("base" if round_index == 1 else "eesd_full") if shared_round else "arm-specific",
            "teacher_round": round_index - 1,
            "teacher_adapter": str(teacher) if teacher and shared_round else None,
            "teacher_adapter_sha256": plans[RULES[0]]["previous_adapter_sha256"] if shared_round and plans else None,
            "shared_scored_sha256": plans[RULES[0]]["scored_sha256"] if shared_round and plans else None,
            "arms": plans,
            "blocked_dependencies": blocked,
            "recursive_run_sha256": digest(lock),
        }
        update_lock = args.output / f"round{round_index}" / "update-inputs.json"
        if update_lock.exists():
            if json.loads(update_lock.read_text()) != update_inputs:
                raise ValueError("recursive teacher/shared bank/budget update binding mismatch")
        else:
            if any((args.output / f"round{round_index}" / rule / "training").exists() for rule in RULES):
                raise ValueError("unbound recursive update inputs; preserve existing training output")
            update_lock.parent.mkdir(parents=True, exist_ok=True)
            with update_lock.open('x') as stream:
                json.dump(update_inputs, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write('\n')
        outcomes = {}
        for rule in RULES:
            if rule in blocked:
                outcomes[rule] = {"status": "blocked_dependency", "pass_at_1": None,
                    "dependency": blocked[rule], "adapter": None, "requested_response_tokens": args.response_token_budget,
                    "actual_response_tokens": 0, "evaluation": None}
                continue
            plan = plans[rule]
            outcome = train_next(
                root=root, model_config=model_config, scored=Path(plan["scored"]), rule=rule,
                previous_adapter=Path(plan["previous_adapter"]) if plan["previous_adapter"] else None,
                output=args.output / f"round{round_index}" / rule / "training",
                seed=args.seed, max_steps=args.max_steps, anchor_beta=args.anchor_beta,
                response_token_budget=args.response_token_budget,
            )
            if (not isinstance(outcome, dict) or outcome.get("status") not in {
                    "trained", "non_estimable_zero_positive_training_weight"}
                    or digest(Path(outcome["receipt"])) != outcome["receipt_sha256"]):
                raise ValueError("invalid or changed recursive training outcome receipt")
            outcomes[rule] = {**outcome, "pass_at_1": None, "evaluation": None,
                              "requested_response_tokens": args.response_token_budget}
            if outcome["status"] != "trained":
                if outcome.get("adapter") is not None:
                    raise ValueError("non-estimable update cannot provide a teacher adapter")
                outcomes[rule]["actual_response_tokens"] = 0
                current[rule] = None
                continue
            if not isinstance(outcome.get("adapter"), str) or not Path(outcome["adapter"]).is_absolute():
                raise ValueError("trained outcome requires an absolute adapter path")
            current[rule] = Path(outcome["adapter"])
            outcomes[rule]["actual_response_tokens"] = args.response_token_budget

            bank = args.output / f"round{round_index}" / rule / "primary-bank"
            generate_bank(
                root=root, domain=args.domain, public_root=public_root, family=args.family,
                split="primary", output=bank, seed=args.seed, adapter=current[rule], greedy=True,
            )
            evaluation = args.output / f"round{round_index}" / rule / "fresh-eval"
            evaluate_primary(
                root=root, domain=args.domain, evaluator_root=evaluator_root,
                bank=bank, output=evaluation,
            )
            current_report = evaluation / "report.json"
            vs_base = args.output / f"round{round_index}" / rule / "vs-base.json"
            vs_previous = args.output / f"round{round_index}" / rule / "vs-previous.json"
            compare_reports(root, base_report, current_report, vs_base)
            compare_reports(root, Path(plans[rule]["parent_evaluation"]), current_report, vs_previous)
            previous_eval[rule] = current_report
            outcomes[rule].update(evaluation=str(current_report), evaluation_sha256=digest(current_report),
                pass_at_1=json.loads(current_report.read_text())["fresh_all_tests_pass_at_1"])

        head_to_head = args.output / f"round{round_index}" / "eesd-vs-equal.json"
        comparison = None
        if all(outcomes[rule]["status"] == "trained" for rule in RULES):
            compare_reports(root, Path(outcomes["equal_weight"]["evaluation"]),
                Path(outcomes["eesd_full"]["evaluation"]), head_to_head)
            comparison = json.loads(head_to_head.read_text())
        round_outcome = {"schema": "eesd-recursive-round-outcomes-v1", "round": round_index,
                         "recursive_run_sha256": digest(lock), "update_inputs_sha256": digest(update_lock),
                         "arms": outcomes}
        outcome_path = args.output / f"round{round_index}" / "outcomes.json"
        if outcome_path.exists():
            if json.loads(outcome_path.read_text()) != round_outcome:
                raise ValueError("recursive round outcome/receipt binding mismatch")
        else:
            with outcome_path.open('x') as stream:
                json.dump(round_outcome, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write('\n')
        summaries.append({
            "round": round_index, "update_inputs_sha256": digest(update_lock),
            "outcomes_path": str(outcome_path), "outcomes_sha256": digest(outcome_path),
            "teacher_rule": update_inputs["teacher_rule"], "teacher_round": update_inputs["teacher_round"],
            "equal_weight_pass_at_1": outcomes["equal_weight"]["pass_at_1"],
            "eesd_pass_at_1": outcomes["eesd_full"]["pass_at_1"],
            "eesd_vs_equal": comparison, "arms": outcomes,
        })
        previous_outcomes = outcomes

    report = {
        "schema": "eesd-recursive-study-v1",
        "domain": args.domain,
        "family": args.family,
        "model_config": str(model_config),
        "seed": args.seed,
        "rounds": args.rounds,
        "rules": list(RULES),
        "round0_pass_at_1": json.loads(base_report.read_text())["fresh_all_tests_pass_at_1"],
        "summary": summaries,
        "coverage": {"complete": all(a["status"] == "trained" for row in summaries for a in row["arms"].values()),
            "planned_updates": args.rounds * len(RULES),
            "trained_updates": sum(a["status"] == "trained" for row in summaries for a in row["arms"].values()),
            "non_estimable_updates": sum(a["status"] == "non_estimable_zero_positive_training_weight" for row in summaries for a in row["arms"].values()),
            "blocked_updates": sum(a["status"] == "blocked_dependency" for row in summaries for a in row["arms"].values())},
        "experience_policy": args.experience_policy,
        "study_role": "main_matched_shared_teacher" if shared_mode else "supplemental_arm_specific",
        "teacher_selection": "base at round0, then eesd_full irrespective of evaluation" if shared_mode else "each arm continues itself",
        "vs_previous_definition": "shared teacher before this update" if shared_mode else "same arm previous round",
        "response_token_budget_per_arm_per_round": args.response_token_budget,
        "recursive_run_sha256": digest(lock),
        "experience_description": (
            "each round shares EESD-teacher experience, checkpoint and budget; next teacher is the EESD update"
            if shared_mode else "supplemental: round0 shared; later arm-specific experience and starting checkpoints"
        ),
        "evaluation_policy": "one fresh greedy primary candidate per source; all tests; no selection",
    }
    (args.output / "recursive-report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
