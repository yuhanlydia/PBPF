#!/usr/bin/env python3
"""Render completed EESD experiment artifacts into paper-ready LaTeX tables.

This script never invents or back-fills missing cells. It renders "--" for absent
predeclared artifacts and writes a machine-readable coverage report alongside TeX.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import yaml


MECHANISM_ORDER = [
    ("runbugrun", "qwen25_1p5b", "RunBugRun", "Qwen2.5-Coder-1.5B"),
    ("runbugrun", "qwen25_7b", "RunBugRun", "Qwen2.5-Coder-7B"),
    ("runbugrun", "qwen3_8b", "RunBugRun", "Qwen3-8B"),
    ("runbugrun", "deepseek_6p7b", "RunBugRun", "DeepSeek-Coder-6.7B"),
    ("runbugrun", "seed_coder_8b", "RunBugRun", "Seed-Coder-8B"),
    ("runbugrun", "qwen3_coder_30b", "RunBugRun", "Qwen3-Coder-30B-A3B"),
    ("codearc", "qwen25_1p5b", "CodeARC-Replay", "Qwen2.5-Coder-1.5B"),
    ("codearc", "qwen25_7b", "CodeARC-Replay", "Qwen2.5-Coder-7B"),
    ("codearc", "qwen3_8b", "CodeARC-Replay", "Qwen3-8B"),
    ("codearc", "deepseek_6p7b", "CodeARC-Replay", "DeepSeek-Coder-6.7B"),
    ("codearc", "seed_coder_8b", "CodeARC-Replay", "Seed-Coder-8B"),
    ("codearc", "qwen3_coder_30b", "CodeARC-Replay", "Qwen3-Coder-30B-A3B"),
]

TRANSFER_ORDER = [
    ("humaneval", "qwen25_7b", "HumanEval+", "Qwen2.5-Coder-7B"),
    ("mbpp", "qwen25_7b", "MBPP+", "Qwen2.5-Coder-7B"),
    ("humaneval", "deepseek_6p7b", "HumanEval+", "DeepSeek-Coder-6.7B"),
    ("mbpp", "deepseek_6p7b", "MBPP+", "DeepSeek-Coder-6.7B"),
]

TRANSFER_RULES = [
    ("no_update", "Base"),
    ("equal_weight", "Equal-weight SD"),
    ("final_correctness", "Correct-only SD"),
    ("fixed_mass_dirichlet", "Fixed-mass SD"),
    ("eesd_full", "EESD"),
]


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def f4(value):
    return "--" if value is None else f"{float(value):.4f}"


def pct(value):
    return "--" if value is None else f"{100.0*float(value):.2f}"


def mean_std(values):
    values = [float(x) for x in values if x is not None]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def latex_mean_std(values, *, percent=False):
    mean, std = mean_std(values)
    if mean is None:
        return "--"
    scale = 100.0 if percent else 1.0
    return f"{scale*mean:.2f} $\\pm$ {scale*std:.2f}"


def mechanism_table(root: Path, coverage: dict):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Effective-evidence mechanism across datasets and generator models. Same-$\alpha$ gain compares EED with fixed mass at the validation-selected EED $\alpha$ and relevance strength; global-mass gain compares EED with a validation-tuned constant evidence mass. Positive gains mean lower NLL for EED.}",
        r"\label{tab:eesd-mechanism-generated}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Generator & Fixed NLL & EED NLL & Same-$\alpha$ gain & Global-mass gain \\",
        r"\midrule",
    ]
    for dataset, model, dataset_label, model_label in MECHANISM_ORDER:
        path = root / "mechanism" / dataset / model / "report.json"
        report = read_json(path)
        coverage["mechanism"][f"{dataset}/{model}"] = bool(report)
        if report:
            fixed = report["metrics"]["fixed"]["nll"]
            eed = report["metrics"]["effective"]["nll"]
            same = report["bootstraps"]["effective_vs_fixed_at_effective_params"]["gain"]
            global_gain = report["bootstraps"]["effective_vs_tuned_global_mass"]["gain"]
        else:
            fixed = eed = same = global_gain = None
        lines.append(
            f"{dataset_label} & {model_label} & {f4(fixed)} & {f4(eed)} & "
            f"{f4(same)} & {f4(global_gain)} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return lines


def ablation_table(root: Path, coverage: dict):
    # Main causal ablation uses the primary Qwen2.5-7B RunBugRun cell.
    report = read_json(root / "mechanism" / "runbugrun" / "qwen25_7b" / "report.json")
    coverage["tables"]["causal_ablation"] = bool(report)
    rows = []
    if report:
        m = report["metrics"]
        b = report["bootstraps"]
        fixed = m["fixed"]["nll"]
        rows = [
            ("EED vs fixed at fixed-selected params", fixed, m["effective_at_fixed_params"]["nll"],
             b["effective_vs_fixed_at_fixed_params"]["gain"]),
            ("EED vs fixed at EED-selected params", m["fixed_at_effective_params"]["nll"], m["effective"]["nll"],
             b["effective_vs_fixed_at_effective_params"]["gain"]),
            ("EED vs tuned global mass", m["global_mass"]["nll"], m["effective"]["nll"],
             b["effective_vs_tuned_global_mass"]["gain"]),
            ("EED vs constant mean EED mass", m["constant_mean_mass"]["nll"], m["effective"]["nll"],
             m["constant_mean_mass"]["nll"] - m["effective"]["nll"]),
            ("EED vs mean permuted mass", m["permuted_mass"]["mean_nll"], m["effective"]["nll"],
             m["permuted_mass"]["mean_nll"] - m["effective"]["nll"]),
            ("PASS/non-PASS: EED vs fixed", m["binary_fixed_at_effective_params"]["nll"],
             m["binary_effective"]["nll"],
             m["binary_fixed_at_effective_params"]["nll"] - m["binary_effective"]["nll"]),
        ]
    else:
        rows = [(name, None, None, None) for name in (
            "EED vs fixed at fixed-selected params",
            "EED vs fixed at EED-selected params",
            "EED vs tuned global mass",
            "EED vs constant mean EED mass",
            "EED vs mean permuted mass",
            "PASS/non-PASS: EED vs fixed",
        )]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Causal mechanism ablations on the primary RunBugRun/Qwen2.5-Coder-7B cell. Positive NLL gain favors EED.}",
        r"\label{tab:eesd-ablation-generated}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Comparison & Control NLL & EED NLL & NLL gain \\",
        r"\midrule",
    ]
    for name, control, eed, gain in rows:
        lines.append(f"{name} & {f4(control)} & {f4(eed)} & {f4(gain)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


def transfer_table(root: Path, seeds, coverage: dict):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Official EvalPlus downstream transfer after RunBugRun self-distillation. Values are Base+Extra Pass@1 (mean $\pm$ s.d. across locked training seeds).}",
        r"\label{tab:eesd-transfer-generated}",
        r"\small",
        r"\begin{tabular}{ll" + "r"*len(TRANSFER_RULES) + "}",
        r"\toprule",
        "Dataset & Model & " + " & ".join(label for _, label in TRANSFER_RULES) + r" \\",
        r"\midrule",
    ]
    for dataset, model, dataset_label, model_label in TRANSFER_ORDER:
        values = []
        for rule, _ in TRANSFER_RULES:
            scores = []
            for seed in seeds:
                report = read_json(root / "transfer" / dataset / model / rule / f"seed{seed}" / "report.json")
                if report and report.get("evalplus_summary"):
                    scores.append(report["evalplus_summary"]["plus_pass_at_1"])
            coverage["transfer"][f"{dataset}/{model}/{rule}"] = len(scores)
            values.append(latex_mean_std(scores, percent=True))
        lines.append(f"{dataset_label} & {model_label} & " + " & ".join(values) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return lines


def fresh_table(root: Path, seeds, coverage: dict):
    cells = [
        ("runbugrun", "qwen25_7b", "RunBugRun", "Qwen2.5-Coder-7B"),
        ("codearc", "qwen25_7b", "CodeARC-Replay", "Qwen2.5-Coder-7B"),
        ("runbugrun", "deepseek_6p7b", "RunBugRun", "DeepSeek-Coder-6.7B"),
    ]
    rules = [
        ("no_update", "Base"),
        ("equal_weight", "Equal SD"),
        ("final_correctness", "Correct-only"),
        ("fixed_mass_dirichlet", "Fixed mass"),
        ("eesd_full", "EESD"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Fresh-policy all-tests Pass@1 after one self-distillation round. Each policy generates one new greedy primary solution per source; no candidate selection is used. Values are mean $\pm$ s.d. across locked training seeds.}",
        r"\label{tab:eesd-fresh-generated}",
        r"\small",
        r"\begin{tabular}{ll" + "r"*len(rules) + "}",
        r"\toprule",
        "Dataset & Model & " + " & ".join(label for _, label in rules) + r" \\",
        r"\midrule",
    ]
    for dataset, model, dlabel, mlabel in cells:
        vals = []
        for rule, _ in rules:
            scores = []
            for seed in seeds:
                report = read_json(root / "fresh-eval" / dataset / model / rule / f"seed{seed}" / "report.json")
                if report:
                    scores.append(report["fresh_all_tests_pass_at_1"])
            coverage["fresh"][f"{dataset}/{model}/{rule}"] = len(scores)
            vals.append(latex_mean_std(scores, percent=True))
        lines.append(f"{dlabel} & {mlabel} & " + " & ".join(vals) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return lines


def recursive_table(root: Path, seeds, coverage: dict):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Closed-loop recursive self-improvement for Qwen2.5-Coder-7B. Each later round collects new arm-specific experience. Values are all-tests fresh Pass@1 (mean $\pm$ s.d. across seeds).}",
        r"\label{tab:eesd-recursive-generated}",
        r"\small",
        r"\begin{tabular}{clrr}",
        r"\toprule",
        r"Domain & Round & Equal-weight SD & EESD \\",
        r"\midrule",
    ]
    for dataset, label in (("runbugrun", "RunBugRun"), ("codearc", "CodeARC")):
        reports = {}
        for seed in seeds:
            report = read_json(root / "recursive" / dataset / "qwen25_7b" / f"seed{seed}" / "recursive-report.json")
            if report:
                reports[seed] = report
        coverage["recursive"][dataset] = sorted(reports)
        base = [r["round0_pass_at_1"] for r in reports.values()]
        lines.append(f"{label} & 0 & {latex_mean_std(base, percent=True)} & {latex_mean_std(base, percent=True)} \\")
        for round_index in (1, 2, 3):
            equal, eesd = [], []
            for report in reports.values():
                match = next((x for x in report["summary"] if x["round"] == round_index), None)
                if match:
                    equal.append(match["equal_weight_pass_at_1"])
                    eesd.append(match["eesd_pass_at_1"])
            lines.append(
                f"{label} & {round_index} & {latex_mean_std(equal, percent=True)} & "
                f"{latex_mean_std(eesd, percent=True)} \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/experiments/eesd_iclr2027.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--coverage", type=Path)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("schema") != "eesd-iclr2027-v1":
        raise ValueError("locked EESD config required")
    seeds = [int(x) for x in cfg["seeds"]]
    coverage = {
        "mechanism": {},
        "transfer": {},
        "fresh": {},
        "recursive": {},
        "tables": {},
    }
    lines = [
        "% AUTO-GENERATED by scripts/render_eesd_tables.py",
        "% Missing artifacts are rendered as --; never hand-fill generated cells.",
        "",
    ]
    lines += mechanism_table(args.results, coverage)
    lines += ablation_table(args.results, coverage)
    lines += fresh_table(args.results, seeds, coverage)
    lines += transfer_table(args.results, seeds, coverage)
    lines += recursive_table(args.results, seeds, coverage)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    coverage_path = args.coverage or args.output.with_suffix(".coverage.json")
    coverage_path.write_text(json.dumps(coverage, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"tex": str(args.output), "coverage": str(coverage_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
