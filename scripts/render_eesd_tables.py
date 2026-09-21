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


def seed_reports(root: Path, dataset, model, seeds):
    reports, runs = {}, {}
    for seed in seeds:
        path = root / "mechanism" / dataset / model / f"seed{seed}" / "report.json"
        report = read_json(path)
        runs[str(seed)] = {"path": str(path), "present": bool(report)}
        if report:
            reports[seed] = report
    return reports, {
        "complete": len(reports) == len(seeds),
        "expected_seeds": list(seeds),
        "present_seeds": list(reports),
        "missing_seeds": [seed for seed in seeds if seed not in reports],
        "runs": runs,
    }


def seed_summary(values):
    # Each generation seed is one replicate, irrespective of its query count.
    values = list(values)
    mean, std = mean_std(values)
    return {"n": len(values), "mean": mean, "std": std if len(values) > 1 else None}


def latex_seed_summary(summary):
    if not summary["n"]:
        return "--"
    if summary["std"] is None:
        return f'{summary["mean"]:.4f} (n=1)'
    return f'{summary["mean"]:.4f} $\\pm$ {summary["std"]:.4f} (n={summary["n"]})'


def mechanism_values(report):
    return {
        "ordinary_nll": report["metrics"]["ordinary"]["nll"],
        "fixed_nll": report["metrics"]["fixed"]["nll"],
        "effective_nll": report["metrics"]["effective"]["nll"],
        "same_alpha_gain": report["bootstraps"]["effective_vs_fixed_at_effective_params"]["gain"],
        "global_mass_gain": report["bootstraps"]["effective_vs_tuned_global_mass"]["gain"],
    }


def mechanism_table(root: Path, coverage: dict, seeds=(1701, 1702, 1703), order=None):
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Effective-evidence mechanism across datasets and generator models. Values are means and sample standard deviations across generation seeds (n shown); queries are not pooled across seeds. A single seed has no estimated standard deviation. Same-$\alpha$ gain compares EED with fixed mass at the validation-selected EED $\alpha$ and relevance strength; global-mass gain compares EED with a validation-tuned constant evidence mass. Positive gains mean lower NLL for EED.}",
        r"\label{tab:eesd-mechanism-generated}", r"\small",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Dataset & Generator & Ordinary NLL & Fixed NLL & EED NLL & Same-$\alpha$ gain & Global-mass gain \\",
        r"\midrule",
    ]
    keys = ("ordinary_nll", "fixed_nll", "effective_nll", "same_alpha_gain", "global_mass_gain")
    for dataset, model, dataset_label, model_label in (MECHANISM_ORDER if order is None else order):
        reports, cell = seed_reports(root, dataset, model, seeds)
        values = {seed: mechanism_values(report) for seed, report in reports.items()}
        for seed, metrics in values.items():
            cell["runs"][str(seed)]["metrics"] = metrics
        cell["summary"] = {key: seed_summary(v[key] for v in values.values()) for key in keys}
        coverage["mechanism"][f"{dataset}/{model}"] = cell
        columns = [latex_seed_summary(cell["summary"][key]) for key in keys]
        lines.append(f"{dataset_label} & {model_label} & " + " & ".join(columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return lines


def ablation_values(report):
    m, b = report["metrics"], report["bootstraps"]
    return [
        (m["fixed"]["nll"], m["effective_at_fixed_params"]["nll"], b["effective_vs_fixed_at_fixed_params"]["gain"]),
        (m["fixed_at_effective_params"]["nll"], m["effective"]["nll"], b["effective_vs_fixed_at_effective_params"]["gain"]),
        (m["global_mass"]["nll"], m["effective"]["nll"], b["effective_vs_tuned_global_mass"]["gain"]),
        (m["constant_mean_mass"]["nll"], m["effective"]["nll"], m["constant_mean_mass"]["nll"] - m["effective"]["nll"]),
        (m["permuted_mass"]["mean_nll"], m["effective"]["nll"], m["permuted_mass"]["mean_nll"] - m["effective"]["nll"]),
        (m["binary_fixed_at_effective_params"]["nll"], m["binary_effective"]["nll"], m["binary_fixed_at_effective_params"]["nll"] - m["binary_effective"]["nll"]),
    ]


def ablation_table(root: Path, coverage: dict, seeds=(1701, 1702, 1703)):
    reports, cell = seed_reports(root, "runbugrun", "qwen25_7b", seeds)
    coverage["tables"]["causal_ablation"] = cell
    rows = {seed: ablation_values(report) for seed, report in reports.items()}
    for seed, values in rows.items():
        cell["runs"][str(seed)]["comparisons"] = values
    names = (
        "EED vs fixed at fixed-selected params", "EED vs fixed at EED-selected params",
        "EED vs tuned global mass", "EED vs constant mean EED mass",
        "EED vs mean permuted mass", "PASS/non-PASS: EED vs fixed",
    )
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Causal mechanism ablations on the primary RunBugRun/Qwen2.5-Coder-7B cell. Values are means and sample standard deviations across generation seeds (n shown). Positive NLL gain favors EED.}",
        r"\label{tab:eesd-ablation-generated}", r"\small",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Comparison & Control NLL & EED NLL & NLL gain \\", r"\midrule",
    ]
    cell["summary"] = {}
    for index, name in enumerate(names):
        summaries = [seed_summary(row[index][col] for row in rows.values()) for col in range(3)]
        cell["summary"][name] = dict(zip(("control_nll", "effective_nll", "gain"), summaries))
        lines.append(name + " & " + " & ".join(latex_seed_summary(x) for x in summaries) + r" \\")
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
        lines.append(f"{label} & 0 & {latex_mean_std(base, percent=True)} & {latex_mean_std(base, percent=True)} \\\\")
        for round_index in (1, 2, 3):
            equal, eesd = [], []
            for report in reports.values():
                match = next((x for x in report["summary"] if x["round"] == round_index), None)
                if match:
                    equal.append(match["equal_weight_pass_at_1"])
                    eesd.append(match["eesd_pass_at_1"])
            lines.append(
                f"{label} & {round_index} & {latex_mean_std(equal, percent=True)} & "
                f"{latex_mean_std(eesd, percent=True)} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


def configured_mechanism_order(config):
    # Config model paths define the declared scope; labels do not imply active cells.
    aliases = {
        'qwen2.5-coder-1.5b': ('qwen25_1p5b', 'Qwen2.5-Coder-1.5B'),
        'qwen2.5-coder-7b': ('qwen25_7b', 'Qwen2.5-Coder-7B'),
        'qwen3-8b': ('qwen3_8b', 'Qwen3-8B'),
        'deepseek-coder-6.7b': ('deepseek_6p7b', 'DeepSeek-Coder-6.7B'),
        'seed-coder-8b': ('seed_coder_8b', 'Seed-Coder-8B'),
        'qwen3-coder-30b-a3b': ('qwen3_coder_30b', 'Qwen3-Coder-30B-A3B'),
        'starcoder2-15b': ('starcoder2_15b', 'StarCoder2-15B'),
    }
    models = [aliases[Path(path).stem] for path in config['models'].values()]
    return [(dataset, model, label, model_label)
            for dataset, label in [('runbugrun', 'RunBugRun'), ('codearc', 'CodeARC-Replay')]
            for model, model_label in models]


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
    lines += mechanism_table(args.results, coverage, seeds, configured_mechanism_order(cfg))
    lines += ablation_table(args.results, coverage, seeds)
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
