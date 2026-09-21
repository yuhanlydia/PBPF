#!/usr/bin/env python3
"""Resumable orchestrator for the locked EESD mechanism and distillation matrix."""
from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path
import subprocess
import sys

import yaml

from pbpf.eesd.distillation import TRAIN_RULES
from pbpf.apbpf.codearc_bank import file_sha, load_bank


def run(command, *, cwd: Path, env=None):
    print(json.dumps({"command": command}), flush=True)
    subprocess.check_call(command, cwd=cwd, env=env)


def mechanism_generator_script(domain, family):
    """Use the isolated Seed tokenizer compatibility profile; preserve legacy banks."""
    if domain not in ('rbr', 'codearc'):
        raise ValueError('unsupported mechanism domain')
    prefix = 'seed_' if family == 'seed_coder_8b' else ''
    return f'scripts/generate_apbpf_{prefix}{domain}_bank.py'


def verify_mechanism_bank(bank, *, cell, seed, split, root):
    """Verify sealed bytes and their binding to this invocation's public inventory."""
    run_info, rows, digest = load_bank(bank)
    domain = cell['domain']
    script = root / mechanism_generator_script(domain, cell['family'])
    model, revision = runpy.run_path(str(script))['MODELS'][cell['family']]
    public = Path(cell['public_root']).resolve()
    manifest = json.loads((public / 'manifest.json').read_text())
    if file_sha(public / 'tasks.jsonl') != manifest['public_tasks_sha256']:
        raise ValueError('public tasks identity checksum mismatch')
    tasks = [json.loads(line) for line in (public / 'tasks.jsonl').read_text().splitlines()]
    tasks = [row for row in tasks if row['split'] == split]
    if domain == 'codearc':
        grouped = {}
        # Preserve materializer component order; choose the smallest task only
        # within each component, matching the generator's pre-actor inventory.
        for row in tasks:
            component = row['source_component_id']
            if component not in grouped or row['task_id'] < grouped[component]['task_id']:
                grouped[component] = row
        tasks = list(grouped.values())
    limit = int(cell.get('development_components', 200)) if split == 'development' else 0
    if limit:
        tasks = tasks[:limit]
    expected = {
        'schema': f'apbpf-{domain}-generation-v1', 'family': cell['family'],
        'model': model, 'revision': revision, 'seed': seed, 'split': split,
        'candidates': 1, 'adapter_sha256': None, 'offset': 0,
        'components': len(tasks), 'task_ids': [r['task_id'] for r in tasks],
        'source_component_ids': [r['source_component_id'] for r in tasks],
        'public_tasks_sha256': manifest['public_tasks_sha256'],
        'public_manifest_sha256': file_sha(public / 'manifest.json'),
        'source_sha256': file_sha(script),
        'prompt_source_sha256': file_sha(root / 'src/pbpf/apbpf/rbr_prompt.py'),
        'decode_policy': 'temperature0.8-top_p0.95', 'temperature': .8, 'top_p': .95,
        'max_new_tokens': 1024 if domain == 'rbr' else 512,
    }
    if domain == 'rbr':
        expected.update(max_input_tokens=4096,
            inventory_source_sha256=file_sha(root / 'src/pbpf/apbpf/codearc_bank.py'))
    else:
        expected.update(max_input_tokens=4096,
            prompt_policy='adaptively cap public fields while retaining all four example slots',
            codearc_prompt_source_sha256=file_sha(root / 'src/pbpf/apbpf/codearc_prompt.py'))
    mismatches = [key for key, value in expected.items() if run_info.get(key) != value]
    if mismatches:
        raise ValueError(f'bank resume identity mismatch: {bank}: {mismatches}')
    return Path(bank), run_info, digest


def verify_mechanism_cache(cache, *, cell, banks, root, seal=False):
    """Bind cache bytes to verified banks, evaluator inputs and execution sources."""
    payload = json.loads(cache.read_text())
    evaluator = Path(cell['evaluator_root']).resolve()
    evaluator_manifest = json.loads((evaluator / 'manifest.json').read_text())
    tasks_sha = file_sha(evaluator / 'tasks.jsonl')
    if tasks_sha != evaluator_manifest['evaluator_tasks_sha256']:
        raise ValueError('evaluator task checksum mismatch')
    first = banks[0][1]
    identity = [first['model'], first['revision'], first.get('adapter_sha256'), first['seed']]
    expected_banks = [dict(split=info['split'], path=str(path.resolve()),
        complete_sha256=digest, components=info['components']) for path, info, digest in banks]
    actual_banks = [{**entry, 'path': str(Path(entry['path']).resolve())}
                    for entry in payload.get('bank_bindings', [])]
    if (payload.get('schema') != 'eesd-public-query-mechanism-cache-v1'
            or payload.get('dataset') != cell['domain']
            or payload.get('generator_identity') != identity
            or sorted(actual_banks, key=lambda x: x['split']) != sorted(expected_banks, key=lambda x: x['split'])
            or payload.get('evaluator_manifest_sha256') != file_sha(evaluator / 'manifest.json')):
        raise ValueError('cache identity/bank binding mismatch')
    binding = {
        'cache_sha256': file_sha(cache), 'generator_identity': identity,
        'bank_bindings': expected_banks, 'evaluator_tasks_sha256': tasks_sha,
        'evaluator_manifest_sha256': file_sha(evaluator / 'manifest.json'),
        'sources': {name: file_sha(root / name) for name in (
            'scripts/build_eesd_mechanism_cache.py', 'src/pbpf/apbpf/codearc_execution.py',
            'src/pbpf/apbpf/rbr_execution.py')},
    }
    receipt = cache.with_suffix('.binding.json')
    if seal:
        temporary = receipt.with_suffix('.partial')
        with temporary.open('x') as stream:
            json.dump(binding, stream, sort_keys=True)
        if receipt.exists():
            raise FileExistsError(f'cache seal already exists: {receipt}')
        temporary.rename(receipt)
    elif not receipt.exists():
        raise ValueError(f'cache lacks immutable binding seal: {receipt}; preserve and use a new run directory')
    elif json.loads(receipt.read_text()) != binding:
        raise ValueError('cache checksum/source binding mismatch')


def verify_mechanism_report(directory, *, cache, config, cell, seed, root):
    complete = json.loads((directory / 'complete.json').read_text())
    for name in ('report', 'predictions'):
        suffix = '.json' if name == 'report' else '.npz'
        if file_sha(directory / (name + suffix)) != complete[name + '_sha256']:
            raise ValueError('mechanism report artifact checksum mismatch')
    report = json.loads((directory / 'report.json').read_text())
    expected = dict(schema='eesd-evidence-matrix-v1', dataset=cell['dataset'], model=cell['model'],
        seed=seed, visible=int(cell.get('visible', 4)), validation_split='development', assessment_split='primary',
        cache_sha256=file_sha(cache), config_sha256=file_sha(config),
        runner_source_sha256=file_sha(root / 'scripts/run_eesd_evidence_matrix.py'),
        evidence_source_sha256=file_sha(root / 'src/pbpf/eesd/evidence.py'))
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('mechanism report resume identity mismatch')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/experiments/eesd_iclr2027.yaml"))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["mechanism", "prepare-corrections", "generate-corrections", "score", "train", "fresh", "transfer", "recursive", "render"], required=True)
    p.add_argument("--rules", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=None, help="one seed; omit to run all locked seeds")
    p.add_argument("--max-steps", type=int, default=200)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load(args.manifest.read_text())
    if manifest.get("schema") != "eesd-cache-manifest-v1":
        raise ValueError("unexpected EESD cache manifest schema")
    locked = yaml.safe_load(args.config.read_text())
    if locked.get("schema") != "eesd-iclr2027-v1":
        raise ValueError("unexpected EESD experiment config")
    seeds = [args.seed] if args.seed is not None else [int(x) for x in locked["seeds"]]
    args.output.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    if args.stage == "render":
        run(
            [
                python,
                "scripts/render_eesd_tables.py",
                "--results", str(args.output.resolve()),
                "--config", str(args.config.resolve()),
                "--output", str((root / "paper/generated_eesd_results.tex").resolve()),
                "--coverage", str((root / "paper/generated_eesd_results.coverage.json").resolve()),
            ],
            cwd=root,
        )
        return

    if args.stage == "mechanism":
        seen = set()
        for cell in manifest.get("mechanism_cells", []):
            key = cell["dataset"], cell["model"]
            if key in seen:
                raise ValueError(f"duplicate mechanism cell: {key}")
            seen.add(key)
            public_root = Path(cell["public_root"]).resolve()
            evaluator_root = Path(cell["evaluator_root"]).resolve()
            if not public_root.is_dir() or not evaluator_root.is_dir():
                raise FileNotFoundError(f"mechanism materialization roots missing for {key}")
            script = mechanism_generator_script(cell["domain"], cell["family"])
            for generation_seed in seeds:
                bank_root = args.output / "mechanism-banks" / cell["dataset"] / cell["model"] / f"seed{generation_seed}"
                development_bank = bank_root / "development"
                primary_bank = bank_root / "primary"
                if not (development_bank / "complete.json").exists():
                    development_bank.parent.mkdir(parents=True, exist_ok=True)
                    run(
                        [
                            python, script,
                            "--public-root", str(public_root),
                            "--output", str(development_bank.resolve()),
                            "--family", cell["family"],
                            "--split", "development",
                            "--components", str(cell.get("development_components", 200)),
                            "--candidates", "1",
                            "--seed", str(generation_seed),
                        ],
                        cwd=root,
                        env=dict(os.environ),
                    )
                if not (primary_bank / "complete.json").exists():
                    primary_bank.parent.mkdir(parents=True, exist_ok=True)
                    run(
                        [
                            python, script,
                            "--public-root", str(public_root),
                            "--output", str(primary_bank.resolve()),
                            "--family", cell["family"],
                            "--split", "primary",
                            "--candidates", "1",
                            "--seed", str(generation_seed),
                        ],
                        cwd=root,
                        env=dict(os.environ),
                    )
                banks = [verify_mechanism_bank(bank, cell=cell, seed=generation_seed, split=split, root=root)
                         for split, bank in (("development", development_bank), ("primary", primary_bank))]
                cache = args.output / "mechanism-cache" / cell["dataset"] / cell["model"] / f"seed{generation_seed}" / "cache.json"
                cache_created = not cache.exists()
                if cache_created:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    run(
                        [
                            python,
                            "scripts/build_eesd_mechanism_cache.py",
                            "--domain", cell["domain"],
                            "--evaluator-root", str(evaluator_root),
                            "--bank", str(development_bank.resolve()),
                            "--bank", str(primary_bank.resolve()),
                            "--output", str(cache.resolve()),
                        ],
                        cwd=root,
                    )
                verify_mechanism_cache(cache, cell=cell, banks=banks, root=root, seal=cache_created)
                directory = args.output / "mechanism" / cell["dataset"] / cell["model"] / f"seed{generation_seed}"
                if (directory / "complete.json").exists():
                    verify_mechanism_report(directory, cache=cache, config=args.config, cell=cell,
                                            seed=generation_seed, root=root)
                    print(json.dumps({"status": "skip_complete", "cell": key, "seed": generation_seed}), flush=True)
                    continue
                directory.parent.mkdir(parents=True, exist_ok=True)
                run(
                    [
                        python,
                        "scripts/run_eesd_evidence_matrix.py",
                        "--config", str(args.config.resolve()),
                        "--cache", str(cache),
                        "--dataset", cell["dataset"],
                        "--model", cell["model"],
                        "--seed", str(generation_seed),
                        "--validation-split", "development",
                        "--assessment-split", "primary",
                        "--visible", str(cell.get("visible", 4)),
                        "--output", str(directory.resolve()),
                    ],
                    cwd=root,
                )
        return

    cells = manifest.get("correction_cells", [])
    if not cells:
        raise ValueError("manifest has no correction_cells")

    if args.stage == "prepare-corrections":
        experience_seed = int(locked["seeds"][0])
        for cell in cells:
            name = f"{cell['dataset']}/{cell['model']}/round{cell['round']}"
            directory = args.output / "public-corrections" / name
            if (directory / "audit.json").exists():
                print(json.dumps({"status": "skip_complete", "cell": name}), flush=True)
                continue
            public_root = Path(cell["public_root"]).resolve()
            if not public_root.is_dir():
                raise FileNotFoundError(f"public materialization missing: {public_root}")
            script = (
                "scripts/generate_apbpf_rbr_bank.py"
                if cell["domain"] == "rbr"
                else "scripts/generate_apbpf_codearc_bank.py"
            )
            original_root = args.output / "correction-original-banks" / name
            train_bank = original_root / "train"
            development_bank = original_root / "development"
            for split, bank in (("train", train_bank), ("development", development_bank)):
                if (bank / "complete.json").exists():
                    continue
                bank.parent.mkdir(parents=True, exist_ok=True)
                run(
                    [
                        python, script,
                        "--public-root", str(public_root),
                        "--output", str(bank.resolve()),
                        "--family", cell["family"],
                        "--split", split,
                        "--candidates", "1",
                        "--seed", str(experience_seed),
                    ],
                    cwd=root,
                    env=dict(os.environ),
                )
            directory.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    python,
                    "scripts/prepare_eesd_recursive_corrections.py",
                    "--domain", cell["domain"],
                    "--public-root", str(public_root),
                    "--bank", str(train_bank.resolve()),
                    "--bank", str(development_bank.resolve()),
                    "--output", str(directory.resolve()),
                ],
                cwd=root,
            )
        return

    if args.stage == "generate-corrections":
        for cell in cells:
            name = f"{cell['dataset']}/{cell['model']}/round{cell['round']}"
            public_bank = args.output / "public-corrections" / name / "public-corrections.jsonl"
            if not public_bank.exists():
                raise FileNotFoundError(f"prepare-corrections stage missing: {public_bank}")
            directory = args.output / "generated-corrections" / name
            if (directory / "report.json").exists():
                print(json.dumps({"status": "skip_complete", "cell": name}), flush=True)
                continue
            directory.parent.mkdir(parents=True, exist_ok=True)
            command = [
                python,
                "scripts/generate_eesd_corrections.py",
                "--public-bank", str(public_bank.resolve()),
                "--model-config", str((root / cell["model_config"]).resolve()),
                "--domain", cell["domain"],
                "--output", str(directory.resolve()),
                "--relevance-strength", str(cell.get("relevance_strength", 16.0)),
            ]
            if cell.get("previous_adapter"):
                command += ["--adapter", str(Path(cell["previous_adapter"]).resolve())]
            run(command, cwd=root, env=dict(os.environ))
        return

    if args.stage == "score":
        for cell in cells:
            name = f"{cell['dataset']}/{cell['model']}/round{cell['round']}"
            source = args.output / "generated-corrections" / name / "corrections.jsonl"
            if not source.exists():
                raise FileNotFoundError(f"generate-corrections stage missing: {source}")
            directory = args.output / "corrections" / name
            if (directory / "summary.json").exists():
                print(json.dumps({"status": "skip_complete", "cell": name}), flush=True)
                continue
            directory.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    python,
                    "scripts/score_eesd_corrections.py",
                    "--input", str(source),
                    "--config", str(args.config.resolve()),
                    "--output", str(directory.resolve()),
                    "--alpha", str(cell.get("alpha", 0.1)),
                    "--uncertainty-penalty", str(cell.get("uncertainty_penalty", 0.5)),
                ],
                cwd=root,
            )
        return

    if args.stage == "recursive":
        cells = manifest.get("recursive_cells", [])
        if not cells:
            raise ValueError("manifest has no recursive_cells")
        for cell in cells:
            public_root = Path(cell["public_root"]).resolve()
            evaluator_root = Path(cell["evaluator_root"]).resolve()
            if not public_root.is_dir() or not evaluator_root.is_dir():
                raise FileNotFoundError(f"recursive roots missing for {cell['dataset']}/{cell['model']}")
            for seed in seeds:
                directory = args.output / "recursive" / cell["dataset"] / cell["model"] / f"seed{seed}"
                if (directory / "recursive-report.json").exists():
                    print(json.dumps({"status": "skip_complete", "cell": cell["dataset"], "model": cell["model"], "seed": seed}), flush=True)
                    continue
                directory.parent.mkdir(parents=True, exist_ok=True)
                run(
                    [
                        python,
                        "scripts/run_eesd_recursive.py",
                        "--config", str(args.config.resolve()),
                        "--domain", cell["domain"],
                        "--public-root", str(public_root),
                        "--evaluator-root", str(evaluator_root),
                        "--model-config", str((root / cell["model_config"]).resolve()),
                        "--family", cell["family"],
                        "--output", str(directory.resolve()),
                        "--seed", str(seed),
                        "--rounds", str(cell.get("rounds", 3)),
                        "--max-steps", str(args.max_steps),
                    ],
                    cwd=root,
                    env=dict(os.environ),
                )
        return

    if args.stage == "fresh":
        rules = args.rules or [
            "no_update", "equal_weight", "final_correctness",
            "fixed_mass_dirichlet", "eesd_full",
        ]
        unknown = sorted(set(rules) - set(TRAIN_RULES))
        if unknown:
            raise ValueError(f"invalid fresh-eval rules: {unknown}")
        for cell in manifest.get("fresh_cells", []):
            name = f"{cell['dataset']}/{cell['model']}"
            public_root = Path(cell["public_root"]).resolve()
            evaluator_root = Path(cell["evaluator_root"]).resolve()
            if not public_root.is_dir() or not evaluator_root.is_dir():
                raise FileNotFoundError(f"fresh cell roots missing for {name}")
            for seed in seeds:
                base_report = None
                for rule in rules:
                    bank = args.output / "fresh-banks" / name / rule / f"seed{seed}"
                    if not (bank / "complete.json").exists():
                        bank.parent.mkdir(parents=True, exist_ok=True)
                        script = (
                            "scripts/generate_apbpf_rbr_bank.py"
                            if cell["domain"] == "rbr"
                            else "scripts/generate_apbpf_codearc_bank.py"
                        )
                        command = [
                            python, script,
                            "--public-root", str(public_root),
                            "--output", str(bank.resolve()),
                            "--family", cell["family"],
                            "--split", "primary",
                            "--candidates", "1",
                            "--seed", str(seed),
                            "--greedy",
                        ]
                        if rule != "no_update":
                            adapter = (
                                args.output / "training" / cell["source_dataset"] / cell["model"]
                                / f"round{cell['source_round']}" / rule / f"seed{seed}" / "adapter"
                            )
                            if not adapter.is_dir():
                                raise FileNotFoundError(f"training adapter missing: {adapter}")
                            command += ["--adapter", str(adapter.resolve())]
                        run(command, cwd=root, env=dict(os.environ))
                    evaluation = args.output / "fresh-eval" / name / rule / f"seed{seed}"
                    if not (evaluation / "report.json").exists():
                        evaluation.parent.mkdir(parents=True, exist_ok=True)
                        run(
                            [
                                python, "scripts/evaluate_eesd_fresh_bank.py",
                                "--domain", cell["domain"],
                                "--evaluator-root", str(evaluator_root),
                                "--bank", str(bank.resolve()),
                                "--output", str(evaluation.resolve()),
                            ],
                            cwd=root,
                        )
                    report = evaluation / "report.json"
                    if rule == "no_update":
                        base_report = report
                    else:
                        if base_report is None or not base_report.exists():
                            raise FileNotFoundError("base fresh evaluation must run before comparisons")
                        comparison = (
                            args.output / "fresh-comparison" / name / rule / f"seed{seed}.json"
                        )
                        if not comparison.exists():
                            comparison.parent.mkdir(parents=True, exist_ok=True)
                            run(
                                [
                                    python, "scripts/compare_eesd_fresh_eval.py",
                                    "--baseline", str(base_report.resolve()),
                                    "--method", str(report.resolve()),
                                    "--output", str(comparison.resolve()),
                                ],
                                cwd=root,
                            )
        return

    if args.stage == "transfer":
        rules = args.rules or [
            "no_update", "equal_weight", "final_correctness",
            "fixed_mass_dirichlet", "eesd_full",
        ]
        unknown = sorted(set(rules) - set(TRAIN_RULES))
        if unknown:
            raise ValueError(f"invalid transfer rules: {unknown}")
        for cell in manifest.get("transfer_cells", []):
            name = f"{cell['dataset']}/{cell['model']}"
            for seed in seeds:
                for rule in rules:
                    directory = args.output / "transfer" / name / rule / f"seed{seed}"
                    if (directory / "report.json").exists():
                        print(json.dumps({"status": "skip_complete", "cell": name, "rule": rule, "seed": seed}), flush=True)
                        continue
                    directory.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        python,
                        "scripts/run_eesd_evalplus_transfer.py",
                        "--dataset", cell["dataset"],
                        "--model-config", str((root / cell["model_config"]).resolve()),
                        "--output", str(directory.resolve()),
                        "--evaluate",
                    ]
                    if rule != "no_update":
                        adapter = (
                            args.output / "training" / cell["source_dataset"] / cell["model"]
                            / f"round{cell['source_round']}" / rule / f"seed{seed}" / "adapter"
                        )
                        if not adapter.is_dir():
                            raise FileNotFoundError(f"training adapter missing: {adapter}")
                        command += ["--adapter", str(adapter.resolve())]
                    run(command, cwd=root, env=dict(os.environ))
        return

    rules = args.rules or [rule for rule in TRAIN_RULES if rule != "no_update"]
    unknown = sorted(set(rules) - set(TRAIN_RULES))
    if unknown or "no_update" in rules:
        raise ValueError(f"invalid train rules: {unknown or ['no_update']}")
    for cell in cells:
        name = f"{cell['dataset']}/{cell['model']}/round{cell['round']}"
        scored = args.output / "corrections" / name / "scored-corrections.jsonl"
        if not scored.exists():
            raise FileNotFoundError(f"score stage missing: {scored}")
        previous = cell.get("previous_adapter")
        for seed in seeds:
            for rule in rules:
                directory = args.output / "training" / name / rule / f"seed{seed}"
                if (directory / "training-report.json").exists():
                    print(json.dumps({"status": "skip_complete", "cell": name, "rule": rule, "seed": seed}), flush=True)
                    continue
                directory.parent.mkdir(parents=True, exist_ok=True)
                command = [
                    python,
                    "scripts/run_eesd_weighted_sft.py",
                    "--input", str(scored.resolve()),
                    "--model-config", str((root / cell["model_config"]).resolve()),
                    "--rule", rule,
                    "--output", str(directory.resolve()),
                    "--seed", str(seed),
                    "--max-steps", str(args.max_steps),
                    "--anchor-beta", str(cell.get("anchor_beta", 0.03)),
                ]
                if previous:
                    command += ["--previous-adapter", str(Path(previous).resolve())]
                run(command, cwd=root, env=dict(os.environ))


if __name__ == "__main__":
    main()
