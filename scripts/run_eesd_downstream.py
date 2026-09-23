#!/usr/bin/env python3
"""Launch downstream EESD stages with complete arms and explicit scientific settings.

Training launches the seven trainers directly with an explicit response-token
budget; recursive runs explicitly select shared EESD teacher (main) or arm-specific (supplemental) experience. Other stages
use the frozen matrix runner. No mechanism execution source
is modified. The main EESD trust path is parameter-free beyond the Dirichlet prior and KL anchor;
legacy transition utilities may remain in historical configs but are not consumed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import runpy
from pathlib import Path
import subprocess
import sys
import uuid

import yaml

from pbpf.eesd.distillation import TRAIN_RULES
from pbpf.eesd.training_outcomes import NON_ESTIMABLE, read_training_rows, protocol_binding, zero_report


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def adapter_tree_digest(directory: Path) -> str:
    files = sorted(path for path in directory.rglob('*') if path.is_file())
    if not files:
        raise ValueError(f'adapter binding requires a nonempty tree: {directory}')
    result = hashlib.sha256()
    for path in files:
        name = path.relative_to(directory).as_posix().encode()
        result.update(len(name).to_bytes(4, 'big'))
        result.update(name)
        result.update(bytes.fromhex(file_digest(path)))
    return result.hexdigest()


def check_training_binding(task, *, seal=False):
    directory = Path(task['output'])
    receipt_path = directory / 'training-binding.json'
    is_zero = task['eligibility']['status'] == NON_ESTIMABLE
    report_path = directory / ('training-outcome.json' if is_zero else 'training-report.json')
    if is_zero and ((directory / 'adapter').exists() or (directory / 'training-report.json').exists()):
        raise ValueError('non-estimable outcome cannot contain a trained adapter/report')
    if not seal and not receipt_path.is_file():
        raise ValueError(f'unbound existing training output; preserve and use a new output root: {directory}')
    if not report_path.is_file():
        raise ValueError(f'training report missing for binding: {directory}')
    report = json.loads(report_path.read_text())
    mismatches = [key for key, value in task['expected_report'].items() if report.get(key) != value]
    if mismatches:
        raise ValueError(f'training report budget/input/source binding mismatch: {mismatches}: {directory}')
    binding = {
        'schema': 'eesd-training-binding-v1', 'expected_report': task['expected_report'],
        'previous_adapter_tree_sha256': task['previous_adapter_tree_sha256'],
        'training_report_sha256': file_digest(report_path),
        'adapter_tree_sha256': None if is_zero else adapter_tree_digest(directory / 'adapter'),
        'request': task['request'], 'eligibility': task['eligibility'],
        **protocol_binding(Path(__file__).resolve().parents[1]),
    }
    if seal:
        if receipt_path.exists():
            raise FileExistsError(f'refusing to replace training binding: {receipt_path}')
        temporary = receipt_path.with_suffix('.partial')
        with temporary.open('x') as stream:
            json.dump(binding, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.rename(receipt_path)
    elif json.loads(receipt_path.read_text()) != binding:
        raise ValueError(f'training adapter/report checksum or request binding mismatch: {directory}')
    return {'status': NON_ESTIMABLE if is_zero else 'trained',
        'adapter': None if is_zero else str((directory / 'adapter').resolve()),
        'receipt': str(receipt_path.resolve()), 'receipt_sha256': file_digest(receipt_path)}


def make_training_task(*, root, scored, model_config, rule, previous_adapter, output,
                       seed, budget, max_steps, anchor_beta, trainer=None):
    """One direct trainer request and the exact receipt used by both protocols."""
    if budget < 1:
        raise ValueError('response token budget must be positive')
    scored, model_config, output = scored.resolve(), model_config.resolve(), output.resolve()
    if not scored.is_file() or not model_config.is_file():
        raise FileNotFoundError('training scored input/model config missing')
    trainer = Path(trainer).resolve() if trainer else root / 'scripts/run_eesd_weighted_sft.py'
    anchored = runpy.run_path(str(trainer))['ANCHORED_RULES']
    _, eligibility = read_training_rows(scored, rule)
    model_value = yaml.safe_load(model_config.read_text())
    if not isinstance(model_value.get('model_id'), str) or not isinstance(model_value.get('revision'), str) or len(model_value['revision']) != 40:
        raise ValueError('pinned model config required')
    previous = previous_adapter.resolve() if previous_adapter else None
    previous_sha = adapter_tree_digest(previous) if previous else None
    beta = float(anchor_beta)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('training anchor_beta must be finite and nonnegative')
    command = [sys.executable, str(trainer), '--input', str(scored),
        '--model-config', str(model_config), '--rule', rule, '--output', str(output),
        '--seed', str(seed), '--response-token-budget', str(budget),
        '--max-steps', str(max_steps), '--anchor-beta', str(beta)]
    if previous:
        command += ['--previous-adapter', str(previous)]
    task = {'output': str(output), 'command': command,
        'previous_adapter_tree_sha256': previous_sha, 'eligibility': eligibility,
        'request': dict(root=str(root), scored=str(scored), model_config=str(model_config), rule=rule,
            previous_adapter=str(previous) if previous else None, output=str(output), seed=seed,
            budget=budget, max_steps=max_steps, anchor_beta=beta),
        'expected_report': {'input_sha256': file_digest(scored), 'model_config_sha256': file_digest(model_config),
            'trainer_source_sha256': file_digest(trainer), 'seed': seed, 'rule': rule,
            'response_token_budget': budget, 'response_tokens': budget,
            'anchor_beta': beta if rule in anchored else 0.,
            'previous_adapter': str(previous) if previous else None, **protocol_binding(root)},
        'status': 'pending'}
    if eligibility['status'] == NON_ESTIMABLE:
        task['expected_report'] = zero_report(root=root, input_path=scored, model_config=model_config,
            trainer=trainer, rule=rule, seed=seed, budget=budget, previous_adapter=previous,
            anchor_beta=beta if rule in anchored else 0., eligibility=eligibility)
    if output.exists():
        task['outcome'] = check_training_binding(task)
        task['status'] = 'verified_non_estimable' if eligibility['status'] == NON_ESTIMABLE else 'verified_complete'
    return task


def plan_training(*, root, manifest, output, seeds, rules, budget, max_steps):
    """Preflight the entire matrix, refusing old unbound outputs before any job."""
    cells = manifest.get('correction_cells')
    if not isinstance(cells, list) or not cells:
        raise ValueError('train manifest must contain correction_cells')
    tasks, seen = [], set()
    for cell in cells:
        name = f"{cell['dataset']}/{cell['model']}/round{cell['round']}"
        if name in seen:
            raise ValueError(f'duplicate training cell: {name}')
        seen.add(name)
        for seed in seeds:
            for rule in rules:
                tasks.append(make_training_task(root=root,
                    scored=output / 'corrections' / name / 'scored-corrections.jsonl',
                    model_config=root / cell['model_config'], rule=rule,
                    previous_adapter=Path(cell['previous_adapter']) if cell.get('previous_adapter') else None,
                    output=output / 'training' / name / rule / f'seed{seed}', seed=seed,
                    budget=budget, max_steps=max_steps, anchor_beta=cell.get('anchor_beta', .03)))
    return tasks


def plan_recursive(*, root, manifest, output, seeds, budget, max_steps, experience_policy, config):
    cells = manifest.get('recursive_cells')
    if not isinstance(cells, list) or not cells:
        raise ValueError('recursive manifest must contain recursive_cells')
    tasks, seen = [], set()
    for cell in cells:
        name = f"{cell['dataset']}/{cell['model']}"
        if name in seen:
            raise ValueError(f'duplicate recursive cell: {name}')
        seen.add(name)
        public, evaluator = Path(cell['public_root']).resolve(), Path(cell['evaluator_root']).resolve()
        model = (root / cell['model_config']).resolve()
        if not public.is_dir() or not evaluator.is_dir() or not model.is_file():
            raise FileNotFoundError(f'recursive roots/model config missing: {name}')
        for seed in seeds:
            directory = output / 'recursive' / name / f'seed{seed}'
            if directory.is_dir() and any(directory.iterdir()) and not (directory / 'recursive-run.json').is_file():
                raise ValueError(f'unbound legacy recursive output: {directory}')
            command = [sys.executable, str(root / 'scripts/run_eesd_recursive.py'), '--config', str(config),
                '--domain', cell['domain'], '--public-root', str(public), '--evaluator-root', str(evaluator),
                '--model-config', str(model), '--family', cell['family'], '--output', str(directory),
                '--seed', str(seed), '--rounds', str(cell.get('rounds', 3)), '--max-steps', str(max_steps),
                '--response-token-budget', str(budget), '--experience-policy', experience_policy]
            for option, default in [('alpha', .1), ('uncertainty_penalty', .5),
                                    ('anchor_beta', .03), ('relevance_strength', 16.)]:
                command += ['--' + option.replace('_', '-'), str(cell.get(option, default))]
            tasks.append({'output': str(directory), 'command': command, 'status': 'pending'})
    return tasks


def verify_transfer_generation(task):
    directory = Path(task['output'])
    report = json.loads((directory / 'report.json').read_text())
    expected = task['expected_report']
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f'transfer generation data/model binding mismatch: {directory}')
    samples = directory / 'samples.jsonl'
    if file_digest(samples) != report.get('samples_sha256'):
        raise ValueError('transfer samples checksum mismatch')
    rows = [json.loads(line) for line in samples.read_text().splitlines() if line.strip()]
    if (any(set(row) != {'task_id', 'solution'} or not isinstance(row['solution'], str) for row in rows)
            or sorted(row['task_id'] for row in rows) != expected['data_lock']['task_ids']):
        raise ValueError('transfer samples coverage differs from locked public tasks')


def non_estimable_for_cell(*, root, manifest, output, cell, rule, seed):
    """Verify a missing update against CURRENT external cell/input/parent identity."""
    if rule == 'no_update':
        return None
    name = f"{cell['source_dataset']}/{cell['model']}/round{cell['source_round']}"
    directory = output / 'training' / name / rule / f'seed{seed}'
    outcome_path, binding_path = directory / 'training-outcome.json', directory / 'training-binding.json'
    binding = json.loads(binding_path.read_text()) if binding_path.exists() else None
    declared_zero = binding is not None and binding.get('eligibility', {}).get('status') == NON_ESTIMABLE
    if not outcome_path.exists() and not declared_zero:
        return None
    if binding is None:
        raise ValueError('unbound zero training outcome; cannot route fresh/transfer')
    cells = [c for c in manifest.get('correction_cells', []) if c['dataset'] == cell['source_dataset']
        and c['model'] == cell['model'] and c['round'] == cell['source_round']]
    if len(cells) != 1:
        raise ValueError('zero outcome requires one external correction cell to verify its input/parent')
    source = cells[0]
    model_config = (root / cell['model_config']).resolve()
    if (root / source['model_config']).resolve() != model_config:
        raise ValueError('zero outcome external model config mismatch')
    request = binding['request']
    task = make_training_task(root=root, scored=output / 'corrections' / name / 'scored-corrections.jsonl',
        model_config=model_config, rule=rule, previous_adapter=Path(source['previous_adapter']) if source.get('previous_adapter') else None,
        output=directory, seed=seed, budget=request['budget'], max_steps=request['max_steps'],
        anchor_beta=source.get('anchor_beta', .03))
    if task['outcome']['status'] != NON_ESTIMABLE:
        raise ValueError('zero receipt disagrees with current training eligibility')
    return task['outcome']


def plan_fresh(*, root, manifest, output, seeds, rules):
    """Split frozen routing only when a verified missing update must retain a row."""
    tasks = []
    has_zero = False
    for cell in manifest.get('fresh_cells', []):
        for seed in seeds:
            for rule in rules:
                zero = non_estimable_for_cell(root=root, manifest=manifest, output=output, cell=cell, rule=rule, seed=seed)
                task = dict(cell=f"{cell['dataset']}/{cell['model']}", seed=seed, rule=rule,
                    fresh_cell=cell, status=NON_ESTIMABLE if zero else 'pending', command=None)
                if zero:
                    has_zero = True
                    task.update(outcome=zero, main_estimate=None, reason='trained policy unavailable; no fallback adopted')
                tasks.append(task)
    return tasks if has_zero else None


def plan_transfer(*, root, manifest, output, seeds, rules, public_root, public_manifest_sha256):
    from pbpf.eesd.evalplus_public import load_public_dataset
    cells = manifest.get('transfer_cells')
    if not isinstance(cells, list) or not cells:
        raise ValueError('transfer manifest must contain transfer_cells')
    tasks, seen = [], set()
    script = root / 'scripts/run_eesd_evalplus_transfer.py'
    for cell in cells:
        name = f"{cell['dataset']}/{cell['model']}"
        if name in seen:
            raise ValueError(f'duplicate transfer cell: {name}')
        seen.add(name)
        _, data_lock = load_public_dataset(public_root, cell['dataset'], public_manifest_sha256)
        model_config = (root / cell['model_config']).resolve()
        model = yaml.safe_load(model_config.read_text())
        for seed in seeds:
            for rule in rules:
                directory = output / 'transfer' / name / rule / f'seed{seed}'
                zero = non_estimable_for_cell(root=root, manifest=manifest, output=output, cell=cell, rule=rule, seed=seed)
                if zero:
                    tasks.append(dict(command=None, output=str(directory), cell=name, rule=rule, seed=seed,
                        status=NON_ESTIMABLE, outcome=zero, main_estimate=None,
                        reason='trained policy unavailable; no fallback adopted'))
                    continue
                command = [sys.executable, str(script), '--dataset', cell['dataset'],
                    '--model-config', str(model_config), '--output', str(directory),
                    '--public-data-root', str(public_root), '--public-manifest-sha256', public_manifest_sha256]
                adapter_sha = None
                if rule != 'no_update':
                    adapter = output / 'training' / cell['source_dataset'] / cell['model'] / f"round{cell['source_round']}" / rule / f'seed{seed}' / 'adapter'
                    adapter_sha = adapter_tree_digest(adapter)
                    command += ['--adapter', str(adapter)]
                task = {'command': command, 'output': str(directory), 'status': 'pending',
                    'expected_report': {'schema': 'eesd-evalplus-transfer-v1', 'dataset': cell['dataset'],
                        'model_id': model['model_id'], 'revision': model['revision'],
                        'model_config_sha256': file_digest(model_config), 'adapter_sha256': adapter_sha,
                        'generator_source_sha256': file_digest(script), 'data_lock': data_lock,
                        'tasks': data_lock['tasks'], 'max_new_tokens': 1024,
                        'evaluation_status': 'pending-isolated-official-evaluation',
                        'claim_status': 'generation-only-no-efficacy-claim'}}
                if directory.exists():
                    if not (directory / 'report.json').is_file():
                        raise ValueError(f'unbound/incomplete transfer output: {directory}')
                    verify_transfer_generation(task)
                    task['status'] = 'verified_complete'
                tasks.append(task)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['score', 'train', 'fresh', 'transfer', 'recursive'], required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--response-token-budget', type=int,
                        help='Required positive response-token budget per train/recursive update; no default')
    parser.add_argument('--experience-policy', choices=['shared_eesd_teacher', 'arm-specific'])
    parser.add_argument('--public-data-root', type=Path)
    parser.add_argument('--public-manifest-sha256')
    args = parser.parse_args()
    if args.stage == 'transfer' and (args.public_data_root is None or not args.public_manifest_sha256):
        parser.error('transfer requires --public-data-root and --public-manifest-sha256')
    if args.stage != 'transfer' and (args.public_data_root is not None or args.public_manifest_sha256):
        parser.error('public EvalPlus data options are supported only for transfer')
    if args.stage in {'train', 'recursive'} and args.response_token_budget is None:
        parser.error(f'{args.stage} requires an explicit --response-token-budget')
    if args.stage == 'recursive' and args.experience_policy is None:
        parser.error('recursive requires explicit --experience-policy shared_eesd_teacher (main) or arm-specific (supplemental)')
    if args.experience_policy is not None and args.stage != 'recursive':
        parser.error('--experience-policy is supported only for recursive')
    if args.response_token_budget is not None:
        if args.response_token_budget < 1:
            parser.error('--response-token-budget must be positive')
        if args.stage not in {'train', 'recursive'}:
            parser.error('--response-token-budget is supported only for train/recursive')
    root = Path(__file__).resolve().parents[1]

    config_bytes = args.config.read_bytes()
    manifest_bytes = args.manifest.read_bytes()
    config = yaml.safe_load(config_bytes)
    manifest = yaml.safe_load(manifest_bytes)
    if not isinstance(config, dict) or config.get('schema') != 'eesd-iclr2027-v1':
        parser.error('expected eesd-iclr2027-v1 config')
    if not isinstance(manifest, dict) or manifest.get('schema') != 'eesd-cache-manifest-v1':
        parser.error('expected eesd-cache-manifest-v1 manifest')
    seeds = config.get('seeds')
    if (not isinstance(seeds, list) or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        parser.error('config seeds must be a nonempty list of distinct integers')
    if args.seed is not None and args.seed not in seeds:
        parser.error('--seed must belong to the configured seeds')
    if args.max_steps < 1:
        parser.error('--max-steps must be positive')

    rules = None
    if args.stage == 'fresh':
        rules = list(TRAIN_RULES)
    elif args.stage == 'train':
        rules = [rule for rule in TRAIN_RULES if rule != 'no_update']
    elif args.stage == 'recursive':
        rules = ['equal_weight', 'eesd_full']
    elif args.stage == 'transfer':
        rules = ['no_update', 'equal_weight', 'final_correctness', 'fixed_mass_dirichlet', 'eesd_full']
    # Transfer retains the declared five controls; recursive retains its two arms.
    output = args.output.resolve()
    training_tasks = plan_training(root=root, manifest=manifest, output=output,
        seeds=[args.seed] if args.seed is not None else seeds, rules=rules,
        budget=args.response_token_budget, max_steps=args.max_steps) if args.stage == 'train' else None
    recursive_tasks = plan_recursive(root=root, manifest=manifest, output=output,
        seeds=[args.seed] if args.seed is not None else seeds, budget=args.response_token_budget,
        max_steps=args.max_steps, experience_policy=args.experience_policy, config=args.config.resolve()
        ) if args.stage == 'recursive' else None
    transfer_tasks = plan_transfer(root=root, manifest=manifest, output=output,
        seeds=[args.seed] if args.seed is not None else seeds, rules=rules,
        public_root=args.public_data_root.resolve(), public_manifest_sha256=args.public_manifest_sha256
        ) if args.stage == 'transfer' else None
    fresh_tasks = plan_fresh(root=root, manifest=manifest, output=output,
        seeds=[args.seed] if args.seed is not None else seeds, rules=rules) if args.stage == 'fresh' else None
    launch = output / 'downstream-launches' / uuid.uuid4().hex
    launch.mkdir(parents=True, exist_ok=False)
    snapshot_config = launch / 'config.yaml'
    snapshot_manifest = launch / 'manifest.yaml'
    snapshot_config.write_bytes(config_bytes)
    snapshot_manifest.write_bytes(manifest_bytes)
    for task in recursive_tasks or []:
        task['command'][task['command'].index('--config') + 1] = str(snapshot_config)
    command = [sys.executable, str(root / 'scripts/run_eesd_matrix.py'),
        '--config', str(snapshot_config), '--manifest', str(snapshot_manifest),
        '--output', str(output), '--stage', args.stage, '--max-steps', str(args.max_steps)]
    if args.seed is not None:
        command += ['--seed', str(args.seed)]
    if rules is not None:
        command += ['--rules', *rules]
    if training_tasks is not None or recursive_tasks is not None or transfer_tasks is not None:
        command = None
    if fresh_tasks is not None:
        command = None
        for index, task in enumerate(fresh_tasks):
            if task['status'] == NON_ESTIMABLE:
                continue
            cell_manifest = launch / f'fresh-cell-{index}.yaml'
            cell_manifest.write_text(yaml.safe_dump({**manifest, 'fresh_cells': [task['fresh_cell']]}))
            # The frozen runner initializes its baseline per invocation. Include
            # no_update on each update call; its sealed artifacts are reused.
            task_rules = ['no_update'] if task['rule'] == 'no_update' else ['no_update', task['rule']]
            task['command'] = [sys.executable, str(root / 'scripts/run_eesd_matrix.py'),
                '--config', str(snapshot_config), '--manifest', str(cell_manifest), '--output', str(output),
                '--stage', 'fresh', '--seed', str(task['seed']), '--max-steps', str(args.max_steps),
                '--rules', *task_rules]
    source_hashes = {
        str(path.relative_to(root)): digest(path.read_bytes())
        for directory in ('scripts', 'src/pbpf')
        for path in sorted((root / directory).rglob('*.py'))
    }
    record = {
        'schema': 'eesd-downstream-launch-v1', 'status': 'validated', 'created_at': now(),
        'command': command, 'training_tasks': training_tasks, 'recursive_tasks': recursive_tasks, 'transfer_tasks': transfer_tasks, 'fresh_tasks': fresh_tasks, 'working_directory': str(root),
        'original_inputs': {'config': str(args.config.resolve()), 'manifest': str(args.manifest.resolve())},
        'input_sha256': {'config': digest(config_bytes), 'manifest': digest(manifest_bytes)},
        'source_sha256': source_hashes,
        'parameters': {'stage': args.stage, 'output': str(output), 'seed': args.seed,
            'seeds': [args.seed] if args.seed is not None else seeds, 'max_steps': args.max_steps,
            'rules': rules, 'trust_rule': 'posterior_excess_benefit_confidence',
            'response_token_budget': args.response_token_budget,
            'experience_policy': args.experience_policy,
            'public_data_root': str(args.public_data_root.resolve()) if args.public_data_root else None,
            'public_manifest_sha256': args.public_manifest_sha256},
    }
    record_path = launch / 'launch.json'
    def save() -> None:
        if training_tasks is not None:
            coverage = []
            baselines = set()
            for task in training_tasks:
                request = task['request']
                cell = str(Path(task['output']).parent.parent.relative_to(output / 'training'))
                key = (cell, request['seed'])
                if key not in baselines:
                    coverage.append(dict(cell=cell, seed=request['seed'], rule='no_update', status='baseline_not_trained'))
                    baselines.add(key)
                coverage.append(dict(cell=cell, seed=request['seed'], rule=request['rule'],
                    status=task.get('outcome', {}).get('status', task['status']), outcome=task.get('outcome')))
            record['training_coverage'] = coverage
        temporary = record_path.with_suffix('.partial')
        temporary.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + '\n')
        temporary.replace(record_path)
    save()
    print(json.dumps({'launch_record': str(record_path), 'command': command}), flush=True)
    try:
        returncode = 0
        tasks = next((items for items in (training_tasks, recursive_tasks, transfer_tasks, fresh_tasks) if items is not None), None)
        if tasks is None:
            returncode = subprocess.run(command, cwd=root, check=False).returncode
        else:
            for task in tasks:
                if task['status'] in {'verified_complete', 'verified_non_estimable', NON_ESTIMABLE}:
                    continue
                task['status'] = 'running'
                save()
                returncode = subprocess.run(task['command'], cwd=root, check=False).returncode
                if returncode:
                    task.update(status='failed', returncode=returncode)
                    break
                if training_tasks is not None:
                    task['outcome'] = check_training_binding(task, seal=True)
                if transfer_tasks is not None:
                    verify_transfer_generation(task)
                task['status'] = (NON_ESTIMABLE if task.get('outcome', {}).get('status') == NON_ESTIMABLE else 'completed')
                save()
        status = 'generated-awaiting-isolated-evaluation' if transfer_tasks is not None else 'completed'
        if any(t.get('outcome', {}).get('status') == NON_ESTIMABLE or t.get('status') == NON_ESTIMABLE for t in tasks or []):
            status = 'completed_with_non_estimable'
        record.update(status=status if returncode == 0 else 'failed', returncode=returncode)
    except BaseException as error:
        record.update(status='failed', error=repr(error), finished_at=now())
        save()
        raise
    record['finished_at'] = now()
    save()
    if returncode:
        raise SystemExit(returncode)


if __name__ == '__main__':
    main()
