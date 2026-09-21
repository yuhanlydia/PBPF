"""Single-factor, development-only parameter study; never a formal gate run."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import runpy
import time


VARIANTS = {
    'reference': {},
    'particles32': {'particles': 32},
    'learning_rate_low': {'learning_rate': .0001},
    'weight_decay_high': {'weight_decay': .1},
    'association_off': {'association_weight': 0.},
    'association_high': {'association_weight': 10.},
}


def partition_development(records, seed=2701):
    """Inner selection uses original training sources; assessment uses only dev."""
    train = [r for r in records if r['split'] == 'train']
    dev = [r for r in records if r['split'] == 'development']
    source = lambda r: r.get('source_component_id', r['problem_id'])
    sources = {source(r) for r in train}
    if sources & {source(r) for r in dev}:
        raise ValueError('training/development source overlap')
    if len(sources) < 3 or not dev:
        raise ValueError('need at least three training sources and development data')
    ordered = sorted(sources, key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).digest())
    validation = set(ordered[:max(1, round(.2 * len(ordered)))])
    return copy.deepcopy({'train': [r for r in train if source(r) not in validation],
                          'validation': [r for r in train if source(r) in validation],
                          'assessment': dev})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1701, 1702])
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    import numpy as np
    import torch
    from pbpf.train_belief import train_apbpf_step
    repo = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(repo / 'scripts/run_rbr_prediction_gate.py'))
    # No original test record is tensorized, evaluated, or copied into outputs.
    parts = partition_development(json.loads(args.cache.read_text())['records'])
    args.output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (args.output / name).open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    sources = [Path(__file__), repo / 'scripts/run_rbr_prediction_gate.py',
               repo / 'src/pbpf/train_belief.py', repo / 'src/pbpf/belief/model.py',
               repo / 'src/pbpf/belief/losses.py', repo / 'src/pbpf/real_gate.py']
    plan = {'claim_status': 'exploratory-development-only', 'variants': VARIANTS,
            'steps': args.steps, 'seeds': args.seeds, 'torch': torch.__version__,
            'device': 'cpu', 'selection_particles': 32, 'assessment_particles': [8, 128],
            'assessment_inference_seeds': [51701, 51702, 51703],
            'source_sha256': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'partition_sha256': hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest(),
            'counts': {k: len(v) for k, v in parts.items()},
            'selection': 'minimum inner-validation NLL, original training sources only',
            'assessment': 'original development only; repeated comparisons are exploratory',
            'original_test_used': False}
    write('plan.json', plan)
    batches = {k: module['_tensorize'](v, module['FrozenTextEncoder'](256),
               torch.device('cpu'), expected_is_public=True) for k, v in parts.items()}
    labels = {k: b.outcomes[:, 4:].reshape(-1).numpy() for k, b in batches.items()}
    metric = module['compare_predictions']
    grid = (.01, .1, .25, .5, 1., 2., 5., 10.)
    alpha = min(grid, key=lambda a: metric(labels['validation'], {
        'baseline': module['_history_rate'](batches['validation'], a)})['baseline']['nll'])
    dirichlet = module['_history_rate'](batches['assessment'], alpha)
    write('baseline.json', {'alpha': alpha, 'selection_split': 'inner_validation',
          'assessment': metric(labels['assessment'], {'baseline': dirichlet})})
    completed = []
    for seed in args.seeds:
        for name, changes in VARIANTS.items():
            started = time.monotonic()
            config = dict(particles=8, learning_rate=.0003, weight_decay=.01, association_weight=1.)
            config.update(changes)
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = module['NeuralBeliefModel'](256, 32, 192, difficulty_dim=8)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                         weight_decay=config['weight_decay'])
            best, state, selected_step, history = float('inf'), None, None, []
            for step in range(1, args.steps + 1):
                indices = torch.tensor(rng.integers(0, len(parts['train']), size=64))
                losses = train_apbpf_step(model, module['_subset'](batches['train'], indices),
                    optimizer, particles=config['particles'], association_weight=config['association_weight'],
                    invariance_weight=.1, margin=.03, shuffle_seed=seed + step)
                if step == 1 or step % max(1, args.steps // 20) == 0:
                    model.eval()
                    prediction = module['_predict'](model, batches['validation'], particles=32,
                                                     seed=seed + 50000, mode='aligned')
                    nll = metric(labels['validation'], {'baseline': prediction})['baseline']['nll']
                    grad_norm = float(torch.sqrt(sum(p.grad.detach().square().sum()
                                      for p in model.parameters() if p.grad is not None)))
                    history.append(dict(step=step, validation_nll=nll, gradient_norm=grad_norm, **losses))
                    if nll < best:
                        best, state, selected_step = nll, copy.deepcopy(model.state_dict()), step
                    print(json.dumps({'seed': seed, 'variant': name, 'step': step,
                                      'inner_validation_nll': nll}), flush=True)
            model.load_state_dict(state)
            model.eval()
            assessments = []
            for particles in (8, 128):
                for inference_seed in (51701, 51702, 51703):
                    predictions = {arm: module['_predict'](model, batches['assessment'], particles=particles,
                        seed=inference_seed, mode=arm) for arm in
                        ('aligned', 'outcome_shuffled', 'joint_reversed', 'presentation_permuted')}
                    predictions['baseline'] = dirichlet
                    assessments.append({'particles': particles, 'inference_seed': inference_seed,
                                        'metrics': metric(labels['assessment'], predictions)})
            checkpoint = args.output / f'{seed}-{name}.pt'
            torch.save({'model': state, 'config': config, 'seed': seed,
                        'selected_step': selected_step, 'claim_status': 'development-only'}, checkpoint)
            report = {'seed': seed, 'variant': name, 'config': config, 'history': history,
                      'best_inner_validation_nll': best, 'selected_step': selected_step,
                      'assessments': assessments, 'elapsed_seconds': time.monotonic() - started,
                      'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      'claim_status': 'exploratory-development-only'}
            write(f'{seed}-{name}.json', report)
            completed.append({'seed': seed, 'variant': name, 'elapsed_seconds': report['elapsed_seconds']})
            print(json.dumps({'completed': completed[-1]}), flush=True)
    write('complete.json', {'runs': completed, 'original_test_used': False,
          'claim_status': 'exploratory-development-only', 'files': {
              p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()}})


if __name__ == '__main__':
    main()
