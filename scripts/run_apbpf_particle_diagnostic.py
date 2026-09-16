#!/usr/bin/env python3
"""Development-only inference-particle sensitivity of frozen matched models."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.semantic_cache import FrozenSemanticCache
from pbpf.apbpf.stage_prediction import aggregate_predictions
from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, compare_predictions, validate_rbr_cache

COUNTS = (8, 32, 128)
CONTROLS = ('aligned', 'outcome_shuffled', 'joint_reversed', 'presentation_permuted')


def diversity(trace):
    """Track ancestry separately: uniform resampled weights do not imply diversity."""
    batch, steps, particles = trace.log_weights.shape
    lineage = torch.arange(particles).expand(batch, -1).clone()
    rows = []
    for step in range(steps):
        lineage = lineage.gather(1, trace.resampling_indices[:, step])
        ordered = lineage.sort(dim=1).values
        unique = 1 + (ordered[:, 1:] != ordered[:, :-1]).sum(1)
        weights = trace.log_weights[:, step].softmax(-1)
        ess = weights.square().sum(-1).reciprocal()
        rows.append({'prefix': step+1, 'mean_ess': float(ess.mean()),
            'mean_ess_fraction': float((ess/particles).mean()),
            'mean_unique_initial_particles': float(unique.float().mean()),
            'mean_unique_fraction': float((unique.float()/particles).mean()),
            'single_ancestor_fraction': float((unique == 1).float().mean()),
            'resampled_fraction': float(trace.resampled[:, step].float().mean())})
    return rows


def validate_development(payload):
    validate_rbr_cache(payload)
    if (payload['dataset'] != 'codearc_replay' or payload['evaluation_role'] != 'development_assessment_only'
            or any(r.get('original_split') == 'primary' for r in payload['records'])):
        raise ValueError('only explicit CodeARC development assessment with primary sources absent is allowed')
    rows = [r for r in payload['records'] if r['split'] == 'test']
    if (len(rows) != 3200 or len({r['source_component_id'] for r in rows}) != 400
            or any(r['original_split'] != 'development' for r in rows)):
        raise ValueError('all 400 original development sources and eight candidates required')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream = args.development_root.resolve()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    cache = upstream/'development-cache.json'
    payload = json.loads(cache.read_text()); rows = validate_development(payload)
    cache_digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    frozen = json.loads((upstream/'status.json').read_text())
    if frozen['status'] != 'complete' or frozen['development_cache_sha256'] != file_sha(cache):
        raise ValueError('completed source-bound feature experiment required')
    population = [{'candidate_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
    legacy_pop = [{'task_id': r['task_id'], 'source_component_id': r['source_component_id']} for r in rows]
    population_digest = hashlib.sha256(json.dumps(legacy_pop, sort_keys=True).encode()).hexdigest()
    paths = [Path(__file__).resolve(), root/'scripts/run_rbr_prediction_gate.py', *sorted((root/'src/pbpf').rglob('*.py'))]
    sources = {str(p.relative_to(root)): file_sha(p) for p in paths}
    bindings = {}
    for kind in ('lexical', 'semantic'):
        for seed in (1701, 1702, 1703):
            name = f'{kind}-seed{seed}'
            sums = json.loads((upstream/f'{name}.checksums.json').read_text())
            if any(Path(n).name != n or file_sha(upstream/n) != h for n, h in sums.items()):
                raise ValueError('checkpoint/report population receipt mismatch')
            report = json.loads((upstream/f'{name}.json').read_text())
            if (report['cache_sha256'] != cache_digest or report['population_sha256'] != population_digest
                    or report['config']['seed'] != seed or report['config']['particles'] != 8
                    or frozen['runs'][name]['results_sha256'] != file_sha(upstream/f'{name}.json')):
                raise ValueError('model population or frozen training identity differs')
            bindings[name] = sums
    # The plan is written before any new predictions. All budgets/models are retained.
    plan = {'schema': 'apbpf-development-particle-sensitivity-v1', 'cache_sha256': file_sha(cache),
            'source_sha256': sources, 'model_bindings': bindings, 'seeds': [1701, 1702, 1703],
            'training_particles': 8, 'inference_particles': list(COUNTS), 'controls': list(CONTROLS),
            'checkpoint_refitting': False, 'candidate_count': 3200, 'source_count': 400,
            'original_primary_sources_excluded': True,
            'hypothesis': 'finite-particle approximation may contribute to weak association and pair-order sensitivity',
            'selection_rule': 'none; report every fixed cell; no primary evaluation or gate replacement',
            'randomness': 'existing full-batch helper seed+100000; paired control draws at each particle budget; larger budgets do not claim nested draws',
            'scope': 'development-only exploratory numerical diagnostic; no new trained-method or confirmatory claim'}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    state = {'status': 'running', 'pid': os.getpid(), 'plan_sha256': file_sha(output/'plan.json'), 'runs': {}}
    def save():
        p = output/'status.partial'; p.write_text(json.dumps(state, indent=2)+'\n'); p.replace(output/'status.json')
    def check():
        if (any(file_sha(root/n) != sha for n, sha in sources.items()) or file_sha(cache) != plan['cache_sha256']
                or any(file_sha(upstream/n) != h for item in bindings.values() for n, h in item.items())):
            raise ValueError('source, data or frozen model changed')
    save()
    try:
        spec = importlib.util.spec_from_file_location('particle_diagnostic_helper', root/'scripts/run_rbr_prediction_gate.py')
        helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper; spec.loader.exec_module(helper)
        results = {}
        for kind in ('lexical', 'semantic'):
            encoder = FrozenTextEncoder(512) if kind == 'lexical' else FrozenSemanticCache(upstream/'semantic-features')
            if kind == 'semantic' and (encoder.manifest['source_payload_sha256'] != cache_digest
                    or encoder.manifest['evaluation_role'] != 'development_assessment_only'
                    or encoder.manifest['expected_is_public']):
                raise ValueError('semantic feature population or visibility differs')
            batch = helper._tensorize(rows, encoder, torch.device('cpu'), expected_is_public=False)
            labels = batch.outcomes[:, 4:].reshape(-1).numpy()
            by_count = {p: {} for p in COUNTS}
            for seed in plan['seeds']:
                name = f'{kind}-seed{seed}'
                original = json.loads((upstream/f'{name}.json').read_text())
                checkpoint = torch.load(upstream/f'{name}.pt', map_location='cpu', weights_only=True)
                if (any(checkpoint.get(k) != v for k, v in original['config'].items())
                        or (kind == 'semantic' and checkpoint['feature_cache_manifest_sha256'] != encoder.manifest_sha256)):
                    raise ValueError('checkpoint configuration or semantic manifest differs')
                model = NeuralBeliefModel(512, 32, 192, difficulty_dim=8)
                model.load_state_dict(checkpoint['model']); model.eval()
                for particles in COUNTS:
                    check(); key = f'{name}-p{particles}'; state['current'] = key
                    state['runs'][key] = {'status': 'running'}; save()
                    values = {'labels': labels}
                    with torch.no_grad():
                        for control in CONTROLS:
                            state['control'] = control; save()
                            values[control] = helper._predict(model, batch, particles=particles,
                                seed=seed+100000, mode=control)
                        rng = torch.Generator().manual_seed(seed+100000)
                        noise = torch.randn((len(rows), particles, 32), generator=rng)
                        uniforms = torch.rand((len(rows), 4), generator=rng)
                        trace = model.filter(batch, particles=particles, visible_steps=4,
                            proposal_noise=noise, resampling_uniforms=uniforms)
                        diagnostics = diversity(trace)
                        del trace
                    if particles == 8:
                        metrics = compare_predictions(labels, {'baseline': values['aligned'], **{k: values[k] for k in CONTROLS}})
                        if any(abs(metrics[k]['nll']-original['metrics'][k]['nll']) > 1e-6 for k in CONTROLS):
                            raise ValueError('original 8-particle numerical results do not reproduce')
                    path = output/f'{key}.npz'; np.savez_compressed(path, **values)
                    by_count[particles][seed] = values
                    state['runs'][key] = {'status': 'complete', 'predictions_sha256': file_sha(path),
                                          'diversity': diagnostics}; save()
            results[kind] = {}
            for particles, values in by_count.items():
                report = aggregate_predictions(values, {s: population for s in plan['seeds']},
                    bootstrap_seed=201701, replicates=10000)
                report['particle_diversity'] = {str(s): state['runs'][f'{kind}-seed{s}-p{particles}']['diversity'] for s in plan['seeds']}
                results[kind][str(particles)] = report
            (output/f'{kind}-results.json').write_text(json.dumps(results[kind], indent=2)+'\n')
        check()
        (output/'results.json').write_text(json.dumps({'plan_sha256': state['plan_sha256'],
            'results': results, 'scope': plan['scope']}, indent=2)+'\n')
        state.update(status='complete', results_sha256=file_sha(output/'results.json')); save()
    except BaseException as error:
        state.update(status='needs_debug', error=repr(error)); save(); raise


if __name__ == '__main__':
    main()
