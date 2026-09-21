"""Build actor-only repair packets from public history and bound selectors."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch

from pbpf.belief.features import BeliefBatch
from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, public_test_text
from pbpf.registry import OUTCOMES
from .codearc_bank import file_sha
from .repair_materials import actor_rows


def select_repair_rows(cache, targets, context, selection, *, cache_sha256, checkpoint_sha256, seed):
    if (selection['seed'] != seed or selection['cache_sha256'] != cache_sha256
            or selection['belief_checkpoint_sha256'] != checkpoint_sha256):
        raise ValueError('repair selector differs from exact cache, checkpoint or seed')
    rows = actor_rows(cache, targets, context)
    primary = [r for r in rows if r['split'] == 'primary']
    if selection['candidate_order'] != [r['task_id'] for r in primary]:
        raise ValueError('repair selector primary candidate order differs')
    sources = list(dict.fromkeys(r['source_component_id'] for r in primary))
    selected = selection['selections']['particle']
    if selection['sources'] != sources or len(selected) != len(sources) or len(set(selected)) != len(selected):
        raise ValueError('repair must retain one selected candidate per primary source')
    lookup = {r['task_id']: r for r in primary}
    chosen = []
    for source, candidate in zip(sources, selected, strict=True):
        if candidate not in lookup or lookup[candidate]['source_component_id'] != source:
            raise ValueError('repair selector candidate belongs to another source')
        chosen.append(lookup[candidate])
    return [r for r in rows if r['split'] != 'primary'] + chosen


def public_posteriors(rows, checkpoint, *, seed, batch_size=64):
    """Four public tests only; return diagnosis components, excluding difficulty."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (saved['seed'] != seed or saved.get('encoder') != 'frozen_hash_text'
            or not saved['apbpf'] or saved['difficulty_dim'] != 8 or saved['diagnosis_dim'] != 24):
        raise ValueError('repair requires the declared factored belief checkpoint')
    if not rows or batch_size < 1:
        raise ValueError('nonempty actor rows and positive batch size required')
    model = NeuralBeliefModel(saved['feature_dim'], saved['latent_dim'], saved['hidden_dim'], difficulty_dim=8)
    model.load_state_dict(saved['model'])
    model.eval()
    encode = lru_cache(maxsize=20000)(FrozenTextEncoder(saved['feature_dim']))
    generator = torch.Generator().manual_seed(seed + 880000)
    particles, weights = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start+batch_size]
            if any(len(r['tests']) != 4 or len(r['outcomes']) != 4 for r in part):
                raise ValueError('repair posterior accepts exactly four public observations')
            batch = BeliefBatch(
                torch.tensor(np.stack([encode(r['task_text']) for r in part])),
                torch.tensor(np.stack([encode(r['candidate']) for r in part])),
                torch.tensor(np.stack([[encode(public_test_text(t, expected_is_public=saved['expected_is_public']))
                                       for t in r['tests']] for r in part])),
                torch.tensor([[OUTCOMES.index(o) for o in r['outcomes']] for r in part]))
            trace = model.filter(batch, particles=saved['particles'], visible_steps=4, generator=generator)
            particles.append(trace.latents[:, 3, :, 8:].numpy())
            weights.append(trace.log_weights[:, 3].numpy())
    z, logw = np.concatenate(particles), np.concatenate(weights)
    if z.shape != (len(rows), 8, 24) or logw.shape != (len(rows), 8) or not np.isfinite(z).all() or not np.isfinite(logw).all():
        raise ValueError('invalid diagnosis posterior packet')
    if not np.allclose(np.exp(logw).sum(1), 1., atol=1e-5):
        raise ValueError('unnormalized repair posterior weights')
    return z, logw


def write_packet(rows, checkpoint, output, *, seed, binding):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    z, logw = public_posteriors(rows, checkpoint, seed=seed)
    data = output/'actor-rows.jsonl'
    data.write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows))
    with (output/'posteriors.npz').open('xb') as stream:
        np.savez_compressed(stream, diagnosis=z, log_weights=logw)
    counts = {s: sum(r['split'] == s for r in rows) for s in ('train', 'development', 'primary')}
    manifest = {'schema': 'apbpf-repair-actor-packet-v1', 'seed': seed,
                'belief_checkpoint_sha256': file_sha(checkpoint), 'binding': binding,
                'counts': counts, 'primary_sources': len({r['source_component_id'] for r in rows if r['split'] == 'primary'}),
                'files': {n: file_sha(output/n) for n in ('actor-rows.jsonl', 'posteriors.npz')},
                'conditioning': 'diagnosis-only; eight particles; four public observations',
                'visibility': 'public observations plus nonprimary supervised targets; no future tests or primary references',
                'scope': 'supportive exploratory repair after failed upstream gates; not a stage completion'}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
