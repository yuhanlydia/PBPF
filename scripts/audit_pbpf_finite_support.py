#!/usr/bin/env python3
"""Diagnose support loss in archived finite-contract outputs, without refitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(directory):
    summary = json.loads((directory/'summary.json').read_text())
    result = {}
    for split, info in summary['splits'].items():
        values = {arm: [] for arm in summary['arms'] if arm.startswith('particles_')}
        for family in info['family_ids']:
            path = directory/f'family_{family:03d}.npz'
            if sha(path) != summary['files'][path.name]:
                raise ValueError('finite archive differs from original summary')
            with np.load(path, allow_pickle=False) as archive:
                posterior = archive['posteriors']
                exact = posterior[:, summary['arms'].index('exact_bayes')]
                truth, metrics = archive['true_latents'], archive['metrics']
                if not np.isfinite(posterior).all() or not np.allclose(posterior.sum(-1), 1.):
                    raise ValueError('finite posterior archive is not normalized')
                for arm in values:
                    index = summary['arms'].index(arm)
                    approximation = posterior[:, index]
                    floor = np.isclose(approximation, 1e-12, rtol=1e-6, atol=0)
                    missed_mass = (exact*floor).sum(1)
                    floor_kl = np.where(floor, exact*np.log(exact.clip(1e-300)/approximation.clip(1e-300)), 0).sum(1)
                    values[arm].append(np.column_stack([(~floor).sum(1), missed_mass,
                        floor[np.arange(len(truth)), truth], floor_kl, metrics[:, index, 2]]))
        result[split] = {}
        for arm, chunks in values.items():
            v = np.concatenate(chunks)
            if len(v) != info['cases']:
                raise ValueError('finite support audit cannot omit cases')
            result[split][arm] = {'cases': len(v), 'mean_states_above_numerical_floor': float(v[:, 0].mean()),
                'mean_exact_posterior_mass_on_floor_states': float(v[:, 1].mean()),
                'median_exact_posterior_mass_on_floor_states': float(np.median(v[:, 1])),
                'true_latent_on_floor_fraction': float(v[:, 2].mean()),
                'mean_kl_contribution_from_floor_states': float(v[:, 3].mean()),
                'mean_total_kl': float(v[:, 4].mean()), 'median_total_kl': float(np.median(v[:, 4]))}
    return {'summary_sha256': sha(directory/'summary.json'), 'results': result,
            'floor_definition': 'np.isclose(saved posterior, 1e-12, rtol=1e-6, atol=0); stabilizer used by archived finite metric implementation',
            'scope': 'descriptive support audit of existing finite outputs only; no rerun, tuning, new gate or A-PBPF mechanism claim',
            'limitations': ['Final floor states do not distinguish initial proposal omission from later resampling loss.',
                           'Finite discrete support results do not directly identify causes in neural continuous-latent A-PBPF.'],
            'audit_source_sha256': sha(Path(__file__))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = audit(args.input)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({s: v['particles_32'] for s, v in report['results'].items()}), flush=True)


if __name__ == '__main__':
    main()
