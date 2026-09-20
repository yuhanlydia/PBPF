"""Read-only instrumentation of stored real-development diagnostic checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_association_debug import real_batches
from pbpf.belief.diagnostic import InteractionHistoryISBeliefModel, BoundHistoryISBeliefModel, InteractionOnlyHistoryISBeliefModel, HighGainHistoryISBeliefModel


def stats(value):
    value = value.detach().float().flatten()
    return dict(mean=float(value.mean()), median=float(value.median()),
                p90=float(torch.quantile(value, .9)), max=float(value.max()))


@torch.no_grad()
def inspect(run, particles, seed, device, counterfactual_seed):
    saved = torch.load(run / 'model.pt', map_location=device, weights_only=False)
    config = saved['config']
    feature_cache = Path(config['feature_cache']) if config.get('feature_cache') else None
    batches, metadata = real_batches(Path(config['cache']), config['feature_dim'], device, feature_cache)
    for key in ('cache_sha256', 'feature_sha256'):
        if metadata.get(key) != saved['data'].get(key):
            raise ValueError(f'checkpoint {key} does not match current inputs')
    batch = batches['development']
    shuffled, eligible = batch.outcome_counterfactual(visible_steps=4, seed=counterfactual_seed)
    classes = {c.__name__: c for c in (InteractionHistoryISBeliefModel, BoundHistoryISBeliefModel, InteractionOnlyHistoryISBeliefModel, HighGainHistoryISBeliefModel)}
    model = classes[saved['model_class']](config['feature_dim'], config['latent_dim'],
        config['hidden_dim'], difficulty_dim=config['difficulty_dim']).to(device).eval()
    model.load_state_dict(saved['model'])
    encoded, bound, affine, preactivations = [], [], [], []
    for source in (batch, shuffled):
        tests = source.tests[:, :4]
        labels = F.one_hot(source.outcomes[:, :4], 5).to(tests.dtype)
        binding = (labels[..., :, None] * tests[..., None, :]).flatten(-2)
        inputs = binding if isinstance(model, BoundHistoryISBeliefModel) else torch.cat([tests, labels], -1)
        pre = model.pair_encoder[0](inputs)
        preactivations.append(pre)
        encoded.append(model.encode_pairs(tests, labels))
        bound.append(binding.mean(1))
        affine.append(model.pair_encoder[2](pre).mean(1))
    qa, qs = model.history_proposal(batch, 4), model.history_proposal(shuffled, 4)
    noise = torch.randn((len(batch.task), particles, model.latent_dim), device=device,
                        generator=torch.Generator(device=device).manual_seed(seed))
    a = model.filter(batch, particles=particles, proposal_noise=noise)
    s = model.filter(shuffled, particles=particles, proposal_noise=noise)
    z, weights = a.latents[:, -1], a.log_weights[:, -1]
    def predict(latents, log_weights):
        return model.future_predict(batch.task, batch.candidate, batch.tests[:, 4:], latents, log_weights)
    aligned, counterfactual = predict(z, weights), predict(s.latents[:, -1], s.log_weights[:, -1])
    # Reweight the *same* aligned proposal samples by shuffled-history likelihood.
    # This isolates likelihood sensitivity at a fixed finite particle support.
    raw = model.root(batch.task, batch.candidate).log_prob(z) - qa.log_prob(z)
    for i in range(4):
        lp = model.likelihood(z, batch.task, batch.candidate, batch.tests[:, i])
        raw += lp.gather(-1, shuffled.outcomes[:, i, None, None].expand(-1, particles, 1)).squeeze(-1)
    reweighted = predict(z, raw - raw.logsumexp(-1, keepdim=True))
    def nll(lp):
        return -lp.gather(-1, batch.outcomes[:, 4:, None]).squeeze(-1).mean(-1)
    na, ns, nr = nll(aligned), nll(counterfactual), nll(reweighted)
    dim = model.difficulty_dim
    relative_encoding = (encoded[0]-encoded[1]).norm(dim=-1) / encoded[0].norm(dim=-1).clamp_min(1e-8)
    relative_binding = (bound[0]-bound[1]).norm(dim=-1) / bound[0].norm(dim=-1).clamp_min(1e-8)
    return dict(run=str(run), selected_step=saved['step'], model_class=saved['model_class'],
        model_sha256=hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),
        data_cache_sha256=metadata['cache_sha256'], feature_sha256=metadata.get('feature_sha256'), particles=particles, seed=seed, counterfactual_seed=counterfactual_seed,
        eligible=int(eligible.sum()), development_candidates=len(batch.task),
        pair_encoder_relative_change=stats(relative_encoding[eligible]),
        explicit_binding_relative_change=stats(relative_binding[eligible]),
        affine_encoder_shuffle_delta=stats((affine[0]-affine[1]).norm(dim=-1)[eligible]),
        preactivation_abs=stats(preactivations[0].abs()),
        diagnosis_proposal_mean_delta_in_std_units=stats((((qa.mean[:,dim:]-qs.mean[:,dim:])/qa.std[:,dim:]).square().mean(-1).sqrt())[eligible]),
        diagnosis_proposal_log_std_delta=stats((qa.log_std[:,dim:]-qs.log_std[:,dim:]).square().mean(-1).sqrt()[eligible]),
        difficulty_proposal_mean_delta=stats((qa.mean[:,:dim]-qs.mean[:,:dim]).abs()),
        ess_fraction=stats(weights.exp().square().sum(-1).reciprocal()/particles),
        predictive_total_variation=stats((aligned.exp()-counterfactual.exp()).abs().sum(-1)[eligible]/2),
        aligned_nll=float(na.mean()), gap_full=float((ns-na).mean()), gap_eligible=float((ns-na)[eligible].mean()),
        same_proposal_reweighted_gap_full=float((nr-na).mean()),
        same_proposal_reweighted_gap_eligible=float((nr-na)[eligible].mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--particles', type=int, default=64)
    parser.add_argument('--seed', type=int, default=51701)
    parser.add_argument('--counterfactual-seed', type=int, default=51701)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('audit outputs are create-once')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = [inspect(run, args.particles, args.seed, device, args.counterfactual_seed) for run in args.runs]
    report = dict(scope='development_only_mechanism_audit', test_evaluated=False, rows=rows,
        caveat='A changed proposal is not proof of learned diagnosis; importance likelihoods may retain pairing even with invariant proposals. Reweighting uses finite aligned support.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
