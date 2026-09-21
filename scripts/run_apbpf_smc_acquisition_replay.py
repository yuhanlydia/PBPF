#!/usr/bin/env python3
"""Development-only acquisition with the trained SMC proposal/resampling path."""
import argparse
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from pbpf.apbpf.smc_acquisition import acquire, filter_history
from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, public_test_text, validate_rbr_cache
from pbpf.registry import OUTCOMES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('replay outputs are create-once')
    payload = validate_rbr_cache(json.loads(args.cache.read_text()))
    if payload.get('evaluation_role') != 'development_assessment_only':
        raise ValueError('an explicitly development-only cache is required')
    rows = [r for r in payload['records'] if r['split'] == 'test']
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('encoder_type') == 'frozen_code_model':
        raise ValueError('semantic checkpoints require their frozen feature cache; hash fallback forbidden')
    if checkpoint.get('evaluation_role') != 'development_assessment_only' or not checkpoint.get('apbpf'):
        raise ValueError('requires a factored development-only checkpoint')
    model = NeuralBeliefModel(checkpoint['feature_dim'], checkpoint['latent_dim'], checkpoint['hidden_dim'],
                             difficulty_dim=checkpoint['difficulty_dim']).to(args.device)
    model.load_state_dict(checkpoint['model']);model.eval()
    encode = lru_cache(maxsize=20000)(FrozenTextEncoder(checkpoint['feature_dim']))
    policies = ['fixed','random','diagnostic_mi','predictive_entropy','joint_particle_mi']
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root/'src/pbpf/apbpf/smc_acquisition.py',
               root/'src/pbpf/belief/model.py', root/'src/pbpf/real_gate.py']
    identity = {'schema':'apbpf-development-smc-acquisition-replay-v1','seed':args.seed,
        'cache_sha256':sha(args.cache),'checkpoint_sha256':sha(args.checkpoint),
        'source_sha256':{str(x.relative_to(root)):sha(x) for x in sources},
        'device':args.device,'batch_size':args.batch_size,'policies':policies,'budgets':[1,2,4],
        'particles':checkpoint['particles'],'public_pool':4,'future_targets':6,
        'inference':'trained SMC: first selected observation conditions root proposal; same common noise and resampling uniforms for every policy/subset; selected-history replay',
        'diagnostic_mi':'diagnosis components crossed with difficulty marginal; selection only',
        'joint_particle_mi':'original weighted joint SMC particles; required nuisance-sensitive ablation',
        'future_scoring':'original weighted joint SMC posterior for every policy',
        'oracle':'best canonical public subset on evaluator future NLL, upper bound only',
        'scope':'nonconfirmatory development-only; upstream gates failed',
        'budget_semantics':'exact cached public observation counts, not new sandbox executions',
        'population':[{'candidate':r['task_id'],'source':r['source_component_id']} for r in rows]}
    args.output.mkdir(parents=True)
    (args.output/'plan.json').write_text(json.dumps(identity,indent=2)+'\n')
    results=[]
    with torch.no_grad(), (args.output/'records.jsonl').open('x') as stream:
        for start in range(0,len(rows),args.batch_size):
            part=rows[start:start+args.batch_size];n=len(part)
            task=torch.as_tensor(np.stack([encode(r['task_text']) for r in part]),device=args.device)
            code=torch.as_tensor(np.stack([encode(r['candidate']) for r in part]),device=args.device)
            features=torch.as_tensor(np.stack([[encode(public_test_text(t,expected_is_public=checkpoint['expected_is_public']))
                for t in r['tests']] for r in part]),device=args.device)
            labels=torch.tensor([[OUTCOMES.index(o) for o in r['outcomes']] for r in part],device=args.device)
            noises,uniforms,orders=[],[],[]
            for row in part:
                seed=int.from_bytes(hashlib.sha256(f'{args.seed}:{row["task_id"]}'.encode()).digest()[:4],'big')
                generator=torch.Generator().manual_seed(seed)
                noises.append(torch.randn(checkpoint['particles'],model.latent_dim,generator=generator))
                uniforms.append(torch.rand(4,generator=generator))
                orders.append(np.random.default_rng(seed).permutation(4))
            noise=torch.stack(noises).to(args.device);uniform=torch.stack(uniforms).to(args.device)
            order=torch.as_tensor(np.stack(orders),device=args.device)
            records=[{'candidate':r['task_id'],'source':r['source_component_id'],
                      'budgets':{str(b):{} for b in (1,2,4)}} for r in part]

            def score(z,weights):
                predicted=model.future_predict(task,code,features[:,4:],z,weights)
                return -predicted.gather(-1,labels[:,4:,None]).squeeze(-1).cpu().double().numpy()

            # acquire receives only four public feature/outcome slots. Evaluator
            # future features/labels are used separately after each policy returns.
            for policy in policies:
                snapshots=acquire(model,task,code,features[:,:4],labels[:,:4],policy=policy,
                                  noise=noise,uniforms=uniform,random_order=order)
                for budget,(z,weights,selected) in snapshots.items():
                    nll=score(z,weights)
                    for i in range(n):
                        records[i]['budgets'][str(budget)][policy]={'selected':selected[i].cpu().tolist(),'future_nll':nll[i].tolist()}
            for budget in (1,2,4):
                subsets=list(itertools.combinations(range(4),budget));nlls=[]
                for subset in subsets:
                    selected=torch.tensor(subset,device=args.device).expand(n,-1)
                    z,weights=filter_history(model,task,code,features[:,:4],labels[:,:4],selected,noise,uniform)
                    nlls.append(score(z,weights))
                nlls=np.stack(nlls);best=nlls.mean(-1).argmin(0)
                for i in range(n):
                    chosen=nlls[best[i],i];fixed=records[i]['budgets'][str(budget)]['fixed']['future_nll']
                    if chosen.mean()>np.mean(fixed)+1e-7:
                        raise AssertionError('oracle omitted fixed canonical subset')
                    if budget==4 and not np.allclose(chosen,fixed,atol=1e-7):
                        raise AssertionError('all-four canonical oracle must coincide with fixed')
                    records[i]['budgets'][str(budget)]['oracle']={'selected':list(subsets[best[i]]),'future_nll':chosen.tolist()}
            for record in records:
                stream.write(json.dumps(record)+'\n')
            stream.flush();results+=records
            print(json.dumps({'completed':len(results),'total':len(rows)}),flush=True)
    sources=sorted({r['source'] for r in results})
    indices=[np.array([i for i,r in enumerate(results) if r['source']==s]) for s in sources]
    counts=np.array([len(i) for i in indices])
    draws=np.random.default_rng(args.seed).integers(0,len(sources),size=(10000,len(sources)))
    summary={}
    for budget in (1,2,4):
        summary[str(budget)]={}
        fixed=np.array([r['budgets'][str(budget)]['fixed']['future_nll'] for r in results])
        for policy in [*policies,'oracle']:
            values=np.array([r['budgets'][str(budget)][policy]['future_nll'] for r in results])
            gaps=(fixed-values).mean(1);sums=np.array([gaps[i].sum() for i in indices])
            boot=sums[draws].sum(1)/counts[draws].sum(1)
            summary[str(budget)][policy]={'nll':float(values.mean()),'nll_advantage_over_fixed':float(gaps.mean()),
                'ci95':np.quantile(boot,[.025,.975]).tolist()}
    report={**identity,'population':{'candidates':len(rows),'source_components':len(sources)},'summary':summary,
        'bootstrap':'10000 whole-source draws; shared draws across policies; future-example weighted',
        'limitations':['Only four public query candidates; budget four changes ordering only.',
                       'No threshold stopping or confirmatory active-testing gate claim.']}
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'summary':summary,'scope':identity['scope']}),flush=True)


if __name__=='__main__':
    main()
