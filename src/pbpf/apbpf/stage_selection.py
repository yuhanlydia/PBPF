"""Public-only whole-suite utility heads and source-cross-fitted selection."""
import copy
from functools import lru_cache
import json

import numpy as np
import torch

from pbpf.belief.features import BeliefBatch
from pbpf.belief.model import NeuralBeliefModel
from pbpf.real_gate import FrozenTextEncoder, validate_rbr_cache
from pbpf.registry import OUTCOMES
from .codearc_bank import file_sha
from .selection import SuccessHead, cross_fitted_choices, public_batch

NEURAL_BASELINES = ('pair_aware', 'deep_sets', 'no_particle_bottleneck')


def subset(batch, indices):
    return BeliefBatch(*(getattr(batch, key)[indices] for key in ('task','candidate','tests','outcomes')))


def warm_start(head, saved):
    """Transfer public pair/context features from the actual outcome predictor."""
    state = saved['model']
    head.pair.load_state_dict({k.removeprefix('pair.'):v for k,v in state.items() if k.startswith('pair.')})
    head.network[0].load_state_dict({k:state['context.0.'+k] for k in ('weight','bias')})


def fit_selection(payload, belief_checkpoint, deterministic_checkpoints, output, *, seed, cache_sha256, steps=1000):
    validate_rbr_cache(payload)
    if payload.get('evaluation_role')!='exploratory_locked_primary_assessment' or steps<1:
        raise ValueError('full original-split assessment cache and positive fitting budget required')
    saved = torch.load(belief_checkpoint,map_location='cpu',weights_only=True)
    if not saved['apbpf'] or saved['seed']!=seed or saved['encoder']!='frozen_hash_text':
        raise ValueError('requires the exact factored belief checkpoint')
    if set(deterministic_checkpoints)!=set(NEURAL_BASELINES):
        raise ValueError('all three trained deterministic predictor checkpoints required')
    deterministic = {k:torch.load(p,map_location='cpu',weights_only=True) for k,p in deterministic_checkpoints.items()}
    for arm, checkpoint in deterministic.items():
        if (checkpoint['arm']!=arm or checkpoint['kind']!='baselines'
                or any(checkpoint[k]!=saved[k] for k in ('seed','steps','batch_size','learning_rate','feature_dim','hidden_dim','latent_dim','expected_is_public','protocol'))):
            raise ValueError('outcome pretraining identities or budgets differ across arms')
    rows={s:[r for r in payload['records'] if r['split']==s] for s in ('train','development','test')}
    if any(not v for v in rows.values()) or any(r.get('original_split')!='primary' for r in rows['test']):
        raise ValueError('all original partitions are required')
    groups={}
    for i,row in enumerate(rows['test']):groups.setdefault(row['source_component_id'],[]).append(i)
    if len(groups)<5 or any(len(indices)!=8 for indices in groups.values()):
        raise ValueError('source-cross-fitting requires at least five full eight-candidate groups')
    output.mkdir(parents=True,exist_ok=False)
    plan={'schema':'apbpf-full-primary-selection-v1','dataset':payload['dataset'],'seed':seed,'steps':steps,
          'cache_sha256':cache_sha256,'belief_checkpoint_sha256':file_sha(belief_checkpoint),
          'deterministic_checkpoint_sha256':{k:file_sha(p) for k,p in deterministic_checkpoints.items()},
          'input_contract':'task, candidate and exactly four public input/outcome observations; no future test content',
          'target':'all six evaluator-hidden calls pass',
          'training':'train-only BCE; development-only checkpoint/smoothing selection; batch64 AdamW lr.0003 wd.01',
          'pretrained_controls':'pair and context layers transferred from matched outcome predictors, then utility-fitted',
          'fresh_controls':'same architectures trained from fresh initialization as additional deterministic controls',
          'comparator':'five source folds; choose strongest deterministic selector on other folds only',
          'tie_break':'original immutable candidate order','scope':'exploratory full-primary selection; upstream failed gates remain binding'}
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    encode=lru_cache(maxsize=20000)(FrozenTextEncoder(saved['feature_dim']))
    batches={s:public_batch(v,encode,expected_is_public=saved['expected_is_public']) for s,v in rows.items()}
    targets={s:torch.tensor([all(o=='PASS' for o in r['outcomes'][4:]) for r in v],dtype=torch.float32) for s,v in rows.items()}
    model=NeuralBeliefModel(saved['feature_dim'],saved['latent_dim'],saved['hidden_dim'],difficulty_dim=saved['difficulty_dim'])
    model.load_state_dict(saved['model']);model.eval();posterior={}
    with torch.no_grad():
        for split,batch in batches.items():
            zs,ws=[],[]
            for start in range(0,len(batch.task),64):
                trace=model.filter(subset(batch,slice(start,start+64)),particles=saved['particles'],visible_steps=4,
                    generator=torch.Generator().manual_seed(seed+300000+start))
                zs.append(trace.latents[:,-1]);ws.append(trace.log_weights[:,-1])
            posterior[split]=(torch.cat(zs),torch.cat(ws))
    fitting,scores={},{}
    arms=['particle',*NEURAL_BASELINES,*[a+'_pretrained' for a in NEURAL_BASELINES]]
    for offset,name in enumerate(arms):
        torch.manual_seed(seed+offset);arm=name.removesuffix('_pretrained')
        head=SuccessHead(saved['feature_dim'],saved['latent_dim'],saved['hidden_dim'],arm)
        if name.endswith('_pretrained'):warm_start(head,deterministic[arm])
        optimizer=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.01)
        rng=np.random.default_rng(seed);best,best_state,best_step=float('inf'),None,None;history=[]
        for step in range(1,steps+1):
            indices=torch.tensor(rng.integers(0,len(rows['train']),size=64));batch=subset(batches['train'],indices)
            z,w=posterior['train'];prediction=head(batch,z[indices],w[indices]).clamp(1e-7,1-1e-7)
            loss=torch.nn.functional.binary_cross_entropy(prediction,targets['train'][indices])
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step==1 or step%max(1,steps//20)==0:
                with torch.no_grad():
                    prediction=head(batches['development'],*posterior['development']).clamp(1e-7,1-1e-7)
                    value=float(torch.nn.functional.binary_cross_entropy(prediction,targets['development']))
                history.append({'step':step,'development_bce':value})
                if value<best:best,best_state,best_step=value,copy.deepcopy(head.state_dict()),step
        if best_state is None:raise RuntimeError('utility fitting produced no finite checkpoint')
        head.load_state_dict(best_state);head.eval()
        with torch.no_grad():scores[name]=head(batches['test'],*posterior['test']).numpy()
        checkpoint=output/f'{name}.pt'
        with checkpoint.open('xb') as stream:torch.save({'state':best_state,'arm':name,'plan_sha256':file_sha(output/'plan.json')},stream)
        fitting[name]={'best_step':best_step,'development_bce':best,'history':history,'checkpoint_sha256':file_sha(checkpoint),
                       'parameters':sum(p.numel() for p in head.parameters())}
        print(json.dumps({'seed':seed,'utility_arm':name,'best_step':best_step,'development_bce':best}),flush=True)
    pass_index=OUTCOMES.index('PASS');grid=[.01,.1,.25,.5,1.,2.,5.,10.]
    def rate(split,alpha):return (((batches[split].outcomes==pass_index).sum(1).numpy()+alpha)/(4+2*alpha))**6
    def nll(prob):
        y=targets['development'].numpy();p=prob.clip(1e-7,1-1e-7)
        return float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean())
    alpha=min(grid,key=lambda a:nll(rate('development',a)))
    scores['tuned_dirichlet']=rate('test',alpha)
    scores['visible_pass_rate']=(batches['test'].outcomes==pass_index).float().mean(1).numpy()
    fitting['tuned_dirichlet']={'alpha':alpha,'grid':grid,'development_bce':nll(rate('development',alpha))}
    labels=targets['test'].numpy();successes,selections={},{}
    for arm,values in scores.items():
        chosen=[indices[int(values[indices].argmax())] for indices in groups.values()]
        successes[arm]=labels[chosen];selections[arm]=[rows['test'][i]['task_id'] for i in chosen]
    sources=list(groups)
    comparator,folds=cross_fitted_choices(sources,{k:v for k,v in successes.items() if k!='particle'},seed=seed)
    gap=successes['particle']-comparator
    draw=np.random.default_rng(201701).integers(0,len(groups),size=(10000,len(groups)))
    np.savez_compressed(output/'candidate-scores.npz',**scores,labels=labels)
    result={**plan,'population':{'primary_sources':len(groups),'primary_candidates':len(rows['test'])},
            'fitting':fitting,'sources':sources,'candidate_order':[r['task_id'] for r in rows['test']],
            'selected_successes':{k:v.tolist() for k,v in successes.items()},'cross_fitted_successes':comparator.tolist(),
            'crossfit':folds,'selections':selections,'advantage':float(gap.mean()),
            'ci95':np.quantile(gap[draw].mean(1),[.025,.975]).tolist(),
            'selected_pass1':{**{k:float(v.mean()) for k,v in successes.items()},
                'cross_fitted_deterministic':float(comparator.mean()),
                'random_expectation':float(np.mean([labels[i].mean() for i in groups.values()])),
                'oracle':float(np.mean([labels[i].max() for i in groups.values()]))}}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def aggregate_selection(reports):
    if set(reports)!={1701,1702,1703}:raise ValueError('all fixed seeds required')
    sources=reports[1701]['sources']
    if any(r['sources']!=sources or r['seed']!=seed for seed,r in reports.items()):
        raise ValueError('selection source order or seed identity differs')
    differences=np.array([np.array(reports[s]['selected_successes']['particle'])-reports[s]['cross_fitted_successes'] for s in (1701,1702,1703)])
    gap=differences.mean(0);draw=np.random.default_rng(201701).integers(0,len(sources),size=(10000,len(sources)))
    return {'seeds':[1701,1702,1703],'primary_sources':len(sources),'primary_candidates':reports[1701]['population']['primary_candidates'],
            'comparator':'strongest_cross_fitted_deterministic','absolute_selected_pass1_advantage':float(gap.mean()),
            'ci95':np.quantile(gap[draw].mean(1),[.025,.975]).tolist(),
            'seed_estimand':'mean paired selected-success difference over all fixed seeds; whole-source resampling',
            'bootstrap_draws':10000,'per_seed':{str(s):{'advantage':reports[s]['advantage'],'ci95':reports[s]['ci95'],'selected_pass1':reports[s]['selected_pass1']} for s in (1701,1702,1703)}}
