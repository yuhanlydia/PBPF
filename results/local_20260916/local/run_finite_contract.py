"""Standalone finite-contract execution; not a FormalFactory or sealed S1 result."""
import concurrent.futures, hashlib, json, os, time
from pathlib import Path
import numpy as np
import yaml
from pbpf.finite import run_finite_audit
from pbpf.metrics import randomized_hpd_coverage

ROOT = Path('/data/cwj/PBPF')
OUT = Path('/dev/shm/pbpf-additional/finite_contract')
SPEC = yaml.safe_load((ROOT/'configs/iclr/finite.yaml').read_text())
ARMS = ['exact_bayes','map','posterior_mean','uniform','shuffled'] + [f'particles_{n}' for n in SPEC['audit_particles']]

def family_work(family):
    ps=SPEC['probability_spec']; count=ps['cases_per_family']; states=ps['latent_states']; nt=SPEC['tests_per_case']; no=len(SPEC['outcomes']); nv=SPEC['visible_tests']
    priors=np.empty((count,states)); likelihoods=np.empty((count,nt,states,no)); observations=np.empty((count,nt),dtype=np.int8); truths=np.empty(count,dtype=np.int8)
    metrics=np.empty((count,len(ARMS),4)); posteriors=np.empty((count,len(ARMS),states))
    for case in range(count):
        template=case%ps['templates_per_family']
        key=json.dumps([SPEC['generator_version'],SPEC['seed'],family,template,case],separators=(',',':')).encode()
        seed=int.from_bytes(hashlib.sha256(key).digest()[:16],'big')
        rng=np.random.Generator(np.random.PCG64(seed))
        prior=rng.dirichlet(np.ones(states)); probs=rng.dirichlet(np.full(no,.5),size=(nt,states)); truth=int(rng.choice(states,p=prior)); outcomes=np.array([rng.choice(no,p=probs[t,truth]) for t in range(nt)])
        result=run_finite_audit(prior=prior,outcome_probabilities=probs,observed_outcomes=outcomes[:nv].tolist(),future_outcomes=outcomes[nv:].tolist(),particle_counts=SPEC['audit_particles'],rng=rng)
        priors[case]=prior; likelihoods[case]=probs; observations[case]=outcomes; truths[case]=truth
        for index,arm in enumerate(ARMS):
            m=result['arms'][arm]; posterior=np.asarray(m['posterior']); posteriors[case,index]=posterior
            coverage=randomized_hpd_coverage(posterior,truth,mass=SPEC['gate']['hpd_level'],rng=rng)
            metrics[case,index]=[m['future_nll'],m['brier'],m['posterior_kl'],coverage]
    path=OUT/f'family_{family:03d}.npz'
    np.savez_compressed(path,priors=priors,likelihoods=likelihoods,observations=observations,true_latents=truths,metrics=metrics,posteriors=posteriors)
    return family,metrics,hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.time(); allmetrics={}; hashes={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for family,metrics,digest in pool.map(family_work,range(SPEC['probability_spec']['latent_families'])):
            allmetrics[family]=metrics; hashes[f'family_{family:03d}.npz']=digest
            print(json.dumps({'families_completed':len(allmetrics),'cases_completed':len(allmetrics)*500,'elapsed_seconds':time.time()-started}),flush=True)
    report={'scope':'standalone finite-contract diagnostic; not formal S1','spec_sha256':hashlib.sha256((ROOT/'configs/iclr/finite.yaml').read_bytes()).hexdigest(),'generator_seed_encoding':'SHA256 compact JSON array [version,seed,family,template,case], first 16 bytes as big-endian unsigned integer; PCG64','arms':ARMS,'metric_columns':['future_nll','brier','posterior_kl','hpd_coverage'],'splits':{},'files':hashes,'elapsed_seconds':time.time()-started}
    for split,bounds in SPEC['probability_spec']['split_families'].items():
        values=np.concatenate([allmetrics[f] for f in range(bounds[0],bounds[1]+1)])
        target=values[:,ARMS.index('particles_32')]; exact=values[:,ARMS.index('exact_bayes')]; gap=float((target[:,0].mean()-exact[:,0].mean())/exact[:,0].mean()); median=float(np.median(target[:,2])); coverage=float(target[:,3].mean())
        report['splits'][split]={'cases':len(values),'family_ids':list(range(bounds[0],bounds[1]+1)),'metrics':{arm:{'future_nll':float(values[:,i,0].mean()),'brier':float(values[:,i,1].mean()),'median_posterior_kl':float(np.median(values[:,i,2])),'hpd_coverage':float(values[:,i,3].mean())} for i,arm in enumerate(ARMS)},'gate':{'median_kl':median,'relative_future_nll_gap':gap,'hpd_coverage':coverage,'passes':median<=SPEC['gate']['median_kl_max'] and gap<=SPEC['gate']['relative_future_nll_gap_max'] and SPEC['gate']['hpd_coverage'][0]<=coverage<=SPEC['gate']['hpd_coverage'][1]}}
        assert len(values)==SPEC['counts'][split]
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['gate'] for k,v in report['splits'].items()}),flush=True)

if __name__=='__main__': main()
