"""Train-selected kernel/histogram predictor; exploratory development check."""
import sys,json,itertools,hashlib,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_association_debug import real_batches,mean_nll
from pbpf.apbpf.counterfactual import outcome_derangement, joint_permutation
from pbpf.belief.kernel_diagnostic import kernel_predict
from pbpf.real_gate import clustered_nll_gap
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cache',type=Path,required=True)
parser.add_argument('--feature-cache',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():
 raise FileExistsError('results are create-once')

def predict(x,y,alpha,temperature,mix):
 return kernel_predict(x,y[:,:4],visible_steps=4,alpha=alpha,temperature=temperature,mix=mix).reshape(-1,5)

report={'scope':'exploratory_development_only','test_evaluated':False,'selection_split':'train',
 'grid':{'alpha':[.01,.03,.1,.3,1.,3.], 'temperature':[.01,.03,.1,.3,1.], 'mix':[0.,.1,.25,.5,.75,1.]},
 'formula':'(4 * sum_visible [(1-mix)/4 + mix*softmax(cosine/temperature)]*onehot(outcome) + alpha)/(4+5*alpha)',
 'source_sha256':{str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).resolve().parents[1]/'src/pbpf/belief/kernel_diagnostic.py',Path(__file__).resolve().parents[1]/'src/pbpf/belief/semantic_features.py',Path(__file__).resolve().parent/'run_association_debug.py']},
 'caveat':'Adaptive exploratory development result; shuffle seeds are sensitivity checks, not independent replications.'}
for feature in ('lexical','semantic'):
 batches,meta=real_batches(args.cache,256,'cpu',args.feature_cache if feature=='semantic' else None)
 train=batches['train'];x=train.tests.numpy();y=train.outcomes.numpy();labels=y[:,4:].reshape(-1)
 grid=[]
 for a,t,m in itertools.product(*report['grid'].values()):
  grid.append({'alpha':a,'temperature':t,'mix':m,'train_nll':mean_nll(labels,predict(x,y,a,t,m))})
 chosen=min(grid,key=lambda r:r['train_nll']);hist=min((r for r in grid if r['mix']==0),key=lambda r:r['train_nll'])
 dev=batches['development'];dx=dev.tests.numpy();dy=dev.outcomes.numpy();dl=dy[:,4:].reshape(-1)
 config={k:chosen[k] for k in ('alpha','temperature','mix')}; aligned=predict(dx,dy,**config)
 result={'selected':chosen,'train_histogram':hist,'development_nll':mean_nll(dl,aligned),'repeats':[]}
 hp=predict(dx,dy,**{k:hist[k] for k in ('alpha','temperature','mix')})
 result['development_histogram_nll']=mean_nll(dl,hp)
 for seed in (51701,71701,81701,91701):
  shuffled=predict(dx,outcome_derangement(dy,4,seed),**config)
  result['repeats'].append({'seed':seed,'shuffled_nll':mean_nll(dl,shuffled),'bootstrap':clustered_nll_gap(dl,aligned,shuffled,np.repeat(meta['source_ids']['development'],dy.shape[1]-4),seed=201701,replicates=10000)})
 permuted_x,permuted_y=joint_permutation(dx,dy,4,51701)
 result['pair_permutation_max_probability_difference']=float(np.max(np.abs(aligned-predict(permuted_x,permuted_y,**config))))
 result['histogram_comparison_bootstrap']=clustered_nll_gap(dl,aligned,hp,np.repeat(meta['source_ids']['development'],dy.shape[1]-4),seed=201701,replicates=10000)
 result['data']=meta
 result['grid_results']=grid
 report[feature]=result
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as stream:
 stream.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
