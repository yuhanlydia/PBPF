"""Validate completed documented diagnostic bundles and publish a run inventory."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import yaml
from pbpf.artifacts import read_immutable_report
ROOT=Path('/data/cwj/PBPF'); OUT=ROOT/'local/results'; SCRATCH=Path('/dev/shm/pbpf-additional')

def sha(path):
 with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 summary={'scope':'Execute documented experiments; matching published values is not required.','completed':{},'blocked':{},'pending':[]}
 for name in ('evalplus_frozen_164','evalplus_repair_164'):
  root=OUT/name
  if not (root/'summary.json').exists(): summary['pending'].append(name); continue
  rows=json.loads((root/'summary.json').read_text()); assert len(rows)==164 and {r['task_id'] for r in rows}=={f'HumanEval/{i}' for i in range(164)}
  for row in rows:
   bundle=root/row['task_id'].replace('/','-'); report=read_immutable_report(bundle)
   assert len(report['round_pass_at_1'])==4
   assert report['generated_tokens'] <= 10 * 192
   assert report['conditioned_repairs']==4
   assert report['final_pass_at_1']==row['final_pass_at_1']
  summary['completed'][name]={'tasks':164,'bundles_verified':164,'rounds_per_task':4,'initial_tasks_with_passing_candidate':sum(r['initial_passes']>0 for r in rows),'final_solved':sum(r['final_pass_at_1'] for r in rows),'summary_sha256':sha(root/'summary.json')}
 if all(name in summary['completed'] for name in ('evalplus_frozen_164','evalplus_repair_164')):
  profiles={name:{r['task_id']:r for r in json.loads((OUT/name/'summary.json').read_text())} for name in ('evalplus_frozen_164','evalplus_repair_164')}
  ids=[f'HumanEval/{i}' for i in range(164)]
  frozen=np.array([profiles['evalplus_frozen_164'][i]['final_pass_at_1'] for i in ids]); repair=np.array([profiles['evalplus_repair_164'][i]['final_pass_at_1'] for i in ids])
  rng=np.random.default_rng(1701); draw=rng.integers(0,164,size=(10000,164)); difference=repair-frozen
  from scipy.stats import binomtest
  wins=int(((repair==1)&(frozen==0)).sum()); losses=int(((repair==0)&(frozen==1)).sum())
  comparison={'paired_tasks':164,'repair_minus_frozen_pass_at_1':float(difference.mean()),'paired_bootstrap_95ci':np.quantile(difference[draw].mean(axis=1),[.025,.975]).tolist(),'bootstrap_replicates':10000,'bootstrap_seed':1701,'repair_only_pass':wins,'frozen_only_pass':losses,'mcnemar_exact_pvalue':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,'scope':'Historical heuristic diagnostic; no learned PBPF efficacy claim.'}
  (OUT/'evalplus_additional_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
  summary['completed']['evalplus_paired_comparison']=comparison
 for name in ('legacy_exact_smoke','legacy_rbr_6581'):
  root=OUT/name
  if not (root/'manifest.json').exists(): summary['pending'].append(name); continue
  report=read_immutable_report(root)
  summary['completed'][name]={'bundle_verified':True,'rounds':len(report['round_pass_at_1']),'final_pass_at_1':report['final_pass_at_1'],'manifest_sha256':sha(root/'manifest.json')}
 root=OUT/'finite_contract'
 if (root/'summary.json').exists():
  finite=json.loads((root/'summary.json').read_text()); total=0
  for filename,digest in finite['files'].items():
   p=root/filename; assert sha(p)==digest
   with np.load(p,allow_pickle=False) as a:
    assert a['metrics'].shape==(500,10,4) and np.isfinite(a['metrics']).all()
    assert np.allclose(a['priors'].sum(axis=-1),1) and np.allclose(a['likelihoods'].sum(axis=-1),1)
    total+=len(a['metrics'])
  assert total==60000
  summary['completed']['finite_contract']={'cases':total,'splits':{k:v['cases'] for k,v in finite['splits'].items()},'test_gate':finite['splits']['test']['gate'],'scope':'Standalone local generator using repository finite audit; not a sealed formal run.'}
 else: summary['pending'].append('finite_contract')
 audit=json.loads((OUT/'selector_bank_audit.json').read_text()); assert audit==json.loads((ROOT/'results/pbpf_selector_bank_audit.json').read_text())
 summary['completed']['selector_bank_audit']={'matches_committed_audit':True,'decision':audit['decision'],'source_files':audit['source_files']}
 pc=json.loads((OUT/'evalplus_evaluator_positive_control.json').read_text()); assert pc['passes']==pc['tasks']==164
 summary['completed']['evalplus_evaluator_positive_control']={'passed':164,'total':164}
 config=yaml.safe_load((ROOT/'configs/experiments/iclr_pbpf.yaml').read_text())
 summary['blocked']['formal_s0_s4']={'reason':'Repository lacks complete production FormalFactory; operator site, trusted evaluator deployment and approved H200 resources are not provisioned. Exact formal launcher exits 2 before jobs/models start.','exit_code':int((ROOT/'local/logs/formal_launch_all_experiments.exit').read_text()),'cells':config['matrix'],'log':'local/logs/formal_launch_all_experiments.log'}
 for p in (ROOT/'configs/experiments').glob('*.yaml'):
  cfg=yaml.safe_load(p.read_text())
  if cfg.get('claim_status')=='preregistered_configuration_only': summary['blocked'][p.stem]={'reason':'Configuration only; no complete execution entrypoint for its entire arm matrix is shipped.','config':str(p.relative_to(ROOT)),'arms':cfg['arms']}
 summary['blocked']['proposed_selector_followup']={'reason':'docs/PBPF_SELECTOR_GATE.md proposes a harder grouped candidate bank and selector experiment; no such bank or complete fit/evaluation runner is supplied. The existing published bank audit was executed.','document':'docs/PBPF_SELECTOR_GATE.md'}
 (ROOT/'local/additional_completion.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k!='blocked'},indent=2),flush=True)
if __name__=='__main__': main()
