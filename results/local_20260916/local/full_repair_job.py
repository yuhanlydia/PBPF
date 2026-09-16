"""Run and validate the complete 1500-step, eight-candidate local repair diagnostic."""
from pathlib import Path
import json,os,subprocess,time,traceback
root=Path(__file__).resolve().parents[1];status=root/'local/repair_status.json'
def update(**fields):
 row=json.loads(status.read_text());row.update(fields,updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
 temp=status.with_suffix('.tmp');temp.write_text(json.dumps(row,indent=2)+'\n');temp.replace(status)
try:
 update(status='full_repair_running',full_1500_step_pilot_completed=False,full_repair_gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','1')))
 with (root/'local/logs/full_repair_supervisor.log').open('w') as log:
  subprocess.run(['bash','local/run_repair.sh'],cwd=root,stdout=log,stderr=subprocess.STDOUT,env=os.environ,check=True,timeout=14400)
 result=json.loads((root/'local/results/rbr_repair_gate.json').read_text())
 expected={'no_latent','random_latent','heuristic','posterior_mean','posterior_map','pbpf'}
 assert set(result['arms'])==expected
 assert len(result['tasks'])==48 and all(v['tasks']==8 for v in result['summary'].values())
 ids={row['task_id'] for row in result['tasks'] if row['arm']=='no_latent'}
 assert len(ids)==8
 for arm in expected:
  assert {r['task_id'] for r in result['tasks'] if r['arm']==arm}==ids
  assert all(len(r['outcomes'])==10 for r in result['tasks'] if r['arm']==arm)
 import torch
 ckpt=torch.load(root/'local/results/rbr_soft_prefix.pt',map_location='cpu',weights_only=False)
 assert ckpt['step']==1500 and 0<ckpt['best_step']<=1500
 update(status='local_repair_complete',full_1500_step_pilot_completed=True,projector_steps=ckpt['step'],selected_step=ckpt['best_step'],training_items=ckpt['training_items'],summary=result['summary'],scope='Current-code local pilot; upstream dataset/cache differs and formal S0-S4 remains unprovisioned.')
 print(json.dumps({'complete':True,'steps':ckpt['step'],'selected_step':ckpt['best_step'],'summary':result['summary']},indent=2),flush=True)
except Exception as exc:
 update(status='full_repair_failed',full_1500_step_pilot_completed=False,error=f'{type(exc).__name__}: {exc}')
 traceback.print_exc();raise
