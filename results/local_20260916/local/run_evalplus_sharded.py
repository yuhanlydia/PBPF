"""Run the two documented screens, each over three disjoint task shards."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/data/cwj/PBPF'); SCRATCH=Path('/dev/shm/pbpf-additional'); OUTPUT=ROOT/'local/results'
PARTS=[(0,55),(55,55),(110,54)]

def main():
 env=os.environ.copy()
 for k in ['PYTHONPATH','PYTHONHOME']: env.pop(k,None)
 env.update(PATH=str(ROOT/'.venv/bin')+':'+env['PATH'],HF_HOME=str(ROOT/'local/model-cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',HUMANEVAL_OVERRIDE_PATH=str(ROOT/'local/data/evalplus/HumanEvalPlus-v0.1.10.jsonl'))
 os.umask(0o077); jobs=[]
 for index,(start,count) in enumerate(PARTS):
  for mode in ['frozen','repair']:
   script='scripts/run_frozen_7b.sh' if mode=='frozen' else 'scripts/run_repair.sh'
   config='configs/diagnostics/evalplus_frozen_7b_16gb.yaml' if mode=='frozen' else 'configs/diagnostics/evalplus_repair_7b_24gb.yaml'
   output=SCRATCH/f'evalplus_{mode}_164_shard{index}'
   assert not output.exists(),output
   logfile=ROOT/f'local/logs/evalplus_{mode}_164_shard{index}.log'; log=logfile.open('w')
   jobenv=env|{'CUDA_VISIBLE_DEVICES':str(index),'PBPF_NUM_TASKS':str(count),'PBPF_START_INDEX':str(start)}
   command=['bash','local/sandbox.sh','bash',script,config,str(output)]
   process=subprocess.Popen(command,cwd=ROOT,env=jobenv,stdout=log,stderr=subprocess.STDOUT)
   jobs.append({'process':process,'log':log,'mode':mode,'index':index,'start':start,'count':count,'output':output,'command':command,'logfile':str(logfile)})
 started=time.time()
 try:
  while any(job['process'].poll() is None for job in jobs):
   failed=[j for j in jobs if j['process'].poll() not in (None,0)]
   if failed: raise RuntimeError(f"Shard failed: {[(j['mode'],j['index'],j['process'].returncode) for j in failed]}")
   counts={f"{j['mode']}_{j['index']}":len(list(j['output'].glob('HumanEval-*/manifest.json'))) for j in jobs}
   print(json.dumps({'elapsed_seconds':round(time.time()-started),'completed_tasks':counts}),flush=True)
   (ROOT/'local/evalplus_sharded_status.json').write_text(json.dumps({'status':'running','elapsed_seconds':time.time()-started,'completed_tasks':counts},indent=2)+'\n')
   time.sleep(30)
  assert all(j['process'].returncode==0 for j in jobs)
  for mode in ['frozen','repair']:
   destination=OUTPUT/f'evalplus_{mode}_164'; destination.mkdir(exist_ok=False)
   allrows=[]; provenance=[]
   for j in [j for j in jobs if j['mode']==mode]:
    rows=json.loads((j['output']/'summary.json').read_text()); assert len(rows)==j['count']
    assert {r['task_id'] for r in rows}=={f'HumanEval/{i}' for i in range(j['start'],j['start']+j['count'])}
    for row in rows:
     name=row['task_id'].replace('/','-'); shutil.copytree(j['output']/name,destination/name); row['artifact']=str(destination/name); allrows.append(row)
    shutil.copy2(j['output']/'summary.json',destination/f'shard{j["index"]}_summary.json')
    provenance.append({'index':j['index'],'start_index':j['start'],'tasks':j['count'],'torch_seed':0,'gpu':j['index'],'command':j['command'],'log':j['logfile']})
   assert len(allrows)==164
   (destination/'summary.json').write_text(json.dumps(allrows,indent=2)+'\n')
   (destination/'shards.json').write_text(json.dumps(provenance,indent=2)+'\n')
  (ROOT/'local/evalplus_sharded_status.json').write_text(json.dumps({'status':'complete','tasks_per_config':164,'configs':2,'elapsed_seconds':time.time()-started},indent=2)+'\n')
  print('COMPLETED: 164 + 164 tasks, four repair rounds each',flush=True)
 finally:
  for j in jobs:
   if j['process'].poll() is None: j['process'].terminate()
   j['log'].close()

if __name__=='__main__': main()
