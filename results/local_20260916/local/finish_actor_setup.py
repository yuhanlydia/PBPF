"""Finish the pending actor setup and record its actual status without requiring an open terminal."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time,traceback
root=Path('/data/cwj/PBPF'); local=root/'local'
status=local/'repair_status.json'
base=local/'hf/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct'
rev='c03e6d358207e414f1eca0bb1891e29f1db0e242'
def record(stage,**extra):
 row={'status':stage,'updated_at':time.strftime('%Y-%m-%d %H:%M:%S %z'),'repair_smoke_completed':False,'full_1500_step_pilot_completed':False,**extra}
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(status)
try:
 record('model_download_in_progress',note='Downloading into tmpfs to avoid HDD journal stalls; will verify SHA256, persist weights, then run a 2-step/1-task integration check.')
 deadline=time.monotonic()+3600
 files=[]
 for line in (local/'actor-download.urls').read_text().splitlines():
  if line.startswith('  out='): files.append(Path('/dev/shm/pbpf-actor-download')/line.split('=',1)[1])
 while not all(p.exists() and not Path(str(p)+'.aria2').exists() for p in files):
  if time.monotonic()>deadline: raise TimeoutError('Actor download did not complete within one hour; resume the saved HTTP transfer.')
  time.sleep(10)
 record('verifying_and_saving_model')
 for p in files:
  digest=p.name.removesuffix('.http')
  with p.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
  if actual!=digest: raise ValueError(f'Checksum mismatch for {p.name}')
  target=base/'blobs'/digest
  temp=target.with_suffix('.copying')
  shutil.copyfile(p,temp)
  with temp.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
  if actual!=digest: raise ValueError(f'Persistent copy checksum mismatch for {p.name}')
  temp.replace(target)
 os.environ['HF_HOME']=str(local/'hf')
 os.environ['HF_HUB_DISABLE_XET']='1'
 from huggingface_hub import hf_hub_url,get_hf_file_metadata,snapshot_download
 for i in range(1,5):
  name=f'model-{i:05d}-of-00004.safetensors'
  meta=get_hf_file_metadata(hf_hub_url('Qwen/Qwen2.5-Coder-7B-Instruct',name,revision=rev))
  link=base/'snapshots'/rev/name
  if not link.exists(): link.symlink_to(Path('../../blobs')/meta.etag)
 snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct',revision=rev,local_files_only=True)
 # The index gives an offline completeness check even if the Hub has cached a partial snapshot.
 index=json.loads((base/'snapshots'/rev/'model.safetensors.index.json').read_text())
 assert all((base/'snapshots'/rev/name).is_file() for name in set(index['weight_map'].values()))
 record('model_ready_selecting_gpu',model_download_completed=True)
 gpu=None
 for _ in range(60):
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
  for line in raw.splitlines():
   idx,free=map(int,line.split(','))
   if free>=21000: gpu=idx;break
  if gpu is not None: break
  time.sleep(10)
 if gpu is None: raise RuntimeError('Model ready; no GPU with 21 GB free was available for the integration check.')
 record('repair_smoke_running',model_download_completed=True,gpu=gpu)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
 with (local/'logs/repair_smoke_supervisor.log').open('w') as log:
  subprocess.run(['bash','local/run_repair_smoke.sh'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
 result=local/'results/rbr_repair_smoke.json'
 if not result.is_file(): raise RuntimeError('Repair command exited without a result file')
 record('complete',model_download_completed=True,repair_smoke_completed=True,gpu=gpu,result=str(result),note='Only the 2-step/1-task integration check is complete. The 1500-step/8-task paper pilot was not run.')
 with (local/'REPORT.md').open('a') as f:
  f.write('\n## 后续修复集成检查\n\n固定版本 Qwen 模型已下载并校验持久副本；2 步训练、1 个任务的修复集成检查已完成。结果见 `local/results/rbr_repair_smoke.json`。1500 步/8 任务实验未执行。\n')
 # Remove only redundant files created by this setup after verified copies exist.
 for p in files:
  target=base/'blobs'/p.name.removesuffix('.http')
  for extra in (target.with_suffix('.incomplete'),target.with_suffix('.http'),target.with_suffix('.http.aria2'),p):
   if extra.is_file(): extra.unlink()
except Exception as exc:
 record('failed',error=f'{type(exc).__name__}: {exc}',note='Read local/logs/finish_actor_setup.log and local/logs/repair_smoke_supervisor.log. Core prediction environment and results remain available.')
 traceback.print_exc()
 raise
