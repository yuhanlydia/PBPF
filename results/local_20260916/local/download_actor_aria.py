"""Download verified weights with resumable HTTP ranges when Xet transfer fails."""
from pathlib import Path
import os, subprocess, hashlib
root=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(root/'model-cache')
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download
model='Qwen/Qwen2.5-Coder-7B-Instruct'; rev='c03e6d358207e414f1eca0bb1891e29f1db0e242'
base=root/'model-cache/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct'
blobs=base/'blobs'; snap=base/'snapshots'/rev
blobs.mkdir(parents=True,exist_ok=True); snap.mkdir(parents=True,exist_ok=True)
manifest=[]; rows=[]
for i in range(1,5):
 name=f'model-{i:05d}-of-00004.safetensors'; url=hf_hub_url(model,name,revision=rev)
 m=get_hf_file_metadata(url); target=blobs/m.etag
 print(name,m.size,m.etag,flush=True)
 if target.exists() and target.stat().st_size==m.size:
  with target.open('rb') as f: ok=hashlib.file_digest(f,'sha256').hexdigest()==m.etag
  if ok: continue
 temp=blobs/(m.etag+'.http')
 rows.append((temp,target,snap/name))
 manifest.append(f'{url}\n  dir={blobs}\n  out={temp.name}\n  checksum=sha-256={m.etag}\n')
if manifest:
 config=root/'actor-download.urls';config.write_text(''.join(manifest))
 subprocess.run(['aria2c','--input-file='+str(config),'--continue=true','--max-concurrent-downloads=3','--split=16','--max-connection-per-server=16','--min-split-size=8M','--file-allocation=none','--max-tries=5','--retry-wait=3','--connect-timeout=20','--timeout=60','--summary-interval=30','--console-log-level=warn','--download-result=full'],check=True)
 for temp,target,link in rows:
  with temp.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
  assert actual==target.name,(temp,actual)
  temp.replace(target)
  if not link.exists(): link.symlink_to(Path('../../blobs')/target.name)
print(snapshot_download(model,revision=rev,allow_patterns=['*.json','*.safetensors','*.txt','LICENSE','README.md'],max_workers=4),flush=True)
