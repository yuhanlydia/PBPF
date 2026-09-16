import hashlib,json,urllib.request
from pathlib import Path
root=Path('local/data/goav_selector'); root.mkdir(parents=True,exist_ok=True)
tree=json.loads(Path('local/goav-tree.json').read_text()); revision=tree['sha']
manifest={'repository':'https://github.com/yuhanlydia/GOAV','revision':revision,'files':{}}
for name in ['bank.jsonl','bank.npz','sidecar.json']:
 path=f'results/evalplus_full164_v5/bank/{name}'
 url=f'https://raw.githubusercontent.com/yuhanlydia/GOAV/{revision}/{path}'
 payload=urllib.request.urlopen(url,timeout=90).read(); (root/name).write_bytes(payload)
 manifest['files'][name]={'sha256':hashlib.sha256(payload).hexdigest(),'url':url}
(root/'source.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest),flush=True)
