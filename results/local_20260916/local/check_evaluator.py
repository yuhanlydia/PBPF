"""Check the repair evaluator on the official fixed programs for the same eight tasks."""
from pathlib import Path
import gzip,hashlib,json,runpy
root=Path.cwd();api=runpy.run_path('scripts/run_rbr_repair_gate.py',run_name='pbpf_repair_helpers')
payload=json.loads(Path('local/results/rbr_real_gate_dataset.json').read_text())
rows=sorted([r for r in payload['records'] if r['split']=='test'],key=lambda r:hashlib.sha256(f"2701\0{r['task_id']}".encode()).digest())[:8]
wanted={r['task_id'] for r in rows};fixed={}
with gzip.open('local/data/runbugrun-v0.0.1/python_test0.jsonl.gz','rt') as stream:
 for line in stream:
  r=json.loads(line)
  if str(r['id']) in wanted: fixed[str(r['id'])]=r['fixed_code']
assert set(fixed)==wanted
cases=api['_evaluator_cases'](Path('local/data/runbugrun-v0.0.1'),rows)
results=[{'task_id':r['task_id'],'outcomes':api['_execute'](fixed[r['task_id']],c)} for r,c in zip(rows,cases)]
assert all(all(x=='PASS' for x in r['outcomes']) for r in results),results
report={'purpose':'Infrastructure positive control using official fixed programs; not actor evidence','tasks':8,'cases':80,'passed':80,'results':results}
Path('local/results/evaluator_positive_control.json').write_text(json.dumps(report,indent=2)+'\n')
print('Official fixed-code positive control: 80/80 PASS')
