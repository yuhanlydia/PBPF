import datetime,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path('/data/cwj/PBPF'); LOCAL=ROOT/'local'

def digest(p):
 with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

assert json.loads((LOCAL/'evalplus_sharded_status.json').read_text())['status']=='complete'
assert '674 passed' in (LOCAL/'logs/pytest_after_generation_cap.log').read_text()
subprocess.run([sys.executable,'local/summarize_additional.py'],cwd=ROOT,check=True)
a=json.loads((LOCAL/'additional_completion.json').read_text()); assert not a['pending']
subprocess.run([sys.executable,'local/write_execution_report.py'],cwd=ROOT,check=True)
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
(LOCAL/'compatibility.patch').write_bytes(subprocess.check_output(['git','diff'],cwd=ROOT))
completion=json.loads((LOCAL/'completion.json').read_text())
completion.update(updated_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),runnable_documented_experiments_completed=True,all_documented_experiments_completed=False,formal_s0_s4_completed=False,evalplus_configs_completed=2,evalplus_tasks_completed=328,evalplus_rounds_per_task=4,selector_audit_completed=True,finite_contract_cases_completed=60000,additional_report='local/EXPERIMENTS_EXECUTED.md',additional_inventory='local/additional_completion.json',formal_blocker='Complete production scientific factory and operator deployment artifacts are missing; exact formal launcher exits 2 before running.')
(LOCAL/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
model=json.loads((LOCAL/'model_manifest.json').read_text());model['cache']=str((LOCAL/'model-cache').resolve());model['alias']=str(LOCAL/'model-cache');(LOCAL/'model_manifest.json').write_text(json.dumps(model,indent=2)+'\n')
selected=[]
for name in ['REPORT.md','README.md','EXPERIMENTS_EXECUTED.md','environment.json','requirements.lock.txt','model_manifest.json','compatibility.patch','completion.json','additional_completion.json','evalplus_sharded_status.json']:
 selected.append(LOCAL/name)
selected+=list((LOCAL/'results').glob('*.json'))+list((LOCAL/'results').glob('*.pt'))
for name in ['evalplus_frozen_164','evalplus_repair_164','finite_contract','legacy_exact_smoke','legacy_rbr_6581','iclr_verified_after_generation_cap']:
 selected += [p for p in (LOCAL/'results'/name).rglob('*') if p.is_file()]
selected+=list(LOCAL.glob('*.py'))+list(LOCAL.glob('*.sh'))
selected += [p for p in (LOCAL/'logs').glob('*.log') if p.name != 'finalize_all_runs.log']
selected.append(LOCAL/'logs/interrupted_evalplus_artifacts.tar.gz')
selected+=list((LOCAL/'data/goav_selector').glob('*'))
selected.append(LOCAL/'data/evalplus/HumanEvalPlus-v0.1.10.jsonl')
manifest={'files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(set(selected))},'created_at':completion['updated_at']}
(LOCAL/'artifacts.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'all_available_experiments_completed','tests_passed':674,'evalplus_tasks':328,'artifact_files':len(manifest['files']),'formal_matrix':'blocked before execution'},indent=2),flush=True)
