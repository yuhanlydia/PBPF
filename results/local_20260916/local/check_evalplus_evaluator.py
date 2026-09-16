import hashlib,json
from pathlib import Path
from pbpf.sandbox import LocalPythonSandbox
p=Path('local/data/evalplus/HumanEvalPlus-v0.1.10.jsonl')
rows=[json.loads(line) for line in p.read_text().splitlines()]
results=[]
for row in rows:
 harness=row['test']+f"\ncheck({row['entry_point']})\n"
 sandbox=LocalPythonSandbox({'humaneval-base':{'harness':harness}},timeout_seconds=3,python_executable='/usr/bin/python3')
 result=sandbox.execute(row['prompt']+row['canonical_solution'],'humaneval-base')
 results.append({'task_id':row['task_id'],'outcome':result.outcome,'infrastructure_failure':result.infrastructure_failure,'feedback':result.feedback if result.outcome!='PASS' else ''})
report={'tasks':len(results),'passes':sum(r['outcome']=='PASS' for r in results),'dataset_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'results':results}
Path('local/results/evalplus_evaluator_positive_control.json').write_text(json.dumps(report,indent=2)+'\n')
print(f"Positive control: {report['passes']}/{report['tasks']}")
assert report['passes']==164
