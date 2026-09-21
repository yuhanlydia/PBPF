#!/usr/bin/env python3
"""Supplementary primary-only three-seed diversity; never execute code."""
import argparse
import ast
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import runpy
import yaml
from pbpf.apbpf.codearc_bank import load_bank

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(1701,1702,1703)
OUTCOMES={'PASS','WRONG_OUTPUT','COMPILE_ERROR','RUNTIME_EXCEPTION','TIMEOUT'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def mean(values):return sum(values)/len(values) if values else None


def summarize(by_seed):
    require(set(by_seed)==set(SEEDS),'complete three-seed population required')
    indexed={}
    for seed,rows in by_seed.items():
        indexed[seed]={r['source_component_id']:r for r in rows}
        require(rows and len(indexed[seed])==len(rows),'empty or duplicate source population')
        require(all(r['split']=='primary' and len(r['outcomes'])==10 and set(r['outcomes'])<=OUTCOMES for r in rows),'primary-only ten-outcome rows required')
    sources=set(indexed[SEEDS[0]])
    require(all(set(rows)==sources for rows in indexed.values()),'cross-seed source population differs')
    per_source=[]
    for source in sorted(sources):
        rows=[indexed[seed][source] for seed in SEEDS]
        require(len({r['problem_id'] for r in rows})==1 and all(r['test_ids']==rows[0]['test_ids'] for r in rows),'task/test identities differ across seeds')
        code=[r['code'] for r in rows];trees=[]
        for text in code:
            require(isinstance(text,str),'candidate code must be text')
            try:trees.append(ast.dump(ast.parse(text),include_attributes=False))
            except (SyntaxError,ValueError,TypeError,RecursionError):trees.append(None)
        vectors=[tuple(r['outcomes']) for r in rows];correct=[all(x=='PASS' for x in v) for v in vectors]
        valid=[tree for tree in trees if tree is not None];correct_valid=[tree for tree,ok in zip(trees,correct) if ok and tree is not None]
        pairs={}
        for a,b in itertools.combinations(range(3),2):
            pairs[f'{SEEDS[a]}:{SEEDS[b]}']={'shared_task_failure':int(not correct[a] and not correct[b]),
                'task_failure_disagreement':int(correct[a]!=correct[b]),
                'per_test_failure_disagreement':mean([(x=='PASS')!=(y=='PASS') for x,y in zip(vectors[a],vectors[b])]),
                'outcome_category_disagreement':mean([x!=y for x,y in zip(vectors[a],vectors[b])])}
        per_source.append({'source_component_id':source,'problem_id':rows[0]['problem_id'],
            'candidates':3,'exact_code_unique_count':len(set(code)),'exact_code_unique_fraction':len(set(code))/3,
            'empty_candidates':sum(not text.strip() for text in code),'syntax_invalid_candidates':3-len(valid),
            'observed_compile_error_candidates':sum('COMPILE_ERROR' in vector for vector in vectors),
            'valid_ast_candidates':len(valid),'valid_ast_unique_count':len(set(valid)),
            'valid_ast_unique_fraction':len(set(valid))/len(valid) if valid else None,
            'outcome_vector_unique_count':len(set(vectors)),'outcome_vector_unique_fraction':len(set(vectors))/3,
            'correct_candidates':sum(correct),'correct_valid_ast_candidates':len(correct_valid),
            'correct_valid_ast_unique_count':len(set(correct_valid)),
            'correct_valid_ast_unique_fraction':len(set(correct_valid))/len(correct_valid) if correct_valid else None,
            'all_tests_success_by_seed':{str(s):bool(ok) for s,ok in zip(SEEDS,correct)},
            'three_seed_union_all_tests_success':int(any(correct)),'mean_single_seed_all_tests_success':sum(correct)/3,
            'pairs':pairs})
    summary={key:mean([r[key] for r in per_source if r[key] is not None]) for key in ('exact_code_unique_fraction','valid_ast_unique_fraction','outcome_vector_unique_fraction','correct_valid_ast_unique_fraction','three_seed_union_all_tests_success','mean_single_seed_all_tests_success')}
    summary['union_minus_mean_single_seed']=mean([r['three_seed_union_all_tests_success']-r['mean_single_seed_all_tests_success'] for r in per_source])
    summary['per_seed_all_tests_success']={str(s):mean([r['all_tests_success_by_seed'][str(s)] for r in per_source]) for s in SEEDS}
    summary['pairwise']={pair:{metric:mean([r['pairs'][pair][metric] for r in per_source]) for metric in next(iter(per_source))['pairs'][pair]} for pair in per_source[0]['pairs']}
    counts={'sources':len(per_source),'candidates':3*len(per_source)}
    for key in ('empty_candidates','syntax_invalid_candidates','observed_compile_error_candidates','valid_ast_candidates','correct_candidates','correct_valid_ast_candidates'):counts[key]=sum(r[key] for r in per_source)
    summary['pairwise_counts']={pair:{'sources':len(per_source),'shared_task_failures':sum(r['pairs'][pair]['shared_task_failure'] for r in per_source),'task_failure_disagreements':sum(r['pairs'][pair]['task_failure_disagreement'] for r in per_source),'paired_tests':10*len(per_source)} for pair in per_source[0]['pairs']}
    counts['sources_with_valid_ast']=sum(r['valid_ast_candidates']>0 for r in per_source)
    counts['sources_with_correct_candidate']=sum(r['correct_candidates']>0 for r in per_source)
    counts['sources_with_correct_valid_ast']=sum(r['correct_valid_ast_candidates']>0 for r in per_source)
    summary['empty_candidate_rate']=counts['empty_candidates']/counts['candidates'];summary['syntax_invalid_candidate_rate']=counts['syntax_invalid_candidates']/counts['candidates']
    distributions={key:dict(sorted(Counter(r[key] for r in per_source).items())) for key in ('exact_code_unique_count','valid_ast_candidates','valid_ast_unique_count','outcome_vector_unique_count','correct_candidates','correct_valid_ast_unique_count')}
    return {'summary':summary,'counts':counts,'distributions':distributions,'sources':per_source,
        'denominators':'Source means; valid-AST ratios divide by valid candidates within source and omit only zero-denominator sources (counts shown); correct AST analogously conditions on correct AND AST-valid candidates. Empty text parses as a valid empty Module and is counted separately.',
        'interpretation':'Three fixed seed oracle coverage, not deployable selection or independently sampled Pass@3. AST uniqueness is syntax, not semantic/algorithm diversity. No comparative significance or CI claimed.'}


def inspect_seed(*,cell,seed,results_root,direct_cell,manifest_sha256):
    verifier=runpy.run_path(str(ROOT/'scripts/run_eesd_matrix.py'))
    banks=[];rows_by_split={};missing=[]
    for split,count in [('development',200),('primary',500)]:
        path=Path(results_root)/'mechanism-banks'/cell['dataset']/cell['model']/f'seed{seed}'/split
        if not (path/'complete.json').exists():missing.append(f'{split} bank incomplete');continue
        verified=verifier['verify_mechanism_bank'](path,cell=cell,seed=seed,split=split,root=ROOT)
        require(verified[1]['components']==count,'bank source count mismatch')
        _,rows,_=load_bank(path);rows_by_split[split]=rows;banks.append(verified)
    directory=Path(direct_cell) if direct_cell is not None else None
    entry={'status':'missing','reasons':missing,'direct_cell':str(directory) if directory else None}
    if directory is None or not (directory/'complete.json').exists():missing.append('direct cell incomplete');return entry,None
    complete=read(directory/'complete.json')
    require(complete.get('schema')=='eesd-direct-mechanism-complete-v1' and complete.get('execution_profile')=='direct-no-sandbox','wrong direct completion profile')
    expected_files={'cache.json','direct-binding.json','report/report.json','report/predictions.npz','report/complete.json'}
    require(set(complete['files'])==expected_files,'direct cell completion inventory differs')
    require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}==expected_files|{'complete.json'},'extra/missing direct artifacts')
    for name,digest in complete['files'].items():require(sha(directory/name)==digest,'direct artifact checksum mismatch')
    require(not missing,'sealed direct cell depends on missing banks')
    binding=read(directory/'direct-binding.json')
    require(binding.get('schema')=='eesd-direct-mechanism-binding-v1' and binding.get('execution_profile')=='direct-no-sandbox'
        and all(binding.get(k)==v for k,v in [('domain',cell['domain']),('dataset',cell['dataset']),('family',cell['family']),('seed',seed)]),'direct binding identity mismatch')
    lock_ref=binding['execution_lock'];require(sha(lock_ref['path'])==lock_ref['sha256'],'direct execution lock changed')
    lock=read(lock_ref['path'])
    require(lock.get('schema')=='eesd-direct-execution-lock-v1' and lock.get('profile')=='direct-no-sandbox' and lock['manifest']['sha256']==manifest_sha256,'direct execution lock manifest/profile mismatch')
    require(binding['sources']==lock['sources'],'direct source binding mismatch')
    for name,digest in binding['sources'].items():require(sha(ROOT/name)==digest,'direct source changed')
    for path,digest in binding['inputs'].items():require(sha(path)==digest,'direct input changed')
    for field in ('manifest','config'):
        require(sha(lock[field]['path'])==lock[field]['sha256'] and binding['inputs'].get(str(Path(lock[field]['path']).resolve()))==lock[field]['sha256'],'locked input binding missing/changed')
    for field in ('public_root','evaluator_root'):
        for name in ('manifest.json','tasks.jsonl'):
            path=(Path(cell[field])/name).resolve()
            require(binding['inputs'].get(str(path))==sha(path),'materialization binding missing/changed')
    ready=binding['readiness']
    require(ready.get('schema')=='eesd-direct-execution-readiness-v1' and ready.get('status')=='ready' and ready.get('profile')=='direct-no-sandbox'
        and ready['execution_lock_sha256']==lock_ref['sha256'] and ready['sources']==lock['sources'],'direct readiness mismatch')
    require(set(ready['probes'])=={'stdin','call'} and all(r.get('outcome')=='PASS' and r.get('returncode')==0 and r.get('timed_out') is False for r in ready['probes'].values()),'direct probe evidence incomplete')
    expected_banks=[dict(path=str(path.resolve()),split=info['split'],components=info['components'],complete_sha256=digest) for path,info,digest in banks]
    require(binding['banks']==expected_banks,'direct banks differ from verified banks')
    require(binding['cache_sha256']==sha(directory/'cache.json') and binding['report_complete_sha256']==sha(directory/'report/complete.json'),'direct output binding mismatch')
    verifier['verify_mechanism_report'](directory/'report',cache=directory/'cache.json',config=Path(lock['config']['path']),cell=cell,seed=seed,root=ROOT)
    payload=read(directory/'cache.json');info=banks[0][1]
    require(payload.get('generator_identity')==[info['model'],info['revision'],info.get('adapter_sha256'),seed]
        and payload.get('tests_per_candidate')==10 and payload.get('counts')=={'development':200,'primary':500}
        and payload.get('source_counts')=={'development':200,'primary':500},'cache generation/count identity mismatch')
    require(payload.get('bank_bindings')==expected_banks and payload.get('evaluator_manifest_sha256')==sha(Path(cell['evaluator_root'])/'manifest.json'),'cache bank/evaluator binding mismatch')
    require(payload.get('schema')=='eesd-public-query-mechanism-cache-v1' and payload.get('dataset')==cell['domain'],'cache protocol mismatch')
    bankrows=[r for split in ('development','primary') for r in rows_by_split[split]];records=payload['records']
    require(len(records)==len(bankrows)==700,'direct cache population mismatch')
    tasks_path=Path(cell['evaluator_root'])/'tasks.jsonl';tasks_list=[json.loads(x) for x in tasks_path.read_text().splitlines()];tasks={r['task_id']:r for r in tasks_list}
    require(len(tasks)==len(tasks_list),'duplicate evaluator identities')
    selected=[]
    for row,bank in zip(records,bankrows,strict=True):
        candidate=bank['candidates'][0];task=tasks[bank['task_id']]
        require(task['source_component_id']==bank['source_component_id'] and task['split']==bank['split'],'evaluator source/split mismatch')
        expected={'task_id':candidate['candidate_id'],'problem_id':bank['task_id'],'source_component_id':bank['source_component_id'],'split':bank['split'],
            'candidate_code_sha256':hashlib.sha256(candidate['code'].encode()).hexdigest(),'tests':[{'id':str(t['id']),'input':t['input']} for t in task['tests']]}
        require(set(row)==set(expected)|{'outcomes'} and all(row[k]==v for k,v in expected.items()),'cache/bank/input identity mismatch')
        require(len(row['outcomes'])==10 and set(row['outcomes'])<=OUTCOMES and [t['id'] for t in row['tests']]==list(map(str,range(10))),'ordered outcomes invalid')
        if row['split']=='primary':selected.append({'source_component_id':row['source_component_id'],'problem_id':row['problem_id'],'split':'primary','code':candidate['code'],'outcomes':row['outcomes'],'test_ids':[t['id'] for t in row['tests']]})
    require(len(selected)==500 and len({r['source_component_id'] for r in selected})==500,'primary source count differs')
    entry.update(status='verified',complete_sha256=sha(directory/'complete.json'),banks=expected_banks)
    return entry,selected


def run_report(*,manifest,manifest_sha256,results_root,domain,family,direct_cells,output):
    require(sha(manifest)==manifest_sha256,'manifest checksum mismatch')
    value=yaml.safe_load(Path(manifest).read_text());require(value.get('schema')=='eesd-cache-manifest-v1','wrong original-domain manifest')
    cells=[c for c in value['mechanism_cells'] if c['domain']==domain and c['family']==family]
    require(len(cells)==1 and domain in ('rbr','codearc'),'unique original-domain cell required')
    require(set(direct_cells)<=set(SEEDS),'unexpected seed')
    coverage={};data={}
    for seed in SEEDS:
        entry,rows=inspect_seed(cell=cells[0],seed=seed,results_root=results_root,direct_cell=direct_cells.get(seed),manifest_sha256=manifest_sha256)
        coverage[str(seed)]=entry
        if rows is not None:data[seed]=rows
    complete=len(data)==3
    report={'schema':'eesd-supplementary-diversity-v1','status':'supplementary_exploratory_complete' if complete else 'incomplete',
        'domain':domain,'family':family,'split':'primary','execution_profile':'direct-no-sandbox','coverage':coverage,
        'scientific_summary':summarize(data) if complete else None,'manifest_sha256':manifest_sha256,
        'analysis_source_sha256':sha(__file__),'plan_sha256':sha(ROOT/'docs/EESD_SUPPLEMENTARY_EVALUATION_20260921.md'),
        'scope':'Added after initial direct-execution results; supplementary/exploratory. No changes to primary metrics, generation budget, training or selection.'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','results-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--domain',choices=['rbr','codearc'],required=True);p.add_argument('--family',required=True)
    p.add_argument('--direct-cell',action='append',default=[],metavar='SEED=PATH')
    a=p.parse_args();cells={}
    for item in a.direct_cell:
        seed,path=item.split('=',1);seed=int(seed);require(seed not in cells,'duplicate direct seed argument');cells[seed]=Path(path)
    result=run_report(manifest=a.manifest,manifest_sha256=a.manifest_sha256,results_root=a.results_root,domain=a.domain,family=a.family,direct_cells=cells,output=a.output)
    print(json.dumps({'status':result['status'],'output':str(a.output)}))

if __name__=='__main__':main()
