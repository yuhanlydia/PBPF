"""Reconstruct all 49 reserved contrasts; no inference or candidate execution.

A9 returns public-feature masks before cross-seed common-support selection.
Historical scalar point estimates are verified; historical bootstrap intervals
are not reproduced or reused as the amended inferential result.
"""
from __future__ import annotations
import json
from pathlib import Path
import runpy
import numpy as np
import yaml
from pbpf.eesd.mechanism_artifacts import load_mechanism_artifacts,read,require,close,file_sha

PRIMARY_CONTRAST='core/effective_params'
ALPHAS=(.01,.1,.25,.5,1.,2.,5.,10.)
STRENGTHS=(0.,1.,4.,16.)
BINS=((1.,2.),(2.,3.),(3.,4.))
BIN_IDS=('A9/[1,2)','A9/[2,3)','A9/[3,4]')


def contrast_ids():
    return ('A1/relevance',*(f'A4/alpha={a:g}/strength={s:g}' for s in STRENGTHS for a in ALPHAS),
        'A5/tuned_global','A5/same_kernel','A6/assessment_mean','A6/development_mean','A7/permuted',
        'A8/n1','A8/n2','A8/n4','A8/n8',*BIN_IDS,'A10/binary',
        'core/tuned','core/fixed_params','core/effective_params')


def select_same_kernel_mass(runner,examples,selection,masses,ece_bins):
    """Development examples only; fixed EED alpha/kernel, locked mass grid/tie rule."""
    labels,_=runner['labels_clusters'](examples)
    grid=[]
    for mass in masses:
        p,_=runner['predictions'](examples,selection['alpha'],'global',global_mass=mass)
        grid.append(dict(alpha=selection['alpha'],strength=selection['strength'],mass=float(mass),
            **runner['score_predictions'](labels,p,ece_bins=ece_bins)))
    return min(grid,key=lambda x:(x['nll'],x['strength'],x['alpha'],x['mass'])),grid


def reconstruct_contrasts(**loaderargs):
    data,verification=load_mechanism_artifacts(**loaderargs)
    root=Path(loaderargs.get('root') or Path(__file__).resolve().parents[3])
    runner=runpy.run_path(str(root/'scripts/run_eesd_evidence_matrix.py'))
    report=read(Path(loaderargs['report_dir'])/'report.json')
    cache=read(loaderargs['cache']);cfg=yaml.safe_load(Path(loaderargs['config']).read_text());e=cfg['evidence']
    require(tuple(e['alphas'])==ALPHAS and tuple(e['strengths'])==STRENGTHS,'49-slot reconstruction requires locked alpha/strength grids')
    require(e['visible_counts']==[1,2,4,8] and e['permutation_seeds']==[1701,1702,1703]
            and e['concentration_bins_for_n4']==[1,2,3,4],'history/permutation/concentration protocol mismatch')
    dev=[r for r in cache['records'] if r['split']=='development']
    assessment=[r for r in cache['records'] if r['split']=='primary']
    memo={};checked=[]
    def examples(split,n,strength,binary=False):
        key=(split,n,strength,binary)
        if key not in memo:memo[key]=runner['build_examples'](dev if split=='validation' else assessment,n,strength,binary=binary)
        return memo[key]
    def compare(saved,actual,name):
        if saved is None:return False
        for k,v in actual.items():
            require(k in saved,f'{name} missing saved field {k}')
            if v is None:require(saved[k] is None,f'{name}/{k} mismatch')
            else:close(saved[k],v,f'{name}/{k}')
        checked.append(name);return True
    def metrics(p,y):return runner['score_predictions'](y,p,ece_bins=e['ece_bins'])
    def arm(n,selection,rule,*,binary=False,mass=None,overrides=None):
        ex=examples('assessment',n,selection['strength'],binary)
        p,m=runner['predictions'](ex,selection['alpha'],rule,
            global_mass=mass if mass is not None else selection.get('mass'),mass_overrides=overrides)
        return p,m
    def make(proposed,comparators,*,n=4,binary=False,population='all',**extra):
        labels,sources=runner['labels_clusters'](examples('assessment',n,0.,binary))
        ids=[dict(domain=data['domain'],source_component_id=r['source_component_id'],problem_id=r['problem_id'],
            candidate_id=r['task_id'],test_id=t['id'],test_index=i) for r in assessment
            for i,t in enumerate(r['tests']) if i>=n]
        return dict(status='available',labels=labels,proposed=proposed,comparators=comparators,
            sources=sources.tolist(),seeds=[int(data['seed'])]*len(labels),
            query_ids=[json.dumps(q,sort_keys=True,separators=(',',':')) for q in ids],
            population=population,target_history=n,**extra)
    result={key:dict(status='unsupported',reason='corresponding sealed report entry unavailable') for key in contrast_ids()}
    saved_metrics=report.get('metrics',{});p=data['probabilities'];y=data['labels'];sel=report['selected']
    valid={name:compare(saved_metrics.get(name),metrics(prob,y),'metrics/'+name) for name,prob in p.items()}
    for key,proposed,comparator in [('A1/relevance','fixed','ordinary'),('A5/tuned_global','effective','global_mass'),
            ('core/tuned','effective','fixed'),('core/fixed_params','effective_at_fixed_params','fixed'),
            ('core/effective_params','effective','fixed_at_effective_params')]:
        if valid[proposed] and valid[comparator]:result[key]=make(p[proposed],{comparator:p[comparator]})
    if report.get('argmax') is not None:
        compare(report['argmax'],dict(fixed_vs_effective_tuned_agreement=float(np.mean(p['fixed'].argmax(1)==p['effective'].argmax(1))),
            same_effective_params_agreement=float(np.mean(p['fixed_at_effective_params'].argmax(1)==p['effective'].argmax(1)))),'argmax')
    for arm_name,rule in [('ordinary','ordinary'),('fixed','fixed'),('effective','effective'),('global_mass','global')]:
        saved_grid=report.get('selection_grids',{}).get(arm_name)
        if saved_grid is not None:
            _,grid=runner['select_arm'](examples,'validation',4,e['alphas'],[0.] if arm_name=='ordinary' else e['strengths'],
                rule,masses=e['global_mass_grid'] if arm_name=='global_mass' else None,ece_bins=e['ece_bins'])
            require(len(saved_grid)==len(grid),'saved development selection grid size mismatch')
            for i,(saved_row,reconstructed) in enumerate(zip(saved_grid,grid,strict=True)):
                compare(saved_row,reconstructed,f'selection_grid/{arm_name}/{i}')
    factorial=report.get('factorial',[])
    require(len({(r['alpha'],r['strength']) for r in factorial})==len(factorial),'duplicate factorial entry')
    for strength in STRENGTHS:
        for alpha in ALPHAS:
            key=f'A4/alpha={alpha:g}/strength={strength:g}'
            pair=[r for r in factorial if r['alpha']==alpha and r['strength']==strength]
            if not pair:continue
            selection=dict(alpha=alpha,strength=strength)
            fp,_=arm(4,selection,'fixed');ep,_=arm(4,selection,'effective')
            fm,em=metrics(fp,y),metrics(ep,y)
            compare(pair[0],dict(alpha=alpha,strength=strength,fixed_nll=fm['nll'],effective_nll=em['nll'],
                nll_gain=fm['nll']-em['nll'],fixed_brier=fm['brier'],effective_brier=em['brier'],
                accuracy_agreement=float(np.mean(fp.argmax(1)==ep.argmax(1)))),'factorial/'+key)
            aliases=[]
            if alpha in (.1,.01):
                aliases=[f'A{2 if alpha==.1 else 3}/strength={strength:g}']
                duplicates=[r for r in report.get('same_alpha',{}).get(f'{alpha:.2f}',[]) if r['strength']==strength]
                if duplicates:require(duplicates==pair,'same-alpha alias differs from factorial')
            result[key]=make(ep,{'fixed':fp},aliases=aliases)
    same_sel,same_grid=select_same_kernel_mass(runner,examples('validation',4,sel['effective']['strength']),
        sel['effective'],e['global_mass_grid'],e['ece_bins'])
    global_p,_=arm(4,same_sel,'global')
    result['A5/same_kernel']=make(p['effective'],{'same_kernel_global':global_p},
        selection=same_sel,selection_population='development-only',selection_grid=same_grid,
        report_provenance='prospectively-adopted-additional-control-not-in-original-report')
    effective_examples=examples('assessment',4,sel['effective']['strength'])
    masses=np.asarray([x['mass'] for x in effective_examples])
    for key,split,metric_name,selection_name in [
            ('A6/assessment_mean','assessment','constant_mean_mass','constant_mean_effective_mass'),
            ('A6/development_mean','validation','development_mean_mass','development_mean_effective_mass')]:
        mean=float(np.mean([x['mass'] for x in examples(split,4,sel['effective']['strength'])]))
        if selection_name not in sel:continue
        close(sel[selection_name],mean,selection_name)
        control,_=arm(4,sel['effective'],'global',mass=mean)
        if compare(saved_metrics.get(metric_name),metrics(control,y),'metrics/'+metric_name):
            result[key]=make(p['effective'],{metric_name:control},public_mean_mass=mean)
    permutations={};runs=saved_metrics.get('permuted_mass',{}).get('runs',[]);perm_nll=[]
    for permutation_seed in e['permutation_seeds']:
        perm=np.random.default_rng(permutation_seed).permutation(masses)
        control,_=arm(4,sel['effective'],'global',overrides=perm)
        observed=[r for r in runs if r.get('seed')==permutation_seed]
        require(len(observed)<=1,'duplicate saved permutation seed')
        if not observed:continue
        actual=metrics(control,y);compare(observed[0],actual,f'permutation/{permutation_seed}')
        permutations[f'permutation/{permutation_seed}']=control;perm_nll.append(actual['nll'])
    if len(permutations)==3:
        compare(saved_metrics['permuted_mass'],dict(mean_nll=float(np.mean(perm_nll)),std_nll=float(np.std(perm_nll,ddof=1))),'permutation-summary')
        result['A7/permuted']=make(p['effective'],permutations,comparator_aggregation='mean-metrics-not-probabilities')
    for n in (1,2,4,8):
        key=f'A8/n{n}';saved=[x for x in report.get('history_sweep',[]) if x['history_size']==n]
        require(len(saved)<=1,'duplicate saved history')
        if not saved:continue
        if saved[0].get('status')=='unavailable':
            result[key]=dict(status='unsupported',reason=saved[0].get('reason','saved history unavailable'));continue
        if any(saved[0].get(rule+suffix) is None for rule in ('fixed','effective') for suffix in ('_selected','_metrics')):
            result[key]=dict(status='unsupported',reason='saved history selection/metrics incomplete');continue
        tuned={};pred={}
        for rule in ('fixed','effective'):
            tuned[rule],_=runner['select_arm'](examples,'validation',n,e['alphas'],e['strengths'],rule,ece_bins=e['ece_bins'])
            compare(saved[0].get(rule+'_selected'),tuned[rule],f'history/{n}/{rule}/selected')
            pred[rule],_=arm(n,tuned[rule],rule)
            labels,_=runner['labels_clusters'](examples('assessment',n,0.))
            compare(saved[0].get(rule+'_metrics'),metrics(pred[rule],labels),f'history/{n}/{rule}/metrics')
        result[key]=make(pred['effective'],{'fixed':pred['fixed']},n=n,
            target_set_protocol='post-generation history; absolute future targets change with n')
        if n==4:
            close(pred['effective'],p['effective'],'history4 effective alias');close(pred['fixed'],p['fixed'],'history4 fixed alias')
            result[key]['alias_of']='core/tuned'
    # Match the frozen runner's summed prediction weights exactly at bin edges.
    # A6/A7 above deliberately retain effective_mass(raw) from the examples.
    _,bin_masses=arm(4,sel['effective'],'effective')
    for key,(lo,hi) in zip(BIN_IDS,BINS,strict=True):
        mask=(bin_masses>=lo)&((bin_masses<=hi) if hi==4 else (bin_masses<hi))
        matches=[r for r in report.get('concentration_bins',[]) if r['lo']==lo and r['hi']==hi]
        require(len(matches)<=1,'duplicate concentration bin')
        if not matches:continue
        fp=p['fixed_at_effective_params'];ep=p['effective']
        fl=-np.log(fp[np.arange(len(y)),y].clip(1e-12,1));el=-np.log(ep[np.arange(len(y)),y].clip(1e-12,1))
        compare(matches[0],dict(lo=lo,hi=hi,count=int(mask.sum()),mean_mass=float(bin_masses[mask].mean()) if mask.any() else None,
            fixed_nll=float(fl[mask].mean()) if mask.any() else None,effective_nll=float(el[mask].mean()) if mask.any() else None,
            nll_gain=float((fl[mask]-el[mask]).mean()) if mask.any() else None),'concentration/'+key)
        result[key]=make(ep,{'fixed':fp},population='pending-cross-seed-common-source-support',
            public_mask=mask,mask=mask,public_mass=bin_masses.copy(),bin_edges=[lo,hi],upper_inclusive=hi==4)
    bp,_=arm(4,sel['effective'],'effective',binary=True);bf,_=arm(4,sel['effective'],'fixed',binary=True)
    by,_=runner['labels_clusters'](examples('assessment',4,0.,True))
    if (compare(saved_metrics.get('binary_effective'),metrics(bp,by),'metrics/binary_effective')
            and compare(saved_metrics.get('binary_fixed_at_effective_params'),metrics(bf,by),'metrics/binary_fixed_at_effective_params')):
        result['A10/binary']=make(bp,{'binary_fixed':bf},binary=True,class_space=['PASS','NONPASS'])
    for name,path in [('cache',loaderargs['cache']),('config',loaderargs['config']),
                      ('report',Path(loaderargs['report_dir'])/'report.json')]:
        require(file_sha(path)==verification[name+'_sha256'],'artifact changed during reconstruction')
    receipt=dict(schema='eesd-mechanism-contrasts-v1',reserved_slots=49,contrast_ids=list(contrast_ids()),
        verification=verification,checked_saved_point_estimates=checked,
        population_selection_validation='caller-must-verify-locked-public-inventory',
        historical_bootstraps='not reverified or reused; amended inference computed separately',
        source_sha256=file_sha(Path(__file__)),a9_population='awaits cross-seed public-only support sealing',
        unavailable={k:v['reason'] for k,v in result.items() if v['status']!='available'})
    return result,receipt


def derive_public_support(**loaderargs):
    """Derive A9 masks before any assessment outcome field access.

    Trust boundary: the caller must first validate formal public bank inventories
    and existing cache creation seals. This is a public support projection, not a
    replacement for ``load_mechanism_artifacts``. JSON decoding and byte hashing
    are allowed; assessment outcome/metric fields and NPZ arrays are never used.
    Only development outcome fields enter hyperparameter validation. The caller
    must seal common source support before invoking full contrast reconstruction.
    """
    from pbpf.eesd.mechanism_artifacts import (
        FROZEN_RUNNER_SHA256,FROZEN_EVIDENCE_SHA256,CACHE_SOURCES,digest_json)
    root=Path(loaderargs.get('root') or Path(__file__).resolve().parents[3])
    cache_path=Path(loaderargs['cache']);config_path=Path(loaderargs['config'])
    report_dir=Path(loaderargs['report_dir']);seed=loaderargs['seed'];domain=loaderargs['domain']
    require(type(seed) is int and domain in {'rbr','codearc'},'public support domain/seed mismatch')
    runner_path=root/'scripts/run_eesd_evidence_matrix.py'
    require(file_sha(runner_path)==FROZEN_RUNNER_SHA256 and
        file_sha(root/'src/pbpf/eesd/evidence.py')==FROZEN_EVIDENCE_SHA256,'public support frozen source mismatch')
    complete=read(report_dir/'complete.json')
    require(complete.get('report_sha256')==file_sha(report_dir/'report.json') and
        complete.get('predictions_sha256')==file_sha(report_dir/'predictions.npz'),'public support completion checksum mismatch')
    report=read(report_dir/'report.json');cache=read(cache_path);seal=read(cache_path.with_suffix('.binding.json'))
    cfg=yaml.safe_load(config_path.read_text());e=cfg['evidence']
    expected=dict(schema='eesd-evidence-matrix-v1',dataset={'rbr':'runbugrun','codearc':'codearc'}[domain],
        model=loaderargs['model'],seed=seed,visible=4,validation_split='development',assessment_split='primary',
        cache_sha256=file_sha(cache_path),config_sha256=file_sha(config_path),
        runner_source_sha256=FROZEN_RUNNER_SHA256,evidence_source_sha256=FROZEN_EVIDENCE_SHA256)
    require(all(report.get(k)==v for k,v in expected.items()),'public support report identity mismatch')
    require(cfg.get('schema')=='eesd-iclr2027-v1' and seed in cfg['seeds']
        and tuple(e['alphas'])==ALPHAS and tuple(e['strengths'])==STRENGTHS
        and e['concentration_bins_for_n4']==[1,2,3,4],'public support configuration mismatch')
    require(cache.get('schema')=='eesd-public-query-mechanism-cache-v1' and cache.get('dataset')==domain
        and cache.get('tests_per_candidate')==10,'public support cache schema mismatch')
    identity=cache.get('generator_identity')
    require(isinstance(identity,list) and len(identity)==4 and identity[2] is None and identity[3]==seed,
        'public support generation identity mismatch')
    require(seal.get('cache_sha256')==file_sha(cache_path) and seal.get('generator_identity')==identity
        and seal.get('bank_bindings')==cache.get('bank_bindings')
        and seal.get('evaluator_manifest_sha256')==cache.get('evaluator_manifest_sha256')
        and seal.get('sources')=={p:file_sha(root/p) for p in CACHE_SOURCES},'public support cache seal mismatch')
    require(len(cache.get('bank_bindings',[]))==2 and
        {b['split'] for b in cache['bank_bindings']}=={'development','primary'},'public support banks missing')
    for binding in cache['bank_bindings']:
        require(file_sha(Path(binding['path'])/'complete.json')==binding['complete_sha256'],
            'public support bank completion checksum mismatch')
    dev=[];assessment=[];seen=set()
    for row in cache['records']:
        require(row.get('split') in {'development','primary'},'public support unexpected split')
        source=row['source_component_id']
        require(source not in seen,'public support duplicate or overlapping source');seen.add(source)
        tests=row['tests']
        require(len(tests)==10 and [str(t['id']) for t in tests]==list(map(str,range(10)))
            and all(isinstance(t['input'],str) for t in tests),'public support ordered query schema mismatch')
        public={k:row[k] for k in ('task_id','problem_id','source_component_id','split')}
        public['tests']=[{'id':str(t['id']),'input':t['input']} for t in tests]
        if row['split']=='development':
            public['outcomes']=row['outcomes']
            dev.append(public)
        else:
            # Do not access row['outcomes'] here, even to validate its shape/type.
            public['outcomes']=['PASS']*10
            assessment.append(public)
    require(dev and assessment,'public support requires both nonempty populations')
    runner=runpy.run_path(str(runner_path))
    from pbpf.eesd import evidence as imported_evidence
    require(file_sha(Path(imported_evidence.__file__))==FROZEN_EVIDENCE_SHA256,'public support imported evidence mismatch')
    memo={}
    def development_examples(split,n,strength,binary):
        require(split=='validation','support tuning may only use development')
        key=(n,strength,binary)
        if key not in memo:memo[key]=runner['build_examples'](dev,n,strength,binary=binary)
        return memo[key]
    selection,_=runner['select_arm'](development_examples,'validation',4,e['alphas'],e['strengths'],
        'effective',ece_bins=e['ece_bins'])
    saved=report['selected']['effective']
    require(set(saved)==set(selection),'public support development selection schema mismatch')
    for key,value in selection.items():
        if value is None:require(saved[key] is None,'public support development selection mismatch')
        else:close(saved[key],value,f'public support development selection/{key}')
    public_examples=runner['build_examples'](assessment,4,selection['strength'],binary=False)
    # The dummy history affects probabilities, which are discarded; mass depends
    # only on public query/history inputs and the development-selected strength.
    _,masses=runner['predictions'](public_examples,selection['alpha'],'effective')
    ids=[dict(domain=domain,source_component_id=r['source_component_id'],problem_id=r['problem_id'],
        candidate_id=r['task_id'],test_id=t['id'],test_index=i) for r in assessment
        for i,t in enumerate(r['tests']) if i>=4]
    encoded=[json.dumps(q,sort_keys=True,separators=(',',':')) for q in ids]
    sources=[q['source_component_id'] for q in ids]
    result={}
    for key,(lo,hi) in zip(BIN_IDS,BINS,strict=True):
        mask=(masses>=lo)&((masses<=hi) if hi==4 else (masses<hi))
        result[key]=dict(sources=list(sources),seeds=[seed]*len(ids),query_ids=list(encoded),
            public_mask=mask,public_mass=masses.copy(),bin_edges=[lo,hi],upper_inclusive=hi==4)
    receipt=dict(schema='eesd-mechanism-public-support-v1',assessment_outcomes_accessed=False,
        cache_sha256=file_sha(cache_path),config_sha256=file_sha(config_path),
        report_sha256=complete['report_sha256'],cache_binding_sha256=file_sha(cache_path.with_suffix('.binding.json')),
        source_sha256=file_sha(Path(__file__)),runner_source_sha256=FROZEN_RUNNER_SHA256,
        evidence_source_sha256=FROZEN_EVIDENCE_SHA256,development_selected_effective=selection,
        ordered_query_ids_sha256=digest_json(ids),public_mass_sha256=digest_json(masses.tolist()),
        public_mask_sha256={key:digest_json(row['public_mask'].tolist()) for key,row in result.items()},
        population_selection_validation='caller-must-first-verify-locked-public-bank-inventory-and-cache-seals',
        scope='public-only-support-projection-not-full-artifact-verification')
    require(receipt['cache_sha256']==expected['cache_sha256'] and receipt['config_sha256']==expected['config_sha256'],
        'public support artifacts changed during reconstruction')
    return result,receipt
