import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]


def module():
    spec=importlib.util.spec_from_file_location('inference_report',ROOT/'scripts/report_eesd_mechanism_inference.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def public(tmp_path, *, count=500):
    m=module();root=tmp_path/'public';root.mkdir()
    tasks=[{'task_id':f'{split}-{i}','source_component_id':f'{split}-{i}','split':split}
           for split,n in [('development',201),('primary',count)] for i in range(n)]
    (root/'tasks.jsonl').write_text('\n'.join(map(json.dumps,tasks)))
    (root/'manifest.json').write_text(json.dumps({'public_tasks_sha256':m.sha(root/'tasks.jsonl')}))
    return root


def test_expected_population_preserves_order_and_exact_200_500(tmp_path):
    m=module();root=public(tmp_path)
    inventory=m.public_inventory({'public_root':str(root),'domain':'rbr'})
    assert len(inventory['development'])==200 and len(inventory['primary'])==500
    assert inventory['development'][-1]['task_id']=='development-199'


def test_cannot_accept_matching_but_wrong_499_source_subset(tmp_path):
    m=module();root=public(tmp_path,count=499)
    with pytest.raises(ValueError,match='500'):m.public_inventory({'public_root':str(root),'domain':'rbr'})


def test_full_reserved_inventory_and_primary_identity():
    m=module();primary,secondary=m.families()
    assert len(primary)==8 and len(secondary)==1560
    assert not set(primary)&set(secondary)
    assert all('/core/effective_params/nll' in key for key in primary)


def test_mask_common_support_uses_public_masks_and_python_ids():
    m=module();rows={}
    for seed in (1701,1702,1703):
        rows[seed]={'status':'supported','sources':np.array(['a','a','b']),
                    'seeds':np.full(3,seed,dtype=np.int64),'query_ids':[('a',0),('a',1),('b',0)],
                    'labels':np.array([0,1,0]),'proposed':np.array([[.8,.2]]*3),
                    'comparators':{'fixed':np.array([[.6,.4]]*3)},
                    'public_mask':np.array([True,False,seed!=1702])}
    support=m.bin_support(rows)
    assert support['common_sources']==['a']
    assert support['source_counts_by_seed']=={'1701':2,'1702':1,'1703':2}
    args=m.combine(rows,selected_sources=['a'],use_public_mask=True)
    assert args['sources']==['a']*3
    assert all(type(s) is int for s in args['seeds'])
    assert all(type(q) is str for q in args['query_ids'])


def test_incomplete_report_has_no_family_rejections():
    m=module();report=m.incomplete_families('missing reports')
    for family in report.values():
        assert family['complete'] is False
        assert family['reject'] is None and family['adjusted_p'] is None


def test_existing_bad_seal_is_error_not_missing(tmp_path):
    m=module();directory=tmp_path/'report';directory.mkdir()
    (directory/'complete.json').write_text(json.dumps({'report_sha256':'bad','predictions_sha256':'bad'}))
    (directory/'report.json').write_text('{}');(directory/'predictions.npz').write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum'):m.verify_report_bytes(directory)


def test_partial_inventory_still_invokes_formal_verifier_for_existing_sealed_bank(tmp_path):
    m=module();root=public(tmp_path)
    cell={'public_root':str(root),'domain':'rbr','dataset':'runbugrun','model':'qwen25_7b','family':'qwen25_7b'}
    bank=tmp_path/'results/mechanism-banks/runbugrun/qwen25_7b/seed1701/development'
    bank.mkdir(parents=True);(bank/'complete.json').write_text('{}')
    def verifier(path,**kwargs):
        assert kwargs['split']=='development' and kwargs['seed']==1701
        assert kwargs['cell']==cell
        raise ValueError('synthetic decode/source profile mismatch')
    with pytest.raises(ValueError,match='decode/source'):
        m.inspect_cell(cell,tmp_path/'results',ROOT/'configs/experiments/eesd_iclr2027.yaml',{'verify_mechanism_bank':verifier})


def test_missing_cli_preflight_reserves_all_slots_and_writes_seal(tmp_path):
    import yaml
    m=module();root=public(tmp_path)
    cells=[{'public_root':str(root),'domain':domain,'dataset':dataset,'model':model,'family':model,
            'evaluator_root':str(tmp_path/'NEVER_READ_PRIVATE')}
           for dataset,domain in m.DATASETS.items() for model in m.MODELS]
    manifest=tmp_path/'manifest.yaml';manifest.write_text(yaml.safe_dump({'schema':'eesd-cache-manifest-v1','mechanism_cells':cells}))
    lock=ROOT/'runs/eesd-setup/statistical-amendment-lock-20260920.json'
    out=tmp_path/'out'
    report=m.run_report(manifest=manifest,config=ROOT/'configs/experiments/eesd_iclr2027.yaml',
        statistical_lock=lock,statistical_lock_sha256=m.sha(lock),results_root=tmp_path/'none',output=out,preflight_only=True)
    assert report['status']=='preflight_only' and not report['scientific_inference_performed']
    assert len(report['coverage'])==8
    assert all(len(cell)==3 for cell in report['coverage'].values())
    assert report['families']['primary']['family_size']==8
    assert report['families']['secondary']['family_size']==1560
    assert report['families']['primary']['reject'] is None
    assert json.loads((out/'complete.json').read_text())['report_sha256']==m.sha(out/'report.json')
    assert not (out/'a9-public-support.json').exists()


@pytest.mark.parametrize('drift',[None,'mask','query'])
def test_complete_synthetic_flow_seals_support_first_and_shares_alias(tmp_path,monkeypatch,drift):
    m=module()
    cells=[{'dataset':d,'domain':domain,'model':model,'family':model,'public_root':str(tmp_path/'public'),
            'evaluator_root':'unused'} for d,domain in m.DATASETS.items() for model in m.MODELS]
    (tmp_path/'public').mkdir()
    for name in ['manifest.json','tasks.jsonl']:(tmp_path/'public'/name).write_text('{}')
    for name in ['manifest','config','lock']:(tmp_path/name).write_text('{}')
    monkeypatch.setattr(m,'validate_inputs',lambda *a:cells)
    sources=[f's{i:03}' for i in range(500)]
    inventory={split:[{'task_id':s,'source_component_id':s} for s in sources[:n]]
               for split,n in [('development',200),('primary',500)]}
    monkeypatch.setattr(m,'inspect_cell',lambda *a:(inventory,{str(s):{'status':'verified','report_dir':'unused','cache':'unused'} for s in m.SEEDS}))
    def synthetic_rows(**kwargs):
        seed=kwargs['seed'];rows={}
        for contrast in m.contrast_ids():
            row={'status':'available','sources':sources,'seeds':np.full(500,seed,dtype=np.int64),
                 'query_ids':[(s,4) for s in sources],'labels':np.zeros(500,dtype=int),
                 'proposed':np.tile([.8,.2],(500,1)),'comparators':{'fixed':np.tile([.6,.4],(500,1))}}
            if contrast in m.BIN_IDS:row['public_mask']=np.full(500,contrast!=m.BIN_IDS[1],dtype=bool)
            if contrast=='A8/n4':row['alias_of']='core/tuned'
            rows[contrast]=row
        return rows,{'synthetic':True}
    output=tmp_path/'out';calls=[]
    def reconstruct(**kwargs):
        # All 24 public supports must be frozen before the first label-bearing
        # reconstruction, not merely before bootstrap.
        frozen=json.loads((output/'a9-public-support.json').read_text())
        assert len(frozen)==8
        assert (output/'a9-public-support.complete.json').is_file()
        rows,receipt=synthetic_rows(**kwargs)
        if drift=='mask':rows[m.BIN_IDS[0]]['public_mask'][0]=False
        elif drift=='query':rows[m.BIN_IDS[0]]['query_ids'][0]='changed-query'
        return rows,receipt
    def derive(**kwargs):
        rows,_=synthetic_rows(**kwargs)
        return {key:{field:rows[key][field] for field in ('sources','seeds','query_ids','public_mask')}
                for key in m.BIN_IDS},{'synthetic_public_features_only':True}
    monkeypatch.setattr(m,'reconstruct_contrasts',reconstruct)
    monkeypatch.setattr(m,'derive_public_support',derive,raising=False)
    def make_data(**kwargs):
        assert all(type(s) is int for s in kwargs['seeds'])
        assert all(type(q) is str for q in kwargs['query_ids'])
        assert set(kwargs['sources'])==set(sources)
        return kwargs
    monkeypatch.setattr(m,'PairedMechanismData',make_data)
    def bootstrap(*args,**kwargs):
        assert (output/'a9-public-support.json').is_file()
        assert (output/'reconstruction-receipts.json').is_file()
        calls.append(kwargs)
        return {'status':'supported','metrics':{metric:{'status':'supported','p_raw':1/10001,'gain':.2} for metric in m.METRICS}}
    monkeypatch.setattr(m,'bootstrap_gain',bootstrap)
    kwargs=dict(manifest=tmp_path/'manifest',config=tmp_path/'config',statistical_lock=tmp_path/'lock',
        statistical_lock_sha256=m.sha(tmp_path/'lock'),results_root=tmp_path/'none',output=output)
    if drift:
        with pytest.raises(ValueError,match='changed frozen public support'):
            m.run_report(**kwargs)
        assert not calls and not (output/'complete.json').exists()
        return
    report=m.run_report(**kwargs)
    # 49 slots - one alias - one unavailable public bin, per cell.
    assert len(calls)==47*8
    assert len(report['contrasts'])==8
    for values in report['contrasts'].values():
        assert len(values)==49
        assert values['A8/n4']['metrics']==values['core/tuned']['metrics']
        assert values[m.BIN_IDS[1]]['status']=='unsupported'
    assert report['families']['primary']['complete']
    assert all(report['families']['primary']['reject'].values()) # SYNTHETIC p values, not scientific results
    assert report['families']['secondary']['reject'] is None
    assert report['families']['secondary']['family_size']==1560
