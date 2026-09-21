"""Synthetic orchestration tests; no cache execution or real inferential result."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'scripts/report_eesd_replay_inference.py'
spec=importlib.util.spec_from_file_location('replay_reporter',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(tmp_path,monkeypatch,*,missing=False):
    config=tmp_path/'config';config.write_text('{}')
    matrix_path=tmp_path/'matrix';matrix_path.write_text('{}')
    lock=tmp_path/'lock';lock.write_text('{}')
    matrix={'output_root':str(tmp_path/'results'),'cells':[dict(domain=d,family=f,seed=s) for d in m.DOMAINS for f in m.MODELS for s in m.SEEDS]}
    locked={'config':{'path':str(config),'sha256':m.sha(config)},'cache_manifest':{'path':str(matrix_path),'sha256':m.sha(matrix_path)}}
    monkeypatch.setattr(m,'validate_inputs',lambda **kwargs:(locked,matrix,config))
    inventory={split:[{'task_id':f'{split}-{i}','source_component_id':f'{split}-{i}'} for i in range(n)] for split,n in [('development',200),('primary',500)]}
    def inspect(cell,*args,**kwargs):
        is_missing=missing and cell['seed']==1703
        return inventory,{'status':'missing' if is_missing else 'verified','reasons':['report missing'] if is_missing else []},dict(domain=cell['domain'],model=cell['family'],seed=cell['seed'])
    monkeypatch.setattr(m,'inspect_cell',inspect)
    monkeypatch.setattr(m,'source_inventory',lambda:{'synthetic':'0'*64})
    # Test a fixed independent family inventory without requiring a real lock.
    def families():
        p=[];s=[]
        for d in m.DOMAINS:
            for f in m.MODELS:
                for contrast in m.contrast_ids():
                    for metric in m.METRICS:
                        (p if contrast==m.PRIMARY and metric=='nll' else s).append(f'{d}/{f}/{contrast}/{metric}')
        return p,s
    monkeypatch.setattr(m,'families',families)
    return dict(statistical_lock=lock,statistical_lock_sha256=m.sha(lock),output=tmp_path/'out')


def test_missing_keeps_all_24_cells_and_full_families_without_reconstruction(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch,missing=True)
    monkeypatch.setattr(m,'derive_public_support',lambda **kw:pytest.fail('missing inventory must not derive'))
    monkeypatch.setattr(m,'reconstruct_contrasts',lambda **kw:pytest.fail('missing inventory must not reconstruct'))
    result=m.run_report(**args)
    assert result['status']=='incomplete' and result['scientific_inference_performed'] is False
    assert sum(len(seeds) for seeds in result['coverage'].values())==24
    assert result['families']['primary']['family_size']==8
    assert result['families']['secondary']['family_size']==1560
    assert all(f['reject'] is None for f in result['families'].values())

@pytest.mark.parametrize('drift',[None,'mask','query'])
def test_global_support_freeze_before_labels_and_alias_full_slots(tmp_path,monkeypatch,drift):
    args=fixture(tmp_path,monkeypatch);output=args['output'];sources=[f'primary-{i}' for i in range(500)];calls=[];public_calls=[]
    def synthetic(seed):
        result={}
        for contrast in m.contrast_ids():
            row={'status':'available','sources':sources,'seeds':np.full(500,seed,dtype=np.int64),'query_ids':[(s,4) for s in sources],
                 'labels':np.zeros(500,dtype=int),'proposed':np.tile([.8,.2],(500,1)),'comparators':{'fixed':np.tile([.6,.4],(500,1))}}
            if contrast in m.BIN_IDS:row['public_mask']=np.full(500,contrast!=m.BIN_IDS[1],dtype=bool)
            if contrast=='A8/n4':row['alias_of']='core/tuned'
            result[contrast]=row
        return result
    def derive(**kw):
        public_calls.append(kw);rows=synthetic(kw['seed'])
        return {k:{f:rows[k][f] for f in ('sources','seeds','query_ids','public_mask')} for k in m.BIN_IDS},{'synthetic':True}
    def reconstruct(**kw):
        assert len(public_calls)==24
        assert len(json.loads((output/'a9-public-support.json').read_text()))==8
        assert (output/'a9-public-support.complete.json').is_file()
        rows=synthetic(kw['seed'])
        if drift=='mask':rows[m.BIN_IDS[0]]['public_mask'][0]=False
        if drift=='query':rows[m.BIN_IDS[0]]['query_ids'][0]='changed'
        return rows,{'synthetic':True}
    monkeypatch.setattr(m,'derive_public_support',derive);monkeypatch.setattr(m,'reconstruct_contrasts',reconstruct)
    def data(**kw):
        assert all(type(s) is int for s in kw['seeds']) and all(type(q) is str for q in kw['query_ids'])
        assert set(kw['sources'])==set(sources)
        return kw
    monkeypatch.setattr(m,'PairedMechanismData',data)
    def bootstrap(*a,**kw):
        calls.append(kw);assert (output/'reconstruction-receipts.json').exists()
        return {'status':'supported','metrics':{metric:{'status':'supported','p_raw':1.,'gain':0.} for metric in m.METRICS}}
    monkeypatch.setattr(m,'bootstrap_gain',bootstrap)
    if drift:
        with pytest.raises(ValueError,match='changed frozen public support'):m.run_report(**args)
        assert not calls and not (output/'complete.json').exists()
    else:
        report=m.run_report(**args)
        assert len(calls)==47*8
        assert report['families']['primary']['complete'] is True
        assert report['families']['secondary']['reject'] is None
        for values in report['contrasts'].values():
            assert len(values)==49 and values['A8/n4']['metrics']==values['core/tuned']['metrics']
            assert values[m.BIN_IDS[1]]['status']=='unsupported'


def test_existing_corrupt_report_not_missing(tmp_path):
    report=tmp_path/'report';report.mkdir()
    (report/'complete.json').write_text(json.dumps({'report_sha256':'0'*64,'predictions_sha256':'0'*64}))
    (report/'report.json').write_text('{}');(report/'predictions.npz').write_bytes(b'x')
    with pytest.raises(ValueError,match='checksum'):m.verify_report_bytes(report)

def test_existing_report_header_identity_cannot_be_relabelled(tmp_path):
    config=tmp_path/'config';config.write_text('{}');cache=tmp_path/'cache.json';cache.write_text('{}')
    directory=tmp_path/'report';directory.mkdir()
    expected=dict(schema='eesd-evidence-matrix-v1',dataset='apps_replay',model='qwen25_7b',seed=1701,visible=4,validation_split='development',assessment_split='primary',cache_sha256=m.sha(cache),config_sha256=m.sha(config),runner_source_sha256=m.sha(ROOT/'scripts/run_eesd_evidence_matrix.py'),evidence_source_sha256=m.sha(ROOT/'src/pbpf/eesd/evidence.py'))
    (directory/'report.json').write_text(json.dumps(expected))
    m.verify_report_identity(directory,cache=cache,config=config,domain='apps_replay',model='qwen25_7b',seed=1701)
    expected['model']='seed_coder_8b';(directory/'report.json').write_text(json.dumps(expected))
    with pytest.raises(ValueError,match='identity'):m.verify_report_identity(directory,cache=cache,config=config,domain='apps_replay',model='qwen25_7b',seed=1701)


def test_existing_sealed_bank_is_verified_even_when_other_reports_missing(tmp_path,monkeypatch):
    inventory={split:[dict(task_id=f'{split}-{i}',source_component_id=f'{split}-{i}',domain='apps_replay',split=split) for i in range(n)] for split,n in [('development',200),('primary',500)]}
    identities={split:[{k:r[k] for k in ('task_id','source_component_id')} for r in rows] for split,rows in inventory.items()}
    cell={'domain':'apps_replay','family':'qwen25_7b','seed':1701,'output':str(tmp_path/'cache'),'dependencies':{split:dict(bank=str(tmp_path/split),components=len(rows),population_sha256=m.execution.digest(identities[split])) for split,rows in inventory.items()}}
    matrix={'bundle':str(tmp_path/'bundle'),'admission_sha256':'0'*64,'output_root':str(tmp_path),'populations':{'apps_replay':{s:{'ordered_identities':v} for s,v in identities.items()}}}
    bank=tmp_path/'development';bank.mkdir();(bank/'complete.json').write_text('{}')
    monkeypatch.setattr(m.generation,'load_public_split',lambda bundle,sha,domain,split,family:dict(rows=inventory[split],split=split))
    monkeypatch.setattr(m.generation,'build_expected',lambda p,*a:dict(split=p['split']))
    def reject(*a,**kw):raise ValueError('sealed bank tampered')
    monkeypatch.setattr(m.generation,'verify_replay_bank',reject)
    with pytest.raises(ValueError,match='sealed bank tampered'):m.inspect_cell(cell,matrix,tmp_path/'config','0'*64)

def test_artifact_snapshot_rejects_changed_bytes(tmp_path):
    p=tmp_path/'report.json';p.write_text('{}')
    entry={'sealed_artifacts':{str(p):m.sha(p)}}
    m.verify_artifact_snapshot(entry)
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='sealed artifact changed'):m.verify_artifact_snapshot(entry)

def test_real_extension_family_keys_are_separate():
    primary,secondary=m.families()
    assert len(primary)==8 and len(secondary)==1560
    assert len(set(primary+secondary))==1568
    assert all(key.startswith(('apps_replay/','codecontests_replay/')) for key in primary+secondary)
    assert all(key.endswith('/core/effective_params/nll') for key in primary)
