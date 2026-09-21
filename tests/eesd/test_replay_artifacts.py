"""Synthetic cache/report artifacts only; no execution or actual inference data."""
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from pbpf.eesd import replay_artifacts as a
from pbpf.eesd import replay_execution_cache as c
from test_mechanism_artifacts import artifacts,reseal,dump
from test_mechanism_runner import ROOT


def test_original_49_function_is_not_mutated(monkeypatch):
    original=a.contrasts.reconstruct_contrasts.__globals__['load_mechanism_artifacts']
    def sentinel(**kwargs):raise RuntimeError('isolated injected loader')
    monkeypatch.setattr(a,'load_replay_artifacts',sentinel)
    with pytest.raises(RuntimeError,match='isolated injected'):a.reconstruct_contrasts()
    assert a.contrasts.reconstruct_contrasts.__globals__['load_mechanism_artifacts'] is original

from test_mechanism_contrasts import complete_artifacts

@pytest.fixture
def replay(complete_artifacts,monkeypatch,tmp_path):
    """Four-source synthetic math fixture; production 700 gate tested in cache tests."""
    old=complete_artifacts;oldrows=json.loads(old['cache'].read_text())['records']
    monkeypatch.setattr(c.generation,'COUNTS',{'development':2,'primary':2})
    def small_check(inputs):
        assert c.digest(inputs['jobs'])==inputs['bindings']['ordered_jobs_sha256']
        assert len(inputs['jobs'])==4
    monkeypatch.setattr(c,'check_inputs',small_check)
    jobs=[]
    for r in oldrows:
        i=int(r['problem_id'].rsplit('-',1)[1])
        jobs.append(dict(task_id=r['problem_id'],candidate_id=r['task_id'],source_component_id=r['source_component_id'],split=r['split'],code=f'print({i})',tests=[dict(id=t['id'],input=t['input'],expected='private') for t in r['tests']]))
    domain='apps_replay' if old['domain']=='rbr' else 'codecontests_replay'
    bindings={'ordered_jobs_sha256':c.digest(jobs),'watched_files':{},'execution_sources':{'synthetic.py':'a'*64}}
    inputs={'jobs':jobs,'bindings':bindings,'identity':{'domain':domain,'family':'qwen25_7b','seed':1701,'model':c.generation.MODELS['qwen25_7b'][0],'revision':c.generation.MODELS['qwen25_7b'][1],'adapter':None}}
    ready={'schema':'eesd-replay-execution-readiness-v1','status':'ready','profile':c.PROFILE,'execution_lock_sha256':'e'*64,'sources':bindings['execution_sources']}
    directory=tmp_path/'replaycache';measured={'records':oldrows,'readiness':ready,'bindings_sha256':c.digest(bindings)}
    c.publish_cache(directory,inputs,measured)
    kwargs=dict(report_dir=old['report_dir'],cache=directory/'cache.json',config=old['config'],domain=domain,model='qwen25_7b',seed=1701,expected_inputs=inputs,expected_readiness=ready,expected_bindings={'execution_lock_sha256':'e'*64,'statistical_lock_sha256':'f'*64},root=ROOT)
    report=json.loads((old['report_dir']/'report.json').read_text());report.update(dataset=domain,cache_sha256=c.sha(directory/'cache.json'))
    dump(old['report_dir']/'report.json',report);reseal(old['report_dir'])
    rebind(kwargs)
    return kwargs


def rebind(kwargs):
    keys=('cache','config','expected_inputs','expected_readiness','expected_bindings','root')
    dump(kwargs['report_dir']/'replay-binding.json',a.build_report_binding(**{k:kwargs[k] for k in keys}))


def test_replay_six_arms_and_original_49_slots(replay):
    data,receipt=a.load_replay_artifacts(**replay)
    assert len(data['labels'])==12 and len(data['probabilities'])==6
    assert all(q['domain']==replay['domain'] for q in data['query_ids'])
    assert data['query_ids'][0]['test_index']==4
    out,sealed=a.reconstruct_contrasts(**replay)
    assert tuple(out)==a.contrasts.contrast_ids() and len(out)==49
    assert all(row['status']=='available' for row in out.values())
    assert out['A8/n4']['alias_of']=='core/tuned'
    assert len(out['A7/permuted']['comparators'])==3
    assert out['A10/binary']['proposed'].shape[1]==2
    assert sealed['original_contrasts_sha256']==a.FROZEN_CONTRASTS_SHA256


def test_public_support_never_reads_assessment_outcomes_or_full_verifier(replay,monkeypatch):
    original=a.read
    class PublicOnly(dict):
        def __getitem__(self,key):
            if key=='outcomes':raise AssertionError('assessment outcomes accessed')
            return super().__getitem__(key)
    def guarded(path):
        value=original(path)
        if Path(path)==replay['cache']:
            value['records']=[PublicOnly(r) if r['split']=='primary' else r for r in value['records']]
        return value
    monkeypatch.setattr(a,'read',guarded)
    monkeypatch.setattr(c,'verify_replay_cache',lambda *args,**kw:pytest.fail('full labels verifier called'))
    supports,receipt=a.derive_public_support(**replay)
    assert set(supports)==set(a.contrasts.BIN_IDS) and receipt['assessment_outcomes_accessed'] is False
    assert all(len(v['public_mask'])==12 for v in supports.values())


def test_assessment_label_mutation_preserves_public_support(replay):
    before,_=a.derive_public_support(**replay)
    payload=json.loads(replay['cache'].read_text())
    for row in payload['records']:
        if row['split']=='primary':row['outcomes']=['TIMEOUT']*10
    dump(replay['cache'],payload)
    complete_path=replay['cache'].parent/'complete.json';complete=json.loads(complete_path.read_text())
    complete['files']['cache.json']={'sha256':c.sha(replay['cache']),'bytes':replay['cache'].stat().st_size};dump(complete_path,complete)
    report_path=replay['report_dir']/'report.json';report=json.loads(report_path.read_text());report['cache_sha256']=c.sha(replay['cache']);dump(report_path,report);reseal(replay['report_dir']);rebind(replay)
    after,_=a.derive_public_support(**replay)
    for key in before:
        assert before[key]['query_ids']==after[key]['query_ids']
        np.testing.assert_array_equal(before[key]['public_mask'],after[key]['public_mask'])
        np.testing.assert_array_equal(before[key]['public_mass'],after[key]['public_mass'])
    with pytest.raises(ValueError):a.load_replay_artifacts(**replay)


@pytest.mark.parametrize('change',['statlock','seed','query','probability','selected'])
def test_external_or_sealed_artifact_tampering_rejected(replay,change):
    if change=='statlock':replay['expected_bindings']['statistical_lock_sha256']='0'*64
    elif change=='seed':replay['seed']=1702
    elif change=='query':replay['expected_inputs']['jobs'][0]['tests'][4]['input']='different'
    elif change=='selected':
        path=replay['report_dir']/'report.json';r=json.loads(path.read_text());r['selected']['effective']['strength']=123;dump(path,r);reseal(replay['report_dir'])
    else:
        path=replay['report_dir']/'predictions.npz'
        with np.load(path,allow_pickle=False) as stored:arrays={k:stored[k] for k in stored.files}
        arrays['effective'][0]=np.roll(arrays['effective'][0],1);np.savez(path,**arrays);reseal(replay['report_dir'])
    with pytest.raises((ValueError,AssertionError)):a.load_replay_artifacts(**replay)


def test_public_support_still_verifies_development_selection(replay):
    path=replay['report_dir']/'report.json';report=json.loads(path.read_text())
    report['selected']['effective']['alpha']=123.
    dump(path,report);reseal(replay['report_dir'])
    with pytest.raises(ValueError,match='selection'):a.derive_public_support(**replay)


def test_public_projection_rejects_resealed_wrong_order_without_report(replay):
    payload=json.loads(replay['cache'].read_text());payload['records'][2]['tests'][4]['input']='different'
    dump(replay['cache'],payload)
    cp=replay['cache'].parent/'complete.json';complete=json.loads(cp.read_text())
    complete['files']['cache.json']={'sha256':c.sha(replay['cache']),'bytes':replay['cache'].stat().st_size};dump(cp,complete)
    with pytest.raises(ValueError,match='ordered public identity'):
        a.verify_public_cache_projection(replay['cache'],replay['expected_inputs'],replay['expected_readiness'])
