from pathlib import Path
import copy
import pytest
import yaml
from pbpf.eesd.replay_statistics import families,validate_config,protocol


def config():
    return yaml.safe_load((Path(__file__).resolve().parents[2]/'runs/eesd-setup/mechanism-config.lock.yaml').read_text())


def test_separate_complete_families_preserve_every_reserved_slot():
    primary,secondary=families()
    assert len(primary)==8 and len(secondary)==1560
    assert len(set(primary+secondary))==1568
    assert all(k.startswith(('apps_replay/','codecontests_replay/')) for k in primary+secondary)
    assert all(k.endswith('/core/effective_params/nll') for k in primary)
    assert sum('/A8/n4/' in k for k in secondary)==32


def test_original_locked_config_is_accepted():validate_config(config())


@pytest.mark.parametrize('field,value',[('alphas',[.1]),('strengths',[0.]),('bootstrap_draws',9999),('ece_bins',20),('visible_counts',[4])])
def test_no_config_grid_or_budget_relaxation(field,value):
    c=config();c['evidence'][field]=value
    with pytest.raises(ValueError):validate_config(c)


def test_secondary_resolution_limitation_is_predeclared():
    p=protocol()
    assert p['primary_slots']==8 and p['secondary_slots']==1560
    assert 1/(p['bootstrap_draws']+1)>p['alpha']/p['secondary_slots']
    assert p['joint_study_wide_fwer_claim'] is False


@pytest.fixture
def locked(tmp_path, monkeypatch):
    import json
    from pbpf.eesd import replay_statistics as s
    monkeypatch.setattr(s,'ROOT',tmp_path)
    monkeypatch.setattr(s,'SOURCES',('analysis.py',))
    monkeypatch.setattr(s,'DOCUMENTS',('protocol.md',))
    for name in ('analysis.py','protocol.md','generation.json','execution.json','admission.json'):
        (tmp_path/name).write_text('{}')
    cfg=tmp_path/'config.yaml';cfg.write_text(yaml.safe_dump(config()))
    matrix=dict(schema='eesd-replay-cache-matrix-v1',cells=[dict(domain=d,family=m,seed=seed,candidate_jobs=700,test_executions=7000)
        for d in s.DOMAINS for m in s.MODELS for seed in s.SEEDS],
        generation_manifest=s.pointer(tmp_path/'generation.json'),execution_lock=s.pointer(tmp_path/'execution.json'),
        bundle=str(tmp_path),admission_sha256=s.sha(tmp_path/'admission.json'),sources={'analysis.py':s.sha(tmp_path/'analysis.py')})
    mp=tmp_path/'matrix.json';mp.write_text(json.dumps(matrix))
    lock=tmp_path/'lock.json';lock.write_text(json.dumps(s.build_statistical_lock(cfg,mp)))
    return s,lock,s.sha(lock),cfg,mp


def test_statistical_lock_roundtrip_and_explicit_paths(locked):
    s,p,d,c,m=locked
    assert s.validate_statistical_lock(p,d,c,m)['protocol']==s.protocol()
    with pytest.raises(ValueError,match='path mismatch'):s.validate_statistical_lock(p,d,c.parent/'other.yaml')


@pytest.mark.parametrize('name',['analysis.py','protocol.md','generation.json','execution.json','admission.json','config.yaml','matrix.json'])
def test_sealed_input_modification_rejected(locked,name):
    s,p,d,c,m=locked
    target=p.parent/name
    target.write_text(target.read_text()+'\n')
    with pytest.raises(ValueError):s.validate_statistical_lock(p,d)


def test_mutated_reserved_family_rejected_even_with_new_outer_hash(locked):
    import json
    s,p,d,c,m=locked
    lock=json.loads(p.read_text());lock['protocol']['secondary_slots']=1559;p.write_text(json.dumps(lock))
    with pytest.raises(ValueError):s.validate_statistical_lock(p,s.sha(p))
