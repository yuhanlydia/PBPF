import hashlib
import json
from pathlib import Path
import pytest
from pbpf.eesd.generation_ledger import summarize_bank


def write_json(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bank(tmp_path):
    run = {'schema': 'apbpf-codearc-generation-v1', 'model': 'm', 'revision': 'a'*40,
           'seed':1701, 'split':'primary', 'components':2, 'candidates':1,
           'task_ids':['task/a','task/b'], 'source_component_ids':['source/a','source/b']}
    run_sha = write_json(tmp_path/'run.json',run)
    row = {'task_id':'task/a','source_component_id':'source/a','split':'primary',
           'prompt_metadata':{'input_tokens':7,'clipped_fields':[]},
           'elapsed_seconds':1.5,'candidates':[{'generated_tokens':3,'hit_token_cap':False}]}
    row_sha = write_json(tmp_path/'task-a.json',row)
    (tmp_path/'task-a.sha256').write_text(row_sha+'\n')
    return run_sha,row_sha


def test_partial_ledger_ignores_unsealed_inflight_records(tmp_path):
    bank(tmp_path)
    (tmp_path/'task-b.json').write_text('{}')
    out = summarize_bank(tmp_path)
    assert out['status']=='partial'
    assert out['verified_sources']==1
    assert out['expected_sources']==2
    assert out['input_tokens']==7
    assert out['generated_tokens']==3
    assert out['measured_candidate_seconds']==1.5
    assert out['unsealed_records']==['task-b.json']
    assert out['cost_usd'] is None
    assert out['scientific_scoring_complete'] is False


def test_tampered_record_is_not_counted(tmp_path):
    bank(tmp_path)
    (tmp_path/'task-a.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        summarize_bank(tmp_path)


def test_complete_marker_must_cover_full_inventory(tmp_path):
    run_sha,row_sha=bank(tmp_path)
    write_json(tmp_path/'complete.json',{'run_sha256':run_sha,'files':{'task-a.json':row_sha}})
    with pytest.raises(ValueError,match='inventory'):
        summarize_bank(tmp_path)


def test_complete_bank_can_seal_records_without_sidecars(tmp_path):
    run_sha,row_sha=bank(tmp_path)
    run=json.loads((tmp_path/'run.json').read_text())
    run.update(components=1,task_ids=['task/a'],source_component_ids=['source/a'])
    run_sha=write_json(tmp_path/'run.json',run)
    (tmp_path/'task-a.sha256').unlink()
    write_json(tmp_path/'complete.json',{'run_sha256':run_sha,'files':{'task-a.json':row_sha}})
    result=summarize_bank(tmp_path)
    assert result['status']=='complete'
    assert result['verified_sources']==1
    assert result['generated_tokens']==3
