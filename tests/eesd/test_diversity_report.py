import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('diversity',ROOT/'scripts/report_eesd_diversity.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def row(code,outcomes,source='s'):
    return {'source_component_id':source,'problem_id':source,'split':'primary','code':code,'outcomes':outcomes,'test_ids':list(map(str,range(10)))}

def test_hand_computable_union_ast_and_pairwise():
    passed=['PASS']*10;failed=['WRONG_OUTPUT']*10
    data={1701:[row('print(1)',passed)],1702:[row('print(1) #comment',failed)],1703:[row('bad !',failed)]}
    report=m.summarize(data)
    value=report['sources'][0]
    assert value['exact_code_unique_fraction']==1
    assert value['valid_ast_candidates']==2 and value['valid_ast_unique_fraction']==.5
    assert value['syntax_invalid_candidates']==1 and value['outcome_vector_unique_fraction']==2/3
    assert report['summary']['three_seed_union_all_tests_success']==1
    assert report['summary']['mean_single_seed_all_tests_success']==pytest.approx(1/3)
    assert report['summary']['union_minus_mean_single_seed']==pytest.approx(2/3)
    assert value['pairs']['1702:1703']['shared_task_failure']==1
    assert value['pairs']['1701:1702']['per_test_failure_disagreement']==1
    assert value['correct_candidates']==1 and value['correct_valid_ast_unique_fraction']==1


def test_invalid_and_empty_denominators_explicit():
    data={seed:[row('bad !',['COMPILE_ERROR']*10)] for seed in m.SEEDS}
    value=m.summarize(data)['sources'][0]
    assert value['valid_ast_candidates']==0 and value['valid_ast_unique_fraction'] is None
    assert value['correct_valid_ast_unique_fraction'] is None
    data[1701][0]['code']=''
    value=m.summarize(data)['sources'][0]
    assert value['empty_candidates']==1 and value['valid_ast_candidates']==1

@pytest.mark.parametrize('defect',['missing_seed','different_source','development','different_tests','duplicate'])
def test_incomplete_or_misaligned_population_rejected(defect):
    data={seed:[row('print(1)',['PASS']*10)] for seed in m.SEEDS}
    if defect=='missing_seed':del data[1703]
    elif defect=='different_source':data[1702][0]['source_component_id']='other'
    elif defect=='development':data[1702][0]['split']='development'
    elif defect=='different_tests':data[1702][0]['test_ids'][0]='other'
    else:data[1702].append(data[1702][0])
    with pytest.raises(ValueError):m.summarize(data)


def test_missing_seed_inventory_never_summarizes(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'inspect_seed',lambda **kw:({'status':'missing','reasons':['not complete']},None))
    manifest=tmp_path/'manifest.yaml';manifest.write_text('schema: eesd-cache-manifest-v1\nmechanism_cells:\n- {domain: codearc, dataset: codearc, family: qwen25_7b, model: qwen25_7b}\n')
    monkeypatch.setattr(m,'summarize',lambda *a:pytest.fail('incomplete cannot summarize'))
    report=m.run_report(manifest=manifest,manifest_sha256=m.sha(manifest),results_root=tmp_path,domain='codearc',family='qwen25_7b',direct_cells={},output=tmp_path/'report.json')
    assert report['status']=='incomplete' and report['scientific_summary'] is None
    assert len(report['coverage'])==3

def test_corrupt_existing_direct_seal_is_error_not_missing(tmp_path):
    directory=tmp_path/'direct';(directory/'report').mkdir(parents=True)
    names=['cache.json','direct-binding.json','report/report.json','report/predictions.npz','report/complete.json']
    for name in names:(directory/name).write_text('{}')
    import json
    complete={'schema':'eesd-direct-mechanism-complete-v1','execution_profile':'direct-no-sandbox','files':{name:m.sha(directory/name) for name in names}}
    (directory/'complete.json').write_text(json.dumps(complete));(directory/'cache.json').write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):
        m.inspect_seed(cell={'domain':'codearc','dataset':'codearc','model':'qwen25_7b','family':'qwen25_7b'},seed=1701,results_root=tmp_path/'missingbanks',direct_cell=directory,manifest_sha256='0'*64)
