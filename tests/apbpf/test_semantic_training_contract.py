import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pbpf.real_gate import RBR_CACHE_SCHEMA, public_test_text


def test_frozen_features_train_and_bind_exact_source_population(tmp_path):
    spec=importlib.util.spec_from_file_location('semantic_prediction_contract',
        Path(__file__).resolve().parents[2]/'scripts/run_rbr_prediction_gate.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    case={'input':'0','expected':'1','actual':'1','stderr':'','returncode':0,'timed_out':False,'outcome':'PASS'}
    payload={'schema':RBR_CACHE_SCHEMA,'tests_per_candidate':10,'evaluation_role':'development_assessment_only','records':[
        {'task_id':s,'problem_id':s,'source_component_id':s,'split':s,'task_text':'public task',
         'candidate':'print(1)','tests':[dict(case,id=str(i)) for i in range(10)],'outcomes':['PASS']*10}
        for s in ['train','development','test']]}
    texts=['public task','print(1)',public_test_text(case)]
    cache=tmp_path/'features';cache.mkdir()
    np.save(cache/'vectors.npy',np.eye(3,16,dtype=np.float32),allow_pickle=False)
    manifest={'schema':'apbpf-frozen-semantic-cache-v1','dimension':16,
              'vectors_sha256':hashlib.sha256((cache/'vectors.npy').read_bytes()).hexdigest(),
              'text_sha256':[hashlib.sha256(t.encode()).hexdigest() for t in texts],
              'expected_is_public':False,'evaluation_role':'development_assessment_only',
              'source_payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
    (cache/'manifest.json').write_text(json.dumps(manifest))
    options=dict(feature_dim=16,latent_dim=8,hidden_dim=16,particles=2,steps=1,batch_size=2,
                 learning_rate=.001,seed=1701,apbpf=True,difficulty_dim=2,strong_baselines=False,
                 bootstrap_replicates=10,feature_cache=cache)
    report=module.train_and_evaluate(payload,tmp_path/'trained.json',**options)
    assert report['config']['encoder_type']=='frozen_code_model'
    assert report['config']['feature_cache_manifest_sha256']==hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest()
    assert not report['gate']['baseline_fairness_passes']
    payload['records'][0]['task_text']='changed public task'
    with pytest.raises(ValueError,match='another source population'):
        module.train_and_evaluate(payload,tmp_path/'invalid.json',**options)
