import importlib.util
from pathlib import Path
import sys

import numpy as np

from pbpf.real_gate import FrozenTextEncoder


def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[2]/'scripts'/filename)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def test_repair_features_match_prediction_checkpoint_protocol_without_private_feedback():
    repair=module('repair_feature_contract','run_rbr_repair_gate.py')
    prediction=module('prediction_feature_contract','run_rbr_prediction_gate.py')
    rows=[{'task_text':'public task','candidate':'print(1)',
           'tests':[{'input':'2','expected':'3','actual':'PRIVATE_ACTUAL','stderr':'PRIVATE_STDERR'} for _ in range(10)],
           'outcomes':['PASS']*10}]
    encoder=FrozenTextEncoder(16)
    for public in (False,True):
        actual=repair._features(rows,encoder,'cpu',expected_is_public=public)
        expected=prediction._tensorize(rows,encoder,'cpu',expected_is_public=public)
        np.testing.assert_array_equal(actual.tests,expected.tests)
    legacy=repair._features(rows,encoder,'cpu')
    np.testing.assert_array_equal(legacy.tests[0,0],encoder('2'))
    rows[0]['tests'][0]['actual']='CHANGED_PRIVATE';rows[0]['tests'][0]['stderr']='CHANGED_SECRET'
    updated=repair._features(rows,encoder,'cpu',expected_is_public=True)
    np.testing.assert_array_equal(updated.tests,expected.tests)
