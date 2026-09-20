import importlib.util
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')


@pytest.mark.parametrize('changed_key',['cache_sha256','feature_sha256'])
def test_checkpoint_audit_rejects_changed_data_before_scoring(monkeypatch,tmp_path,changed_key):
    path=Path(__file__).resolve().parents[1]/'scripts/inspect_diagnostic_state.py'
    spec=importlib.util.spec_from_file_location('inspect_diagnostic_state_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    recorded={'cache_sha256':'dataset-original','feature_sha256':'feature-original'}
    current={**recorded,changed_key:'changed'}
    saved={'config':{'feature_cache':'features.npz','cache':'dataset.json','feature_dim':8},'data':recorded}
    monkeypatch.setattr(module.torch,'load',lambda *a,**kw:saved)
    monkeypatch.setattr(module,'real_batches',lambda *a,**kw:({},current))
    with pytest.raises(ValueError,match=changed_key):
        module.inspect(tmp_path,4,1,'cpu',2)
