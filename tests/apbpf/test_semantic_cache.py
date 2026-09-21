import hashlib
import json

import numpy as np
import pytest

from pbpf.apbpf.semantic_cache import FrozenSemanticCache


def make_cache(path):
    vectors=np.eye(2,16,dtype=np.float32)
    np.save(path/'vectors.npy',vectors,allow_pickle=False)
    manifest={'schema':'apbpf-frozen-semantic-cache-v1','dimension':16,
              'vectors_sha256':hashlib.sha256((path/'vectors.npy').read_bytes()).hexdigest(),
              'text_sha256':[hashlib.sha256(t.encode()).hexdigest() for t in ['public code A','public code B']]}
    (path/'manifest.json').write_text(json.dumps(manifest))
    return vectors


def test_semantic_cache_uses_only_content_and_fails_without_matching_text(tmp_path):
    vectors=make_cache(tmp_path);encoder=FrozenSemanticCache(tmp_path)
    np.testing.assert_array_equal(encoder('public code B'),vectors[1])
    with pytest.raises(ValueError,match='no hash fallback'):
        encoder('candidate-ID-123')
    with pytest.raises(TypeError,match='public text'):
        encoder({'actual':'hidden feedback'})


def test_semantic_cache_rejects_changed_vectors(tmp_path):
    make_cache(tmp_path)
    with (tmp_path/'vectors.npy').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        FrozenSemanticCache(tmp_path)
