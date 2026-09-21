"""Read content-addressed, frozen public-text features without label access."""
import hashlib
import json
from pathlib import Path

import numpy as np


class FrozenSemanticCache:
    def __init__(self, root):
        self.root = Path(root)
        path = self.root / 'manifest.json'
        self.manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest = json.loads(path.read_bytes())
        if self.manifest.get('schema') != 'apbpf-frozen-semantic-cache-v1':
            raise ValueError('invalid frozen semantic cache schema')
        vectors_path = self.root / 'vectors.npy'
        with vectors_path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        if checksum != self.manifest['vectors_sha256']:
            raise ValueError('semantic feature checksum mismatch')
        self.vectors = np.load(vectors_path, allow_pickle=False, mmap_mode='r')
        self.dimension = self.manifest['dimension']
        keys = self.manifest['text_sha256']
        if (self.vectors.shape != (len(keys), self.dimension) or len(set(keys)) != len(keys)
                or not np.isfinite(self.vectors).all()):
            raise ValueError('invalid semantic feature inventory')
        self.indices = {key: i for i, key in enumerate(keys)}

    def __call__(self, text):
        if not isinstance(text, str):
            raise TypeError('semantic encoder accepts public text only')
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self.indices:
            raise ValueError('public text absent from frozen semantic cache; no hash fallback')
        return self.vectors[self.indices[key]]
