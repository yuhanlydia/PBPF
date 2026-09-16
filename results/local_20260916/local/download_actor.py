"""Fetch the exact actor revision used by the upstream repair diagnostic."""
import os
from pathlib import Path
root = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(root / 'model-cache')
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1')
from huggingface_hub import snapshot_download
print(snapshot_download(
    'Qwen/Qwen2.5-Coder-7B-Instruct',
    revision='c03e6d358207e414f1eca0bb1891e29f1db0e242',
    allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE', 'README.md'],
    max_workers=4,
), flush=True)
