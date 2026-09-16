#!/usr/bin/env python3
"""Download and verify the locked DeepSeek replication model (no GPU run)."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    if args.manifest.exists():
        raise FileExistsError("model verification manifest already exists")
    from huggingface_hub import HfApi, snapshot_download
    model = "deepseek-ai/deepseek-coder-6.7b-instruct"
    revision = "e5d64addd26a6a1db0f9b863abf6ee3141936807"
    metadata = HfApi().model_info(model, revision=revision, files_metadata=True)
    if metadata.sha != revision:
        raise ValueError("replication model revision mismatch")
    names = [s.rfilename for s in metadata.siblings
             if s.rfilename.endswith(".safetensors") or s.rfilename in
             {"config.json", "generation_config.json", "model.safetensors.index.json",
              "tokenizer.json", "tokenizer_config.json", "LICENSE", "README.md"}]
    print(json.dumps({"phase": "download", "model": model, "revision": revision, "files": names}), flush=True)
    path = Path(snapshot_download(model, revision=revision, cache_dir=args.cache_dir,
                                 allow_patterns=names, max_workers=2))
    files = []
    for sibling in metadata.siblings:
        if sibling.rfilename not in names:
            continue
        file = path / sibling.rfilename
        with file.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = sibling.lfs.sha256 if sibling.lfs is not None else None
        if expected is not None and actual != expected:
            raise ValueError(f"downloaded model LFS checksum mismatch: {sibling.rfilename}")
        files.append({"file": sibling.rfilename, "bytes": file.stat().st_size,
                      "sha256": actual, "publisher_lfs_sha256": expected})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x") as stream:
        json.dump({"model": model, "revision": revision, "snapshot": str(path),
                   "files": files, "status": "downloaded-and-lfssha-verified-not-yet-evaluated"}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"phase": "complete", "snapshot": str(path)}), flush=True)


if __name__ == "__main__":
    main()
