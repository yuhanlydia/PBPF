#!/usr/bin/env python3
"""Deterministic anonymous source package, gated by a reviewed protocol snapshot."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

FIXED_TIME = (2026, 1, 1, 0, 0, 0)
DESIGN = "docs/superpowers/specs/2026-09-14-pbpf-iclr2027-design.md"
CONFIG = "configs/experiments/iclr_pbpf.yaml"
REQUIRED = ("pbpf_iclr2027.tex", "references.bib", "results_pending.tex",
            "experimental_setup.tex", "variants/titles.tex", "README.md",
            "PROVENANCE.md", "Makefile", "validate_variants.py",
            "iclr2027_conference.sty", "iclr2027_conference.bst",
            "math_commands.tex", "natbib.sty", "fancyhdr.sty")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def protocol_paths(repo_root: Path) -> list[str]:
    paths = [DESIGN, CONFIG]
    paths += [p.relative_to(repo_root).as_posix() for p in sorted((repo_root / "configs/iclr").glob("*.yaml"))]
    if len(paths) <= 2 or any(not (repo_root / name).is_file() for name in paths):
        raise ValueError("Complete formal configuration and approved design are required for protocol lock")
    return sorted(paths)


def lock_protocol(paper: Path, repo_root: Path, acknowledged: bool = False) -> dict:
    if not acknowledged:
        raise ValueError("Protocol lock requires --acknowledge-config-reviewed after Task 7 review")
    paths = protocol_paths(repo_root)
    payloads = {name: (repo_root / name).read_bytes() for name in paths}
    for name, data in payloads.items():
        target = paper / "protocol" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    lock = {"schema": "pbpf-paper-protocol-lock-v1",
            "review_status": "explicit_config_review_acknowledgment",
            "claim_status": "prospective_no_formal_results",
            "files": {name: sha256(data) for name, data in payloads.items()}}
    (paper / "protocol-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock


def validate_lock(paper: Path, repo_root: Path) -> dict:
    path = paper / "protocol-lock.json"
    if not path.is_file():
        raise ValueError("Missing protocol lock; finish Task 7 review before packaging")
    lock = json.loads(path.read_text())
    if lock.get("schema") != "pbpf-paper-protocol-lock-v1" or lock.get("review_status") != "explicit_config_review_acknowledgment":
        raise ValueError("Invalid protocol lock schema or review acknowledgment")
    expected = set(protocol_paths(repo_root))
    if set(lock.get("files", {})) != expected:
        raise ValueError("Protocol lock file inventory differs from current configuration")
    for name, digest in lock["files"].items():
        for location in (repo_root / name, paper / "protocol" / name):
            if not location.is_file() or sha256(location.read_bytes()) != digest:
                raise ValueError(f"Stale protocol lock: {name}")
    return lock


def archive_bytes(payloads: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payloads.items()):
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("Archive paths must be relative and remain inside the package")
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def build_package(paper: Path, repo_root: Path, output: Path) -> dict:
    lock = validate_lock(paper, repo_root)
    required = [*REQUIRED, "protocol-lock.json"]
    required += [f"variants/{kind}_{i}.tex" for kind in ("abstract", "intro") for i in range(1, 4)]
    required += [f"protocol/{name}" for name in lock["files"]]
    missing = [name for name in required if not (paper / name).is_file()]
    if missing:
        raise ValueError(f"Missing manuscript assets: {missing}")
    payloads = {name: (paper / name).read_bytes() for name in sorted(required)}
    # File allowlisting prevents accidental publication of logs, local paths, or data.
    for name, data in payloads.items():
        # A scanner's own bare-prefix literals are not concrete private paths.
        private_path = re.search(rb"/(?:workspace|Users)/[^\s\"'<>]+", data)
        if private_path or b"BEGIN PRIVATE KEY" in data:
            raise ValueError(f"Private path or key marker in package member: {name}")
    manifest = {"schema": "pbpf-paper-source-manifest-v1",
                "claim_status": "prospective_no_formal_results",
                "protocol_lock_sha256": sha256(payloads["protocol-lock.json"]),
                "files": {name: sha256(data) for name, data in payloads.items()}}
    payloads["MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    data = archive_bytes(payloads)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / "PBPF_ICLR_2027.zip"
    archive_path.write_bytes(data)
    (output / "MANIFEST.json").write_bytes(payloads["MANIFEST.json"])
    (output / "PBPF_ICLR_2027.zip.sha256").write_text(f"{sha256(data)}  PBPF_ICLR_2027.zip\n")
    return {"archive": archive_path.as_posix(), "sha256": sha256(data),
            "members": len(payloads), "claim_status": manifest["claim_status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--paper-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lock-protocol", action="store_true")
    parser.add_argument("--acknowledge-config-reviewed", action="store_true")
    args = parser.parse_args()
    paper = args.paper_dir or args.repo_root / "paper"
    if args.lock_protocol:
        lock = lock_protocol(paper, args.repo_root, args.acknowledge_config_reviewed)
        print(json.dumps({"protocol_files_locked": len(lock["files"]), "package_created": False}))
    else:
        print(json.dumps(build_package(paper, args.repo_root, args.output_dir or paper / "dist")))


if __name__ == "__main__":
    main()
