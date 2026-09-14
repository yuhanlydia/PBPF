"""Paper interface checks; no formal results or release artifacts are produced."""
from pathlib import Path
import importlib.util
import io
import json
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def test_required_manuscript_assets_exist():
    required = ["pbpf_iclr2027.tex", "references.bib", "math_commands.tex",
                "iclr2027_conference.sty", "iclr2027_conference.bst",
                "natbib.sty", "fancyhdr.sty", "README.md", "Makefile",
                "PROVENANCE.md", "results_pending.tex", "variants/titles.tex",
                "experimental_setup.tex"]
    assert not [name for name in required if not (PAPER / name).is_file()]


def test_variant_interfaces_and_text_contracts():
    titles = (PAPER / "variants/titles.tex").read_text()
    assert len(re.findall(r"\\PBPFTitleChoice\{\d+\}", titles)) == 12
    for i in range(1, 4):
        abstract = (PAPER / f"variants/abstract_{i}.tex").read_text().strip()
        assert 170 <= len(abstract.split()) <= 210
        assert "\n\n" not in abstract
        intro = (PAPER / f"variants/intro_{i}.tex").read_text()
        assert intro.count(r"\item") == 3
    source = (PAPER / "pbpf_iclr2027.tex").read_text()
    assert r"\author{Anonymous authors}" in source
    assert r"\iclrfinalcopy" not in re.sub(r"(?m)%.*$", "", source)
    assert r"\input{experimental_setup}" in source
    assert r"\PackageError{PBPF}" in source


def test_pending_only_results_and_citation_integrity():
    source_files = [p for p in PAPER.rglob("*.tex") if p.name != "math_commands.tex"]
    sources = "\n".join(p.read_text() for p in source_files)
    assert "50 -> 84" not in sources and "84 / 164" not in sources
    assert "/workspace/" not in sources and "/Users/" not in sources
    assert sources.count(r"\section{Experimental Setup}") == 1
    pending = (PAPER / "results_pending.tex").read_text()
    assert "PENDING" in pending
    assert not re.search(r"\\newcommand\{\\Result\w+\}\{\d", pending)
    keys = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "references.bib").read_text()))
    cited = set()
    for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", sources):
        cited.update(group.split(","))
    assert cited <= keys
    assert {"delmoral2006", "gilks2001", "maddison2017", "li2021prefix",
            "guo2017", "selfdebug", "ldb", "rex", "rlef", "roulette",
            "runbugrun", "evalplus", "livecodebench", "codearc"} <= cited


def test_packager_refuses_unfrozen_configuration(tmp_path):
    script = ROOT / "scripts/package_paper.py"
    assert script.is_file()
    spec = importlib.util.spec_from_file_location("package_paper", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.build_package(tmp_path / "unlocked-paper", ROOT, tmp_path / "output")
    except ValueError as exc:
        assert "protocol lock" in str(exc).lower()
    else:
        raise AssertionError("An unreviewed configuration must not create a release package")
    assert not list(tmp_path.iterdir())


def load_packager():
    spec = importlib.util.spec_from_file_location("package_paper", ROOT / "scripts/package_paper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_members_are_sorted_fixed_timestamp_and_deterministic():
    module = load_packager()
    first = module.archive_bytes({"z.tex": b"last", "a.tex": b"first"})
    second = module.archive_bytes({"a.tex": b"first", "z.tex": b"last"})
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["a.tex", "z.tex"]
        assert archive.read("a.tex") == b"first"
        assert all(item.date_time == (2026, 1, 1, 0, 0, 0) for item in archive.infolist())
        assert all(item.external_attr >> 16 == 0o100644 for item in archive.infolist())


def test_archive_rejects_escaping_paths():
    module = load_packager()
    for name in ("/private.tex", "../private.tex", "paper/../../private.tex"):
        try:
            module.archive_bytes({name: b"payload"})
        except ValueError:
            pass
        else:
            raise AssertionError(f"Escaping path accepted: {name}")


def test_protocol_lock_requires_acknowledgment_and_rejects_mutation(tmp_path):
    module = load_packager()
    repo = tmp_path / "repo"
    paper = repo / "paper"
    paper.mkdir(parents=True)
    for name in (module.DESIGN, module.CONFIG, "configs/iclr/models.yaml"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture: prospective\n")
    try:
        module.lock_protocol(paper, repo)
    except ValueError as exc:
        assert "acknowledge" in str(exc)
    else:
        raise AssertionError("Protocol locked without review acknowledgment")
    module.lock_protocol(paper, repo, acknowledged=True)
    module.validate_lock(paper, repo)
    (repo / "configs/iclr/models.yaml").write_text("fixture: changed\n")
    try:
        module.validate_lock(paper, repo)
    except ValueError as exc:
        assert "Stale protocol lock" in str(exc)
    else:
        raise AssertionError("Stale protocol lock accepted")


def test_real_paper_sources_package_under_temporary_fixture_lock(tmp_path):
    """Exercise successful packaging without locking or releasing the real protocol."""
    module = load_packager()
    repo = tmp_path / "fixture-repo"
    paper = repo / "paper"
    names = [*module.REQUIRED]
    names += [f"variants/{kind}_{i}.tex" for kind in ("abstract", "intro") for i in range(1, 4)]
    for name in names:
        target = paper / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PAPER / name, target)
    for name in (module.DESIGN, module.CONFIG, "configs/iclr/models.yaml"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture: not_the_formal_protocol\n")
    module.lock_protocol(paper, repo, acknowledged=True)
    first = module.build_package(paper, repo, tmp_path / "fixture-output-one")
    second = module.build_package(paper, repo, tmp_path / "fixture-output-two")
    assert first["sha256"] == second["sha256"]
    with zipfile.ZipFile(first["archive"]) as archive:
        assert set(names) <= set(archive.namelist())
        assert "MANIFEST.json" in archive.namelist()
        assert archive.namelist() == sorted(archive.namelist())
        assert not any(name.endswith((".pdf", ".aux", ".log")) for name in archive.namelist())
        manifest = json.loads(archive.read("MANIFEST.json"))
        for name, expected_hash in manifest["files"].items():
            assert module.sha256(archive.read(name)) == expected_hash
    with (paper / "README.md").open("a") as handle:
        handle.write("\nPrivate path: /workspace/private-example/secret.txt\n")
    try:
        module.build_package(paper, repo, tmp_path / "must-not-publish")
    except ValueError as exc:
        assert "Private path" in str(exc)
    else:
        raise AssertionError("Concrete private path accepted")
    assert not (tmp_path / "must-not-publish").exists()


def test_equivalent_integer_selectors_render_the_selected_title(tmp_path):
    """Leading zeros and a plus sign are integers, not undefined macro names."""
    if not shutil.which("pdflatex") or not shutil.which("pdftotext"):
        import pytest
        pytest.skip("TeX and Poppler are optional outside paper-build jobs")
    command = (r"\def\TitleVersion{01}\def\AbstractVersion{+1}"
               r"\def\IntroVersion{01}\input{pbpf_iclr2027.tex}")
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                             f"-output-directory={tmp_path}", "-jobname=canonical-selectors", command],
                            cwd=PAPER, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:]
    first = subprocess.run(["pdftotext", "-layout", "-f", "1", "-l", "1",
                            str(tmp_path / "canonical-selectors.pdf"), "-"],
                           capture_output=True, text=True, check=True).stdout
    first = re.sub(r"(?m)^\s*\d{3}\s*", "", first)
    actual = re.sub(r"[^a-z]", "", first.lower())
    assert "afailedtestisnotadiagnosisparticlebeliefprogramfilteringforcoderepair" in actual
