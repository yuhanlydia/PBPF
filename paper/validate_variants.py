#!/usr/bin/env python3
"""Validate source contracts and compile all 108 anonymous manuscript variants."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import json
from pathlib import Path
import re
import shutil
import subprocess

PAPER = Path(__file__).resolve().parent


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def check_sources(paper: Path = PAPER) -> dict:
    title_source = (paper / "variants/titles.tex").read_text()
    titles = dict(re.findall(r"\\PBPFTitleChoice\{(\d+)\}\{([^\n]+)\}", title_source))
    if set(titles) != {str(i) for i in range(1, 13)}:
        raise ValueError("Expected exactly twelve title choices")
    abstract_words = []
    for i in range(1, 4):
        abstract = (paper / f"variants/abstract_{i}.tex").read_text().strip()
        count = len(abstract.split())
        if not 170 <= count <= 210 or "\n\n" in abstract:
            raise ValueError(f"Abstract {i}: expected one 170--210 word paragraph, got {count}")
        abstract_words.append(count)
        intro = (paper / f"variants/intro_{i}.tex").read_text()
        if intro.count(r"\item") != 3:
            raise ValueError(f"Introduction {i}: expected three contribution bullets")
    sources = "\n".join(p.read_text() for p in paper.rglob("*.tex")
                        if p.name != "math_commands.tex" and "build" not in p.parts)
    if sources.count(r"\section{Experimental Setup}") != 1:
        raise ValueError("Expected one shared Experimental Setup")
    if any(token in sources for token in ("/workspace/", "/Users/", "50 -> 84", "84 / 164")):
        raise ValueError("Private path or legacy diagnostic value in manuscript")
    main = (paper / "pbpf_iclr2027.tex").read_text()
    if r"\author{Anonymous authors}" not in main:
        raise ValueError("Manuscript must be anonymous")
    if r"\iclrfinalcopy" in re.sub(r"(?m)%.*$", "", main):
        raise ValueError("Anonymous manuscript enables finalcopy")
    if "human review" not in main or "AI use statement" not in main:
        raise ValueError("Missing explicit AI assistance/review-status disclosure")
    # Unescaped braces must balance; comments are excluded, not rewritten.
    for path in [paper / "pbpf_iclr2027.tex", paper / "experimental_setup.tex",
                 paper / "results_pending.tex", *sorted((paper / "variants").glob("*.tex"))]:
        balance = 0
        for token in re.findall(r"(?<!\\)[{}]", re.sub(r"(?m)(?<!\\)%.*$", "", path.read_text())):
            balance += 1 if token == "{" else -1
            if balance < 0:
                raise ValueError(f"Unbalanced brace: {path.name}")
        if balance:
            raise ValueError(f"Unbalanced brace: {path.name}")
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", (paper / "references.bib").read_text()))
    citations = {key for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", sources)
                 for key in group.split(",")}
    if citations - bibkeys:
        raise ValueError(f"Unresolved citations: {sorted(citations - bibkeys)}")
    return {"titles": len(titles), "abstract_words": abstract_words,
            "contribution_bullets": [3, 3, 3], "cited_keys": len(citations),
            "titles_by_id": titles}


def compile_variant(selectors: tuple[int, int, int], titles: dict) -> dict:
    title, abstract, intro = selectors
    job = f"pbpf-t{title}-a{abstract}-i{intro}"
    result = subprocess.run(["make", f"TITLE={title}", f"ABSTRACT={abstract}", f"INTRO={intro}"],
                            cwd=PAPER, text=True, capture_output=True)
    (PAPER / "build" / f"{job}.validation.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{job}: TeX build failed; see build/{job}.validation.log")
    log = (PAPER / "build" / f"{job}.log").read_text()
    if "undefined" in log or "Overfull" in log:
        raise RuntimeError(f"{job}: unresolved reference or overfull box")
    aux = (PAPER / "build" / f"{job}.aux").read_text()
    match = re.search(r"\\newlabel\{sec:main-end\}\{\{[^}]*\}\{(\d+)\}", aux)
    if not match or int(match.group(1)) > 9:
        raise RuntimeError(f"{job}: main text exceeds nine pages or boundary missing")
    first = subprocess.run(["pdftotext", "-layout", "-f", "1", "-l", "1", str(PAPER / "build" / f"{job}.pdf"), "-"],
                           check=True, text=True, capture_output=True).stdout
    # Physical layout keeps right-aligned small-caps title words in reading order.
    # Remove the official review line-number gutter before comparing text.
    first = re.sub(r"(?m)^\s*\d{3}\s*", "", first)
    if normalized(titles[str(title)]) not in normalized(first):
        raise RuntimeError(f"{job}: wrong title in rendered PDF")
    opening = " ".join((PAPER / f"variants/abstract_{abstract}.tex").read_text().split()[:10])
    if normalized(opening) not in normalized(first):
        raise RuntimeError(f"{job}: wrong abstract in rendered PDF")
    full = subprocess.run(["pdftotext", str(PAPER / "build" / f"{job}.pdf"), "-"],
                          check=True, text=True, capture_output=True).stdout
    intro_opening = (PAPER / f"variants/intro_{intro}.tex").read_text().splitlines()[1]
    if normalized(" ".join(intro_opening.split()[:10])) not in normalized(full):
        raise RuntimeError(f"{job}: wrong introduction in rendered PDF")
    if "Anonymous authors" not in first or "PENDING" not in full or "aiusestatement" not in normalized(full):
        raise RuntimeError(f"{job}: anonymity, pending status, or AI disclosure missing")
    return {"title": title, "abstract": abstract, "intro": intro,
            "main_text_last_page": int(match.group(1)),
            "underfull_boxes": log.count("Underfull"), "status": "passed"}


def invalid_selectors() -> list[dict]:
    cases = [("TitleVersion", "0"), ("TitleVersion", "13"),
             ("AbstractVersion", "0"), ("AbstractVersion", "4"),
             ("IntroVersion", "0"), ("IntroVersion", "4"),
             ("TitleVersion", "1.5"), ("AbstractVersion", "bad")]
    outcomes = []
    for index, (selector, value) in enumerate(cases):
        job = f"invalid-selector-{index}"
        command = rf"\def\{selector}{{{value}}}\input{{pbpf_iclr2027.tex}}"
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                                 "-output-directory=build", f"-jobname={job}", command],
                                cwd=PAPER, text=True, capture_output=True)
        if result.returncode == 0:
            raise RuntimeError(f"Invalid {selector}={value} compiled successfully")
        if value.isdecimal() and "Invalid" not in result.stdout:
            raise RuntimeError(f"Invalid {selector}={value} failed for an unexpected reason")
        outcomes.append({"selector": selector, "value": value, "status": "rejected"})
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    sources = check_sources()
    print(json.dumps({k: v for k, v in sources.items() if k != "titles_by_id"}), flush=True)
    if args.static_only:
        return
    for tool in ("make", "latexmk", "pdflatex", "bibtex", "pdftotext"):
        if not shutil.which(tool):
            raise SystemExit(f"Missing {tool}; static checks passed, PDF variants not compiled")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    (PAPER / "build").mkdir(exist_ok=True)
    variants = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(compile_variant, selectors, sources["titles_by_id"])
                   for selectors in product(range(1, 13), range(1, 4), range(1, 4))]
        for future in as_completed(futures):
            variants.append(future.result())
            if len(variants) % 12 == 0:
                print(f"Compiled and checked {len(variants)}/108 variants", flush=True)
    invalid = invalid_selectors()
    report = {"source_checks": {k: v for k, v in sources.items() if k != "titles_by_id"},
              "compiled_variants": sorted(variants, key=lambda x: (x["title"], x["abstract"], x["intro"])),
              "invalid_selectors": invalid, "formal_experiments": "not_run"}
    (PAPER / "build" / "variant-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(variants)} variants; {len(invalid)} invalid selectors rejected", flush=True)


if __name__ == "__main__":
    main()
