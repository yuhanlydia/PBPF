# PBPF ICLR 2027 prospective manuscript

This is a buildable, anonymous research draft, not a completed experimental
report or a submission-ready claim of empirical improvement. All result slots
are explicitly pending. Official style assets are unmodified.

## Build and select a narrative

Requirements: a TeX installation with `pdflatex`, `bibtex`, `latexmk`, and
standard AMS, Times, hyperref, booktabs, and array packages. Python 3.11+ is
needed only for validation and packaging tools.

```bash
make
make TITLE=3 ABSTRACT=2 INTRO=2
make variants
```

The output is `build/pbpf-t1-a1-i1.pdf` for the recommended combination.
Selectors are integers `TitleVersion=1..12`, `AbstractVersion=1..3`, and
`IntroVersion=1..3`. Defaults are all 1. Invalid selectors fail compilation.
`make variants` builds every one of the 108 combinations and checks six
out-of-range boundaries plus fractional and nonnumeric selectors. `make` also works after unpacking the source
archive without access to the implementation repository.

There are exactly twelve titles in `variants/titles.tex`; the first is
recommended. Abstracts and introductions are complete independent alternatives:

| Story | Main question | Use condition |
| --- | --- | --- |
| 1 (recommended) | Failed test versus diagnosis | Primary framing, without an unobserved success claim |
| 2 | Whole-sequence coherence | Prefer only if the eventual coherence controls support it |
| 3 | Compact belief with frozen actor | Prefer only if the eventual model replications support it |

Every abstract is 170–210 whitespace-delimited words and one paragraph. Every
introduction ends with exactly three contribution bullets. The arithmetic
AA/BB example illustrates a distributional identity; it is not a measurement.
The single Experimental Setup is shared by all variants.

## Evidence and release gates

Never replace `results_pending.tex` from a smoke test or legacy output. A future
empirical revision requires immutable formal result artifacts, completed gates,
denominators, intervals, model/dataset cells, seed provenance, and resource
ledgers. This package deliberately does not auto-import results.

This release reconciles the Experimental Setup with implementation commit
`4ff9010bfb509aa8f37d79aa94ef9f82b9b5355b`. For a future release, review the
configuration and reconcile the setup before running from the repository root:

```bash
python scripts/package_paper.py --lock-protocol --acknowledge-config-reviewed
python scripts/package_paper.py
```

The first command snapshots the reviewed formal YAML files and approved design
into `paper/protocol/` and writes `paper/protocol-lock.json`. It is an explicit
post-review operation, not automatic evidence of review. The second command
refuses missing, stale, or mismatched locks, writes sorted fixed-timestamp
`paper/dist/PBPF_ICLR_2027.zip`, and emits a SHA-256 manifest. Neither is created
before configuration review. Repeated packaging of identical input bytes is
deterministic. Auxiliary files, PDFs, logs, caches, private paths, and benchmark
payloads are excluded from the source ZIP.

The staged source previously passed all 108 narrative combinations, eight
invalid-selector checks, and nine packaging tests. At the user's instruction,
tests and LaTeX builds were not rerun after final configuration reconciliation.
The release ZIP therefore contains sources, not a newly compiled release PDF;
historical staging checks do not certify the final edited layout or full
production execution. All scientific results remain pending.

The AI-use statement must be checked for completeness before submission. It
truthfully describes AI assistance without claiming completed human review.
See `PROVENANCE.md` for primary sources and the corrected REx attribution.
