# Paper provenance

This source package is a prospective manuscript. It contains no completed
confirmatory results and does not establish a public preregistration.

## Conference assets

The unmodified files `iclr2027_conference.sty`, `iclr2027_conference.bst`,
`math_commands.tex`, `fancyhdr.sty`, and `natbib.sty` are extracted byte-for-byte
from the [official ICLR 2027 distribution](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip).
The distribution contains no separate LICENSE file. Original copyright and
license notices in the files are retained; this package does not assign a new
license to them. In particular, the vendored third-party style files retain
their own terms. The author's formatting example was used as structural
guidance, not copied as manuscript prose.

Official archive SHA-256:
`0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`.

| Unmodified asset | SHA-256 |
| --- | --- |
| `iclr2027_conference.sty` | `797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6` |
| `iclr2027_conference.bst` | `2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5` |
| `math_commands.tex` | `90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9` |
| `natbib.sty` | `88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d` |
| `fancyhdr.sty` | `b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea` |

The [author guide](https://iclr.cc/Conferences/2027/AuthorGuidelines) and
[AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors) were checked
on 2026-09-14: anonymous submission, nine main-text pages, required AI-use
disclosure, references/appendix excluded, and separate non-counting disclosure
and reproducibility statements. The build does not modify the conference style.
The authors must recheck current rules before actual submission. This package
is not an assertion that a prospective-only manuscript satisfies scientific
submission requirements.

## Scientific inputs and citation correction

The approved design and implementation plan dated 2026-09-14, paper research
notes, dataset provenance notes, and current formal YAML configuration were
the drafting inputs. This release attaches the reviewed configuration snapshots
from implementation commit `4ff9010bfb509aa8f37d79aa94ef9f82b9b5355b`;
the packager refuses an absent or mismatched protocol lock. The final setup
explicitly distinguishes configured identities from materialized artifact
hashes, and shipped launch interfaces from the unshipped production factory.

REx was incorrectly conflated with “Learning to Repair” in an earlier research
note. This manuscript instead cites Tang et al., *Code Repair with LLMs Gives
an Exploration-Exploitation Tradeoff*, [arXiv:2405.17503](https://arxiv.org/abs/2405.17503),
and its [official REx repository](https://github.com/haotang1995/REx), matching
the configured candidate-allocation algorithm. The different Amazon repository
is not cited as REx. The configured revision must still be verified at runtime.

Primary metadata sources are linked in `references.bib`. FIVO author metadata
comes from its arXiv page; VSMC, calibration, and RLEF use PMLR; prefix tuning
and LDB use ACL Anthology. The published ICML RLEF record lists six authors;
its metadata is used consistently rather than mixing that record with the
different author list in the revised arXiv version. RunBugRun cites its
2023 preprint, rather than implying that the pinned snapshot is a 2026 release.
REx, Rollout Roulette, and benchmark adaptations are labeled local paper-spec
or modified-protocol implementations, not audited official reproductions.

## AI assistance and review status

AI assistance covered method/design, mathematical formulation, implementation,
testing, literature discovery/summarization, and manuscript/artifact drafting.
The paper does not assert that a human has reviewed all AI-assisted material.
Formal run measurements remain pending. Software verification and visual
inspection can establish build behavior, not scientific validity or empirical
success. No author identity, private repository URL, or machine-specific path
belongs in the anonymous paper package.

Earlier staged sources underwent software and LaTeX checks. Final integration
and wording reconciliation were not followed by test or build reruns, honoring
the user's instruction to stop verification. The requested archive was packaged
once; its member listing and SHA-256 were inspected as delivery metadata only.
