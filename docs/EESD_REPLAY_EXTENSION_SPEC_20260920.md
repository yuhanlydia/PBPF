# Prospective APPS and CodeContests Replay extension

Status: design adopted before extension generation or execution outcomes;
population admission still requires the checks below. This is a new extension,
not an original preregistration or an official benchmark evaluation.

## Purpose and scope

The user retains four model families and requests broader dataset coverage.
Prepare two additional mechanism domains, `apps_replay` and
`codecontests_replay`, with the same four pinned models and generation seeds
1701, 1702, 1703. Each admitted domain has exactly 200 development and 500 primary
source components, shared by all four models. Existing RunBugRun/CodeARC cells,
generation sources, statistical families, and downstream training scope remain
unchanged. This extension measures prediction/calibration of a fixed execution
tool's categorical outcomes, not official benchmark pass@1 or semantic validity.

## Input versions and test provenance

- APPS: fixed local revision `21e74ddf8de1a21436da12e3e653065c5213e9d1`.
  Study development/primary derive from the official test split; eligible
  official train tasks are reserved, not added to the current SFT study.
- CodeContests: revision `802411c3010cb00d1b05bad57ca77365a3c699d6`, the four
  already selected official-train shards recorded in
  `runs/eesd-data/raw/codecontests/inventory-lock.json`. Do not characterize this
  fixed shard subset as the full official training population.
- APPS uses native string-valued stdin input/output pairs. CodeContests may use
  official public, private, and generated tests, preserving all memberships
  when a pair occurs in multiple pools. Generated provenance is never erased.
- Keep inputs at most 8192 characters and outputs at most 4096 characters,
  measured before whitespace normalization. Deduplicate matching pairs and
  exclude inputs having conflicting expected outputs. No model outcome,
  reference-program success, or generated-program quality selects tasks.
- Retain complete statements up to 6000 characters; never clip statements or
  the four designated input/output pairs to obtain token-fit eligibility.

## Source graph and common population

Build the complete joint source graph before token-fit or representative
selection. Edges record canonical URL identity, normalized-statement identity,
identical substantial test-pair sets, or at least three distinct shared test
pairs. The latter are conservative overlap candidates, not confirmed duplicates.
Keep ineligible nodes and their transitive connections. Quarantine components
touching the actual locked 1221 RunBugRun sources and APPS components spanning
its official train/test boundary. Raw unused RunBugRun overlap is a sensitivity
column, not the primary cross-domain exclusion set.

Probe all eligible members of surviving components against all four pinned
tokenizers. A member must fit 4096 input tokens in every model. Then assign a
component to CodeContests if it contains a fitting CodeContests member; otherwise
assign it to APPS if it contains a fitting APPS official-test member. A component
cannot supply both domains. Within its assigned domain select the fitting member
with the smallest SHA256 of UTF-8 `eesd-replay-representative-v1|` followed by its
stable task ID, breaking a hash tie by task ID. Do not choose the shortest prompt.

Rank selected source components separately per domain by SHA256 of UTF-8
`eesd-replay-split-v1|1701|` followed by domain, `|`, and the stable component ID;
break ties by component ID. First 200 become development, next 500 primary,
remaining components stay reserve. If either domain has fewer than 700 fitting
independent components, stop its admission. Do not change limits or select a
different outcome-dependent population to fill the gap.

## Visibility and generation

Each public prompt contains the complete statement plus four designated
input/output pairs. These four may be newly disclosed from a released test pool;
record their original provenance and do not call them four original public
examples. Prioritize available native/identified public examples, with remaining
choices determined by the frozen input-only selection rule. Select six other
targets whose input is not among the identified originally disclosed inputs.
Seal the exact ordered ten-test identities and selection-code SHA before use.

Unparsed statements may contain additional examples. Report this limitation:
the study withholds six **execution outcomes**, not a certified set of six
originally hidden benchmark answers. Expected outputs, source solutions, target
tests and evaluator records must not be added to the generator's public view.

Use the full synthesis prompt/chat-template/tokenization policy from the public
token probe, after its exact bytes and per-model token IDs are sealed. No buggy
program is fabricated. Generation uses one candidate per source, temperature
0.8, top-p 0.95, maximum 1024 new tokens, and each of the three fixed seeds.
All four models use the same public population. Record input/output token counts,
cap hits and source hashes. A changed prompt or tokenizer invalidates admission.

## Execution and inference boundary

The proposed common execution profile reuses the immutable `execute_stdin`
implementation: Python 3.10.12 candidate runtime, bubblewrap 0.6.1, 6-second
deadline, terminal input newline normalization, case-sensitive line/token
comparison with absolute numeric tolerance 1e-4 and zero relative tolerance.
Runtime/environment/source checks must pass before execution; the present
namespace failure remains an external blocker. Infrastructure failure is not a
model outcome. The builder environment remains Python 3.11.16.

Exclude detected file-I/O, interactive and special-judge cases before generation.
The initial deterministic screen excludes nonempty CodeContests `input_file` or
`output_file`, and statements matching the case-insensitive regular expression
`\binteractive\b|\bspecial\s+(?:judge|checker)\b|\b(?:input|output)\.txt\b`.
Record matching text/reasons. This conservative text screen can exclude negated
mentions and does not certify that remaining tasks have unique valid answers.
Do not claim an unverified native SPJ index mapping is an exclusion certificate.
Enumerate unresolved judge semantics as a limitation. The
comparison tool's PASS is not evidence that a program satisfies the original
contest's judge. Preserve PASS, WRONG_OUTPUT, COMPILE_ERROR, RUNTIME_EXCEPTION,
TIMEOUT as the five measured execution categories, for every selected test.

Only after candidate sealing may the prediction cache receive the ten test
inputs and evaluator-derived outcome labels. Predictions use the four designated
execution observations; subsequent labels are scoring targets. New schemas and
entry points must identify synthesis extensions explicitly rather than disguising
them as RBR repair or CodeARC replay.

The extension has its own eight primary and 1560 secondary hypothesis slots,
following the existing 49-contrast/source-paired/three-fixed-seed definitions,
each family at alpha .05 and 10000 draws. This does not claim joint study-wide
FWER .05 across original and extension families. The secondary p-value resolution
precludes rejection at the first Holm threshold; report that power limitation.
Lock exact contrast/support rules before any extension assessment inference.

## Admission evidence

The final materialization must bind raw versions/hashes, full joint graph,
excluded nodes, task representatives, ten-test provenance/role selection, four
tokenizer identities, exact prompt/token hashes, split selection, and public and
evaluator file checksums. Public records contain only task/source/split identity,
statement and four designated tests. Keep target identities and output strings
in evaluator-only records. Independent verification must confirm 200/500 sources
per domain, no cross-domain component reuse, unchanged full statements, four
public/six target disjoint inputs, and equal populations across models/seeds.

Admission is data preparation. It does not establish candidate generation,
execution scores, favorable results, or completion of the wider experiment goal.
