# Dataset expansion review

Status: candidate review; no additional mechanism domain has been admitted.

The user's latest scope keeps four model families and requests more datasets.
The active mechanism queue remains RunBugRun and CodeARC. HumanEval+ and
MBPP+ are transfer benchmarks, so these four names do not constitute four
mechanism domains.

## Protocol clarification and corrected priorities (2026-09-20)

Earlier review language incorrectly treated four **original statement-visible
examples** as a universal mechanism admission requirement. The existing replay
materializers do not impose that condition:

- `src/pbpf/apbpf/rbr_materialize.py:57–67` hashes tests from `tests_all.jsonl.gz`,
  selects ten, exposes the first four, and marks the remaining six evaluator-only.
  Lines 111–118 identify an exploratory split of official-training sources.
- `src/pbpf/apbpf/codearc_materialize.py:83–96` orders ten invocations by their
  existing indices and exposes the first four under CodeARC-Replay.
- `docs/EESD_ICLR2027_EXPERIMENTS.md` section 1 requires four public executions;
  it does not require four examples originally printed in a problem statement.

Distinguish **native-statement-visible** protocols, preserving original example
visibility, from separately declared **replay** protocols that deterministically
assign four visible and six evaluator-only tests. Previous native-public counts
remain useful lower-bound evidence; they are not absolute replay eligibility
limits. Any replay expansion requires a new prospective lock before generation
and scoring. No additional domain is admitted by this clarification.

The existing **eight cells and their primary/secondary Holm families remain
unchanged**. Additional datasets need an explicitly separate expansion inventory
and statistical plan; do not silently enlarge or pool into the locked families.

1. **APPS-Replay:** first mechanism expansion candidate. Existing inventory finds
   171 train and 2,534 test tasks with ten bounded distinct tests, before complete
   source/judge/visibility validation. The 9 train / 258 test tasks with four
   verified original examples plus six other tests apply only to the
   native-statement-visible screen; the other tests are not certified originally
   hidden. Development allocation and source deduplication must be locked.
   There are 441 test tasks with at least three test pairs shared with RunBugRun,
   which are overlap candidates, not proven duplicate problems.
   Inventory: `runs/eesd-data/raw/apps/eligibility-report.json`.
2. **CodeContests-Replay:** next mechanism inventory candidate, preferably using
   an explicitly declared source-disjoint partition of official train. Official
   valid/test sizes cannot provide 200/500, but this does not rule out replay
   splits of train. Preserve public/private/generated provenance and audit
   overlap with APPS/RBR; different dataset names do not guarantee independent
   sources. Together with APPS-Replay this is the preferred direction for broader
   source coverage, conditional on those audits, not a claim already established.
3. **TACO-Replay:** alternative rather than an automatic independent addition to
   APPS. It explicitly reuses APPS and CodeContests material and has substantial
   measured APPS URL overlap. Its native-example screen does not cap replay
   eligibility; see the separate counts below. Special-judge and split work remain.
4. **LiveCodeBench:** retain as temporal transfer first. Full and lite pools must
   be distinguished; lite removes tests. The native-public inventory below does
   not establish impossibility of every separately declared replay design, but
   no such mechanism expansion is presently proposed for this transfer benchmark.

xCodeEval's official synthesis validation split has only 106 tasks, insufficient
for 200 development without a new split protocol. Official sources:
https://github.com/LiveCodeBench/LiveCodeBench,
https://github.com/FlagOpen/TACO,
https://github.com/ntunlp/xCodeEval/blob/main/program_synthesis.md.

## Admission requirements

- Pin the dataset revision, files, and hashes before candidate generation.
- Count eligible independent sources after length limits and deduplication.
- Lock native-statement-visible versus replay visibility explicitly. For replay,
  pin the test pool, deterministic selection/order and four/six assignment before
  any model outcome is observed. Preserve original test provenance separately.
- Separate model-visible evidence from evaluator-only tests. Audit whether the
  original statement already exposes a held-out test or its answer; predeclare
  example removal or test exclusions where needed. A `hidden` flag alone does
  not establish that isolation.
- Lock source-disjoint development/primary/training populations and compatible
  generation/execution adapters. A new program-generation task is not automatically
  the same repair task as RunBugRun. Unsupported judges must be handled or excluded
  before results, without selecting by model success.
- Audit within-dataset and cross-dataset overlap, including RunBugRun/CodeARC.
- Declare any changed development size or evidence protocol prospectively;
  report that result separately from the current locked mechanism protocol.
- Preserve the existing four-model generation queue while reviewing additions.

This review is not an assertion that six datasets are downloaded, runnable,
or evaluated. There are currently two mechanism domains and two prepared
transfer benchmarks; APPS-Replay and CodeContests-Replay are prioritized mechanism
inventory candidates, and LiveCodeBench remains a transfer candidate.

## Completed LiveCodeBench inventory

All six release_v6 files of `livecodebench/code_generation_lite`, revision
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, have been downloaded and verified
against official LFS SHA256 values. There are 1,055 rows: 602 AtCoder,
444 LeetCode and 9 Codeforces. Actual stored dates span 2023-05-07 to 2025-04-06.
Only 78 rows have at least four supplied public tests (64 stdin, 14 functional).
It cannot supply 200-development/500-primary under the native supplied-public
protocol. This is not a general replay upper bound. No private test payload was
decoded in this inventory. Keep this release as a transfer candidate; any different
visibility protocol would need its own explicit prospective declaration.

Receipt: `runs/eesd-setup/livecodebench-release-v6-receipt.json`.

## CodeContests and TACO: pinned lightweight review (2026-09-20)

In the initial lightweight phase only metadata, documentation and one CodeContests
Viewer row were fetched. The later authorized TACO test-only download and inventory
are described below. No candidate program was executed.
The following are inventory candidates, not admitted mechanism domains.

### CodeContests

- Pin `deepmind/code_contests` revision
  `802411c3010cb00d1b05bad57ca77365a3c699d6`. Hub blob metadata lists 41 Parquet
  files totaling **7,624,659,530 bytes**; downloading the whole repository is
  unsuitable with approximately 6 GiB free. The valid shard is
  `data/valid-00000-of-00001-5e672c5751f060d3.parquet` (**51,829,044 bytes**);
  test is `data/test-00000-of-00001-9c49eeff30aacaa8.parquet`
  (**63,077,400 bytes**). LFS SHA256 values are respectively
  `02e8c1ccedae716f1e43cc813fcb7823c3db666ea92638820aba80e8cef451ab` and
  `aa426cbdb202bf8703b658bcb31fd1878ca7cfd33ca07d3b703dc94ca6a2b651`.
- Fixed metadata contains **13,328 train / 117 valid / 165 test** problems.
  Thus the official valid/test splits cannot meet 200 development / 500 primary,
  even before eligibility filtering. Using a new source-disjoint partition of
  train would require an explicit new split protocol, not an enlarged official
  test-set claim. Such a train-derived replay partition is a legitimate candidate
  extension, consistent with the existing RBR materializer; eligibility is unmeasured.
- `public_tests`, `private_tests`, and `generated_tests` have distinct provenance.
  For a native-public protocol, count four original public and six private tests
  separately. A new replay protocol may instead lock ten eligible tests and assign
  four visible/six evaluator-only, retaining their original provenance; inclusion
  of generated tests must be declared. Only the designated four may be generator
  evidence. The six held-out inputs become mechanism predictor queries after
  candidate sealing, with their outputs restricted to evaluator labels.
- Aizu/AtCoder data comes from CodeNet, giving a concrete shared upstream source
  with RunBugRun; other contest sources also intersect APPS's coverage. This is
  overlap risk, not a measured duplicate count. Match source/problem identifiers,
  normalized statements and test-pair fingerprints before partitioning.
- Stdin execution is the cheapest adapter route, but `input_file`/`output_file`
  tasks must be excluded or explicitly supported. Judge semantics and time limits
  need a declared adapter; do not assume RunBugRun's numeric comparator applies.
  Use the pinned Hub schema to decode categorical source values: its encoded
  values need not equal the original protobuf enum numbers.

Sources: [pinned metadata](https://huggingface.co/datasets/deepmind/code_contests/raw/802411c3010cb00d1b05bad57ca77365a3c699d6/dataset_infos.json),
[dataset card](https://huggingface.co/datasets/deepmind/code_contests/blob/802411c3010cb00d1b05bad57ca77365a3c699d6/README.md),
[original test schema](https://github.com/google-deepmind/code_contests/blob/main/contest_problem.proto).

### TACO

- Pin `BAAI/TACO` revision `d593ed0a2becbbc952230bb89be09189bf1056dc`.
  Use only `ALL/*.parquet`: ten shards total **2,419,844,942 bytes**; Arrow files
  are another representation and should not also be downloaded.
  `ALL/test-00000-of-00001.parquet` is **245,784,461 bytes**, LFS SHA256
  `5d99adc603500c05751aff9f61bed7bbd54a9ad5ea569d108b645346655aae44`.
- Official counts are **25,443 train / 1,000 test**, without a validation split.
  A prospective source-disjoint development holdout from train is needed. These
  totals do not establish native-public or replay eligibility, or 200/500 support.
- Tests are a JSON string `input_output` with `inputs`, `outputs`, and optional
  `fn_name`; there is no explicit public/private test field in the published
  schema. In a native-statement-visible protocol, public examples must be traced
  to statements and unresolved provenance retained. A separately locked replay
  protocol can define four tests as visible, but must identify that new assignment
  explicitly rather than claim those were the dataset's original public tests.
- The official acknowledgements explicitly include material curated from
  **APPS and CodeContests**. Accordingly this cannot be treated as an independent
  domain without overlap auditing; CodeNet-derived overlap with RBR is also a
  possibility through CodeContests. `url`, `source`, statement and test hashes
  provide matching keys, with transitive components retained across datasets.
- Mixed stdin/function-call problems require separate adapters. The official
  project published a special-judge test-problem list; exact-output scoring would
  be invalid for these cases without the declared judge. Pin that supplementary
  list separately at GitHub revision
  `245eba3beb2d23a07082307de303de2589e4321a`, `output_spj.jsonl` (42,672 bytes).
  Exclude unsupported cases before generation, rather than after seeing failures.
- The Viewer `/splits?dataset=BAAI/TACO` returned HTTP 501 (legacy script dataset)
  during this review. Use its existing pinned Parquet files directly, not remote
  dataset-script execution; parse `input_output` with JSON, not `eval`.

Sources: [pinned card/schema](https://huggingface.co/datasets/BAAI/TACO/blob/d593ed0a2becbbc952230bb89be09189bf1056dc/README.md),
[official source acknowledgements and evaluator updates](https://github.com/FlagOpen/TACO/blob/245eba3beb2d23a07082307de303de2589e4321a/README.md),
[special-judge inventory](https://github.com/FlagOpen/TACO/blob/245eba3beb2d23a07082307de303de2589e4321a/output_spj.jsonl).
File sizes and revisions were checked using the official Hub dataset metadata
endpoint with `blobs=true`; these are compressed file sizes, not memory estimates.

### Historical bounded inventory plan (completed/stopped as recorded below)

1. Save the two pinned Hub metadata responses (paths, sizes, LFS hashes), cards,
   and the pinned TACO special-judge list into a small inventory receipt.
2. First inspect TACO test and the first train shard using HTTP Range: fetch the
   Parquet footer, inspect row-group compressed column sizes, then project only
   `question`, `input_output`, `url`, `source`, `name`, `starter_code`. Exclude
   solution columns. Use an explicit **64 MiB total transfer cap**, at most 100
   rows per split and a small in-memory cache; refuse servers ignoring Range or
   chunks exceeding the cap. Record the exact revision, shard, row-group/row
   offsets, fetched ranges and observed row counts. This is a feasibility sample,
   not a population eligibility estimate. Do not execute any task or reference.
3. Emit counts for malformed/missing tests, stdin versus function calls, bounded
   distinct tests, statement-verified public examples, unresolved visibility,
   special-judge exclusions and overlap candidates against local APPS/RBR.
   Do not claim full-shard SHA verification from partial Range reads.
4. Proceed to a separately budgeted projected full inventory only if visibility
   can actually be established. For CodeContests, resolve the insufficient
   official split sizes before spending bandwidth on an eligibility census.
   No selection may use model success or evaluator outcomes.

That inventory phase has concluded. Current priorities are APPS-Replay and
CodeContests-Replay under new source/split/visibility locks; the official-split
size blocker applies to unchanged CodeContests valid/test, not all possible
replay partitions. No candidate is currently certified as a new 200/500 domain.
The existing formal matrix, Holm families and frozen generation remain unchanged.

### Completed TACO test inventory

The 64 MiB range phase stopped after 102,689 bytes: the test shard has one
1,000-row group and its `input_output` compressed column alone is 228,911,210
bytes. That original receipt is preserved. A separate 300 MiB test-only phase
then downloaded the 245,784,461-byte shard and verified its full official LFS
SHA256. It read only the six inventory columns; no solutions were decoded,
train shard downloaded, or program executed.

The complete test inventory contains 945 stdin and 55 function-call tasks.
932 stdin tasks have at least ten bounded distinct input/output pairs after
removing inputs with conflicting outputs. The conservative statement parser
verifies four public examples and six other tests for 36 tasks under the
native-statement-visible screen. This is not a replay eligibility upper bound. These
public-example counts are lower bounds, and the other tests are not certified
hidden. There are 372 canonical-URL overlaps with the local APPS inventory and
104 tasks with at least three shared test pairs with RunBugRun; the latter are
overlap candidates rather than confirmed duplicate source problems.

The official SPJ list flags 218 row indices, but its ordering equivalence to the
pinned Hub revision is not certified. Provisionally applying those exclusions
leaves 34 of the 36 joint candidates, then 11 after the observed APPS/RBR overlap
exclusions. These numbers do not establish an eligible primary population.
No training split or development population has been inventoried.

Receipt: `runs/eesd-data/raw/taco/inventory-receipt.json`; per-task inventory:
`runs/eesd-data/raw/taco/eligibility-inventory.jsonl`. The test shard remains a
reviewed expansion candidate, not a new formal mechanism domain.

### TACO replay screen from the existing test inventory

The following counts were computed from the same 1,000-row inventory, without
requiring four original statement examples or executing any code:

| Replay screen | Task rows | Distinct nonempty canonical URLs |
|---|---:|---:|
| Valid matching IO counts, stdin, at least ten bounded unambiguous pairs | 932 | 896 |
| Above, provisionally excluding official-index SPJ flags | 719 | 684 |
| Above, also excluding RBR three-pair overlap candidates | 623 | 590 |
| Above, excluding both APPS URL overlaps and RBR candidates | 432 | 399 |

URL counts are not certified independent source components. SPJ index alignment,
statement leakage, transitive duplicates, judge semantics and development sources
remain unaudited. The final row falls below 500 under that particular simultaneous
exclusion policy, while the earlier rows leave replay feasibility open. Neither
fact certifies final eligibility. Do not count overlapping APPS/TACO populations
as independent domains simply to increase the dataset total. Prioritize inventory
of APPS-Replay and CodeContests-Replay, retaining TACO as an alternative whose
source contribution must be demonstrated.
