# APPS / CodeContests Replay adapter review

**Status: PROPOSAL — not adopted or implemented.** Read-only review, 2026-09-20.
No candidate execution, new data download, or change to the original eight-cell
mechanism family. These would be additional program-synthesis domains, not RBR
repair rows with renamed dataset fields.

## 1. Execution semantics are not interchangeable

| Profile | Observed behavior and consequence |
|---|---|
| Current RBR executor | Executes a complete Python stdin program with bwrap, adds a missing terminal input newline, defaults to 6 seconds, compares line/token structure case-sensitively, and permits numeric absolute error ≤1e-4 with zero relative tolerance. This is a local repair-replay protocol, not either benchmark's official judge. |
| APPS upstream evaluator | `fn_name` absent/None selects stdin; list-valued input/output is newline-joined. It wraps source in a function, adds imports, patches stdin and `open`, and uses a 4-second alarm. Its comparison includes stripped strings, NumPy `allclose` defaults, unordered sets and rounding numbers to three decimals. Thus it is not simply exact match or RBR's numeric comparator. Runtime exceptions and timeouts share return code −1; −2 denotes compilation/setup failure. [Pinned evaluator](https://github.com/hendrycks/apps/blob/362aedc3c71cd7d9bd2fc96a6c80e11dbc38c7a5/eval/testing_util.py). |
| CodeContests upstream comparator | Tokenizes whitespace, lowercases strings, checks token counts/order and accepts numeric absolute difference strictly below 1e-5. Integer-looking strings also parse as doubles, so reproduce implementation behavior rather than assuming an integer-only branch. The tester accepts an optional comparator callback. [Comparator implementation](https://github.com/google-deepmind/code_contests/blob/fa7a4f8139aab08362503f3344778eb86901709a/execution/tester_sandboxer.cc), [tester interface](https://github.com/google-deepmind/code_contests/blob/fa7a4f8139aab08362503f3344778eb86901709a/execution/tester_sandboxer.h). |

The reviewed upstream repository commits are pinned above; dataset revisions need
separate locks. CodeContests distinguishes public/private/generated tests and
includes problem time/memory limits plus input_file/output_file metadata. Those
fields do not by themselves implement file I/O or a special judge. Its reference
example sends tests through stdin/stdout and can stop on first failure; that is
insufficient for an EED cache requiring every selected outcome.
[Dataset schema](https://github.com/google-deepmind/code_contests/blob/fa7a4f8139aab08362503f3344778eb86901709a/contest_problem.proto),
[example runner](https://github.com/google-deepmind/code_contests/blob/fa7a4f8139aab08362503f3344778eb86901709a/execution/solve_example.cc).

**Recommendation:** start with an explicitly named isolated Python3 stdin Replay
profile, not a claim of native official evaluation. Lock a comparator separately
for each dataset, with conformance fixtures against its pinned upstream function.
Porting comparison behavior alone does not reproduce source wrapping, interpreter,
limits, imports, file handling or contest runtime. APPS's patched `open` is not
native contest file-I/O support. No per-problem SPJ dispatcher is established by
the reviewed evaluators; CodeContests' callback is only an extension mechanism.

Initially exclude interactive, file-I/O and multiple-valid-output/SPJ problems
unless a pinned, independently validated judge is available. Record eligibility
and exclusion reasons before generation, without filtering by model outcomes.
A generic text comparator can reject a semantically valid constructive answer,
or accept a semantically wrong one. Distinguish “passes released comparison
function,” “passes this Replay profile,” and “satisfies original task semantics.”
The upstream project itself cautions that executions may differ from the original
contest environment. [Official README](https://github.com/google-deepmind/code_contests/blob/fa7a4f8139aab08362503f3344778eb86901709a/README.md).

## 2. Public examples and evaluator leakage

Two visibility protocols are admissible and must be distinguished in a new
prospective lock, following the [dataset expansion clarification](EESD_DATASET_EXPANSION_REVIEW.md):

- **Native-statement-visible:** preserve the original visibility boundary. A
  requirement for ≥4 genuine original public examples applies to this profile,
  not to every Replay experiment. CodeContests `public_tests` can seed that
  inventory; APPS's combined `input_output` does not itself establish original
  visibility. The earlier native-example counts are a narrower screen only.
- **Researcher-defined Replay:** from a pinned, publicly released original test
  pool, deterministically select/order tests and designate four observations as
  model-visible before generation. Disclose their inputs and expected outputs,
  and retain at least six distinct evaluator targets whose answers are absent
  from the prompt. This may reassign tests originally labeled private/generated;
  preserve that original provenance separately from the new Replay role. These
  four observations are not thereby *native public examples*, and the resulting
  evaluation is not an official benchmark score. Lock selection and assignment
  before observing any generated candidate or its execution outcomes. This
  four-observation construction is consistent with the existing RBR/CodeARC
  Replay approach; it does not authorize changing their frozen protocol.

For either profile, inventory **all** examples already present in the original
statement, explanations and sample blocks. Four selected observation slots do
not imply that only four answers are disclosed. Keep statement semantics and
original examples intact; exclude already disclosed input/output pairs from
undisclosed evaluator targets. If that exclusion is not possible, explicitly
limit the estimand to include disclosed targets rather than claiming hidden
outcome generalization. Do not delete arbitrary occurrences of common output
numbers to manufacture privacy, hide test IDs while leaving answers visible, or
use prompt truncation as leakage remediation. Fewer than six established
undisclosed targets fails admission to the proposed four/six private-outcome
profile, without implying that all other prospectively defined studies are
invalid. Original-to-Replay IDs and statement-disclosure audit results must be
sealed. Neither the APPS inventory nor a test's stored role certifies that its
answer is absent from the prompt.

Preserve actual model execution outcomes for all selected tests. After candidate
sealing, EED may receive the six target **inputs** as prediction queries, with
their expected outputs and candidate execution outcomes confined to the
evaluator. This is a public-query/private-outcome Replay estimand, not an
assertion that released tests were never available in model pretraining or an
official hidden-test repair score.

## 3. Minimal extension-only interfaces

Proposed new files/namespaces (none created by this review):

1. `materialize_eesd_replay_extension.py`: separate public/evaluator manifests,
   schema `eesd-stdin-synthesis-public-v1`, dataset key `apps_replay` or
   `codecontests_replay`, revision, source/task/split IDs, statement, four protocol-visible
   observations and provenance (original visibility and Replay role separately). No `buggy_code`, gold solution or incorrect reference
   solution in the generator view. Seal ordered ten-test IDs, evaluator-role
   provenance and statement-disclosure audit separately.
2. `generate_eesd_replay_extension_bank.py`: whitelisted synthesis prompt:
   “Write a complete Python3 program that reads standard input and writes standard
   output,” followed by the statement and four protocol-visible observations. Lock chat template,
   model revision, decoding, input cap, clipping markers and candidate IDs. Use a
   new generation schema; never satisfy `RBR-generated-repair` by fabricating a
   buggy program. Preserve all four slots when bounding text.
3. `execute_replay_stdin` plus extension cache builder: separate execution profile
   and comparator ID/SHA; mount only code/input, never expected answers or judge
   data. Return PASS, WRONG_OUTPUT, COMPILE_ERROR, RUNTIME_EXCEPTION or TIMEOUT
   from actual structured execution events. APPS's combined −1 cannot retrospectively
   separate timeout/runtime: instrument the isolated run. Infrastructure/sandbox
   failures and unsupported judges are explicit errors/statuses, not fabricated
   WRONG_OUTPUT. Preserve all ten outcomes; disable early-stop shortcuts.
4. `analyze_eesd_replay_extension.py`: new sealed cache adapter to generic EED
   prediction/statistics functions, retaining query/source/seed identities and
   validation-only selection. Reuse `codearc_bank.load_bank`'s inventory checksum
   mechanics where schema-compatible, adding an extension-specific profile and
   public-population verifier. Do not call the frozen RBR/CodeARC builder with
   disguised dataset names. Existing reconstruction code hardcodes their schemas
   and source hashes; it is not an extension adapter.

`mechanism_inference.PairedMechanismData` can consume independently verified
extension arrays. Its reuse does not authorize inserting new hypotheses into the
original 8-primary/1560-secondary family. Give extension estimates their own
predeclared scope/families, tables, output roots and coverage; keep original
claims unchanged.

## 4. Required new locks and dependencies

Lock dataset/repository revisions; official split mapping; overlap with RBR/APPS/
CodeContests source URLs and near-duplicate statements; native versus Replay
visibility profile, released test pool, deterministic four/six assignment and
original visibility provenance; statement-disclosure exclusions or explicit
estimand limitations; source eligibility, counts and deterministic representative selection; test
ordering/deduplication; synthesis prompt and token budgets; seed/model matrix;
Python/import environment; stdin encoding/newline behavior; sandbox and resource
limits; comparator order/tolerances/case/NaN/Inf handling; SPJ/file-I/O exclusions;
class mapping; bootstrap estimand and extension multiplicity family.

APPS-native replication additionally needs its wrapping/runtime dependencies and
pinned NumPy behavior. CodeContests-native replication needs its C++/Bazel,
Sandbox2/Abseil stack and data-reader tooling; runtime file-I/O/SPJ wiring remains
separate work. A Replay Python adapter is smaller but requires an honest protocol
label. Existing namespace isolation remains an execution prerequisite; neither
new dataset bypasses the present sandbox gate. Dataset qualification and source
separation, not generator naming, are the first blockers.
