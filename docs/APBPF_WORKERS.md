# A-PBPF stage-worker contract

The repository bundles adapters for all twenty-one real stages: `materialize`,
`hard_bank_lock`, `execution_cache`, `hard_bank`, `hard_bank_gate`,
`train_baselines`, `train_belief`, `baseline_fairness_gate`, `association`,
`association_gate`, `pair_invariance_gate`, `oracle_headroom`,
`oracle_headroom_gate`, `active_testing`, `active_testing_gate`, `selection`,
`selection_gate`, `repair`, `replication`, `replication_gate`, and `paper_tables`.
Only materialization has completed as a real two-domain stage so far. Public
candidate import has also been checked on all 1112 CodeARC source groups; worker
contract fixtures are separate from scientific execution evidence. The full local
pipeline is queued behind unfinished RBR generation and execution; complete
adapter coverage does not mean a completed or scientifically validated run.
The site template intentionally leaves every command unprovisioned. Existing
research scripts need explicit adapters to satisfy this contract; do not simply
point a stage at an unrelated script and treat its exit status as evidence.

The independent schema is `apbpf-iclr-v1`; legacy `pbpf-iclr` is unchanged.
`local_exploratory` uses real workers and permanently labels every stage
`exploratory-predeclared`, including before any gate fails. It cannot become
main-table eligible, even if every descriptive gate later passes.
`local_24gb`, `slurm_h200x16`, and `real` use only real commands. Profile names
describe operator resources; the runner itself does not allocate GPUs or submit
Slurm jobs. Slurm adapters must wait for completion and return the actual worker
result. `local_smoke` is the sole fake backend and is always
`smoke-only-no-claim`.

## Provisioning and invocation

Copy `configs/apbpf/site_local_24gb.example.yaml` to an operator-owned location.
Supply three existing directories (`dataset_root`, `model_cache`,
`artifact_cache`), the working directory, every stage command, and an immutable
SHA-256 worker identity for every command. The identity may be the adapter-file
digest, immutable container digest, or a canonical bundle digest; it enters the
run fingerprint, so changing worker code requires a new run identity. When a
local adapter is declared under `worker_files`, `doctor` recomputes its SHA-256
and refuses a mismatch. Immutable container/bundle identities remain an operator
attestation and must be independently audited. Command
values are argv arrays, never shell snippets. Never put access tokens or other
credentials in argv because commands are recorded in run artifacts; inject them
through the operator's secret manager or inherited environment instead. The exact tokens `{request}` and
`{result}` are replaced by absolute paths. No other interpolation occurs.
Alternatively workers may read the environment variables below.

```bash
pbpf-apbpf doctor --profile local_24gb --site /absolute/apbpf-site.yaml
bash scripts/run_apbpf_iclr.sh --profile local_24gb --site /absolute/apbpf-site.yaml --resume
pbpf-apbpf report --profile local_24gb --site /absolute/apbpf-site.yaml
pbpf-apbpf verify --profile local_24gb --site /absolute/apbpf-site.yaml
pbpf-apbpf rerun-stage --profile local_24gb --site /absolute/apbpf-site.yaml --stage association
```

Set `PBPF_APBPF_PYTHON` to an installed project Python if `.venv/bin/python`
is absent. `PBPF_APBPF_SITE` supplies the site path when `--site` is omitted.
All commands default to `configs/experiments/apbpf_iclr2027.yaml` and
`--output-root runs/apbpf`. `doctor` is read-only: it lists every missing command,
missing directory, and unavailable executable and exits 2 when unprovisioned.
Readiness does not validate adapter scientific correctness or start experiments.

## Implemented local materialization

`scripts/run_apbpf_materialize_worker.py` reads the three provisioned site paths
and the runner-owned request. The dataset directory must contain
`runbugrun-v0.0.1/`, `project_codenet/problem_descriptions.tar.gz`, and `codearc/`.
It validates the raw checksums and writes separate public/evaluator directories
for both domains below the attempt's `outputs/`. Reference programs and future
calls are evaluator-only. RBR descriptions are read directly from the pinned
tar archive, so mutable extracted HTML cannot change the input silently.

This worker requires `local_exploratory`: the generated RBR bank uses a new
source partition of the 1221 eligible official-training problems, with
321 training, 400 development, and 500 primary sources. Official test-only
sources remain excluded. The earlier exploratory belief checkpoints saw some
of these problems; downstream workers must train new belief/utility models
on the new fitting sources and bind checkpoint identities to this manifest.
No source is removed based on reference-program or generated-candidate success.
The worker materializes CodeARC from its pinned raw data as well, preserving
its 212/400/500 source partition.

For incremental provisioning, `doctor`, `run`, and `rerun-stage` accept
`--through-stage STAGE`. Readiness then covers that stage and its ordered
predecessors. Execution stops at the requested stage, retaining the ordinary
gate-stop rules; missing later stages remain pending. `verify` always requires
all 21 stages. Omit this option for the full pipeline. A partial failure's
recorded retry commands preserve the same terminal stage.

Provision `commands.materialize` as the installed Python plus the materialize
worker path, and set its SHA-256 in `worker_revisions.materialize` and its path
in `worker_files.materialize`. Other stage commands may remain null for:

```bash
pbpf-apbpf doctor --profile local_exploratory --site /absolute/site.yaml --through-stage materialize
pbpf-apbpf run --profile local_exploratory --site /absolute/site.yaml --through-stage materialize --output-root /absolute/runs
```

The completed root stage proves data preparation and provenance only. It does
not prove generation, execution, hard-bank quality, inference, or efficacy.

## Declared candidate reuse and the first five stages

The materializer optionally accepts `--candidate-cache-manifest PATH` and
`--candidate-cache-sha256 SHA256`. Both exact values must appear in the frozen
site command. The manifest has schema `apbpf-public-generation-cache-v1` and a
`domains` object with `rbr` and `codearc` lists; each entry contains only an
absolute `path` and the frozen SHA-256 of that bank's `complete.json`.

This is an explicit public candidate-data import at the root of the new run.
It records `fresh_generation: false`; it does not claim that generation occurred
again or that previous primary exposure disappeared. The imported generation
must match the primary model revision, seed, fixed eight-candidate budget,
decoding parameters, and public task bytes. It must cover every source exactly
once across train/development/primary. CodeARC uses the first task representative
of each source in materialized order, matching its original generator; its 1114
task IDs represent 1112 source components. Uniform syntax-based extraction is
required. Existing raw and bounded prompt policies remain recorded in each bank.

Only `run.json`, `complete.json` and declared candidate JSON files are copied,
byte-for-byte, into the materialize evidence inventory. Hidden fields and
undeclared side files are rejected or excluded. In particular, old evaluation
reports and gate outcomes are never imported. The candidate manifest's own
checksum enters the configuration fingerprint through the stage command.
All four downstream adapters consume only exact same-run stage artifacts.

`run_apbpf_bank_lock_worker.py` executes all 500 primary groups in each domain
inside the public-only evaluator sandbox. It creates separate domain locks and
a combined 1000-source v2 lock. `run_apbpf_execution_cache_worker.py` verifies
both complete primary locks before opening any evaluator task file, then freshly
executes all training/development candidates and the primary hidden tests.
It also merges the exact primary visible/hidden records and publishes
`rbr-full-cache.json` and `codearc-full-cache.json`, with all candidates retained
and primary sources mapped only to the fitter's test split. Six future expected
answers are redacted before feature extraction. Separate `*-development-only-cache.json`
artifacts exclude primary sources and split original training into fitting and
validation; their test slot represents original development assessment.
The execution index binds these caches, their source counts, and their hashes.
`run_apbpf_hard_bank_worker.py` audits all 800 development groups and all 1000
locked primary groups; domain-level development counts accompany the combined
pilot. `run_apbpf_hard_bank_gate_worker.py` binds the resulting decision to the
exact lock dependency. The runner independently recomputes the unchanged gate.
A failing pilot count remains a sealed negative report and stops ordinary runs.

`scripts/run_local_apbpf_bank_prefix.py --run-root /absolute/local/longgoal
--output /absolute/new-prefix-attempt` waits for the declared Qwen inventories,
checks the source files frozen when queued, creates the cache manifest and site
overlay, then launches real execution through `hard_bank_gate`. It provisions
only those five stages. A scientific gate failure is retained with exit 20;
later stages and full-DAG verification remain incomplete. Candidate import and
the new prefix are always exploratory, including if the hard-bank gate passes.

To additionally provision actual checkpoint training, pass
`--through-stage train_belief`. `--continue-exploratory` explicitly permits
training after a failed hard-bank gate and preserves the failed-gate lineage;
it does not change thresholds or permit confirmatory claims. A waiting prefix
must be archived and redeclared if captured source files change before launch.

`run_apbpf_train_worker.py --kind baselines` and `--kind belief` implement the two
training stages. Each consumes the declared full execution caches for both
domains and fits seeds 1701–1703. Defaults are 1000 steps, batch size 64, learning
rate 0.0003, feature width 256 and hidden width 192. These exploratory workers use
frozen hash-text features, not code-model semantic embeddings. The frozen source
and site command record the budget; protocol particle count and loss weights
come from the request. Only training outcomes enter optimization; original
development outcomes select checkpoints and Dirichlet smoothing. Primary
predictions are produced after fitting. Future expected-answer text is absent.

The baseline stage saves all three neural predictor state dictionaries and the
tuned Dirichlet parameter. The belief stage saves the factored model checkpoint.
Both preserve prediction matrices, primary candidate/source order, complete
training histories, and cache identities. These stages do not decide scientific
gates or claim cross-fitted selection performance. Contract fixtures exercise
both training workers but do not count as real stage runs.

`--through-stage pair_invariance_gate` also provisions the next four adapters.
The fairness worker compares all four baseline arms against the exact belief
prediction population and checks matching data, features, optimization budgets,
development checkpoint selection, and source order. The association worker
restores each actual belief checkpoint and verifies its aligned predictions,
then replays outcome-only shuffling, joint reversal, presentation permutation,
orderless history, masked semantics, wrong-candidate and random-latent controls.
Every domain retains all 500 primary sources and eight candidates per source.
Prespecified ambiguity strata are descriptive and cannot replace the population.

The fixed-seed estimand averages paired per-example losses over seeds
1701–1703, not probabilities or a selected best seed. Each source is one bootstrap
cluster across all seeds, with 10000 draws; seeds are not independent new sources.
Per-seed results and all raw prediction arrays remain available. Fairness uses
the minimum aggregate gap and minimum confidence lower bound across domains for
each comparator, requiring both domains to pass. Association and pair-invariance
use each domain's full-population aggregate directly. All original numerical
thresholds remain unchanged and the runner recomputes each gate decision.

The standalone `run_local_stage_training_diagnostic.py` can exercise these
trainers and controls on an existing, checksummed full cache while complete
two-domain generation is pending. Its output is explicitly a standalone
exploratory diagnostic and never counts as a sealed DAG stage. The replication
bridge described below supplies the declared cross-family input provenance.

`--through-stage active_testing_gate` provisions the oracle and active-query
workers as well. Association produces an evaluator-only assessment bundle with
the exact full cache and all three belief checkpoints. The association gate,
oracle stage and oracle gate forward byte-identical bundles as declared output
artifacts. Downstream workers consume only their direct dependencies; no implicit
ancestor lookup or untracked external result import is used. These bundles must
never be mounted in a generator sandbox.

Oracle replay exhausts canonical subsets at budgets 1, 2 and 4, using the same
proposal noise and resampling uniforms. Fixed and uniformly random subsets use
the same canonical ordering. The local exploratory oracle gate is predeclared
at budget 2; all budgets remain reported. With only four public tests, the
four-test canonical oracle equals fixed testing. This is a limitation of the
current materialization, not evidence that test selection has no value in a
larger pool.

Active replay uses the trained joint SMC posterior, a nuisance-marginalized
diagnostic MI selector, and fixed/random/predictive-entropy/joint-particle-MI
controls. Policies receive only four public feature/outcome slots; six future
targets enter evaluator scoring separately. Every fixed-budget policy observes
exactly its declared number of distinct public tests. At budget4, the pool
allows order differences only. The gate uses advantage over both fixed and
random testing, without changing the 0.03 threshold.

Stopping thresholds are chosen on original development sources before primary
replay. The chosen thresholds and complete threshold--quality curves remain
separate from the fixed-budget table. A savings claim requires at least25%
fewer observations and no worse primary NLL than fixed, random and same-policy
four-test references, with nonnegative source-cluster lower bounds. All budgets
count cached public observations; no wall-clock or physical execution savings
are claimed. Failed upstream gates continue to block confirmatory claims.

`--through-stage selection_gate` additionally provisions actual utility fitting
and candidate selection. It binds the original hard-bank hidden success labels,
full training cache and all three belief/baseline checkpoints. Utility heads
consume candidate/task text and four public observations only. The six future
outcomes supply fitting labels on training sources, checkpoint selection on
development sources, and final scoring on primary sources. Primary labels never
enter model fitting. All 500 sources and eight candidates per source are retained
for each domain. The comparator is selected using five source folds, with no
overlap between comparator-choice and assessment sources. It includes fresh and
pretrained deterministic neural heads, tuned Dirichlet and visible pass rate.
Each neural head gets the same 1000-step utility budget; checkpoint selection
uses original development sources only. Three-seed paired differences receive
10000 whole-source bootstrap draws. Both domains must exceed the unchanged
0.03 absolute Pass@1 threshold with positive confidence lower bounds.

`run_local_selection_diagnostic.py` can exercise selection on completed standalone
model outputs while the full two-domain bank is pending. It remains exploratory.
`run_local_codearc_semantic_debug.py` separately excludes all original primary
sources and compares lexical and frozen Qwen features at the same 512 dimensions
and 1000-step training budget. Only verified label-free public text is mounted
in the feature extractor sandbox; execution evidence and reference answers are
excluded. GPU2 extraction waits for the declared generation process to finish.

Repair prerequisites are now emitted by `execution_cache` as three separate
artifacts per domain under `repair-materials/`: `training-targets.json` contains
reference programs for original training and development sources only;
`evaluator-tests.json` preserves the exact ten-test execution protocol, including
CodeARC expected exceptions; `public-context.json` contains four public examples
only. No primary reference program is exported. Targets are used for training
gradients or development checkpoint selection according to their recorded split.
The evaluator artifact must never be mounted in an actor process.

The selection gate forwards original per-domain/per-seed selection reports as
declared artifacts, so repair can use the actual selected candidate per source
without reading an undeclared ancestor. These reports contain evaluator labels
and also must never be mounted in the actor. `repair_packets` whitelists four
public observations, keeps the selected candidate for every primary source,
and computes eight diagnosis-only particles from the original belief checkpoint.
Training/development rows carry supervised targets; primary rows have none.
`prepare_apbpf_repair_packets.py` exercises this preparation on completed
standalone CodeARC evidence. Packet preparation is not projector training,
repair generation, new execution, or a completed `repair` DAG stage.

`--through-stage repair` now provisions the real supportive repair worker. It
checks association/cache/model lineage, builds actor-only packets, trains a
diagnosis-only eight-token prefix on frozen pinned Qwen, generates all six
repair arms, seals each seed's complete generated inventory before opening
private tests, and freshly executes every generated program. Both domains use
all three seeds and every one of the 500 particle-selected primary sources;
this is 3000 generated repairs per domain/seed. GPU0 is assigned after Qwen
bank generation finishes; an occupied device is never preempted.

The fixed projector budget is 1500 steps, AdamW learning rate0.0003, two sampled
components per training sequence, context cap1536, eight prefix tokens, token
RMS cap0.02 and diagnosis-dependent delta RMS cap0.002. Normalization is fitted
on training posteriors only. Checkpoint selection runs every250 steps on the
lexicographically first candidate per original development source. Oversized
training/validation sequences are recorded as token-cap exclusions. Primary
sources are never dropped; generation uses a matched4096-token prompt cap and
512-token continuation cap, recording original lengths, truncation and actual
generated token IDs. The token-remix fault remains intentionally incorrect and
slower than fixed-component decoding. It is not a valid serving comparator.

`run_local_packet_repair.py` runs the same actor in public-only generator
sandboxes. Its companion `run_local_packet_repair_evaluation.py` verifies the
entire source/arm inventory, writes a pre-hidden lock, and scores new programs
under the original six-second evaluator timeout. Small real-data engineering
smokes are explicitly labeled and never become scientific or stage evidence.
The staged adapter and standalone runs remain exploratory after failed gates.

`build_apbpf_full_replay_cache.py` verifies all four completed replay banks,
the pinned generator identity, public/evaluator manifests, phase-specific
execution records and the primary pre-hidden lock before writing a redacted
full cache and source proof. `run_local_full_replication_cell.py` waits for one
complete domain/family replay, then uses the same three-seed training,
counterfactual replay and utility-selection implementations as the existing
full-population diagnostics. Each cell retains all500 primary sources and
4000 candidates. Standalone cells are not sealed replication stages.

The replication gate adapter requires every locked domain/family cell, all
three seeds and full populations. Its numerical rule remains positive
association and positive selection advantage in every cell; it does not
replace stricter upstream margin/CI gates. The paper-tables adapter reads its
eight direct dependencies and exports gate decisions, unchanged gate metrics,
per-seed selection/repair results and replication effects. It preserves failed
gates and permanently marks tables ineligible for confirmatory main-table use.
Missing confidence bounds are never inferred from lower bounds or averaged
across seed reports. The full supervisor provisions all 21 stages, including
replication and both downstream adapters; the older prefix remains available
only for intentionally partial runs.

## Complete local replication and remaining ablations

`run_local_apbpf_full_pipeline.py --run-root /absolute/local/longgoal
--materialized-root /absolute/materialize/outputs --output /absolute/new-attempt
--gpu 0` waits for all Qwen public banks and both completed DeepSeek replays.
It freezes sources when queued, writes immutable candidate and replication
manifests, provisions every adapter, and runs the permanently exploratory DAG.
Use a new output directory for a new declaration. Current GPU work is never
preempted; GPU0 repair begins only after its Qwen bank generation is finished.

`run_apbpf_replication_bridge.py` wraps the unchanged materialize,
execution-cache, association and association-gate workers. At materialization,
both replication cache paths and proofs are bound by the root manifest checksum
in the frozen command. Each cache is reconstructed from raw bound candidate
execution and compared byte-for-byte, with matching generator revision, decoding
budget and materialized task inventories. The bridge forwards these evaluator
inputs through exact direct dependencies; no generator receives them. Existing
DeepSeek candidate execution is explicitly reused. Qwen candidates are freshly
executed by the ordinary bank/cache stages.

`run_apbpf_replication_worker.py` reads same-run Qwen association and selection
evidence, freshly fits all three DeepSeek model seeds and utility heads on each
declared cache, and emits the complete two-domain/two-family matrix. Standalone
trained results are not imported. Positive direction is checked by the original
replication gate; stricter failed upstream gates remain binding.

The prediction ablation `history_rate` uses exactly four visible categorical
outcomes and fixed Laplace smoothing alpha=1. It differs from the tuned Dirichlet
baseline. `run_apbpf_history_rate_ablation.py` adds this supplementary score using
the exact existing aligned predictions, all three seeds and all 500 primary
sources. It fits no parameters and does not change any gate or checkpoint. The
association bridge records it for Qwen; the replication worker records it for
DeepSeek. Paper tables export the complete four-cell control comparison.
Standalone instances can wait for the existing replication-cell supervisors,
but do not count as sealed stages. Public-only probability construction and
exact population/label bindings are checked independently.

For development-only numerical diagnosis, `run_apbpf_particle_diagnostic.py
--development-root /absolute/codearc-semantic-debug-v1 --output /absolute/new-run`
holds all six completed lexical/semantic checkpoints fixed and evaluates 8, 32
and 128 inference particles on all 400 original development sources. It rejects
primary records, validates model/data receipts, and first reproduces each
original eight-particle result. All budgets, seeds and pair-preserving controls
are retained. ESS and surviving initial-particle ancestry are reported separately;
resampling can produce uniform weights without restoring lost particle diversity.
`summarize_apbpf_particle_diagnostic.py --run /absolute/new-run --cache
/absolute/codearc-semantic-debug-v1/development-cache.json --output /absolute/summary.json`
requires the complete 18-cell run and adds paired source-bootstrap comparisons
against eight particles. This is an exploratory inference-budget diagnostic,
not retraining, primary evaluation, or a replacement for the locked eight-particle
protocol and its failed gates.

After the bounded32-particle development refits, use
`summarize_apbpf_particle_training.py --training-root /absolute/refits
--development-root /absolute/codearc-semantic-debug-v1 --sensitivity-root
/absolute/particle-sensitivity --output /absolute/new-comparison --wait`.
It waits for every fixed seed, restores the completed checkpoints, verifies
reproduction of their reported NLLs, and compares both training variants at32
inference particles using paired source-bootstrap intervals. This separates
training-budget effects from simply increasing inference computation. It also
aggregates the new association contrast, retaining original per-seed gates;
neither comparison evaluates primary sources or changes the original protocol.

For a completed lexical generated-development study, run
`summarize_generated_development.py --development-root /absolute/completed-study
--output /absolute/new-summary`. It requires all three fixed seeds, verifies
source and checkpoint receipts, and replays aligned, outcome-shuffled,
joint-reversed and presentation-permuted predictions. Each arm must reproduce
its original reported NLL within 1e-6 before aggregation. All 400 development
sources remain included; the 10000-draw bootstrap keeps a source in the same
cluster across seeds. Raw predictions and original per-seed gates are retained.
This is a development summary, not a new fit or a sealed primary stage.

`audit_pbpf_finite_support.py --input /absolute/finite_contract --output
/absolute/support-audit.json` verifies every existing finite-family archive and
reports posterior mass at the numerical floor for every original particle arm.
It does not rerun the finite generator or distinguish initial support omission
from later resampling loss. Its discrete-state findings cannot be presented as
an established explanation for the neural continuous-latent experiments.

## Worker inputs and outputs

The `hard_bank_lock` stage receives only public materialization and must finish
before `execution_cache` mounts or produces evaluator-hidden outcomes. The
later `hard_bank` stage audits that exact locked source/candidate inventory.

Every worker receives:

- `APBPF_REQUEST`: absolute `request.json`, containing the resolved config,
  stage/fingerprint, declared dependency checksums and result-file paths,
  confirmatory/claim labels, failed-gate lineage, and output directory.
- `APBPF_RESULT`: absolute `worker-result.json` to create before successful exit.
- `APBPF_OUTPUTS`: an empty attempt-owned directory for evidence artifacts.

Read declared inputs; do not fetch another run's outputs, mutate request or
identity files, or expose evaluator-only hidden labels to generator code. The
runner verifies checksums and provenance, but workers remain responsible for
the scientific computation and public/hidden information firewall. Bootstrap
units must be source-problem components with 10,000 draws, not candidates.

Return a JSON object of this shape (the strings below are explanatory):

```json
{
  "schema": "apbpf-stage-result-v1",
  "stage": "the exact request stage",
  "fingerprint": "the exact request fingerprint",
  "dependencies": {"upstream_stage": "exact digest copied from request.dependencies"},
  "summary": {"description": "actual computation and population"},
  "artifacts": [{"path": "outputs/evidence.json", "sha256": "actual SHA-256"}]
}
```

Copy the complete `request.dependencies` mapping verbatim; its values identify
the direct dependencies' immutable completion records and are not a checksum of
the current `request.json`. Do not write `runner_lineage`: after the worker
returns, the runner adds that reserved field and transitively binds every
ancestor's exact completion checksum. Confirmatory repair validates this chain,
not merely equal configuration fingerprints. The focused repair helper also
requires real, unfailed confirmatory gate attempts and binds the supplied belief
checkpoint bytes to the inherited `train_belief` evidence artifact.

Every real stage needs at least one evidence artifact below `outputs/`, including
the paper-tables stage. Artifact paths are relative to the attempt directory.
Do not write symlink evidence or refer to mutable external files. Copy durable
checkpoint/data evidence into the output inventory as appropriate. Workers may
read model/data caches from the explicitly provisioned paths in the request.
The runner captures `stdout.log` and `stderr.log` and enforces the site timeout.
Nonzero exit, timeout, missing result, invalid provenance, or checksum mismatch
is an execution failure, never a reason to fall back to smoke output.

## Gate metrics

Gate results additionally contain
`"gate": {"passed": true, "reason": "...", "metrics": {...}}`.
`passed` must agree with a decision recomputed by the runner from the locked
thresholds. All numerical metrics must be finite. Required keys:

| Stage | Metrics |
| --- | --- |
| `hard_bank_gate` | v2 `lock_schema`/`audit_schema`; lock/candidate/source SHA-256s, with the lock SHA matching an artifact of the exact `hard_bank_lock` dependency; `lock_precedes_hidden_execution`, `candidate_inventory_bound`, `source_inventory_bound`, `pilot_primary_source_disjoint`; group counts and visible/oracle Pass@1 |
| `baseline_fairness_gate` | `baselines`, keyed by all four names in `gates.yaml`, each with `nll_advantage` and `clustered_lower_bound` |
| `association_gate` | `population: full_locked_population`; `domains` rows with `gap_nats_per_test`, `clustered_lower_bound` |
| `pair_invariance_gate` | `domains` rows with `shuffle_gap`, `joint_reversal_degradation`, `presentation_permutation_degradation` |
| `oracle_headroom_gate` | `domains` rows with `nll_advantage_over_fixed`, `nll_advantage_over_random` |
| `active_testing_gate` | `domains` rows with `matched_hidden_quality`, `test_reduction`, `fixed_budget`, `all_policies_execute_exact_budget`, `nll_advantage_at_four_tests` |
| `selection_gate` | `comparator: strongest_cross_fitted_deterministic`; `domains` rows with `absolute_selected_pass1_advantage`, `clustered_lower_bound` |
| `replication_gate` | `cells`: rows with `domain`, `family`, `association_gap`, `selection_advantage` |

`domains` must contain exactly the configured confirmatory domain names
(`runbugrun`, `codearc_replay`); EvalPlus is audit-only. Replication must cover
the full confirmatory-domain/model-family cross product without duplicates.
Report all metrics, including failing values. Baseline advantages and selection
advantages use the positive-is-better direction; association is shuffled NLL
minus aligned NLL. Supporting evidence must preserve full-population results,
ambiguity strata, confidence interval construction, model/data locks, seeds,
and controls; the dispatcher does not recreate them from scalar summaries.

## Immutability, gate stops, and recovery

Resolved scientific configs, source-file hashes, profile, site commands/paths,
and declared worker revisions determine the fingerprint. A change creates a new run directory; it cannot be
resumed into another identity. Runs are serialized by a process lock.

Each `stages/STAGE/attempt-NNNNNN/` retains `identity.json`, `request.json`,
`command.json`, `result.json`, `work.json`, `summary.json`, and a checksum inventory
in `complete.json`. Real workers add their logs, raw result and evidence files.
Publication is create-once. Completed attempts are verified before reuse;
corruption is rejected rather than overwritten. Interrupted/failed attempts are
retained and `--resume` appends a fresh attempt. `rerun-stage` retains all prior
attempts and reruns the target and complete later stage suffix, because a changed
gate can alter downstream global claim labels, not just direct dependency edges.

Execution failures include immutable `failure.json` with exact argv (when
provisioned), return code, error, a same-identity `rerun-stage` command, and a
separate `run` command for the new fingerprint after code/config/worker changes. Missing
commands are explicitly null, never fabricated as executable. Failed scientific
decisions add `gate-failure.json`; a normal run exits 20 before downstream work.
`--continue-exploratory` permits continuation but propagates failed-gate checksums
and a nonconfirmatory label to every subsequent artifact. It cannot restore main
table eligibility. Repair occurs only after association and selection gates,
unless this explicit exploratory override is used.

`report` inspects completed, failed, interrupted, stale and pending stages read-only.
`verify` additionally requires the complete DAG. Neither command creates or
repairs artifacts. `main_table_eligible` is true only for a complete real DAG
with all gates passing and no exploratory lineage; it is an infrastructure
eligibility check, not an independent audit of the worker's scientific validity.

## Explicit offline smoke path

```bash
pbpf-apbpf run --profile local_smoke --output-root runs/apbpf-smoke
pbpf-apbpf run --profile local_smoke --output-root runs/apbpf-smoke --resume
pbpf-apbpf verify --profile local_smoke --output-root runs/apbpf-smoke
```

Do not supply a site file (or `PBPF_APBPF_SITE`) for smoke. Every smoke stage
contains `scientific_measurements: false`; successful verification does not make
it confirmatory. `--fail-smoke-gate association_gate` deliberately fails that
synthetic gate; the run's immutable smoke control is reused on resume. Use a
different output root for a different smoke-control scenario. These commands
are examples, not a record of tests performed in this implementation session.
