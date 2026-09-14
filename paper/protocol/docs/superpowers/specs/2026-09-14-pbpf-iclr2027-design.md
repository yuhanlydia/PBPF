# PBPF ICLR 2027 Confirmatory Design

**Status:** approved implementation specification. This document defines the
prospective experiment; legacy `evalplus_full164_v1` numbers are diagnostic and
must not appear as PBPF evidence.

## Scientific claim

A test result is an observation, not a diagnosis. PBPF maintains a calibrated,
candidate-specific posterior over mutually incompatible latent failure
hypotheses and conditions one complete repair on one posterior draw. The paper
tests two linked claims:

1. the learned posterior predicts evaluator-withheld execution outcomes better
   than matched deterministic memories; and
2. preserving posterior multimodality and continuation-level coherence improves
   hidden-test repair under equal generation and execution budgets.

PBPF does not claim to be the first Bayesian debugger. Its proposed contribution
is the connection between sequential candidate-version diagnosis and coherent
posterior-conditioned code generation.

## State semantics and probability model

For task `x`, candidate slot `g`, and repair round `r`, candidate version
`c[g,r]` owns an independent particle approximation to a latent failure state
`z[g,r]`. Candidate samples and diagnostic particles are different axes.

The state is piecewise static:

- while ordered tests execute on unchanged source code, `z[g,r]` is fixed and
  only the observation likelihood is accumulated;
- when a patch creates `c[g,r+1]`, a child state is sampled from a learned
  proposal conditioned on its selected parent, code diff, and first child
  observation;
- siblings and unrelated candidates can never exchange posterior mass.

Along one lineage,

```text
p(z_0 | x,c_0) p(E_0 | z_0,x,c_0)
prod_r p(z_r | z_{r-1},x,c_r,delta_r) p(E_r | z_r,x,c_r),
```

where ordered visible evidence `E_r=(e[r,1],...,e[r,J])` factorizes into a
five-way categorical likelihood over `PASS`, `WRONG_OUTPUT`,
`RUNTIME_EXCEPTION`, `TIMEOUT`, and `COMPILE_ERROR`. Infrastructure failures
are not a sixth model outcome.

### Root and child proposals

Root particles are sampled from a learned diagonal Gaussian
`q_root(z | x,c,e_1)`. A child particle is sampled as

```text
z_r^m ~ q_phi(z | z_parent^a, x, c_r, delta_r, e[r,1]).
```

Ancestor selection is explicit and has no default. Let `P` be the number of
child particles and let normalized parent weights be `w_parent`. The caller
must select one of two schemes:

- `deterministic_enumeration`: `a[1:P]` must be a permutation of every parent
  index. No ancestor proposal terms are permitted. The base log mass is
  `b_m = log w_parent[a_m]`.
- `sampled`: draw `a_m ~ rho(a)`; repeated indices are permitted. Supply the
  finite categorical log probability `log_ancestor_proposal[m] = log rho(a_m)`
  at each actual draw. The proposal must cover the parent target's support.
  The base log mass is `b_m = -log(P) + log w_parent[a_m] - log rho(a_m)`.
  The supplied per-draw probabilities need not sum to one, since repeated
  draws repeat the same probability. In particular, weighted sampling with
  `rho = w_parent` gives uniform base masses `1/P`; parent mass is not counted
  twice. Uniform sampling with `rho = 1/P` leaves `w_parent[a_m]` as the base
  mass, but unlike deterministic enumeration may repeat parent indices.

The first unnormalized child log importance mass is

```text
b_m
+ log p_psi(z_r^m | z_parent^a,x,c_r,delta_r)
+ log p_theta(e[r,1] | z_r^m,x,c_r,test_1)
- log q_phi(z_r^m | z_parent^a,x,c_r,delta_r,e[r,1]).
```

Its incremental log evidence is `logsumexp_m(log importance mass_m)`, computed
before normalization or resampling. The ancestor base masses are never
renormalized first: doing so would corrupt the sampled-ancestor Monte Carlo
evidence estimate. Subtract this incremental log normalizer to obtain the
normalized child weights, and add it to the parent's cumulative log evidence.

For test `j>1` on the same source code,

```text
log w[r,j]^m = log w[r,j-1]^m
             + log p_theta(e[r,j] | z_r^m,x,c_r,test_j).
```

No transition or proposal term is legal unless a new latent sample was drawn.
Every Gaussian correction is evaluated at the actual sampled latent.

The filter records the incremental log normalizer, normalized entropy,
effective sample size, unique ancestor count, resampling decisions, proposal
noise, and selected generation particle. If `ESS < P/2`, evaluation uses
systematic resampling followed by one Metropolis-adjusted Langevin
resample--move step on the small latent target. Uncorrected jitter is only a
named biased ablation. During training, discrete ancestor choices are detached.

## Learned modules

The primary implementation uses `d_z=32`, `P=8`, and eight soft-prefix tokens.
Torch modules are:

- `SemanticEncoder`: actor-hidden-state pooling or a compact frozen code encoder
  for task, code, diff, test, and observation features;
- `RootProposal`: diagonal Gaussian parameters for a root candidate;
- `PatchTransition`: diagonal Gaussian prior for a child candidate;
- `ChildProposal`: evidence-conditioned diagonal Gaussian proposal;
- `OutcomeLikelihood`: calibrated five-way future outcome distribution;
- `SoftPrefixProjector`: maps one latent to eight actor input embeddings.

The Stage-B loss is

```text
L_B = -E[log Z_hat_SMC] + lambda_future L_future,
L_future = -sum_{j>k} log sum_m w_k^m p_theta(y_j | z_m,x,c,test_j).
```

The data loader samples test prefixes `k in {1,2,4}` and candidate lineages.
The primary endpoint fixes `k=4`. The likelihood is trained with an unweighted
proper categorical NLL. One scalar temperature is fitted on a source-disjoint
calibration split and is reused inside filtering and reporting.

Actor training freezes the belief model. The correct repair likelihood is a
mixture of whole sequences:

```text
L_A = -log sum_m stopgrad(w_m)
      exp(sum_t log pi(a*_t | a*_<t,x,c,E,z_m)).
```

For bounded memory, the implementation first computes component
responsibilities without gradients, then serially replays one component graph
at a time. QLoRA is a secondary replication; the frozen-actor soft-prefix model
is primary. Low-rank K, V, and K+V injection are ablations, not headline
methods.

At inference, PBPF samples a particle once and holds it for every token of one
patch. With several repair calls it systematic-samples component indices across
calls, while holding each index within its continuation. Token-wise remix is a
fault ablation because a product of mixtures contains chimeric cross-diagnosis
sequences that are absent from the mixture of sequences.

## Evaluator firewall and immutable data

Generator and evaluator execute in separate processes with separate manifests.
The generator process can read task text, candidate code, ordered visible test
identifiers, and bounded visible feedback. It cannot read hidden test source,
expected hidden outputs, gold code, gold patch, or final hidden outcomes.
Candidate hashes are sealed before hidden evaluation starts.

All manifests pin dataset revision, model revision, prompt revision, container
digest, split seed, task cluster, test order, and SHA-256 content hashes. Resume
rejects any fingerprint mismatch. Every arm uses the same immutable candidate
bank and derived per-task RNG streams.

### Data protocols

1. **Finite audit:** 50,000/5,000/5,000 synthetic cases split by latent family
   and generator template; ten tests per case, four visible and six future.
2. **PBPF-RBR:** Python RunBugRun records with at least ten active tests, fixed
   code passing all tests, buggy code failing at least one, and code length at
   most 2,048 actor tokens. Preserve official boundaries, then remove connected
   leakage components defined by problem identity and normalized code/test
   signatures. Use deterministic caps 12,000/2,000/2,000 and split development
   into tune/select halves. Tests are SHA-256 ordered; first four are visible.
3. **CodeARC-Replay:** anonymous variant only; invocation indices 0--3 visible
   and 4--9 evaluator-only. Target code and expected outputs never enter the
   generator process. This is explicitly a replay protocol, not an official
   interactive leaderboard claim.
4. **PBPF-EvalPlus:** all HumanEval+ 164 and MBPP+ 378 tasks. Base tests provide
   visible feedback; plus-only tests are hidden and run once after final hashes
   are sealed. This is a repair protocol, not zero-shot EvalPlus leaderboard
   evaluation.
5. **LiveCodeBench:** the release-v5-to-v6 new slice is a contamination-reduced
   locked replication for Qwen2.5 and the selected top three arms.

Pinned upstream revisions are stored in data manifests, never reused across
unrelated datasets. Dataset preparation is deterministic, resumable, and emits
counts plus exclusion reasons.

## Models

| Role | Model | Required scope |
|---|---|---|
| CI smoke | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | opt-in tiny integration |
| primary | `Qwen/Qwen2.5-Coder-7B-Instruct` | all main experiments and development ablations |
| architecture replication | `Qwen/Qwen3-8B` with thinking disabled | top three locked arms |
| cross-family replication | `deepseek-ai/deepseek-coder-6.7b-instruct` | top three locked arms |

Formal inference is BF16, context length 8,192, output cap 1,024, temperature
0.8, and top-p 0.95. Every model uses a full immutable revision and its native
chat template.

## Confirmatory stages and gates

### S0: integrity

Run sixteen tasks per dataset. Required: all hashes present, hidden fields absent
from generator artifacts, identical banks and budgets across arms, deterministic
resume, and infrastructure failure at most one percent.

### S1: exact finite audit

Compare exact Bayes with `P in {1,4,8,16,32}`. At `P=32`, median posterior KL
must be at most 0.05 nat, future NLL must be within two percent of exact, and
randomized 90% HPD coverage must lie in `[0.87,0.93]`.

### S2: frozen future-outcome prediction

Train on PBPF-RBR, select once on the development-select split, and evaluate the
sealed RBR test plus CodeARC-Replay. Prediction arms are prior, last outcome,
full transcript, matched GRU, matched DeepSets, scalar correctness belief, MAP,
posterior mean, PBPF, shuffled evidence, wrong candidate, and matched random
latent.

PBPF advances only if, against the one development-selected strongest
deterministic comparator:

- task-macro prefix-four future NLL improves by at least five percent;
- the paired 95% cluster-bootstrap lower bound on log relative improvement is
  above zero;
- Brier satisfies the development-locked one-sided non-inferiority margin; and
- shuffled, wrong-candidate, and random controls each remove at least eighty
  percent of the gain.

### S3: equal-budget repair

Per task and seed, build eight genuine actor samples once. Share exact hashes
across all arms. Report initial selected Pass@1, hidden pass@8 oracle ceiling,
and final Pass@1 separately. Every matched repair arm receives exactly four
repair decodes, the same visible tests and order, the same maximum tokens, and
no early stopping. Exactly one final program is sealed and submitted.

Primary arms are independent repair, Self-Debug/full transcript, REx,
development-selected deterministic belief, paper-spec Rollout Roulette, and
PBPF sample-once. A no-repair selector is a lower-compute reference. LDB and
trace-rich methods appear only in a separate enhanced-information table.

PBPF advances if final hidden Pass@1 exceeds both Self-Debug and the strongest
matched deterministic arm by at least three absolute points, both paired 95%
confidence lower bounds are above zero, and the S2 prediction advantage remains.

### S4: locked replication

Run PBPF and the two preselected strongest baselines on full EvalPlus with
Qwen3-8B and DeepSeek-Coder-6.7B, and on CodeARC-Replay plus the locked
LiveCodeBench slice with Qwen2.5. Direction must agree on both extra model
families and the fixed-equal-weight cross-model confidence interval must exceed
zero. SWE-bench is excluded from the core claim; it may be added after all core
gates pass.

## Ablations

Development ablations use 256 RBR tasks and three seeds, one factor at a time.
They are not a Cartesian sweep. The locked confirmatory set contains only the
best configuration and five causal controls.

- particles `P={1,2,4,8,16,32}`;
- uniform/learned proposal, likelihood, transition; proposal correction off;
- resampling never/every step/ESS; rejuvenation off;
- last/window-two/window-four/full/orderless/shuffled/wrong-candidate/masked;
- sample-once/MAP/posterior mean/token-remix/full ensemble;
- no-op/random/text/soft-prefix/K/V/KV; prefix length `2,4,8,16`;
- candidate-specific/shared belief, where shared is labeled a faulty control;
- status-only versus exception/full trace in a secondary information table.

## Statistics and accounting

Formal seeds are `1701,1702,1703`. The independent unit is a source problem
cluster. Metrics first aggregate hidden tests within candidate, candidate within
task, and seeds within task. Inference uses 10,000 paired stratified cluster
bootstrap replicates with common resample indices across arms and 95% intervals.
Secondary comparisons use Holm correction. Every model-by-dataset cell is shown;
pooled significance alone is insufficient.

Budget ledgers separately report actor generation calls, belief forward calls,
verifier cases and suites, input/output tokens, CPU seconds, GPU hours, wall time,
peak allocated memory, and disk bytes. Calling a cheap belief scorer is not an
actor generation. Any particle that performs a full decode increases the
effective decode multiplicity and cannot be called compute matched.

## Execution contract

The only formal user entry point is:

```bash
bash scripts/run_iclr.sh \
  --config configs/experiments/iclr_pbpf.yaml \
  --profile slurm_h200x16 \
  --resume
```

The internal DAG is `doctor -> manifest -> prepare -> finite -> bank -> visible
execute -> train/freeze belief -> prediction -> Gate B -> repair rounds -> seal
hashes -> isolated hidden evaluation -> Gate C -> locked replication -> tables`.
Gate-B failure stops repair unless `--force-after-failed-gate` is supplied; forced
outputs are labeled nonconfirmatory and excluded from main-table generation.

Stable work keys use SHA-256 and shard by `int(sha256(key),16) % num_shards`.
Each shard writes private JSONL/Parquet files using temporary-file, fsync, atomic
rename, and a checksum completion marker. Merge rejects missing, duplicate, or
hash-mismatched keys. No shared NFS SQLite file is allowed.

`doctor --dry-run` prints every cell's tasks, actor decodes, token bounds,
sandbox suites, memory profile, and projected cost. A disjoint 64-task
calibration measures throughput and aborts if projected cost differs by more than
25%, memory exceeds 90%, or infrastructure failures exceed 1%.

## Required behavioral tests

- an observation for candidate A cannot change candidate B;
- siblings inherit only their selected parent and remain isolated;
- proposal/transition correction is rejected without a sampled child latent;
- exact finite posterior convergence and incremental normalizers;
- deterministic systematic resampling and corrected MALA acceptance;
- whole-sequence mixture rejects the token-remix `AA/BB` chimera;
- hidden tests cannot enter prompts, training batches, selection, or resume state;
- configured model, method, dataset, and immutable revisions match runtime;
- all arms share banks and equal actor-generation budgets;
- EvalPlus base and plus-only tests are disjoint and atomic;
- stable sharding and interrupted-run resume reproduce identical hashes;
- shuffled/wrong-candidate/random controls are first-class executable arms;
- a tiny fake backend executes the complete DAG without downloading a model.

CPU CI never downloads model weights. GPU integration and external datasets are
explicit opt-in jobs.

## Paper and artifact contract

`paper/pbpf_iclr2027.tex` must compile with the official ICLR 2027 style and
contain switchable title, abstract, and introduction variants. The recommended
variant is the observation-versus-diagnosis story. A single Experimental Setup
matches this specification. Numerical claims remain explicit pending-result
macros until immutable formal artifacts exist. The submission ZIP contains the
TeX, bibliography, style files, math commands, and a short build README.

The repository README distinguishes runnable infrastructure from completed
empirical evidence and gives exact checkout, install, doctor, smoke, SLURM,
resume, aggregate, and verification commands.
