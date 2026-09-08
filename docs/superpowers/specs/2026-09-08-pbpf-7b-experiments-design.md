# PBPF 7B Coding-Repair Experiment Design

**Status:** preregistered implementation target; no 7B repair gain is claimed.

## Claim and falsifiable hypotheses

PBPF maintains a posterior over hidden program/failure hypotheses as fixed-order execution evidence arrives, predicts unobserved test outcomes with a proper scoring rule, and conditions one coherent repair continuation on a sampled posterior particle. It is not a generic rollout ensemble and not merely additional KV state.

For particle `m`, proposal correction is mandatory:

`log w_t = log w_{t-1} + log p(o_t|z_t) + log p(z_t|z_{t-1}) - log q(z_t|z_{t-1},o_t)`.

Systematic resampling occurs when ESS is below half the particle count. Sequence generation samples one particle once and holds it for the whole continuation. Token-wise re-mixing is a deliberately incorrect ablation because product-of-mixtures is not mixture-of-sequences.

Hypotheses are: PBPF approaches exact finite posteriors; lowers held-out future-outcome NLL versus matched deterministic states; and the predictive gain causally improves fixed-budget round-4 repair Pass@1. If prediction fails, repair RL stops.

## Repository boundary

The repository owns belief/proposal/likelihood models, particle filtering, history and KV/soft-prompt controls, fixed-order repair, model/benchmark adapters, immutable trajectory banks, statistics and QLoRA entry points. Test selection remains fixed and cannot use GOAV.

## Models

| Role | Hugging Face ID | Scope |
|---|---|---|
| plumbing | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | end-to-end smoke |
| primary | `Qwen/Qwen2.5-Coder-7B-Instruct` | frozen mechanism and online repair |
| architecture replication | `Qwen/Qwen3-8B` | strongest two-arm replication |
| cross-family | `deepseek-ai/deepseek-coder-6.7b-instruct` | frozen transfer |

Every revision is resolved to a full commit SHA. Models are reported separately.

## Benchmarks and roles

| Dataset | Role |
|---|---|
| finite affine/DSL programs | exact-posterior identification, 50k/5k/5k family split |
| RunBugRun Python | grouped real repair training and OOD mechanism test |
| CodeARC | interactive program-induction transfer with evaluator firewall |
| HumanEval+ / MBPP+ | public functional repair smoke |
| LiveCodeBench v6 | public fixed-version secondary coding transfer |
| SWE-bench Lite/Verified | expensive repository-level secondary transfer only |

RunBugRun groups source problem, submission/user lineage, translations and shared test source before splitting. Loaders never expose future expected outputs, gold solution/patch, SWE-bench `test_patch`, `FAIL_TO_PASS` or `PASS_TO_PASS`. Public target code and tests remain hidden-by-protocol, not called unseen.

## Baselines and provenance

Controlled representations are raw full transcript, last observation, sliding window 1/2/4, orderless set, pass-rate state, matched GRU, matched exchangeable set encoder, MAP, posterior mean, P-way ensemble, random matched-norm latent, single shared KV delta, PBPF soft prompt and PBPF low-rank KV.

Repair baselines include independent sampling, Self-Debug/raw transcript, REx, RLEF paper-spec and LDB under matched feedback. Rollout Roulette is the primary particle-inference neighbour but its particles are language trajectories rather than execution-conditioned program hypotheses. RSP is a random-latent negative control; UpSkill is an optional visible discrete-latent control; LaDi-RL is optional and must pass an independent code audit.

Each baseline records `paper_url`, `repository_url`, pinned commit/license hashes and one of `official_adapter`, `paper_spec_reimplementation`, or `controlled_ablation`. RLEF has no verified author code and cannot be marked official.

## Experiment stages

### A exact finite posterior

Use family-held-out finite programs and exact enumeration. Compare P=1/4/8/16/32, exact Bayes, MAP, posterior mean, GRU, exchangeable set encoder, orderless/shuffled/wrong-task/random controls. Primary endpoint is NLL of all unseen outcomes after prefix four; also report Brier, ECE, posterior KL, randomized 90% HPD coverage, ESS and unique ancestors.

### B frozen 7B prediction

Freeze the actor and train only belief/adapter modules on grouped RunBugRun. Transfer to CodeARC and EvalPlus. Every arm receives the same immutable candidates, test order, outcome alphabet and visible-token budget. Gate: macro-task future-outcome NLL improves at least 5% relative to the strongest matched-compute deterministic baseline with paired cluster-bootstrap CI above zero, while Brier does not regress. Shuffled evidence and matched random latent must remove at least 80% of the gain.

### C fixed-order repair

Use G=8 immutable initial candidates and four mandatory repair rounds. Six candidates are model samples at temperature .8/top-p .95 and two are registered deterministic mutants. Tests cannot be chosen, skipped, repeated, reordered or early-stopped by an arm. One active slot is selected by predicted remaining-suite success with generation log-probability as fixed tie-break; exactly one final program is submitted.

First run a frozen actor, then QLoRA online repair with predictive and actor objectives separated by stop-gradient boundaries. Primary endpoint is round-4 Pass@1. Advance only if PBPF is at least 3 absolute points above the strongest matched baseline with paired CI above zero and retains the future-NLL advantage.

### D repository transfer

Evaluate a frozen mini-SWE-agent scaffold on Lite for development and Verified once after freeze. Resolved rate, tests, tool calls, tokens, wall time and failures are secondary evidence.

## Ablations

- Filtering: P=1/4/8/16/32; learned/uniform weights; no/every-step/ESS resampling; no rejuvenation; omit proposal correction; independent versus correlated likelihood; leave-one-candidate-out.
- History: last only, window 1/2/4, full, orderless, shuffled, wrong-task and outcome-masked. Report `NLL(last-k)-NLL(full)` as an operational conditional-information proxy.
- State injection: text summary, soft prompt, K-only, V-only, K+V, last 4/8/all layers, shared delta, matched-rank and Frobenius-norm random delta, no-op, P-way full-forward ensemble and prefix reuse on/off.
- Feedback: status only (primary), status+output diff, exception trace and full test source, always in separate tables.
- Mixture: coherent sequence-level particle versus token-level re-mixture fault ablation.

## Statistics, budgets and artifacts

The base task/source cluster is the statistical unit. Formal online runs use three seeds and 10,000 paired hierarchical bootstrap replicates; secondary results use Holm correction. Report future NLL, Brier/ECE, posterior KL/coverage, ESS/resampling/ancestors, round-wise and final Pass@1, regressions, patch size, tokens, executions, CPU seconds, wall time, GPU hours and physical KV bytes.

All arms match candidate bank, execution count, four rounds and visible-token limit. Runs are immutable and contain config/data/model/container hashes, task rows, events, arrays, ledgers, gates and checksums. Gold sidecars are inaccessible to the trainer.

## Hardware profiles

- `16gb`: NF4/BF16 compute, context 2,048, microbatch 1, accumulation 16–32, P=8 sequential, last-4 layers, delta rank 4–8, actor LoRA rank 16/alpha 32 smoke.
- `24gb`: NF4 QLoRA rank 32/alpha 64, context 3,072–4,096, microbatch 1–2, P=8 in chunks 2–4, last-8 layers, online pilot only.
- `4x24gb`: separated rollout/training for screening.
- `h200_formal`: BF16 LoRA and formal three-seed online runs.

Peak memory above 90%, infrastructure failure above 1%, or projected cost overrun above 25% aborts scaling.

## Failure handling and tests

The categorical outcomes are PASS, WRONG_OUTPUT, RUNTIME_EXCEPTION, TIMEOUT and COMPILE_ERROR; infrastructure failure is separate. Config and manifest validation fail closed on leakage, revision ambiguity, variable test order, arm-specific stopping and unmatched budgets. Tests cover normalized log weights, proposal correction, ESS, deterministic systematic resampling, coherent sequence mixtures, permutation equivariance, exact-posterior convergence, history masking, KV-delta shape/norm controls, artifact tampering and deterministic fake-backend integration. GPU tests are opt-in and CPU CI never downloads weights.

