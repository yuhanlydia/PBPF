# A-PBPF Information-Theoretic Test-Time Inference Design

**Status:** implementation landed; runtime verification was deferred at the
project owner's request. Existing PBPF measurements motivate the method; they
are not evidence that A-PBPF succeeds.

## Scientific claim

Execution feedback contains two separable signals: candidate-level difficulty
and a test-specific diagnosis. A predictor can improve future-outcome NLL using
the first signal while discarding the second. A-PBPF learns a factored particle
state, explicitly tests whether aligned test--outcome pairs outperform
counterfactual pairings, uses diagnostic mutual information to acquire tests,
and samples one diagnostic state for one complete repair continuation.

## Non-negotiable controls

- Source-problem components, not candidates, are the split/bootstrap unit.
- A counterfactual preserves the test multiset and outcome histogram and changes
  only their assignment. Constant histories are ineligible for the auxiliary
  association loss but remain in ordinary prediction/calibration evaluation.
- Joint presentation permutations that preserve pairs must be approximately
  invariant; outcome-only shuffles, semantic masking, and wrong-candidate
  evidence must degrade association-aware predictions.
- The primary association statistic is
  `NLL(outcome-shuffled) - NLL(aligned)` over the full locked population, with
  ambiguity strata reported separately. The method cannot pass by reporting only
  a favorable post-hoc subset.
- Particle-weight entropy is not diagnostic entropy. Test acquisition uses mutual
  information for an explicit discrete diagnostic component.
- Fixed-budget policies execute exactly the same number of tests. Threshold
  stopping is reported as a separate budget--quality curve.
- One posterior component is sampled once per repair continuation. Token-wise
  re-mixture is an intentionally incorrect ablation.
- Existing saturated EvalPlus groups are an audit set only. A confirmatory hard
  bank must leave measurable headroom beyond visible pass rate.
- Failed upstream gates stop confirmatory downstream claims. An explicit
  exploratory override may collect artifacts but must label them nonconfirmatory.

## Model

For a candidate `c` and history `H={(t_i,y_i)}`, each particle is split into
`z=(z_d,z_g)`: `z_d` is test-invariant difficulty and `z_g` is diagnosis. Outcome
logits are additive:

```text
logits(y|t,H,z) = difficulty_head(x,c,z_d)
                + diagnosis_head(x,c,t,z_g).
```

The existing importance-corrected, piecewise-static SMC remains the inference
backbone. The aligned path retains FIVO and held-out future NLL. An outcome-only
derangement is evaluated with common proposal noise. On eligible examples:

```text
L_assoc = mean relu(margin + NLL_aligned - NLL_shuffled)
L_inv   = symmetric KL(difficulty_aligned || difficulty_shuffled)
L       = L_FIVO + lambda_future L_future
          + lambda_assoc L_assoc + lambda_inv L_inv.
```

Zero eligible examples produce an exact differentiable zero auxiliary loss.
The shuffle seed, eligibility mask, dimensions, coefficients, and config hash
are checkpointed.

This coordinate factorization is an operational inductive bias, not a theorem
that the latent representation is statistically identifiable. The primary claim
therefore concerns counterfactual pair use and decision value, not complete
latent disentanglement. Joint-particle MI, nuisance-interventional MI, shuffle,
and pair-preserving controls must be reported separately; stronger independence
regularizers are a follow-up ablation rather than an assumed property.

## Information value

Diagnosis is represented as component probabilities `q_k` with component-wise
outcome predictions `p_k(y|t,H)`. For each unexecuted public test:

```text
EIG(t) = H(sum_k q_k p_k(.|t,H)) - sum_k q_k H(p_k(.|t,H)).
```

This is the mutual information between diagnosis component and prospective
outcome. The policy selects maximum EIG, never repeats a test, and either uses a
fixed budget `B in {1,2,4}` or a development-locked stopping threshold.

## Data and feedback

RunBugRun records retain input, expected output, bounded actual output/stderr,
timeout/return-code metadata, and categorical outcome. Generator-visible fields
remain separately firewalled. Association training uses expected/observed
semantics only when the protocol explicitly declares them public. Cache schema
versions fail closed rather than silently accepting old input-only records.

## Experiment gates

0. Baseline fairness: A-PBPF must beat a matched deterministic pair-aware
   predictor, exchangeable Deep Sets/histogram model, tuned Dirichlet rate, and
   no-particle bottleneck by at least `0.02` NLL with a clustered lower bound
   above zero before particles are claimed as necessary.
1. Hard bank: at least 300 mixed pilot groups and 500--1000 confirmatory groups;
   visible-pass selection must be below the hidden oracle ceiling.
2. Association: gap at least `0.03` nats/test with source-cluster 95% lower bound
   above zero in every confirmatory domain.
3. Pair invariance: joint-reversal and presentation-permutation degradation are
   each at most 25% of the shuffled gap.
4. Active testing: an oracle subset must first establish at least `0.03` NLL
   headroom over fixed/random testing. A-PBPF then needs either at least 25%
   fewer tests at matched hidden quality or at least `0.03` lower NLL at fixed
   four-test budget.
5. Selection: at least `+0.03` absolute selected Pass@1 over the strongest
   cross-fitted deterministic selector with clustered lower bound above zero.
6. Repair is downstream/exploratory until association and selection pass.
7. Replication requires direction agreement on two data protocols and two model
   families.

## Executable artifact

A new `pbpf-apbpf` command and `scripts/run_apbpf_iclr.sh` own a separate
`apbpf-iclr-v1` fingerprint. They do not weaken or rewrite the frozen legacy
`pbpf-iclr` contract. Commands are `doctor`, `run`, `verify`, `report`, and
`rerun-stage`. Every stage writes identity, immutable work records, checksum,
summary, and actionable failure metadata. `--resume` accepts only the same
fingerprint; `--continue-exploratory` propagates a nonconfirmatory label after a
failed gate.

## Success interpretation

Association identifiability is the primary paper claim. Active testing and hard
selection establish decision value. Repair is supportive only if the new
belief-to-actor bridge succeeds; a negative repair result does not get rewritten
as positive evidence.
