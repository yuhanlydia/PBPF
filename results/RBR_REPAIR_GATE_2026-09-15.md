# RunBugRun frozen-actor repair pilot — 2026-09-15

## Question

This pilot tests the claim that a PBPF state which predicts future execution
outcomes can improve a frozen actor's actual program repairs. It is the actor
conditioning follow-up to `RBR_GATE_B_2026-09-15.md`, rather than another
future-outcome prediction result.

An initial implementation omitted the programming-problem statement. That made
the repair request underdetermined and produced zero solved tasks in every arm.
The repaired dataset joins RunBugRun programs and tests to the public IBM
Project CodeNet HTML problem descriptions, converts them to bounded plain text,
and includes the same statement in every actor arm. Future test inputs, outcomes,
and reference repairs remain hidden.

## Protocol

- Checksum-verified RunBugRun v0.0.1 Python programs and tests.
- Source-problem-disjoint 282/54/79 train/development/test candidates from
  112/27/66 problems; ten deterministic tests per candidate.
- Four visible test inputs and categorical outcomes; six tests held out for
  evaluation.
- Frozen `Qwen/Qwen2.5-Coder-7B-Instruct` at revision
  `c03e6d358207e414f1eca0bb1891e29f1db0e242`, loaded in 4-bit on one RTX 4090.
- Eight input soft-prefix tokens, greedy decoding, and identical prompts and
  generation budgets across arms.
- The projector trains on 98 fixed repairs that fit the 1,024-token cap. Fifteen
  development repairs fit the same cap. Training ran for 1,500 steps and selected
  step 750 by exact posterior-mixture development NLL.
- Prefix token RMS is bounded at 0.02 and the latent-dependent delta RMS at
  0.002. These bounds were added after unbounded prefixes exceeded the actor's
  native embedding scale and corrupted generation.

Development token NLL at the selected checkpoint was 0.11173 for a zero latent
and 0.11153 for the exact PBPF posterior mixture. This is a very small
teacher-forced advantage.

## Matched eight-candidate pilot

| Arm | Solved | Future-test pass fraction |
|---|---:|---:|
| No latent | 1/8 | 35.42% |
| Random matched-norm latent | 1/8 | 35.42% |
| Outcome-histogram latent | 1/8 | 35.42% |
| PBPF posterior mean | 0/8 | 22.92% |
| PBPF posterior MAP | 0/8 | 22.92% |
| PBPF sampled particle | 0/8 | 22.92% |

The deterministic posterior mean and MAP controls reproduce the sampled-particle
failure, so sampling variance does not explain the result. On the one task solved
by the no-latent arm, all three PBPF posterior conditions changed the correct
one-based output index into an incorrect zero-based index. The latent-dependent
prefix therefore caused the measured regression.

## Decision

This pilot is a negative repair gate. It preserves the earlier finding that a
small learned state predicts future outcome categories well, but it does not
support the stronger claim that the state is a reusable repair skill. The
current state mostly represents a candidate's exchangeable failure distribution;
the prompt already exposes the same four categorical observations, and neither
representation supplies richer diagnostic content such as expected-versus-actual
output traces.

The preregistered expansion condition was a positive small repair gate. It was
not met, so this experiment should not be expanded to more repair tasks or seeds
and should not support an ICASSP/ICLR repair-effectiveness claim.

## Reproduction

`scripts/run_rbr_prediction_gate.py --prepare` builds the description-aware,
execution-verified cache. `scripts/run_rbr_repair_gate.py` trains the bounded
soft-prefix projector with development checkpoint selection and evaluates the
matched arms. Raw generations, logs, and model checkpoints remain local; the
compact machine-readable pilot record is `rbr_repair_gate_pilot.json`.
