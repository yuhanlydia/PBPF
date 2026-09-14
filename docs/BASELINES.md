# Formal arms and ablations

`configs/iclr/arms.yaml` is the executable Task-5 registry, separate from legacy
diagnostic baselines. Every repair arm has validated provenance mode, URL,
revision, license and deviations. REx and Rollout Roulette are pinned paper-spec
reimplementations; others are controlled implementations. None claims official
upstream execution.

Prediction includes prior, last, full transcript, windows two/four, DeepSets,
matched GRU, scalar correctness, MAP, posterior mean, PBPF and five causal controls.
Learned comparators match all unique learned Stage-B parameters within 5%,
including heads/projections. Shared frozen actor/encoder and particle values are
excluded. Parameter matching is not a claim of equal FLOPs.

Repair compares independent, Self-Debug, REx, development-selected strongest
deterministic belief, Rollout Roulette, PBPF sample-once, and lower-compute
no-repair. Matched arms share eight genuine actor samples, four complete repair
decodes with multiplicity one, four ordered visible tests and no early stop. The
legacy six-sample/two-mutant bank is prohibited. Roulette's additional partial
and discarded-prefix work is explicitly metered; four complete continuations do
not imply identical calls or FLOPs. Task-5 arms exist, but the complete production
training/evaluation factory remains an integration requirement.

The primary actor is frozen BF16, context 8,192/output 1,024, temperature .8 and
top-p .95. PBPF uses P=8, dz=32 and eight soft-prefix tokens held for an entire
continuation. Native templates and disabled Qwen3 thinking are mandatory. Final
selection uses visible pass fraction, then lower round, lower initial slot,
lexical source hash and version hash, matching Task-5 behavior.

`configs/iclr/ablations.yaml` has explicit one-factor patches for 256 RBR
development tasks and seeds 1701/1702/1703, never Cartesian expansion. Locked
causal controls are shuffled evidence, wrong candidate, masked outcomes,
matched-norm random latent and faulty shared belief. Token remix and shared belief
are explicitly faulty controls. Trace-rich information is a separate secondary
table, never pooled into the primary comparison.

The experiment config prospectively declares frozen actor-hidden-state masked-mean
pooling, AdamW/LR/batch/step/future-loss/MALA grids, deterministic selection ties,
absolute .01 multiclass Brier non-inferiority margin, bootstrap strata and
model-specific retrain/freeze policies. These choices are preregistered or later
development-selected, not measured results. Train, temperature calibration,
development-select and locked test have distinct roles; actual learned locks must
be frozen before confirmatory work. No test labels may enter fitting.

Initial selected Pass@1, initial hidden pass@8 oracle and final selected Pass@1
are separate endpoints. An oracle cannot satisfy a final Pass@1 gate. Legacy
RLEF/LDB/SWE-agent claims are not part of this core matrix.
