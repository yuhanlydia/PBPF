# Real-data diagnostic parameter debugging

The user requested actual debugging and more runs, then explicitly requested a
long-running goal. The objective is a reproducible improvement with limitations,
not a relaxed binary gate or ordinary prediction improvement alone.

This is sequential, adaptive development exploration. Each stage's settings
were saved before launching that stage under
`/root/pbpf-runs/parameter-debug-20260917/`; it is not a preregistered study.
No held-out test records are encoded or evaluated.

- [x] Four single-factor, seed-1701 trials against semantic fixed-prefix reference:
  evidence weight .1; invariance weight 0; association weight 10; training K32.
  All other settings remain 1000 steps, K64 evaluation, fixed training histories.
- [x] Instrument actual pair encodings, Gaussian proposals, ESS and likelihood
  reweighting at fixed particle support.
- [x] Test and implement outcome-conditioned feature binding; train seeds
  1701/1702, adding a seed-1702 reference. Disclose larger encoder capacity.
- [x] Test and implement removal of the additive diagnosis MLP; retain original
  encoder and only bilinear test × g diagnosis logits. Train both seeds.
- [x] Test a tenfold interaction-head **initialization** on the interaction-only
  model. Other initialized tensors are identical at a matched seed. Train both.
- [x] Fix the sampling/corruption seed confound in evaluation; preserve defaults
  and old records. Reevaluate frozen selected checkpoints under independent
  sampling, corruption and K64/K256 sensitivity panels.
- [x] Compare Monte Carlo averaged probabilities and source-cluster bootstrap
  results for reference, interaction-only and high-gain checkpoints.
- [x] Retain every positive/negative result, verify artifacts and document scope.

Completed: eleven new 1000-step real-data training runs. The high-gain arm
reproduces a much larger positive association gap in both optimization seeds,
across sampling/corruption panels and higher evaluation budgets. This is a
current-development-population improvement; it does not establish interpretable
bug types, fresh-source generalization, active-testing gains or repair control.
