# Evidence-Shapley relevance correction — 2026-09-24

## Why the previous formulation can fail

A Shapley value is only as meaningful as its coalition value function. Scoring the
entire corrected program is poorly aligned with EESD relevance because most response
tokens are copied or unchanged source code. Their language-model probability can
dominate the small edit that execution evidence actually caused.

This branch therefore defines coalition value on the edited region only:

[
v(S)=overline{log p(a^1_{Delta}mid x,a^0,E_S)}
-overline{log p(a^0_{Delta}mid x,a^0,E_S)}.
]

The bars denote a mean over the token positions participating in the token-level
original/correction diff. The score is a **contrastive edit log-probability**, not a
binary log-odds model.

For four public executions, exact signed Shapley attribution is

[
phi_i=sum_{Ssubseteq Nsetminus{i}}
rac{|S|!(n-|S|-1)!}{n!}
[v(Scup{i})-v(S)].
]

EED receives nonnegative influence magnitude:

[
s_i=|phi_i|.
]

If all attributions are numerically zero, relevance falls back to uniform rather
than turning tiny numerical noise into a concentrated evidence distribution.

## Important separation

- `signed_attribution`: whether an execution pushes model preference toward or away from the observed edit.
- `relevance = abs(signed_attribution)`: how influential the execution is.
- `effective_mass(relevance)`: how broadly that influence is distributed.

The sign is preserved in artifacts for diagnostics; it is not discarded silently.

## Exact vs LOO

`--mode exact` uses all 16 coalitions for the locked four-execution protocol.
`--mode loo` uses the full coalition and four leave-one-out coalitions. LOO is
reported as LOO, never mislabeled as Shapley.

## Run

```bash
python scripts/score_eesd_shapley_relevance.py \
  --input <generated-corrections>/corrections.jsonl \
  --model-config configs/models/qwen2.5-coder-7b.yaml \
  --output <new-shapley-directory> \
  --mode exact
```

For an adapter-generated correction bank, pass the same `--adapter` used at generation.

The output `corrections-shapley.jsonl` preserves the original lexical relevance in
`legacy_relevance`, replaces `relevance` with Shapley influence magnitude, and records:
- every coalition value;
- signed attribution;
- normalized relevance;
- effective evidence mass;
- edit-token counts;
- `game_span = max_S v(S)-min_S v(S)`;
- attribution L1 magnitude.

The last two diagnostics are important. If `game_span` is almost zero, poor EED
performance should not be interpreted as a failure of the Shapley summation; the
model simply does not change its edit preference when public evidence is removed.

## Verification

Pure CPU tests cover:
- exact Shapley on additive games;
- pair-interaction splitting;
- Shapley efficiency;
- exact Shapley vs leave-one-out distinction;
- signed-to-absolute relevance conversion;
- uniform fallback for zero attribution;
- token edit masks for replace/insert/delete;
- exact coalition inventory;
- causal-LM label masking;
- coalition prompt filtering while preserving stored clipped content.

A real model run is still required to validate GPU memory/runtime and the scientific
quality of the resulting relevance distribution.
