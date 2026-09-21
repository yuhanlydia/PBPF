# Real association: representation and history-view experiment

The user requested continuing until the real-data diagnosis issue is addressed.
This stage tests representation and training-history hypotheses; ordinary NLL
improvement and synthetic success do not satisfy its association criterion.

## Locked comparison

Use the existing source-disjoint train/development cache only. Do not encode or
score held-out test records. All four runs use the interaction model, seed 1701,
1000 steps, 8 training particles and 64 evaluation particles:

1. Lexical features, fixed visible prefix.
2. Lexical features, uniformly resampled training test order.
3. Frozen CodeBERT features, fixed visible prefix.
4. Frozen CodeBERT features, uniformly resampled training test order.

CodeBERT model: `microsoft/codebert-base`, revision
`3b0952feddeffad0063f274080e3c23d75e7eb39`. Encode public task/code and contextual
test input only. Pool input-token states separately from context. Reduce to
256 features with a fixed random projection after training-only centering;
normalize per row. No labels are used to fit the feature transformation.
Record truncation, file hashes, feature transform and cache identities.

Training-order augmentation jointly permutes all ten test/outcome pairs for
training rows only. It never changes pairing or evaluation prefixes. Apply the
same minibatch/view schedule to deterministic baselines in augmented runs.
All deterministic baseline checkpoints must also be scored with outcome
shuffle and pair-preserving controls, not just absolute NLL.

## Implementation and verification

- [x] Test public-field whitelisting, input-only pooling, train-only projection,
  cache ID binding, and pair-preserving training augmentation.
- [x] Build immutable frozen semantic feature cache for train/development.
- [x] Extend runner with explicit feature-cache, history-view and eval-budget
  options. Preserve defaults and report deterministic association diagnostics.
- [x] Run the locked four-cell matrix. Replicate promising association results
  on another optimization seed, without using held-out tests to choose settings.
- [x] Verify artifacts and report all cells, including failures. Keep gate
  threshold 0.03 nats/test; descriptive development bootstrap is not confirmation.

## Exploratory extension (after launching the locked matrix)

A fixed cosine-similarity control exposed pairing signal despite the neural
model's near-zero gaps. Added a query-conditioned kernel control retaining
individual pairs, with alpha/temperature/mixture selected on training NLL only.
This is explicitly adaptive development exploration, not a fifth preregistered
cell or a particle success. Verify exact histogram limit, joint permutation,
visible-label-only API, and bootstrap gains relative to both shuffle and the
train-selected histogram. The bootstrap comparison to histogram must not be
replaced with a shuffle-only significance claim.

All four neural settings failed the locked association screen; no neural seed
replication was triggered. The kernel control is deterministic and uses four
corruption sensitivity checks, not optimization-seed replications.
