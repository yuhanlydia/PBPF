# Association debugging: a causal, exchangeable reference path

This is an exploratory redesign authorized by the user after the seed-1701
RunBugRun pilot failed. It does not replace or relabel the frozen pilot.

## Evidence and scope

The pilot used lexical hashing, 440 training candidates, and only 30 eligible
held-out candidates. Its best validation checkpoint was step 250, before strong
training association appeared. The original proposal reads only the first pair;
later observations reweight/resample a fixed set. Difficulty invariance acts on
five outcome probabilities, not the latent marginal. Evidence/future losses are
sums, whereas association is per-test. These are hypotheses to isolate, not
proof that the research idea succeeds.

Implement a separate reference model/trainer and development-only experiment
driver. Preserve old constructors, checkpoints, tests, gates, and test inventory.
Use the same lexical features first to isolate inference/objective changes.
No repair or claim of identifiable bug semantics is in scope.

## Inference

For each prefix h independently, construct a Gaussian proposal q_h(z|c,H_h).
Its difficulty slice reads code/task and the outcome histogram; its diagnosis
slice additionally reads a Deep Sets mean of test/outcome pair embeddings.
The count is provided explicitly. No suffix tests or outcomes enter q_h.
Draw K particles and compute log importance weights
log p(z|c) + sum_{i<=h} log p(y_i|c,t_i,z) - log q_h(z|c,H_h).
Normalize these weights for predictions; logmeanexp gives log Z_h.
This is prefix importance sampling, not the legacy SMC/FIVO estimator.
Within-prefix pair order invariance holds up to floating-point summation.
Aligned and shuffled histories use identical Gaussian noise.

## Objective and invariance

Evidence loss is -log Z_H/H. Future loss averages tests per prefix, then prefixes
{1,2,4}. Association uses relu(m + aligned - stop_gradient(shuffled)) on eligible
histories. Detaching blocks the direct bad-shuffle gradient; shared parameters
can still degrade shuffled predictions, so absolute quality remains required.
Use weighted multiscale RBF MMD on difficulty particles, standardized by a
detached root-prior scale. This compares empirical latent marginals; it is not
an exact KL and does not prove disentanglement. Record its definition explicitly.

## Validation and experiments

Retain snapshots at the declared evaluation schedule. Select the lowest-NLL
checkpoint passing development association and pair-invariance constraints
within 0.02 NLL of the best checkpoint. If none qualifies, return the best-NLL
snapshot with an explicit failed development gate, never a passing fallback.
Development controls must use common random numbers. Test data are never loaded
into feature tensors or scored by this diagnostic driver.

First use a generated controlled task with independent source instances, known
test-trigger associations, balanced visible outcome histograms, and a histogram
baseline. Synthetic success is only a learnability check. Then compare the
original training path, normalized/latent-objective-only changes, and the full
history-IS redesign on the existing train/development split. Record failures as
well as successes. Do not tune against the previously observed held-out test.

## Acceptance

Tests cover causal prefixes, pair permutation, importance correction against a
hand-computed finite particle example, latent-MMD sensitivity with constant
output heads, detached comparator gradients, loss units, and fail-closed
checkpoint selection. Existing belief/APBPF tests remain passing. A real-data
gain is an experimental outcome, not a completion requirement or promised fact.

## Controlled-task-driven extension

At step 450 the concatenation-only history-IS model still produced roughly
log(2) NLL and almost no association gap on the balanced sign-trigger task.
Keep that run intact and add a separately named `interaction` arm with an
explicit bilinear residual `W_g(g * W_t t)` in the diagnosis likelihood. All
other inference/loss/selection choices remain the same. This tests whether
making test/diagnosis interactions easy to represent helps learning, without
changing the task, exposing the hidden diagnosis, or asserting identifiability.
Add a deterministic interaction control using outcome-conditioned visible-test
summaries and a bilinear future-test decoder. Improvement over a concatenation
MLP alone cannot establish a need for particles. Archive its selected weights,
step and development trajectory just like the other deterministic baselines.
