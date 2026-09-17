# Sampling-robust checkpoint selection and particle-budget follow-up

The checkpoint evaluator now averages predictive probabilities across four
independent particle draws before computing NLL. Corruption remains fixed and
aligned/shuffled controls share sampling seeds. Selection, selected-model
scoring and bootstrap inputs use the same estimator. Verification ensembles
use disjoint sampling seeds. The .03 association screen and .02 NLL guard are
unchanged. `--selection-replicates 1` reproduces the earlier runner protocol.

This correction addresses sensitivity to one favorable draw, not adaptive
development-set reuse. Four K64 self-normalized estimates are not one pooled
K256 estimate. We did not run a matched experiment establishing the magnitude
of variance reduction or a causal NLL gain from the evaluator change itself.

## Third optimization seed

Four matched runs use seed 1703, 1000 steps, the same semantic cache, fixed
four-test histories, and four K64 draws per checkpoint evaluation. The held-out
test split was not tensorized or scored. All four selection screens fail; the
selected checkpoints are the minimum-NLL fallback, at steps 450, 750, 700 and
950 respectively. Failures have not been relabeled.

The following independent audit freezes those checkpoints and averages
probabilities over sampling seeds 51701, 71701, 81701, 91701, with corruption
fixed at 51701. These draws are disjoint from selection and the runner's three
verification ensembles. NLL is lower-is-better; the gap is shuffled minus
aligned NLL. Bootstrap intervals resample the 31 development source components.

| Model | Training particles | NLL | Association gap | Descriptive 95% gap interval |
|---|---:|---:|---:|---|
| Original interaction | 8 | .452601 | .000123 | [-.000125, .000425] |
| Interaction-only | 8 | .430485 | .000280 | [.000049, .000661] |
| Interaction-only, 10x initialized weights | 8 | .419254 | .017798 | [.004619, .034003] |
| Interaction-only, 10x initialized weights | 32 | .408513 | .028673 | [.005332, .057959] |

High-gain K8 improves NLL over the original by .033347 (descriptive interval
[.013368,.057171]). K32 improves over high-gain K8 by .010740
([.001603,.019411]). The K8 difference from interaction-only is .011231 with
an interval spanning zero ([-.003752,.026724]). These are adaptive development
comparisons, not confirmatory statistical claims.

The primary selection-corruption gaps are .000556, -.000017, .010385 and
.019130; the two high-gain gaps are smaller than in the audit table. Both
sampling and corruption realizations differ; corruption sensitivity may
contribute. The independent panel does not retrospectively change
the failed primary screens or select another checkpoint.

Across four individual K64 sampling draws at fixed corruption, high-gain K8
has gaps .01570–.02011; K32 has .02529–.03285. Separate corruption perturbations
retain positive gaps. At evaluation K256, the K8-trained model has gaps .02159
and .01825, and the K32-trained model has .02880 and .03216. Equal seeds across
different particle budgets do not imply nested particles for all candidates.

## Particle-budget replication

After the seed-1703 audit, a recorded follow-up plan added exactly two runs at
seed 1704: high-gain K8 and K32, with every other setting unchanged. This was
an adaptive follow-up, not a preregistered independent-data experiment.
Both runs completed 1000 steps and failed the primary .03 screen; minimum-NLL
fallback selected steps 850 and 950. Primary gaps were .016368 and .023055.

The same independent probability-averaged audit gives:

| Training particles | NLL | Association gap | Descriptive 95% gap interval |
|---|---:|---:|---|
| 8 | .436196 | .028106 | [.005760, .056084] |
| 32 | .431240 | .037388 | [.006804, .072329] |

K32 improves NLL by .004957, but the paired descriptive interval
[-.010455,.022073] includes zero. Thus both new seeds have better point estimates
for K32 prediction and association, while the additional prediction benefit is
not clearly separated from population uncertainty in the replication. One
post-selection audit gap above .03 does not change the failed primary screen.
K32 remains an explicit experimental setting; the default training budget has
not been changed. The initialization result now has positive audit association
in four optimization seeds, all on the same adaptively reused population.

## Scope and reproduction

The population is unchanged: 91 development candidates from 31 source
components, with 40 nonconstant visible histories. This repeats initialization
sensitivity on additional optimization seeds, not on a new dataset. It does not
establish interpretable bug identity, active testing or repair control.

Use `scripts/run_association_debug.py` with `--source runbugrun`, the immutable
semantic cache, `--seed 1703 --steps 1000 --eval-particles 64
--selection-replicates 4`, and the model/particle choices in the table.
Exact commands, source snapshots, data hashes, checkpoints and negative
controls are recorded under `/root/pbpf-runs/selection-fix-20260917/`.
The pre-run plan is `plan.json`; the independent panel is `independent-audit/`.
`sampling-averaged-comparison.json` and its prediction NPZ preserve the inputs
to the table and bootstrap comparisons.
Seed-1704 counterparts are `replication-audit/` and
`replication-sampling-averaged-comparison.json`. The complete follow-up has six
training runs, with none dropped. The runner regression suite passes all 10
tests, including mean-probability scoring, fixed corruption, independent
verification seeds and artifact generation. This is a focused check, not a new
full-suite run. Independent review found no implementation/provenance blockers;
two overbroad statements about primary-versus-audit gaps were corrected.
All 834 files listed by the six run and two audit checksum manifests were
verified, and every training source snapshot matches the current source.
The companion `diagnostic_selection_fix_2026-09-17.json` summarizes all six runs,
both audits, both averaged-prediction comparisons and verification metadata.
