# Baselines and ablations

`configs/baselines.yaml` is the machine-readable provenance ledger. Each entry
records a paper URL, repository URL when verified, pinned upstream or local
implementation revision, license hash, execution availability, and one of
`official_adapter`, `paper_spec_reimplementation`, or
`controlled_ablation`.

The representation matrix includes raw transcript, last observation, windows
1/2/4, orderless set, pass rate, equal-parameter GRU and exchangeable encoders,
MAP, posterior mean, P-way full-forward ensemble, matched-norm random latent,
single shared KV delta, PBPF soft prompt, and PBPF low-rank KV. State injection
separates text, soft prompt, K-only, V-only, K+V, active-layer count, prefix reuse,
rank, and Frobenius norm. The token-wise re-mixture arm is a named fault ablation;
it is not a PBPF implementation.

Repair plans name independent sampling, Self-Debug/raw transcript, REx,
RLEF paper-spec, and LDB under identical feedback, executions, rounds, candidate
bank, and visible-token budget. REx and LDB are pinned upstream references, but
their runtime adapters are not integrated and execution validation rejects them.
They are not results-producing arms in this repository.
RLEF is training with execution feedback; no verified author repository is
registered, so it is explicitly a paper-spec reimplementation and never labeled
official. The LDB paper consumes block-level runtime state; this repository does
not claim to provide that official adapter.

Rollout Roulette is the primary particle-inference neighbor, but its particles
are language-generation trajectories. PBPF particles represent hidden
program/failure hypotheses updated by ordered execution evidence, including the
transition/proposal ratio. RSP is a matched-norm random-latent negative control.
UpSkill is optional. LaDi-RL remains disabled until an independent code audit.

The GRU and exchangeable CPU controls have equal trainable parameter counts but
different measured matrix-vector operation counts. They are therefore not
matched-compute baselines: execution validation rejects the legacy
`matched_gru` and `matched_exchangeable` arm names until an audited compute
matching protocol is registered.

Filtering ablations cover P=1/4/8/16/32, learned versus uniform weights,
no/every-step/ESS resampling, rejuvenation off, omitted proposal correction,
independent/correlated likelihoods, and leave-one-candidate-out. History and
feedback variants are reported in separate tables; they are never silently pooled.
