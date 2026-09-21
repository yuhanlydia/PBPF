# EESD execution settings locked before new generation (2026-09-20)

This execution supplement resolves missing runner settings before any new
mechanism result is observed. The alpha and relevance grids remain unchanged.

- Candidate-bank seeds: 1701, 1702, 1703. Each seed gets separate banks, cache,
  and reports. Use each generator's existing sampling policy (temperature 0.8,
  top-p 0.95), one candidate per task. The old orchestrator's greedy decoding
  would produce deterministic duplicates across seeds and is corrected here.
- Global mass search: 0.25, 0.5, 1, 2, 4, 8, 16; select using development NLL.
- ECE: 10 bins. Paired source-cluster bootstrap: seed 314159, 10,000 draws.
- Mass permutations: seeds 1701, 1702, 1703.
- Effective mass bins for history size 4: [1,2), [2,3), [3,4], including 4.
- A2/A3 report paired fixed/effective metrics at alpha 0.10/0.01 for every
  predeclared strength, without assessment selection.
- A6 uses mean effective mass computed from assessment public input features,
  never assessment outcomes, to hold the assessment mean mass exactly fixed.
  Development mean mass remains an additional diagnostic.
- A8 uses post-generation history lengths 1,2,4,8. Different lengths change the
  future target set. An unavailable length is explicitly reported with its
  reason and is not silently dropped.

These are prospective technical settings, not result-selected changes.

## User scope amendment before generation

The user reduced the required model matrix to four different model families:
Qwen2.5-Coder-7B, DeepSeek-Coder-6.7B, Seed-Coder-8B and
StarCoder2-15B-Instruct-v0.1. StarCoder2 uses `bigcode/starcoder2-15b-instruct-v0.1`
at revision `ffb8dd9776ba9a66d655ecd962e882f3013e9f7c`.
Thus two datasets times four models times three seeds require 24 reports.
Qwen2.5-Coder-1.5B, Qwen3-8B and Qwen3-Coder-30B-A3B remain optional historical
configurations; they are excluded from the required manifest and downloads
before observing new experiment results. Other phases retain their declared scope.

## Resume integrity and paired controls

Every mechanism resume verifies candidate-bank checksums and the current model,
revision, seed, split, public inventory, sampling settings and generation sources.
Caches are sealed at creation with a `.binding.json` receipt tying their bytes to
those banks, evaluator inputs and execution source files. Existing caches without
a seal are preserved and rejected for automatic reuse. Completed evidence reports
must match their artifact checksums, config, cache, seed and estimator sources.
Development and assessment source components must be disjoint. Each predeclared
A2/A3 alpha/strength pair reports the same paired source-cluster bootstrap used
elsewhere (10,000 draws, seed 314159).


## CodeARC public prompt length protocol

Before first generation, CodeARC public prompts are capped at 4,096 input tokens
by adaptively clipping public field text while retaining all four public example
slots. This preserves the predeclared example count; it does not select examples
using outcomes or silently truncate the tail of the assembled prompt. The bank
identity records `max_input_tokens=4096`, the field-clipping `prompt_policy`, and
SHA-256 of `src/pbpf/apbpf/codearc_prompt.py`; resume validates all three against
the frozen execution protocol. The shared chat-format source digest remains
bound separately to `src/pbpf/apbpf/rbr_prompt.py`.

## Seed tokenizer compatibility amendment (before Seed formal generation)

A real FP4 public-only readiness probe found Seed-Coder's tokenizer emits
`token_type_ids` that its model refuses. Two explicit Seed-only generator copies
disable that unused tensor (`return_token_type_ids=False`). The original RBR and
CodeARC generators, prompts, token IDs, sampling settings and populations are
unchanged. The matrix and operational queue select the Seed-specific source and
verify its own SHA. Existing Qwen banks still pass the identity verifier.

The prior execution lock is retained; the additional immutable receipt is
`runs/eesd-setup/execution-lock-v2-seed-compatibility.json`. Pre-amendment runner
and queue source snapshots are retained in the setup directory. No scientific
outcomes had been computed or consulted for this compatibility amendment.
