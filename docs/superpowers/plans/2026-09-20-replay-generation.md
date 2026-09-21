# Replay Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and seal the 24 new Replay domain/model/seed banks, each containing 200 development and 500 primary sources, without changing original experiment generation or scheduling.

**Architecture:** A new public-only library validates admission and delegates exact prompt reproduction to root-owned `pbpf.eesd.replay_prompt`. A new generator uses the existing pinned Transformers/BitsAndBytes loading recipe and immutable completion helpers. A separate manifest planner routes 24 cells to 48 split directories; it never imports the old matrix dispatcher.

**Tech Stack:** Python 3.11, Transformers 4.57.6, PyTorch, BitsAndBytes FP4, pytest, JSON.

**Spec:** `docs/EESD_REPLAY_EXTENSION_SPEC_20260920.md`.

**Status:** Implementation proposal only. This document does not launch generation, change the active queue, or authorize execution of candidate programs.

## Global Constraints

- Domains: `apps_replay`, `codecontests_replay`; families: `qwen25_7b`, `deepseek_6p7b`, `seed_coder_8b`, `starcoder2_15b`; seeds 1701/1702/1703.
- Exactly 200 development and 500 primary unique source components per domain, same ordered populations for all models/seeds. 24 domain/model/seed banks mean 48 split inventories and 16,800 candidates total.
- One sampled candidate/source, temperature .8, top_p .95, max_new_tokens 1024, input cap 4096; no truncation, clipping, alternate samples or filtering by completion quality.
- Public input whitelist: task_id/source_component_id/split/domain/statement/visible_tests; each visible test has only input/output, exactly four.
- New generation must never open evaluator/tasks.jsonl or raw dataset solution fields. Admission metadata may contain evaluator *hashes*, not test contents.
- All source/model/template/tokenizer identity changes invalidate resume. No adapter, offset, train split, greedy mode or alternate scientific parameters in this first entrypoint.
- Do not edit old generators, `rbr_prompt.py`, `codearc_bank.py`, `run_eesd_matrix.py`, `run_eesd_evidence_matrix.py`, `runs/eesd-setup/generate_mechanism_banks.py`, or the current queue/manifests.
- No GPU or candidate execution during implementation validation. Production launch is a separate integration step owned by the root agent.

## Review Focus

1. Public loader accidentally follows evaluator paths: poison evaluator files/read spies must prove generation reads public data and audit metadata only (Task 1).
2. A superficially identical prompt changes JSON escaping/newlines: require all three exact sealed hashes and input token count, before model loading (Tasks 1–2).
3. A prior record is internally hashed but belongs to another source/seed: compare external planned identity and recomputed task seed, not complete.json alone (Task 3).
4. Interrupted candidate/checksum publication can duplicate sampling: recover only verified immutable record or stop with an explicit incomplete-publication error; never silently overwrite/re-generate (Task 3).
5. Seed tokenizer supplies token_type_ids, StarCoder requires `###` stop: assert exact generate inputs and family stop policy with fakes (Task 2).

## Observed immutable inputs and code reuse

Published bundle: `runs/eesd-data/replay-extension-locked-20260920`.
Admission SHA256: `73d96f7b0054726aef27b007e8a789ea22a739cfd338e9d096360261dbfbd40f`.
The admission schema is `eesd-replay-admission-v1`, status `data-admitted-not-generated-not-scored`.
It seals public manifest/tasks and `selected-token-audit.jsonl` through `{sha256,bytes}` entries in outputs.
Each public manifest is `eesd-replay-public-v1`, task_kind `stdin_synthesis`, domain, counts `{development:200,primary:500}`, tasks checksum/bytes.
Each selected audit row uses `source_id` (map explicitly to public `source_component_id`), task_id, domain, message_sha256 and models[family] with input_tokens/rendered_prompt_sha256/token_ids_sha256/fits_4096.

Actual successful raw-IO probe:
- `runs/eesd-setup/replay-token-probe/raw-joint-3033/receipt.json`
- `runs/eesd-setup/replay-token-probe/raw-joint-3033/prompt-template.txt`
- Template SHA256 `3718d7f2222d7d3c75276467a2fd3cf249b59daa8a0d5bdb03286aa2f3a46d60`.
- Policy `eesd-replay-synthesis-full-public-v1`; Transformers 4.57.6.
- Admission evidence seals the token receipt path/hash and each model revision/chat template/tokenizer-file hashes. Verify that chain, not an arbitrary user-supplied template.

`runs/eesd-setup/generate_mechanism_banks.py` is a dispatcher tied to old matrix/domain verifiers and executes argument parsing at import; do not reuse it by import or route Replay through it.
`generate_apbpf_rbr_bank.py` and `generate_apbpf_seed_rbr_bank.py` have inline model loading, not a reusable loader API. Copy the short loading recipe into the new generator; do not refactor frozen sources.
Reuse imports only `extract_program`, `generation_stop_kwargs`, `decode_completion` from `pbpf.apbpf.rbr_prompt`, binding its SHA. Never call its `prompt_for` or `chat_prompt`: these construct repair prompts and may clip fields or change role layout.
`pbpf.apbpf.codearc_bank.load_bank` can check generic record hashes/inventory, but cannot certify Replay schema/seed/public identity. New Replay verifier owns those checks and may delegate generic integrity to it, recording its SHA if used. Do not call old `verify_hidden_lock` (allowed provenance prefixes are only RBR/CodeARC).
Do not call `replay_admission.verify_inputs` in generation: admission publication validation opens private candidate projections. Implement the narrower public-chain reader separately.

## Exact prompt/token construction (root-owned shared helper)

Root is implementing `src/pbpf/eesd/replay_prompt.py`: sealed TEMPLATE/messages_for plus render_and_verify. Generation must import that helper and bind its source SHA, not duplicate its template or rendering logic. The code below documents the exact required behavior; helper argument naming should follow the actual root implementation at integration.

Read the sealed template bytes with UTF-8 and retain its trailing newline. The actual template is:

```text
Write a complete Python3 program that reads standard input and writes standard output.
Solve the task below. Return only the complete program, without explanation.
The four designated observations are JSON strings preserving the exact input/output text.

TASK STATEMENT
{statement}

FOUR DESIGNATED VISIBLE OBSERVATIONS
{observations}
```

Use the exact successful probe serialization, not a paraphrase:

```python
observations = [
    f'Observation {i}\nInput: ' + json.dumps(t['input'], ensure_ascii=False)
    + '\nOutput: ' + json.dumps(t['output'], ensure_ascii=False)
    for i, t in enumerate(row['visible_tests'], 1)
]
messages = [{'role': 'user', 'content': template.format(
    statement=row['statement'], observations='\n\n'.join(observations))}]
message_bytes = json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tokenizer(prompt, add_special_tokens=False, return_token_type_ids=False,
                truncation=False)['input_ids']
ids_bytes = json.dumps(ids, separators=(',', ':')).encode()
```

Compare SHA256(message_bytes), SHA256(prompt.encode()), SHA256(ids_bytes) and len(ids) to the matching selected audit row. No thinking/system-message overrides. Revalidate tokenizer file/chat-template hashes and exact revision before these checks. Perform complete split CPU token preflight before creating/loading the GPU model. Build generation tensor from those verified ids, or retokenize with the same flags and demand exact equality, not merely equal lengths.

## Task 1: Public admission loader and exact prompt preflight

**Files:** Create `src/pbpf/eesd/replay_generation.py`; test `tests/eesd/test_replay_generation.py`.

**Interfaces:**
- `load_public_split(bundle: Path, admission_sha256: str, domain: str, split: str, family: str) -> dict`: returns rows in public-file order, matching audit rows, model identity, template text and verified binding digests; opens no evaluator data.
- `prepare_prompt(tokenizer, row: dict, audit: dict, template: str) -> dict`: narrow wrapper around root-owned `replay_prompt.render_and_verify`, returning messages/prompt/ids and verified three hashes/count; additionally requires the supplied sealed template to equal the shared helper TEMPLATE bytes. Do not independently implement rendering.

- [ ] Write failures for wrong admission SHA, public bytes/hash drift, unknown fields, duplicate task/source, wrong domain/split/count, missing/duplicate audit identity, any model not in the exact four, audit all_four_fit false, and altered tokenizer/template. Use a fixture factory with 200/500 small public rows, four-model audit metadata and poisoned evaluator files.
- [ ] Add deterministic fake tokenizer tests: `assert prepare_prompt(... )['ids'] == expected_ids`; mutate CRLF, Unicode, JSON whitespace, final newline or one id and require `ValueError`. Assert a monkeypatched file reader raises if evaluator/raw paths are opened; successful public preflight must not trigger it.
- [ ] Run `.venv/bin/python -m pytest tests/eesd/test_replay_generation.py -q` and record expected missing-module failures.
- [ ] Implement whitelist validation and admission→token-receipt→template/model bindings; preserve task order. Delegate prompt construction/hash checking to the root-owned `replay_prompt.render_and_verify`; require its TEMPLATE hash to equal the sealed template hash.
- [ ] Re-run focused tests to green; CPU-only real preflight should match all 1,400 selected task prompt hashes for each of four tokenizers (no model load).

## Task 2: Independent sampled synthesis CLI

**Files:** Create `scripts/generate_eesd_replay_bank.py`; test `tests/eesd/test_replay_generator_cli.py`.

**Interface:** Required `--bundle`, `--admission-sha256`, `--domain`, `--family`, `--split`, `--seed`, `--output`; `--preflight-only` performs public/token validation and exits before torch/model loading. Seeds/splits/families restricted as above. Scientific limits fixed rather than adjustable CLI defaults.

- [ ] Write fake model/tokenizer tests asserting no GPU model constructed if one prompt audit fails; assert one generate call per source with do_sample=True, num_return_sequences=1, max_new_tokens=1024, temperature=.8, top_p=.95, and no token_type_ids.
- [ ] Write tests for deterministic task seed `int.from_bytes(sha256(f'{seed}:{task_id}'.encode()).digest()[:4], 'big')`, `torch.manual_seed(task_seed)`, StarCoder's dual EOS/### stop, and no candidate rejection on empty/invalid Python completion.
- [ ] Run focused test file and observe missing entrypoint failures.
- [ ] Implement the existing loading recipe locally: pinned `AutoTokenizer.from_pretrained(...,local_files_only=True)`, pad_token=eos_token, padding_side=left; `AutoModelForCausalLM.from_pretrained(...,device_map={'':0},torch_dtype=torch.bfloat16,local_files_only=True,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_quant_type='fp4',bnb_4bit_use_double_quant=False))`; model.eval/inference_mode. Existing omitted quant type defaults to FP4; make it explicit and record the realized quantization config/dependency versions.
- [ ] Record raw completion, extracted code, input_tokens, generated_tokens, stop policy and hit_token_cap from the immutable helpers; keep all sampled outputs. Re-run fake tests, no GPU.

## Task 3: New Replay bank schema, atomic resume and verification

**Files:** Extend new `replay_generation.py` and new generator only; test `tests/eesd/test_replay_generation_resume.py`.

**Interfaces:** `verify_replay_bank(bank: Path, expected: dict) -> dict` returns verified run/rows/complete digest. Expected is built from current public preflight, never solely from the bank itself.

Output per split directory:
- `run.json`, schema `eesd-replay-generation-v1`: task_kind stdin_synthesis, domain/family/model/revision/split/seed, exact ordered task_ids/source_component_ids, components, candidates=1, all fixed decode parameters, explicit quantization policy, admission/public-manifest/public-tasks/audit/template/token-receipt hashes, generator/library/replay_prompt/immutable-helper source hashes and tokenizer bindings, adapter=None. Persist package versions and task-seed derivation.
- `<task_id.replace('/', '-')>.json` and `.sha256`: task_id/source_component_id/domain/split/task-derived seed; three prompt hashes/count; one candidate ID `{task_id}/{family}/0`, raw_completion/code/generated_tokens/hit_token_cap. Reject filename collisions before writes.
- `complete.json`, schema `eesd-replay-generation-complete-v1`: run_sha256 and exact filename→SHA map, written only after verifying the whole expected inventory. No metrics/outcome labels.

- [ ] Write failure tests for changed source/admission/seed/model, record tampering, checksum absent, extra/missing files, duplicate IDs, filename collisions and crosssplit/source swaps. Simulate interrupted record/complete writes.
- [ ] Implement create-once run identity, temporary-file atomic publication, strict immutable resume and complete verification; never skip solely because complete.json exists. Resume existing records only after validating external identity and recomputed task/prompt bindings. Stop on an orphan partial artifact with an actionable infrastructure error; retain it without overwriting scientific data.
- [ ] Verify fake partial resume makes no generate call for verified records, and complete resume makes no model load. Re-run this test file plus Tasks 1–2.

## Task 4: Independent manifest and launch plan, no active queue integration

**Files:** Create `scripts/plan_eesd_replay_generation.py`; test `tests/eesd/test_replay_generation_plan.py`. Its output manifest is a new requested path, not an existing experiment configuration.

**Interface:** `--bundle --admission-sha256 --output-root --manifest-out` emits create-once schema `eesd-replay-generation-matrix-v1` with 24 cells, each containing domain/family/seed, 200/500 split commands, sealed input/source hashes and expected output locations. Planner executes no commands.

Output layout: `<output-root>/replay-mechanism-banks/{domain}/{family}/seed{seed}/{development|primary}`. This avoids masquerading as old mechanism domain artifacts.

- [ ] Test exact Cartesian coverage 2×4×3, 48 commands, source population shared across model/seed, shell-free argv arrays, no evaluator CLI arguments and no change to original manifest files.
- [ ] Build argv directly for `generate_eesd_replay_bank.py`; fail if required admission population or family is missing. Write generation-only status, not experiment completed.
- [ ] Run all four new focused test files; perform actual public/token CPU preflight and save its receipt with hashes/counts; do not launch GPU work.
- [ ] Handoff proposed manifest and test evidence to root for separate scheduler integration. Existing GPU reservation/orphan detection policies are operational reuse, not a reason to modify the running dispatcher here.

## Freeze and scientific boundary review

The generic bank layout does not make old cache/evidence validators accept these domains. Execution, Replay cache construction and new 49-contrast inference-family registration are separate work after candidate sealing; this plan does not change them. Existing `execute_stdin` environment remains blocked by namespace support and must never be bypassed. Do not report Replay generation as official APPS/CodeContests pass@1.

Minimum freeze set before first new bank: new generator/library/planner/replay_prompt source hashes, imported completion-helper SHA, manifest SHA, admission/public/audit/token/template identities and explicit model loading/decode policy. Any later code change requires a new run identity/output; it must not silently resume an earlier bank.

No commits or queue changes are part of this bounded plan delivery.

## Implementation evidence (2026-09-20)

- New public loader, prompt preparation, generation CLI, immutable bank verifier and manifest planner implemented. Root independently ran the four focused test files: 44 passed. A final library/CLI integration check with fake model execution is pending before scheduler handoff.
- Published production manifest: `runs/eesd-setup/replay-generation-manifest-20260920.json`, SHA `f0fcced94556984ec554f272a5adcf2df1f05d1d2d62f09a3f5120fa82f81f52`. Its read audit confirms no evaluator/raw reads. It contains 24 cells, 48 split commands, 16,800 planned candidates.
- Frozen generator SHA `1ea1dd4a54609a08ef47b77ce48ba8752cd5848be8dcb92f1c4b11ee72eaa188`, public library SHA `70f3ce971df405b4159863598f983c59aed79b57b61b5cfcb7e046cf1865394c`, prompt SHA `13d2cc7e0fe43992884cc26eebeb39107d8407a4df79bf9c8fb383ca3808112e`.
- Ruling: avoid a redundant whole-population four-tokenizer rerun. The original sealed probe covers every admitted member; the new shared helper checks all 1,400 message hashes and eight real tokenizer samples; production performs full-split exact rendered/token checks before every GPU load. The public loader has additionally passed all 16 domain/split/family combinations. This does not replace the mandatory per-bank preflight.
- Existing official model weight proof SHA is now bound in each run; loader checks revision/status, nonempty safetensor inventory, published SHA equality and current local file sizes. It does not rehash all weight bytes for every bank.
- Restart limitation: published or partially published attempts are protected. A crash after decoding but before any publication artifact exists may cause a same-seed retry. No claim of exactly-once decode attempts.
- No Replay GPU generation has started at this checkpoint. Operational queue integration is separate from these frozen scientific sources.
