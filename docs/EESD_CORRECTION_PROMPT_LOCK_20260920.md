# Prospective correction prompt input policy (2026-09-20)

This policy is chosen from public-only CPU token-length measurements, before
correction generation or hidden correction evaluation. No hidden outcomes inform
its selection. Evidence is recorded in
`runs/eesd-setup/correction-length-probe/{report.json,counts.jsonl,findings.md}`.
CodeARC task 508 has 1,128,840 public task-text characters; its unbounded repair
prompt exceeds 1.99 million tokens. The prior training limit of 1536 also fails to
reserve a 512-token answer for many ordinary public inputs.

## Locked model view

- Correction input defaults to 4096 tokens; response remains 512 tokens.
  Training must accommodate at least 4608 tokens and use the saved generation
  messages, rather than reconstructing a user-only prompt.
- Preserve the complete generated original program and all four public slots,
  including their IDs, outcomes and optional expected_error values.
- Only task_text and each slot's input, expected, actual and stderr may be
  shortened. For a clipped non-string value, render canonical sorted JSON first;
  the shortened model-view value is a string carrying an explicit marker.
- If total eligible public text is at most 16384 characters (four times the
  input-token budget), try the unchanged view first. This guard avoids first
  tokenizing megabytes. Otherwise start bounded construction directly.
- Try shared per-field character caps 2048, floor(3/4 of previous cap), down to
  zero. A clipped field retains ceil(cap/2) head and floor(cap/2) tail characters,
  separated by `[PUBLIC TEXT TRUNCATED: original_characters=N; retained_characters=C]`.
  Keep the first view whose actual chat-template tokenization fits. At zero,
  clipped fields retain only their explicit marker. If this minimal view cannot
  fit, reject the entire bank before loading model weights; never drop a task.
- Store exact system+user messages, the last user content as prompt, policy,
  character cap, clipped field names/lengths, actual input-token count and SHA256
  of compact JSON input token IDs. Generation rechecks exact token identity.
  Report the helper source SHA256 and input/output budgets.
- Render chat template with generation prompt, then encode without extra special
  tokens and with return_token_type_ids=False. Existing decoding is unchanged.

Execution and relevance continue to use the full original public tests and
execution descriptors. This changes only the model-visible text. All correction
arms use the same input policy. A future policy/budget change requires a new lock
and regenerated aligned data; previous and new policies must not be silently
pooled. Synthetic feedback probes are engineering checks, not repair results.
No candidate execution or GPU model loading is part of this CPU validation.
