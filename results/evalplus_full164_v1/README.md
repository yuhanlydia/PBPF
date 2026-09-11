# EvalPlus HumanEval 164 real-model repair run

This directory records the aggregate result of the pinned Qwen2.5-Coder-7B
repair screen run on all 164 HumanEval tasks. The immutable per-task artifacts
remain in the worker run directories under `/root/pbpf-runs/`; this checked-in
summary is the publication pointer and does not replace those reports.

The run used the repository's executable local Python sandbox and the pinned
Qwen revision from `configs/experiments/repair_7b_24gb.yaml`. It is a real
repair-chain diagnostic: initial candidates, four fixed repair rounds, and
post-repair evaluator checks were recorded for every task. The particle
likelihood in this pilot was intentionally fixed and therefore this result is
not a learned PBPF-versus-matched-baseline comparison or formal evidence.

The three immutable worker batches were:

- `evalplus-full164-v1` (tasks 0--67)
- `evalplus-full164-v1-part2` (tasks 68--128)
- `evalplus-full164-v1-part3` (tasks 129--163)

