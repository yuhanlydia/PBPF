# EESD ICLR 2027 execution checklist

Branch: `research/eesd-iclr2027`

This is the single checklist for completing the paper experiments. Do not delete
or overwrite unfavorable cells.

## Phase 0 — verify repository

```bash
git pull
python -m pip install -e '.[ml,experiment,test]'
python -m pytest -q -m 'not gpu and not network'
python -m compileall -q src scripts tests
```

## Phase 1 — mechanism factorial

Run ordinary / fixed-mass / effective-evidence under the complete alpha x strength
grid and the same-alpha controls on RunBugRun and CodeARC. Required outputs:

- selected-system table;
- same-alpha 0.10 table;
- same-alpha 0.01 table;
- tuned global-mass control;
- constant-mean-mass control;
- permuted-mass control;
- visible-count sweep 1/2/4/8;
- concentration-bin report;
- five-class vs binary report.

## Phase 2 — cross-model mechanism replication

Required model configs:

- Qwen2.5-Coder-7B-Instruct
- DeepSeek-Coder-6.7B-Instruct
- Qwen3-8B
- Qwen2.5-Coder-1.5B-Instruct

Run all predeclared cells on RunBugRun and CodeARC.

## Phase 3 — correction trust analysis

Create original->repair transition records:
FIX, REGRESSION, PRESERVED, UNRESOLVED.

Produce quartile tables for EESD utility / effective evidence against:
fix rate, regression rate, net gain and final correctness.

## Phase 4 — one-round self-distillation

Matched correction bank and matched optimization budget across:
no-update, equal-weight SFT, correctness-filter SFT, scalar-confidence weighting,
fixed-mass Dirichlet SD, EED mean utility, EED no-KL, full EESD.

Primary endpoints: Pass@1 and regression rate.

## Phase 5 — recursive self-improvement

Rounds 0,1,2,3 on Qwen2.5-Coder-7B for RunBugRun and CodeARC.
Freeze evaluation population across rounds. Generate new training/correction data
each round. Report retained correctness and accumulated regressions.

## Phase 6 — downstream transfer

Evaluate the learned adapters/policies on EvalPlus (HumanEval+ and MBPP+) and the
locked LiveCodeBench temporal slice. These are downstream correctness evaluations,
not reinterpreted as public-input/private-outcome mechanism data.

## Phase 7 — paper aggregation

Every JSON result must map to a row in:
`docs/EESD_ICLR2027_EXPERIMENTS.md`.

The paper is experiment-complete only when all mandatory checkboxes in that file
are filled or explicitly reported as null/negative.
