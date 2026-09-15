# PBPF candidate-selector gate

PBPF's verified positive result is future-outcome prediction, while direct
soft-prefix repair conditioning failed its small gate. The lower-risk follow-up
is therefore candidate selection: generate a fixed group of real programs,
observe the same cheap executions for every candidate, and select exactly one
program using a candidate-specific future-success estimate.

## Required data shape

The independent unit is a source-problem group, not an individual program. Each
group needs:

- 8 real samples from the same frozen actor and prompt;
- at least 4 immutable cheap tests per candidate;
- one evaluator-held full-suite label per candidate;
- source-problem-disjoint train, development and test partitions;
- enough mixed groups containing both successful and unsuccessful candidates.

All-fail and all-pass groups measure task difficulty but contain no within-group
selection information. A useful pilot should contain at least 300 mixed groups;
the confirmatory target is 500–1,000 test groups plus separate fitting data.
Report the number of mixed groups rather than only the total number of programs.

The matched primary comparison is PBPF-selected Pass@1 against visible pass
rate, a cross-fitted deterministic predictor, and random selection. Oracle
Pass@K is only the ceiling. PBPF advances only if it improves selected Pass@1
by at least 3 absolute points over the strongest deterministic selector and the
source-problem-clustered 95% confidence interval is above zero.

## Audit of the existing GOAV EvalPlus bank

The published GOAV diagnostic bank contains 164 HumanEval+ groups and 8 frozen
Qwen2.5-Coder-7B candidates per group, for 1,312 programs. Its trusted labels
contain 89 all-fail groups, no all-pass groups, and only 75 mixed groups.

| Selector | Full 164-task Pass@1 |
|---|---:|
| Random candidate expectation | 12.73% |
| First visible cheap test pass rate | 45.12% |
| First two cheap tests pass rate | 45.12% |
| First four cheap tests pass rate | **45.73%** |
| Oracle Pass@8 | **45.73%** |

On the 75 mixed groups, four-test pass rate selects a trusted-success candidate
in 75/75 cases. It therefore reaches the Oracle Pass@8 ceiling before PBPF is
introduced. This bank is useful for validating the audit code, but it cannot
identify an advantage over the required simple baseline and must not be used as
a positive PBPF selector experiment.

The next bank must include harder near-miss candidates for which cheap pass
counts tie or conflict with the full-suite outcome. Diversity means varied
failure behavior within each task, not merely more candidates or more repeated
tests.

## Reproduction

Run `scripts/audit_selector_bank.py` against the GOAV `bank.jsonl`, `bank.npz`
and trusted `sidecar.json`. The committed machine-readable result is
`results/pbpf_selector_bank_audit.json`.
