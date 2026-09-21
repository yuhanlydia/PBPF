# Replay Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Produce verified, separate public/evaluator 200-development/500-primary task bundles for APPS-Replay and CodeContests-Replay from the joint graph and four-model token audit.

**Architecture:** A pure selection/validation module consumes hash-bound research inventories and emits a deterministic common population. A create-once CLI writes both views under one new bundle directory. It neither loads model weights nor executes candidate programs; generation/execution integration is a subsequent subsystem.

**Tech Stack:** Python 3.11, pathlib, hashlib, json, existing pytest environment.

**Spec:** `docs/EESD_REPLAY_EXTENSION_SPEC_20260920.md`.

## Global Constraints

- Four pinned model families and seeds 1701, 1702, 1703; exactly 200 development and 500 primary components per admitted domain.
- The complete source graph precedes token filtering and includes ineligible nodes and transitive connections.
- Require all four tokenizers to fit 4096 input tokens, without clipping the statement or four designated tests.
- CC-first assignment occurs after all-model token fit; one component supplies at most one domain.
- Preserve official test-pool provenance; exclude identified public inputs from the six targets.
- Do not change frozen mechanism sources, the current eight-cell matrix, its families, or existing training scope.
- This deliverable creates no generated bank, execution score, or formal efficacy claim.

## Review Focus

- A token audit for a different prompt or task population must fail checksum/identity validation (Task 1).
- Two fitting domain members in one component must yield only the CC representative (Task 1).
- An ineligible bridge to a quarantined source must remain excluded, even if another member fits (Task 1).
- Public projection must not contain target tests, provenance-only expected outputs, or solution/reference fields (Task 2).
- Existing output roots or an insufficient domain population must fail before publishing a partial bundle (Task 2).

### Task 1: Deterministic population selection

**Files:**
- Create: `src/pbpf/eesd/replay_materialization.py`
- Test: `tests/eesd/test_replay_materialization.py`

**Interfaces:**
- Consumes public member records with `domain`, `task_id`, `source_id`, `statement`, `visible_tests`; the receipt-bound full graph identifies quarantined components.
- Consumes token rows with exact task/source/domain identity, message SHA, four model identities and token counts/hashes, plus the probe receipt binding public input bytes and prompt source/template.
- Produces `select_population(public_rows, token_rows, quarantined_sources, *, development=200, primary=500)` returning per-domain lists of `(split, member)` in deterministic source rank order. Production CLI permits only 200/500; smaller counts are solely for pure-function tests.

- [x] Write failing tests with shared-source members, a non-fitting CC member and fitting APPS member, a quarantined bridge, missing model token rows, duplicate task IDs, and a domain with too few sources.

```python
def test_cc_first_uses_fitting_member_only():
    rows = [member('apps_replay', 'apps:1', 'shared'),
            member('codecontests_replay', 'cc:1', 'shared')]
    selected = choose_representatives(rows, token_rows(rows, fits=True), set())
    assert [r['task_id'] for r in selected] == ['cc:1']

def test_quarantine_survives_fitting_members():
    rows = [member('apps_replay', 'apps:1', 'bridge')]
    assert choose_representatives(rows, token_rows(rows, fits=True), {'bridge'}) == []
```

`member` and `token_rows` are local test helpers producing complete records;
`choose_representatives(public_rows, token_rows, quarantined_sources)` is the
pure representative-selection helper exposed by this module.

- [x] Run `.venv/bin/python -m pytest tests/eesd/test_replay_materialization.py -q` and confirm the new imports/tests fail before implementation.
- [x] Implement exact identity checks and selection, using no outcomes or reference code:

```python
def rank(prefix, *parts):
    return hashlib.sha256((prefix + '|'.join(parts)).encode('utf-8')).hexdigest()

# Group by source_id after validating the fixed graph/identity mapping.
# Among all-model-fitting members, choose the CC domain when present,
# then minimum (rank('eesd-replay-representative-v1|', task_id), task_id).
# Sort representatives within each domain by
# (rank('eesd-replay-split-v1|1701|', domain, source_id), source_id).
# Select first development, then primary; insufficient counts raise ValueError.
```

- [x] Confirm focused tests pass and inspect that token filtering never reconstructs or weakens the source graph. Thirteen tests pass; independent review found and prompted a tested fix for a bare-string quarantine argument. These checks do not certify the upstream graph or receipts.

### Task 2: Bind private records and publish separated bundles

**Files:**
- Extend: `src/pbpf/eesd/replay_materialization.py`
- Create: `scripts/materialize_eesd_replay_extension.py`
- Extend tests: `tests/eesd/test_replay_materialization.py`

**Interfaces:**
- `project_views(selected_rows, evaluator_members)` returns public and evaluator rows for exactly the chosen IDs, source IDs and splits.
- CLI arguments: `--joint-receipt`, `--public-candidates`, `--evaluator-candidates`, `--token-receipt`, `--token-results`, `--output`. Receipts bind all input paths/bytes and complete graph evidence; actual field names must be checked against the completed inventory/probe receipts before coding the adapter.
- Output: one create-once bundle containing per-domain `public/tasks.jsonl`, `public/manifest.json`, `evaluator/tasks.jsonl`, `evaluator/manifest.json`, and a bundle-level admission receipt. Public/evaluator schemas identify stdin synthesis explicitly.

- [x] Add failing tests for target/public input collision, test order changes, missing provenance, source mismatch, extra private fields in the public source row, token-receipt input tampering and pre-existing output.

```python
def test_projection_uses_public_allowlist():
    row = selected_member_with_injected_private_fields()
    public, private = project_views([row], evaluator_members_for(row))
    assert set(public[0]) == {
        'task_id', 'source_component_id', 'split', 'domain',
        'statement', 'visible_tests',
    }
    assert len(public[0]['visible_tests']) == 4
    assert len(private[0]['tests']) == 10
    assert all('original_pools' in t for t in private[0]['tests'])
```

The fixture helpers above provide exact raw string IO and known-public input
fingerprints, with ten unique inputs and four designated visible slots.

- [x] Run the new tests and confirm meaningful failures.
- [x] Validate all inputs before creating output. Preserve raw statement/IO bytes; use an explicit public field allowlist. Place exact ten-test provenance and target data only in evaluator rows. Bind graph, raw versions, selected IDs, prompt, tokenizer identities, token hashes and spec SHA in the admission receipt.
- [x] Write both views beneath a sibling staging directory, then publish by one directory rename after all counts/checksums pass. Refuse existing destinations. A failure leaves no apparently complete public bundle.
- [x] Rerun focused tests, then use the real joint inventory and token audit only if both domains satisfy 700-source admission.
- [x] Independently verify file hashes, 200/500 source counts, cross-domain disjoint components, exact public statement/test equality, ten-test order and four-model input-length proofs. Record the verification receipt and limitations; do not label data admission an experiment result.

## Execution handoff

The user already authorized autonomous experiment preparation and supervision.
Use the existing agents for independent data/token preparation and a separate
review of materialization. Review the completed joint/probe schemas before
implementing this bounded subsystem; continue only from actual receipts, not
provisional inventory counts. Preserve the dirty experiment branch and make no
unrelated source, dependency or frozen-run changes.

### Execution record

2026-09-20: 49 focused tests passed. Final raw-IO joint receipt `2d965273dd5383e9ca5e95d35f765f9742c88667a047fbac6f6d5d49d616dd9a`; token receipt `8257ad4e5b03daf1e176baf6b249059d142ad26be08637bf2fddc9d8156921b4`. All-model fitting independent sources: APPS 1758, CC 995. Production CLI exit 0 published `runs/eesd-data/replay-extension-locked-20260920` with fixed 200/500 per domain. Independent final bundle verification is in progress. No Replay generation or candidate execution has started.

Final independent verification passed: `runs/eesd-setup/replay-extension-independent-verification-20260920.json`. Independently recomputed population; re-read all original statements and 14,000 native IO pairs; no candidate execution.
