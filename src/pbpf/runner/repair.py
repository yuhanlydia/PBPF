"""Public-only repair driver and an explicitly nonconfirmatory offline DAG.

OfflineExperiment is a CI/operator smoke backend: synthetic decoders and belief
probabilities exercise the real data, seal, RPC, statistics and repair-session
interfaces without model downloads. Its outputs are not experimental evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from pbpf.arms.base import ArmCandidate, RepairCondition, VisibleEvent
from pbpf.arms.repair import DecodeResult, FormalBank, RepairArm, selection_manifest
from pbpf.data.schema import PublicTask, PublicTest
from .aggregate import aggregate_stage_b, aggregate_stage_c, infrastructure_valid, require_main_tables
from .budget import BudgetLedger, MeteredActor, Usage, validate_matched
from .doctor import doctor
from .evaluator import HiddenEvaluator, VisibleVerifier, score_predictions, seal_final, seal_predictions
from .shard import atomic_write, canonical_bytes, digest, work_id
from .stage import STAGES
from .stats import ClusterPlan, PlanLock, reduce_metrics


def training_records(rows):
    rows = list(rows)
    if not rows or any(row.get("split") not in {"train", "dev-tune", "calibration"} for row in rows):
        raise ValueError("locked evaluation/dev-select labels cannot enter training")
    return rows


@dataclass(frozen=True)
class RootRequest:
    task: PublicTask
    seed: int
    slot: int
    max_new_tokens: int = 1024
    num_return_sequences: int = 1


def build_bank(task, actor, *, seed, model_id, model_revision, prompt_revision):
    """Eight genuine backend draws, never copied/padded or mutant roots."""
    if type(task) is not PublicTask or type(seed) is not int or seed not in (1701, 1702, 1703):
        raise ValueError("public task and fixed formal seed required")
    if type(actor) is not MeteredActor:
        raise ValueError("bank dispatch requires runner-owned MeteredActor and mandatory ledger")
    candidates = []
    for slot in range(8):
        result = actor.generate_root(RootRequest(task, seed, slot))
        if type(result) is not DecodeResult or result.output_tokens > 1024:
            raise ValueError("root actor must respect typed decode result and output cap")
        candidates.append(ArmCandidate(task.task_id, slot, 0, result.source, "actor_sample", model_id,
            model_revision, prompt_revision, sample_seed=seed, output_tokens=result.output_tokens))
    return FormalBank(tuple(candidates))


def run_repair_session(session, verifier):
    """Task-5 session owns selection; trusted verifier returns visible events only."""
    if type(session) is not RepairArm or type(verifier) is not VisibleVerifier:
        raise TypeError("typed repair session and visible verifier required")
    if type(session.actor) is not MeteredActor:
        raise ValueError("repair dispatch requires runner-owned MeteredActor and mandatory ledger")
    for _ in range(0 if session.name == "no_repair" else 4):
        child = session.propose()
        session.observe(child.version_id, verifier.execute(session.task, child))
    return session.finish()


class _OfflineBelief:
    """Deterministic smoke stub, explicitly never a trained PBPF model."""
    name = "pbpf"

    def __init__(self):
        self.accounting = {"belief_forwards": 0}

    def initialize(self, task, candidate):
        self.accounting["belief_forwards"] += 1

    def observe(self, version, event):
        self.accounting["belief_forwards"] += 1

    def spawn_child(self, parent, child, event):
        self.accounting["belief_forwards"] += 1

    def condition(self, version):
        self.accounting["belief_forwards"] += 1
        return RepairCondition(version, latent=(1.,), particle_index=0)


class _OfflineActor:
    def __init__(self, experiment, arm, work):
        self.experiment, self.arm, self.work = experiment, arm, work

    def generate_root(self, request):
        self.experiment.arm_inputs.append(asdict(request))
        self.experiment.actor_requests += 1
        return DecodeResult(f"print('bad') # {request.seed}:{request.slot}", 2)

    def generate(self, request):
        self.experiment.arm_inputs.append(asdict(request))
        self.experiment.actor_requests += 1
        source = "print('ok')" if self.arm == "pbpf" else "print('bad')"
        return DecodeResult(source + f" # {request.rng_seed}", 2)

    def native_usage(self):
        return Usage(input_token_ids=(11, 12), output_token_ids=(21, 22))


class OfflineExperiment:
    def __init__(self, root, *, fingerprint, fail_gate_b=False):
        self.root, self.fingerprint = Path(root), fingerprint
        self.fail_gate_b = fail_gate_b
        self.calls, self.actor_requests, self.arm_inputs = [], 0, []
        self.ledger = BudgetLedger(self.root / "budget")
        self.sandbox = [sys.executable, "-I", "-c",
            "import json,sys; r=json.load(sys.stdin); print(json.dumps({'outcome':'PASS' if \"print('ok')\" in r['source'] else 'WRONG_OUTPUT'}))"]
        self.arms = ("pbpf", "self_debug", "independent")

    def handlers(self):
        def handler(name):
            def execute(context):
                if context.confirmatory:
                    raise ValueError("offline fake backend cannot claim confirmatory execution")
                self.calls.append(name)
                return getattr(self, "_" + name)(context)
            return execute
        return {name: handler(name) for name in STAGES}

    def _task(self):
        rows = [json.loads(line) for line in (self.root / "public" / "tasks.jsonl").read_text().splitlines()]
        row = rows[0]
        row["visible_tests"] = tuple(PublicTest(**test) for test in row["visible_tests"])
        row["group_ids"] = tuple(row["group_ids"])
        return PublicTask(**row)

    def _private(self):
        path = self.root / "private" / "manifest.json"
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def _verifier(self):
        manifest, checksum = self._private()
        return VisibleVerifier(manifest, manifest_hash=checksum, sandbox_command=self.sandbox, budget_root=self.ledger.root)

    def _plan(self):
        path = self.root / "bootstrap.json"
        plan = ClusterPlan.create(task_keys=["smoke-task"], source_clusters=["smoke-source"], strata=["smoke"], seed=44,
            upstream_identity=self.fingerprint.digest)
        if path.exists():
            return ClusterPlan.load(path, lock=PlanLock.from_plan(plan))
        plan.save(path)
        return plan

    def _doctor(self, context):
        return {"cells": doctor([{"dataset": "PBPF-EvalPlus", "model": "fake", "tasks": 1,
            "arms": 3, "visible_cases": 1, "hidden_cases": 1, "seeds": 3}]), "mode": "offline_nonconfirmatory"}

    def _manifest(self, context):
        cells = context.input("doctor")["cells"]
        return {"fingerprint": self.fingerprint.digest, "matrix": cells,
                "scientific_inputs": self.fingerprint.to_dict(), "backend": "synthetic_offline"}

    def _prepare(self, context):
        manifest = context.input("manifest")
        from pbpf.data import prepare_dataset
        from pbpf.data.firewall import PROTOCOL_PROVENANCE
        dataset, revisions = PROTOCOL_PROVENANCE["PBPF-EvalPlus"]
        private = self.root / "private" / "manifest.json"
        if not private.exists():
            prepared = prepare_dataset("PBPF-EvalPlus", [{"task_id": "smoke-task", "prompt": "VISIBLE_CANARY: repair output",
                "candidate_code": "print('bad')", "base_tests": [{"id": "v", "source": "# visible", "expected_output": "ok"}],
                "plus_tests": [{"id": "h", "source": "# HIDDEN_SOURCE_CANARY", "expected_output": "HIDDEN_OUTPUT_CANARY"}],
                "gold_code": "GOLD_CODE_CANARY", "gold_patch": "GOLD_PATCH_CANARY", "official_split": "test"}],
                generator_root=self.root / "public", evaluator_root=self.root / "private",
                configured_dataset_id=dataset, runtime_dataset_id=dataset,
                configured_revisions=revisions, runtime_revisions=revisions, licenses={"fixture": "synthetic"},
                experiment_fingerprint=manifest["fingerprint"])
        return {"task": asdict(self._task()), "public_checksum": hashlib.sha256((self.root / "public" / "manifest.json").read_bytes()).hexdigest(),
                "private_checksum": self._private()[1], "split": "locked_test"}

    def _finite(self, context):
        prepared = context.input("prepare")
        from pbpf.finite import run_finite_audit
        probabilities = np.tile(np.asarray([[[.8, .05, .05, .05, .05], [.2, .2, .2, .2, .2]]]), (10, 1, 1))
        result = run_finite_audit(prior=np.asarray([.5, .5]), outcome_probabilities=probabilities,
            observed_outcomes=[0]*4, future_outcomes=[0]*6, rng=np.random.default_rng(1701))
        return {"audit": result, "prepared_checksum": prepared["public_checksum"], "scope": "tiny_smoke_not_S1"}

    def _bank(self, context):
        prepared, finite = context.input("prepare"), context.input("finite")
        banks = []
        for seed in (1701, 1702, 1703):
            actor = MeteredActor(_OfflineActor(self, "bank", f"bank/{seed}"), ledger=self.ledger, work_prefix=f"bank/{seed}")
            bank = build_bank(self._task(), actor, seed=seed,
                model_id="fake", model_revision="a"*40, prompt_revision="smoke")
            banks.append({"seed": seed, "bank_hash": bank.content_hash, "candidates": [asdict(c) for c in bank.candidates]})
        return {"banks": banks, "finite_checksum": digest(finite)}

    def _visible_execute(self, context):
        prepared, bank = context.input("prepare"), context.input("bank")
        verifier = self._verifier()
        events = {}
        for row in bank["banks"]:
            for item in row["candidates"]:
                candidate = ArmCandidate(**item)
                rows = verifier.execute(self._task(), candidate)
                events[candidate.version_id] = [asdict(e) for e in rows]
        return {"events": events, "bank_checksum": digest(bank), "public_checksum": prepared["public_checksum"]}

    def _train_belief(self, context):
        visible, prepared = context.input("visible_execute"), context.input("prepare")
        # The smoke fit uses a separate synthetic train split, never the locked
        # prepared task's labels. Visible eval records are identity-only here.
        train = training_records([{"split": "train", "source": "disjoint-smoke-training", "labels": [0, 1, 1, 1]}])
        counts = [1 + train[0]["labels"].count(i) for i in range(5)]
        weights = [value/sum(counts) for value in counts]
        lock = digest({"weights": weights, "train_split_hash": digest(train), "comparator": "independent"})
        return {"weights": weights, "model_lock": lock, "selection_hash": digest([lock, "smoke-dev-select"]),
            "train_split_hash": digest(train), "eval_input_checksum": digest(visible),
            "public_checksum": prepared["public_checksum"], "training_scope": "synthetic_disjoint"}

    def _predict(self, context):
        bank, trained, visible = context.input("bank"), context.input("train_belief"), context.input("visible_execute")
        predictions, code = [], {}
        for row in bank["banks"]:
            for item in row["candidates"]:
                candidate = ArmCandidate(**item)
                code[candidate.source_hash] = candidate.source
                for arm in ("pbpf", "independent", "shuffled_evidence", "wrong_candidate", "random_latent"):
                    p = (.19 if self.fail_gate_b else .5) if arm == "pbpf" else (.2 if arm == "independent" else .21)
                    predictions.append({"key": [candidate.task_id, row["seed"], arm, candidate.version_id, "h"],
                        "source_hash": candidate.source_hash, "probabilities": [(1-p)/4, p, (1-p)/4, (1-p)/4, (1-p)/4]})
        seal = self.root / "prediction-seal.json"
        if not seal.exists():
            seal_predictions(seal, predictions=predictions, expected_keys=[r["key"] for r in predictions],
                candidate_inventory=list(code), model_lock=trained["model_lock"], fingerprint=self.fingerprint.digest)
        else:
            sealed = json.loads(seal.read_bytes())
            if sorted(sealed["predictions"], key=lambda r: work_id(r["key"])) != sorted(predictions, key=lambda r: work_id(r["key"])):
                raise ValueError("immutable prediction seal conflict")
        return {"predictions": predictions, "code": code, "prediction_seal_hash": json.loads(seal.read_bytes())["seal_hash"],
            "model_lock": trained["model_lock"], "selection_hash": trained["selection_hash"], "visible_checksum": digest(visible)}

    def _gate_b(self, context):
        predicted = context.input("predict")
        manifest, checksum = self._private()
        scored = score_predictions(manifest, self.root / "prediction-seal.json", code=predicted["code"],
            fingerprint=self.fingerprint.digest, model_lock=predicted["model_lock"], manifest_hash=checksum,
            sandbox_command=self.sandbox, budget_root=self.ledger.root)
        if any(row["infrastructure_failure"] for row in scored["scores"]):
            raise ValueError("smoke future evaluation infrastructure failure")
        records = [dict(task=row["key"][0], seed=row["key"][1], arm=row["key"][2], candidate=row["key"][3],
            test=row["key"][4], source_cluster="smoke-source", stratum="smoke", nll=row["nll"], brier=row["brier"]) for row in scored["scores"]]
        reduced = reduce_metrics(records, expected_keys=[row["key"] for row in predicted["predictions"]])
        metrics = {arm: {metric: [task[metric] for task in tasks.values()] for metric in ("nll", "brier")} for arm, tasks in reduced.items()}
        return aggregate_stage_b(metrics=metrics, plan=self._plan(), comparator="independent", brier_margin=.01,
            selection_hash=predicted["selection_hash"], prediction_seal_hash=scored["prediction_seal_hash"])

    def _repair(self, context):
        bank, visible, trained, gate = (context.input(name) for name in ("bank", "visible_execute", "train_belief", "gate_b"))
        selections, code, matched = [], {}, []
        for row in bank["banks"]:
            formal_bank = FormalBank(tuple(ArmCandidate(**candidate) for candidate in row["candidates"]))
            initial_events = {c.version_id: tuple(VisibleEvent(**e) for e in visible["events"][c.version_id]) for c in formal_bank.candidates}
            per_seed = []
            for arm in self.arms:
                session = RepairArm(arm, task=self._task(), bank=formal_bank, initial_events=initial_events,
                    actor=MeteredActor(_OfflineActor(self, arm, f"repair/{row['seed']}/{arm}"), ledger=self.ledger,
                        work_prefix=f"repair/{row['seed']}/{arm}"), seed=row["seed"],
                    belief=_OfflineBelief() if arm == "pbpf" else None)
                selection = run_repair_session(session, self._verifier())
                selections.append(selection)
                selected = next(c for c in session.candidates if c.version_id == selection.final_version_id)
                code[selected.source_hash] = selected.source
                per_seed.append(dict(bank_hash=formal_bank.content_hash, roots=8, genuine=True, arm=arm,
                    actor_requests=session.accounting["actor_decodes"], full_decodes=session.accounting["actor_decodes"],
                    multiplicities=[1]*4, visible_opportunities=[[test.test_id for test in self._task().visible_tests]]*4,
                    max_new_tokens=1024, label=selection.compute_class))
                self.ledger.record_attempt(f"repair-visible/{row['seed']}/{arm}", Usage(
                    belief_forwards=session.accounting["belief_forwards"]), accepted=True)
            validate_matched(per_seed)
            matched.extend(per_seed)
        return {"selection_manifest": selection_manifest(selections), "code": code, "matched": matched,
            "model_lock": trained["model_lock"], "gate_b_hash": gate["decision_hash"]}

    def _seal(self, context):
        repair = context.input("repair")
        path = self.root / "final-seal.json"
        if not path.exists():
            seal_final(path, mapping=repair["selection_manifest"]["selections"],
                expected_keys=[row["key"] for row in repair["selection_manifest"]["selections"]],
                fingerprint=self.fingerprint.digest, code=repair["code"])
        sealed = json.loads(path.read_bytes())
        if sealed["selections"] != sorted(repair["selection_manifest"]["selections"], key=lambda r: work_id(r["key"])):
            raise ValueError("final selection seal conflict")
        return {"seal": sealed, "code": repair["code"]}

    def _hidden_eval(self, context):
        sealed = context.input("seal")
        manifest, checksum = self._private()
        result = HiddenEvaluator(manifest, fingerprint=self.fingerprint.digest, output_root=self.root / "hidden-receipts",
            mode="smoke", manifest_hash=checksum, sandbox_command=self.sandbox, budget_root=self.ledger.root).evaluate(self.root / "final-seal.json", code=sealed["code"])
        result.pop("worker_pid")  # operational metadata must not perturb scientific resume bytes
        expected = sealed["seal"]["expected_keys"]
        if not infrastructure_valid(expected_ids=expected, failed_ids=[r["work_id"] for r in result["results"] if r["infrastructure_failure"]]):
            raise ValueError("hidden infrastructure failure exceeds fixed expected denominator")
        return result

    def _gate_c(self, context):
        hidden, stage_b = context.input("hidden_eval"), context.input("gate_b")
        metrics = {arm: [sum(row["outcome"] == "PASS" for row in hidden["results"] if row["key"][4] == arm)/3] for arm in self.arms}
        return aggregate_stage_c(metrics=metrics, plan=self._plan(), comparator="independent", stage_b=stage_b,
            expected_stage_b_hash=stage_b["decision_hash"], endpoint="final_selected_pass1")

    def _replication(self, context):
        gate = context.input("gate_c")
        return {"gate_c_hash": gate["decision_hash"], "status": "offline_fixture_only_not_empirical_replication",
                "locked_arms": list(self.arms), "direction": {name: row["delta"] > 0 for name, row in gate["comparisons"].items()}}

    def _tables(self, context):
        replication, c, b = (context.input(name) for name in ("replication", "gate_c", "gate_b"))
        if context.confirmatory:
            require_main_tables([{"confirmatory": context.confirmatory, "failed_gate_hash": context.failed_gate_hash}])
        # The durable trusted ledger retains honest resource telemetry outside
        # the scientific run. Time measurements are operational observations,
        # so they must not perturb immutable leaves or package identity.
        budget = self.ledger.totals()
        for name in ("cpu_seconds", "gpu_seconds", "gpu_hours", "wall_seconds"):
            budget.pop(name)
        return {"main_table_eligible": False, "scope": "offline_smoke_only", "gate_b": b["passed"], "gate_c": c["passed"],
            "replication": replication["status"], "budget": budget}
