"""Append-only attempt charges, distinct from accepted-work cache receipts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import json
import math
from pathlib import Path

from .shard import atomic_write, canonical_bytes, work_id
from .shard import digest
from pbpf.arms.repair import DecodeResult, PartialResult


@dataclass(frozen=True)
class Usage:
    actor_requests: int = 0
    full_decodes: int = 0
    belief_forwards: int = 0
    visible_verifier_suites: int = 0
    visible_verifier_cases: int = 0
    hidden_verifier_suites: int = 0
    hidden_verifier_cases: int = 0
    input_token_ids: tuple = ()
    output_token_ids: tuple = ()
    cpu_seconds: float = 0.
    gpu_seconds: float = 0.
    wall_seconds: float = 0.
    peak_allocated_bytes: int = 0
    disk_bytes: int = 0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key.endswith("_token_ids"):
                if not isinstance(value, (list, tuple)) or any(type(v) is not int or v < 0 for v in value):
                    raise ValueError("native token IDs must be nonnegative integers")
                object.__setattr__(self, key, tuple(value))
            elif (isinstance(value, bool) or not isinstance(value, (float, int))
                  or not math.isfinite(value) or value < 0
                  or (not key.endswith("seconds") and type(value) is not int)):
                raise ValueError("budget measurements must be finite nonnegative numbers; counts are integers")


class BudgetLedger:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def charge(self, work, attempt, usage, *, accepted):
        if type(usage) is not Usage or type(accepted) is not bool or not work or not attempt:
            raise ValueError("typed usage and explicit work/attempt acceptance required")
        row = dict(work=work, attempt=attempt, accepted=accepted, usage=asdict(usage))
        payload = canonical_bytes(row)
        path = self.root / (work_id([work, attempt]) + ".json")
        with (self.root / ".ledger.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                if path.read_bytes() != payload:
                    raise ValueError("attempt charge conflicts with immutable accepted accounting")
                return
            if accepted and any(r["work"] == work and r["accepted"] for r in self._rows()):
                raise ValueError("work already has an accepted attempt")
            atomic_write(path, payload, create_once=True)

    def _rows(self):
        return [json.loads(path.read_bytes()) for path in sorted(self.root.glob("*.json"))]

    def record_attempt(self, work, usage, *, accepted):
        """Charge an actual new execution, even if its result was accepted before.

        Call at dispatch to charge request/sequence reservations through crashes;
        later token/time observations use separate attempt events. A cache hit
        does not dispatch and must not call this method.
        """
        if type(usage) is not Usage or type(accepted) is not bool or not work:
            raise ValueError("typed usage and explicit acceptance required")
        with (self.root / ".ledger.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prior = [row for row in self._rows() if row["work"] == work]
            attempt = f"execution-{len(prior)+1}"
            row = dict(work=work, attempt=attempt,
                accepted=accepted and not any(r["accepted"] for r in prior), usage=asdict(usage))
            atomic_write(self.root / (work_id([work, attempt])+".json"), canonical_bytes(row), create_once=True)
            return attempt

    def totals(self):
        totals = {key: 0 for key in asdict(Usage()) if not key.endswith("_token_ids")}
        totals.update(input_tokens=0, output_tokens=0)
        for row in self._rows():
            usage = asdict(Usage(**row["usage"]))
            for key, value in usage.items():
                if key.endswith("_token_ids"):
                    totals[key.replace("_token_ids", "_tokens")] += len(value)
                elif key == "peak_allocated_bytes":
                    totals[key] = max(totals[key], value)
                else:
                    totals[key] += value
        totals["gpu_hours"] = totals["gpu_seconds"] / 3600
        return totals


class MeteredActor:
    """Runner-owned dispatch accounting, independent of actor cooperation.

    Backends must expose native_usage() even after failed/partial requests; it
    returns IDs and measured resources only, never request/decode counters.
    Reservations precede dispatch and survive a crash without reconciliation.
    """
    def __init__(self, backend, *, ledger, work_prefix):
        if type(ledger) is not BudgetLedger or not work_prefix or not callable(getattr(backend, "native_usage", None)):
            raise ValueError("mandatory ledger and backend native telemetry capability required")
        self.backend, self.ledger, self.work_prefix = backend, ledger, work_prefix

    def _call(self, method, request, **kwargs):
        multiplicity = request.num_return_sequences
        if type(multiplicity) is not int or multiplicity < 1:
            raise ValueError("decode multiplicity requires positive exact integer")
        if type(request.max_new_tokens) is not int or request.max_new_tokens < 1:
            raise ValueError("output cap requires positive exact integer")
        if method == "generate_partial":
            if type(kwargs.get("complete")) is not bool:
                raise ValueError("partial complete must be an exact boolean")
            if (type(kwargs.get("token_budget")) is not int or not 0 < kwargs["token_budget"] <= request.max_new_tokens
                    or type(kwargs.get("prefix")) not in (tuple, list)
                    or any(type(token) is not str or not token for token in kwargs["prefix"])):
                raise ValueError("partial token budget and ordered prefix tokens required")
        function = getattr(self.backend, method)
        full = multiplicity if method != "generate_partial" or kwargs.get("complete") is True else 0
        work = self.work_prefix + "/" + method
        attempt = self.ledger.record_attempt(work, Usage(actor_requests=1, full_decodes=full), accepted=False)
        result, failure, reconciliation = None, None, None
        try:
            result = function(request, **kwargs)
            if type(result) is not (PartialResult if method == "generate_partial" else DecodeResult):
                raise ValueError("typed decoder result required")
            if type(result.output_tokens) is not int or result.output_tokens < 0:
                raise ValueError("decoder token counts require exact nonnegative integers")
            if result.output_tokens > (kwargs["token_budget"] if method == "generate_partial" else request.max_new_tokens):
                raise ValueError("actor output cap exceeded")
            if method == "generate_partial" and (
                    (kwargs["complete"] and (type(result.source) is not str or not result.source))
                    or (not kwargs["complete"] and result.source is not None)
                    or result.prefix[:len(kwargs["prefix"])] != tuple(kwargs["prefix"])):
                raise ValueError("partial completion semantics or retained prefix mismatch")
            return result
        except BaseException as error:
            failure = type(error).__name__
            raise
        finally:
            usage = self.backend.native_usage()
            if type(usage) is not Usage:
                raise ValueError("backend must reconcile native token/resource telemetry")
            counters = ("actor_requests", "full_decodes", "belief_forwards", "visible_verifier_suites",
                "visible_verifier_cases", "hidden_verifier_suites", "hidden_verifier_cases")
            if any(getattr(usage, key) for key in counters):
                raise ValueError("backend cannot self-report runner-owned dispatch counters")
            # Native observations are incurred costs even if result validation
            # fails. Charge before reconciliation and preserve its failure.
            self.ledger.charge(work + "/telemetry", attempt, usage, accepted=False)
            if failure is None and len(usage.output_token_ids) != result.output_tokens:
                failure, reconciliation = "ValueError", "output_token_count_mismatch"
            receipt = {"work": work, "attempt": attempt, "status": "failed" if failure else "accepted",
                "failure_class": failure, "usage_hash": digest(asdict(usage)),
                "result_hash": digest(asdict(result)) if type(result) in (DecodeResult, PartialResult) else None,
                "reconciliation_failure": reconciliation}
            atomic_write(self.ledger.root / "actor-receipts" / (work_id([work, attempt])+".json"),
                canonical_bytes(receipt), create_once=True)
            if reconciliation:
                raise ValueError("native output-token IDs disagree with decoder token count")

    def generate_root(self, request):
        return self._call("generate_root", request)

    def generate(self, request):
        return self._call("generate", request)

    def generate_partial(self, request, **kwargs):
        return self._call("generate_partial", request, **kwargs)


def validate_matched(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("matched inventory required")
    common = None
    for row in rows:
        if any(type(row[k]) is not int for k in ("roots", "full_decodes", "actor_requests", "max_new_tokens")):
            raise ValueError("budget counts require exact nonboolean integers")
        if (type(row["visible_opportunities"]) not in (list, tuple)
                or any(type(n) is not int for n in row["multiplicities"])
                or any(type(suite) not in (list, tuple) or not suite or any(type(t) is not str or not t for t in suite)
                       or len(suite) != len(set(suite)) for suite in row["visible_opportunities"])):
            raise ValueError("nonempty ordered unique visible test IDs and integer multiplicities required")
        if (row["arm"] in {"no_repair", "full_ensemble"} or row["label"] != "matched_four_decodes"
                or row["roots"] != 8 or row["genuine"] is not True
                or row["full_decodes"] != 4 or row["actor_requests"] < 4
                or row["multiplicities"] != [1]*4 or row["max_new_tokens"] != 1024
                or len(row["visible_opportunities"]) != 4):
            raise ValueError("not a matched four-decode, eight-genuine-root experiment")
        identity = (row["bank_hash"], row["visible_opportunities"], row["max_new_tokens"])
        if common is not None and identity != common:
            raise ValueError("matched arms require identical banks and ordered visible opportunities")
        common = identity
    return True
