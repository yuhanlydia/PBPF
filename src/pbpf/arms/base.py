"""Small explicit generator-side contracts; never accept evaluator dictionaries."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import dis
import hashlib
import inspect
import json
import math
from types import CodeType, ModuleType
from typing import Any

from pbpf.data.schema import PublicTask, PublicTest
from pbpf.registry import OUTCOMES


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def source_hash(source: str) -> str:
    """Exact UTF-8 program bytes, not timestamps, lineage, or decode telemetry."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def derived_seed(seed: int, *parts: str) -> int:
    return int.from_bytes(hashlib.sha256(canonical([seed, *parts]).encode()).digest()[:8], "big")


def _public_json(value):
    """Only JSON primitives/arrays/string-key objects, with no lossy coercion."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("public JSON numbers must be finite")
        return value
    if type(value) is list:
        return [_public_json(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _public_json(item) for key, item in sorted(value.items())}
    raise TypeError("public inputs support only finite JSON structures with string keys")


def freeze_public_test(test):
    if type(test) is not PublicTest or type(test.test_id) is not str or not test.test_id:
        raise TypeError("public test must have a nonempty string identity")
    if test.invocation_index is not None and (type(test.invocation_index) is not int or test.invocation_index < 0):
        raise ValueError("public invocation index must be a nonnegative integer")
    source = _public_json(test.source)
    # Canonical source text is the immutable semantic input. Never retain the
    # caller's nested containers or stringify arbitrary Python objects.
    if source is not None and not isinstance(source, str):
        source = canonical(source)
    return PublicTest(test.test_id, source, test.invocation_index)


def freeze_public_task(task):
    if type(task) is not PublicTask:
        raise TypeError("public typed task required")
    if any(type(getattr(task, field)) is not str for field in
           ("task_id", "protocol", "task_text", "candidate_code", "split")):
        raise TypeError("public task text and identity fields must be strings")
    if not isinstance(task.visible_tests, (tuple, list)) or not isinstance(task.group_ids, (tuple, list)):
        raise TypeError("public task tests/groups must be arrays")
    if any(type(group) is not str for group in task.group_ids):
        raise TypeError("public task group identities must be strings")
    return PublicTask(task.task_id, task.protocol, task.task_text, task.candidate_code,
                      tuple(freeze_public_test(test) for test in task.visible_tests), task.split, tuple(task.group_ids))


def _callable_value(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is tuple:
        return {"tuple": [_callable_value(item) for item in value]}
    raise ValueError("callable default/closure state must be immutable canonical values")


def callable_identity(function):
    implementation = function if inspect.isfunction(function) or inspect.ismethod(function) else type(function).__call__
    try:
        implementation_source = inspect.getsource(implementation)
    except (TypeError, OSError):
        implementation_source = getattr(implementation, "__qualname__", type(implementation).__qualname__)
    result = {"module": getattr(implementation, "__module__", ""),
              "name": getattr(implementation, "__qualname__", type(implementation).__qualname__),
              "implementation_sha256": source_hash(implementation_source)}
    target = implementation.__func__ if inspect.ismethod(implementation) else implementation
    if inspect.isfunction(target):
        result["defaults"] = [_callable_value(value) for value in target.__defaults__ or ()]
        result["kwdefaults"] = {key: _callable_value(value) for key, value in sorted((target.__kwdefaults__ or {}).items())}
        try:
            result["closure"] = {name: _callable_value(cell.cell_contents)
                                 for name, cell in zip(target.__code__.co_freevars, target.__closure__ or ())}
        except ValueError as error:
            raise ValueError("callable closure state is empty, mutable or noncanonical") from error
    return result


def _module_state_digest(module):
    import torch
    digest = hashlib.sha256()
    # Traverse the actual registrations backing named_parameters/named_buffers,
    # independent of overridable state_dict or enumeration methods. Aliased
    # registered names and nonpersistent buffers are deliberately retained.
    pending = [("", module, frozenset())]
    while pending:
        prefix, current, ancestors = pending.pop()
        if id(current) in ancestors:
            raise ValueError("encoder/module registration cycles are unsupported")
        lineage = ancestors | {id(current)}
        for namespace, values in (("parameter", current._parameters), ("buffer", current._buffers)):
            for name, value in sorted(values.items()):
                if value is None:
                    continue
                if not isinstance(value, torch.Tensor):
                    raise ValueError("encoder registered state must contain tensors")
                digest.update(f"{prefix}/{namespace}/{name}".encode())
                value = value.detach().cpu().contiguous()
                if not torch.isfinite(value).all():
                    raise ValueError("encoder checkpoint tensors must be finite")
                digest.update(canonical([str(value.dtype), list(value.shape)]).encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        if type(current).get_extra_state is not torch.nn.Module.get_extra_state:
            digest.update(canonical(_public_json(current.get_extra_state())).encode())
        pending.extend((f"{prefix}/{name}", child, lineage) for name, child in sorted(current._modules.items()) if child is not None)
    return digest.hexdigest()


def _reject_explicit_rng(callback, *, seen=frozenset()):
    """Conservative static guard, never execute a semantic encoder as a probe.

    Inspectable Python helpers are traversed. Dynamic lookup and opaque callable
    aliases are unsupported rather than treated as evidence of determinism.
    Stock eval-mode Dropout is the sole stochastic-name exception.
    """
    target = callback.__func__ if inspect.ismethod(callback) else callback
    if not inspect.isfunction(target):
        raise ValueError("semantic encoder callable indirection must be inspectable Python")
    if id(target) in seen:
        return
    seen = seen | {id(target)}
    pending, names, bytecode = [target.__code__], set(), []
    while pending:
        code = pending.pop()
        names.update(code.co_names)
        instructions = list(dis.get_instructions(code))
        bytecode.extend((instruction, instructions[index - 1] if index else None)
                        for index, instruction in enumerate(instructions))
        pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
    if names & {"getattr", "__getattribute__", "eval", "exec", "__import__", "globals", "locals", "vars"}:
        raise ValueError("semantic encoder dynamic callable indirection is unsupported")
    rng_names = {"random", "secrets", "rand", "randn", "randint", "randint_like", "randperm",
        "rand_like", "randn_like", "randbelow", "randbits", "getrandbits", "randrange",
        "choice", "choices", "sample", "shuffle", "permutation", "uniform", "uniform_",
        "normal", "normal_", "gauss", "normalvariate", "standard_normal", "random_sample",
        "default_rng", "RandomState", "Generator", "SystemRandom", "multinomial", "bernoulli",
        "bernoulli_", "poisson", "poisson_", "exponential_", "geometric_", "cauchy_",
        "token_bytes", "token_hex", "token_urlsafe", "dropout", "dropout1d", "dropout2d",
        "dropout3d", "alpha_dropout", "feature_alpha_dropout"}
    owner = callback.__self__ if inspect.ismethod(callback) else None
    if owner is not None:
        import torch
        dropout_types = (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d,
                         torch.nn.Dropout3d, torch.nn.AlphaDropout, torch.nn.FeatureAlphaDropout)
        if type(owner) in dropout_types and owner.training is False:
            rng_names -= {"dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout"}
        elif isinstance(owner, torch.nn.Module):
            for name, child in owner._modules.items():
                uses = [(instruction, previous) for instruction, previous in bytecode if instruction.argval == name]
                if (type(child) in dropout_types and child.training is False and uses
                        and all(instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"} and previous is not None
                                and previous.opname == "LOAD_FAST" and previous.argval == target.__code__.co_varnames[0]
                                for instruction, previous in uses)):
                    rng_names.discard(name)  # only actual self.<eval-child> accesses
    if names & rng_names:
        raise ValueError("stochastic RNG-based semantic encoder implementation is unsupported")
    for name in names:
        value = target.__globals__.get(name)
        if callable(value):
            if not inspect.isfunction(value):
                raise ValueError("semantic encoder opaque callable alias is unsupported")
            _reject_explicit_rng(value, seen=seen)


def encoder_identity(encoder):
    """Explicit frozen semantic encoder protocol, never arbitrary object pickle.

    Stateful callable objects and bound owners expose semantic_identity() with
    exactly implementation/revision/config. Config must include all ordinary
    JSON instance fields. Torch-managed fields are represented by architecture,
    training flags and a recomputed tensor/buffer digest. Unsupported state is
    rejected instead of omitted. Plain stateless functions need no protocol.
    """
    owner = encoder.__self__ if inspect.ismethod(encoder) else encoder
    if inspect.isfunction(encoder):
        _reject_explicit_rng(encoder)
        callable_identity(encoder)  # validates kw-only defaults and all closures too
        captures = list(encoder.__defaults__ or ()) + [cell.cell_contents for cell in encoder.__closure__ or ()]
        captures += [encoder.__globals__[key] for key in encoder.__code__.co_names if key in encoder.__globals__]
        for value in captures:
            if isinstance(value, ModuleType) or inspect.isroutine(value) or isinstance(value, type):
                continue
            if value is not None and type(value) not in (str, bool, int, float):
                raise ValueError("stateful encoder function requires an explicit callable-owner protocol")
        if encoder.__dict__:
            raise ValueError("encoder function has unsupported mutable instance state")
        return {"kind": "stateless_function", "implementation": callable_identity(encoder),
                "globals": {key: _public_json(encoder.__globals__[key]) for key in encoder.__code__.co_names
                            if key in encoder.__globals__ and type(encoder.__globals__[key]) in (str, bool, int, float)}}
    protocol = getattr(owner, "semantic_identity", None)
    if not callable(protocol):
        raise ValueError("stateful encoder requires explicit semantic_identity implementation/revision/config")
    declared = protocol()
    if (type(declared) is not dict or set(declared) != {"implementation", "revision", "config"}
            or any(type(declared[key]) is not str or not declared[key].strip() for key in ("implementation", "revision"))
            or type(declared["config"]) is not dict):
        raise ValueError("encoder semantic_identity has an invalid schema")
    try:
        declared = _public_json(declared)
        import torch
        is_module = isinstance(owner, torch.nn.Module)
        managed = set(torch.nn.Module().__dict__) if is_module else set()
        ordinary = {key: _public_json(value) for key, value in vars(owner).items() if key not in managed}
    except (TypeError, ValueError) as error:
        raise ValueError("encoder has unsupported non-JSON mutable configuration") from error
    if any(key not in declared["config"] or declared["config"][key] != value for key, value in ordinary.items()):
        raise ValueError("encoder semantic_identity config must bind every ordinary instance field")
    result = {"kind": "declared_encoder", **declared,
              "callable": callable_identity(encoder), "identity_method": callable_identity(protocol)}
    if is_module:
        try:
            baseline = torch.nn.Module().__dict__
            represented = {"training", "_parameters", "_buffers", "_modules", "_non_persistent_buffers_set"}
            for module in owner.modules():
                if module.training is not False:
                    raise ValueError("semantic encoder root and every submodule must already be in eval mode")
                _reject_explicit_rng(module.forward)
                if any(vars(module).get(key) != value for key, value in baseline.items() if key not in represented):
                    raise ValueError("encoder hooks or other uncheckpointed mutable module state are unsupported")
            module_config = {name: {key: _public_json(value) for key, value in vars(module).items() if key not in managed}
                             for name, module in owner.named_modules()}
        except (TypeError, ValueError) as error:
            raise ValueError("encoder module has unsupported mutable configuration") from error
        result.update(architecture=repr(owner), training={name: module.training for name, module in owner.named_modules()},
                      forward_implementations={name: callable_identity(module.forward) for name, module in owner.named_modules()},
                      module_config=module_config, state_digest=_module_state_digest(owner))
    else:
        _reject_explicit_rng(encoder if inspect.ismethod(encoder) else inspect.getattr_static(type(encoder), "__call__"))
        if inspect.ismethod(encoder) and callable(owner):
            _reject_explicit_rng(inspect.getattr_static(type(owner), "__call__"))
        result["state_digest"] = source_hash(canonical(ordinary))
    return result


def belief_digest(belief):
    """Bind behavioral configuration and implementation as well as weights."""
    identity = belief.behavior_identity()
    digest = hashlib.sha256(canonical(identity).encode())
    for namespace, module in (("model", belief.model), ("projector", belief.projector)):
        if module is None:
            continue
        digest.update(namespace.encode())
        digest.update(_module_state_digest(module).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class ArmCandidate:
    task_id: str
    slot: int
    round: int
    source: str
    origin: str
    model_id: str
    model_revision: str
    prompt_revision: str
    parent_version_id: str | None = None
    sample_seed: int = 1701
    wall_seconds: float = 0.
    output_tokens: int = 0

    def __post_init__(self):
        if (not self.task_id or not self.source or self.slot < 0 or self.round < 0
                or self.origin not in {"actor_sample", "repair", "mutant"}
                or not self.model_id or not self.prompt_revision
                or len(self.model_revision) != 40
                or any(c not in "0123456789abcdef" for c in self.model_revision)):
            raise ValueError("candidate requires source, identity and immutable model provenance")
        if self.round == 0 and self.parent_version_id is not None:
            raise ValueError("root cannot have a parent")
        if self.round > 0 and not self.parent_version_id:
            raise ValueError("repair version requires a parent")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds < 0 or self.output_tokens < 0:
            raise ValueError("invalid telemetry")

    @property
    def source_hash(self) -> str:
        return source_hash(self.source)

    @property
    def version_id(self) -> str:
        record = asdict(self)
        for key in ("wall_seconds", "output_tokens", "source"):
            record.pop(key)
        record["source_hash"] = self.source_hash
        return source_hash(canonical(record))


@dataclass(frozen=True)
class VisibleEvent:
    task_id: str
    version_id: str
    event_id: str
    test_id: str
    ordinal: int
    outcome: str
    feedback: str = ""

    def __post_init__(self):
        if (not all((self.task_id, self.version_id, self.event_id, self.test_id))
                or self.ordinal < 0 or self.outcome not in OUTCOMES
                or not isinstance(self.feedback, str) or len(self.feedback) > 2048):
            raise ValueError("visible event requires one of five outcomes and bounded feedback")


def validate_event(task: PublicTask, candidate: ArmCandidate, event: VisibleEvent, position: int):
    if type(task) is not PublicTask or type(candidate) is not ArmCandidate or type(event) is not VisibleEvent:
        raise TypeError("only typed public task, candidate and visible event are accepted")
    if event.task_id != task.task_id or event.version_id != candidate.version_id:
        raise ValueError("event owner task/version mismatch")
    if (position >= len(task.visible_tests) or event.ordinal != position
            or task.visible_tests[position].test_id != event.test_id):
        raise ValueError("event must follow the visible test order exactly once")


@dataclass(frozen=True)
class RepairCondition:
    version_id: str
    text: str = ""
    latent: tuple[float, ...] | None = None
    particle_index: int | None = None
    # A nested immutable tuple is safe to retain for a complete continuation.
    soft_prefix: tuple[tuple[float, ...], ...] | None = None


class BeliefArm(ABC):
    """No method performs actor decoding. Forward calls are separately metered."""

    @abstractmethod
    def initialize(self, task, candidate): ...

    @abstractmethod
    def observe(self, version_id, event): ...

    @abstractmethod
    def spawn_child(self, parent_version_id, candidate, first_event): ...

    @abstractmethod
    def predict(self, version_id, test): ...

    @abstractmethod
    def condition(self, version_id): ...

    @abstractmethod
    def snapshot(self, version_id): ...
