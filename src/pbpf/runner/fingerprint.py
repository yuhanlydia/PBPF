"""Scientific inputs only; publication paths and telemetry are not run identity."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from .shard import canonical_bytes


def _hash(value, length=64):
    if type(value) is not str or len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"immutable lowercase {length}-character content hash required")


@dataclass(frozen=True)
class CodeIdentity:
    tree: str
    content_hash: str

    def __post_init__(self):
        _hash(self.tree)
        _hash(self.content_hash)


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    revision: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_hash: str
    chat_template_hash: str
    weights_hash: str

    def __post_init__(self):
        if not self.model_id or not self.tokenizer_id:
            raise ValueError("exact model/tokenizer identifiers required")
        _hash(self.revision, 40)
        _hash(self.tokenizer_revision, 40)
        _hash(self.tokenizer_hash)
        _hash(self.chat_template_hash)
        _hash(self.weights_hash)


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str
    revision: str
    adapter_hash: str
    split_hash: str
    raw_hash: str

    def __post_init__(self):
        if not self.dataset_id:
            raise ValueError("dataset identifier required")
        _hash(self.revision, 40)
        for value in (self.adapter_hash, self.split_hash, self.raw_hash):
            _hash(value)


@dataclass(frozen=True)
class PromptIdentity:
    revision: str
    template_hash: str

    def __post_init__(self):
        _hash(self.revision)
        _hash(self.template_hash)


@dataclass(frozen=True)
class ScientificLocks:
    checkpoint_hash: str
    selection_hash: str
    calibration_hash: str
    brier_margin: float

    def __post_init__(self):
        for value in (self.checkpoint_hash, self.selection_hash, self.calibration_hash):
            _hash(value)
        if type(self.brier_margin) not in (int, float) or not math.isfinite(self.brier_margin) or self.brier_margin < 0:
            raise ValueError("explicit finite nonnegative calibration margin required")


def _typed(kind, value):
    try:
        return asdict(value if type(value) is kind else kind(**value))
    except (TypeError, KeyError) as error:
        raise ValueError(f"complete {kind.__name__} schema required") from error


def _scientific(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or any(token in key.lower() for token in
                    ("path", "timestamp", "wall_seconds", "telemetry", "prepared_manifest")):
                raise ValueError("fingerprint admits only scientific inputs, not paths/telemetry/prepared manifests")
            _scientific(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _scientific(child)


def scientific_config(value):
    """Closed scientific domain. Operational settings are separate API inputs.

    Extending this versioned allowlist requires an explicit schema/code change;
    arbitrary names or nested payloads cannot smuggle job/path/time metadata.
    """
    if type(value) is not dict or not value:
        raise ValueError("nonempty typed scientific config required")
    integers = {"roots", "repair_decodes", "max_new_tokens", "particles", "training_steps", "batch_size", "bootstrap_seed"}
    nonnegative = {"temperature", "learning_rate", "weight_decay"}
    probabilities = {"top_p", "dropout"}
    allowed = integers | nonnegative | probabilities | {"mode", "experiment_hash"}
    if not set(value) <= allowed:
        raise ValueError("unknown scientific config key; operational inputs must remain separate")
    for key, item in value.items():
        if key == "experiment_hash":
            _hash(item)
            valid = True
        elif key == "mode":
            valid = type(item) is str and item in {"smoke", "formal"}
        elif key in integers:
            valid = type(item) is int and item >= (0 if key == "bootstrap_seed" else 1)
        else:
            valid = type(item) in (int, float) and math.isfinite(item) and item >= 0
            valid = valid and (key not in probabilities or item <= 1)
        if not valid:
            raise ValueError("invalid typed scientific config value")
    return dict(value)


@dataclass(frozen=True)
class ScientificIdentity:
    code: dict
    models: dict
    datasets: dict
    prompt: dict
    container_digest: str
    config: dict
    seeds: tuple
    locks: dict
    schema: str

    def __post_init__(self):
        values = dict(vars(self))
        _scientific(values)
        if not all((self.code, self.models, self.datasets, self.prompt, self.config, self.locks, self.schema)):
            raise ValueError("all scientific identity domains and locks are required")
        if type(self.container_digest) is not str or not self.container_digest.startswith("sha256:"):
            raise ValueError("immutable container digest required")
        _hash(self.container_digest[7:])
        if tuple(self.seeds) != (1701, 1702, 1703) or any(type(seed) is not int for seed in self.seeds):
            raise ValueError("formal seeds must be 1701, 1702, 1703")
        values["code"] = _typed(CodeIdentity, self.code)
        values["models"] = {name: _typed(ModelIdentity, model) for name, model in self.models.items()}
        values["datasets"] = {name: _typed(DatasetIdentity, dataset) for name, dataset in self.datasets.items()}
        values["prompt"] = _typed(PromptIdentity, self.prompt)
        values["locks"] = _typed(ScientificLocks, self.locks)
        values["config"] = scientific_config(self.config)
        # Snapshot the complete canonical payload now; later mutation of input
        # dictionaries cannot silently change the identity of an active run.
        object.__setattr__(self, "_payload", canonical_bytes(values))

    @property
    def digest(self):
        return hashlib.sha256(self._payload).hexdigest()

    def to_dict(self):
        return json.loads(self._payload)


RunFingerprint = ScientificIdentity


@dataclass(frozen=True)
class DevelopmentIdentity:
    """Phase A binds inputs/procedures, without fictitious future learned locks.

    Phase B calls freeze only after actual checkpoint, selection and temperature
    artifacts have been published and verified. Their hashes then bind all
    sealed test prediction, repair, and replication leaves.
    """
    code: dict
    models: dict
    datasets: dict
    prompt: dict
    container_digest: str
    config: dict
    seeds: tuple
    schema: str

    def __post_init__(self):
        values = dict(vars(self))
        _scientific(values)
        if not all((self.code, self.models, self.datasets, self.prompt, self.config, self.schema)):
            raise ValueError("complete development input identity required")
        if not isinstance(self.container_digest, str) or not self.container_digest.startswith("sha256:"):
            raise ValueError("immutable container digest required")
        _hash(self.container_digest[7:])
        if tuple(self.seeds) != (1701, 1702, 1703) or any(type(seed) is not int for seed in self.seeds):
            raise ValueError("formal seeds must be 1701, 1702, 1703")
        values["code"] = _typed(CodeIdentity, self.code)
        values["models"] = {name: _typed(ModelIdentity, model) for name, model in self.models.items()}
        values["datasets"] = {name: _typed(DatasetIdentity, dataset) for name, dataset in self.datasets.items()}
        values["prompt"] = _typed(PromptIdentity, self.prompt)
        values["config"] = scientific_config(self.config)
        values["phase"] = "development_inputs"
        object.__setattr__(self, "_payload", canonical_bytes(values))

    @property
    def digest(self):
        return hashlib.sha256(self._payload).hexdigest()

    def to_dict(self):
        return json.loads(self._payload)

    def freeze(self, **locks):
        values = self.to_dict()
        values.pop("phase")
        return ScientificIdentity(**values, locks=_typed(ScientificLocks, locks))


def code_identity(root, files):
    root = Path(root).resolve()
    rows = []
    for name in sorted(files):
        path = root / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(root):
            raise ValueError("code inventory must remain inside source root")
        rows.append([name, hashlib.sha256(path.read_bytes()).hexdigest()])
    if not rows:
        raise ValueError("code content inventory cannot be empty")
    return {"tree": hashlib.sha256(canonical_bytes(rows)).hexdigest(),
            "content_hash": hashlib.sha256(canonical_bytes([row[1] for row in rows])).hexdigest()}
