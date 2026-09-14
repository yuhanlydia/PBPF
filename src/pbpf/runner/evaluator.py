"""Trusted verifier RPC; generators receive only ordered visible status events.

The same-UID local subprocess is a smoke/pilot mechanism, never a secrecy claim.
Formal callers supply verified OS isolation and an external isolation-capable
sandbox command. One immutable accepted result per sealed work ID is guaranteed;
a process crash may cause physical execution to be retried before acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import resource
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import threading

from pbpf.arms.base import ArmCandidate, VisibleEvent, source_hash
from pbpf.data.schema import PublicTask
from pbpf.registry import OUTCOMES
from .shard import atomic_write, canonical_bytes, digest, work_id
from .budget import BudgetLedger, Usage


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _envelope(kind, payload):
    row = dict(format=kind, **payload)
    return dict(row, seal_hash=digest(row))


def _parse_seal(snapshot, kind, fingerprint):
    row = json.loads(snapshot)
    if row.get("format") != kind or row.get("fingerprint") != fingerprint:
        raise ValueError("seal format/fingerprint mismatch")
    if row.get("seal_hash") != digest({k: v for k, v in row.items() if k != "seal_hash"}):
        raise ValueError("seal checksum mismatch")
    inventory = row.get("selections" if kind == "pbpf-final-seal-v1" else "predictions", [])
    ids = [work_id(item["key"]) for item in inventory]
    if (not ids or len(ids) != len(set(ids)) or sorted(ids) != row.get("expected_keys")
            or sorted({item["source_hash"] for item in inventory}) != row.get("candidate_hashes")
            or any(not _sha(item["source_hash"]) or len(item["key"]) != 5 for item in inventory)):
        raise ValueError("sealed exact work/candidate inventory mismatch")
    return row


def _read_seal(path, kind, fingerprint):
    return _parse_seal(Path(path).read_bytes(), kind, fingerprint)


def _check_code(code, hashes):
    if set(code) != set(hashes) or any(source_hash(source) != key for key, source in code.items()):
        raise ValueError("sealed candidate code inventory or bytes changed")


def seal_final(path, *, mapping, expected_keys, fingerprint, code):
    mapping = sorted(mapping, key=lambda row: work_id(row["key"]))
    ids = [work_id(row["key"]) for row in mapping]
    if (not mapping or len(ids) != len(set(ids)) or sorted(ids) != sorted(work_id(k) for k in expected_keys)
            or any(len(row["key"]) != 5 or not _sha(row["source_hash"]) or not _sha(row["version_id"]) for row in mapping)):
        raise ValueError("final exact (dataset,model,task,seed,arm) selection inventory required")
    hashes = sorted({row["source_hash"] for row in mapping})
    _check_code(code, hashes)
    row = _envelope("pbpf-final-seal-v1", dict(fingerprint=fingerprint, selections=mapping,
        expected_keys=sorted(ids), candidate_hashes=hashes))
    atomic_write(path, canonical_bytes(row), create_once=True)
    return Path(path)


def seal_predictions(path, *, predictions, expected_keys, candidate_inventory, model_lock, fingerprint):
    predictions = sorted(predictions, key=lambda row: work_id(row["key"]))
    ids = [work_id(row["key"]) for row in predictions]
    hashes = sorted(set(candidate_inventory))
    if not _sha(model_lock) or not hashes or any(not _sha(h) for h in hashes):
        raise ValueError("candidate inventory and frozen model lock required")
    if (not ids or len(ids) != len(set(ids)) or sorted(ids) != sorted(work_id(k) for k in expected_keys)
            or {r["source_hash"] for r in predictions} != set(hashes)):
        raise ValueError("prediction seal requires exact keyed prediction/candidate inventory")
    for row in predictions:
        probabilities = row["probabilities"]
        if (len(row["key"]) != 5 or len(probabilities) != 5 or any(isinstance(p, bool) or not math.isfinite(p) or p < 0 for p in probabilities)
                or not math.isclose(sum(probabilities), 1., abs_tol=1e-8)):
            raise ValueError("keyed normalized five-way probabilities required")
    row = _envelope("pbpf-prediction-seal-v1", dict(fingerprint=fingerprint, predictions=predictions,
        expected_keys=sorted(ids), candidate_hashes=hashes, model_lock=model_lock))
    atomic_write(path, canonical_bytes(row), create_once=True)
    return Path(path)


_CAPABILITY_TOKEN = object()
_IN_WORKER = False


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode))


def _trusted_path(path):
    """Prove a resolved launch/authority path cannot be replaced by a generator.

    Trusted root/evaluator writers remain part of the deployment trust base.
    Sticky shared ancestors cannot replace their trusted-owned children.
    """
    for entry in (path, *path.parents):
        info = entry.stat()
        if info.st_uid not in {0, os.getuid()}:
            raise ValueError("formal path must be root/evaluator owned")
        if stat.S_IMODE(info.st_mode) & 0o022 and not (entry != path and stat.S_ISDIR(info.st_mode) and info.st_mode & stat.S_ISVTX):
            raise ValueError("formal path has generator-writable file or parent")


def _trusted_dependency_path(path):
    # Preserve declared symlink paths as well as resolved targets. In
    # particular a generator-owned link in /tmp is not a trusted dependency.
    for entry in (path, *path.parents):
        info = entry.lstat()
        if info.st_uid not in {0, os.getuid()}:
            raise ValueError("sandbox dependency link or parent is not trusted-owned")
        if not stat.S_ISLNK(info.st_mode) and stat.S_IMODE(info.st_mode) & 0o022:
            if not (entry != path and stat.S_ISDIR(info.st_mode) and info.st_mode & stat.S_ISVTX):
                raise ValueError("sandbox dependency has generator-writable parent or file")
    _trusted_path(path.resolve(strict=True))


@dataclass(frozen=True)
class BackendIdentity:
    command: tuple
    container_digest: str
    files: tuple
    dependencies: tuple = ()

    @classmethod
    def capture(cls, command, container_digest, dependency_files=()):
        if not isinstance(container_digest, str) or not container_digest.startswith("sha256:") or not _sha(container_digest[7:]):
            raise ValueError("explicit immutable backend container digest required")
        command = tuple(command or (sys.executable, str(Path(__file__).parents[1] / "sandbox.py")))
        executable = shutil.which(command[0]) or command[0]
        files = []
        for index, argument in enumerate(command):
            path = Path(executable if index == 0 else argument.split("=", 1)[-1])
            if index == 0 or path.is_file():
                resolved = path.resolve(strict=True)
                files.append((index, str(resolved), hashlib.sha256(resolved.read_bytes()).hexdigest()))
        dependencies = tuple(sorted((str(Path(path).absolute()), str(Path(path).resolve(strict=True)),
            hashlib.sha256(Path(path).resolve(strict=True).read_bytes()).hexdigest()) for path in dependency_files))
        return cls(command, container_digest, tuple(files), dependencies)

    def to_dict(self):
        return {"command": list(self.command), "container_digest": self.container_digest,
                "files": [list(row) for row in self.files], "dependencies": [list(row) for row in self.dependencies]}

    def validate(self):
        for declared, resolved, checksum in self.dependencies:
            _trusted_dependency_path(Path(declared))
        if type(self).capture(self.command, self.container_digest, [path for path, resolved, checksum in self.dependencies]) != self:
            raise ValueError("sandbox backend executable/script/config content identity changed")


@dataclass(frozen=True)
class ExternalSandboxSpec:
    """Trusted launch declaration: exhaustive non-container file dependencies.

    Runtime libraries belong to the immutable container digest. All wrapper,
    executable, configuration and auxiliary files outside that runtime must be
    listed explicitly by the trusted deployment operator. No command templates,
    inline code, module lookup or automatic dependency inference are accepted.
    """
    command: tuple
    dependency_files: tuple
    container_digest: str

    def __post_init__(self):
        if (type(self.command) not in (list, tuple) or not self.command
                or any(type(arg) is not str or not arg for arg in self.command)
                or type(self.dependency_files) not in (list, tuple) or not self.dependency_files
                or any(type(path) is not str or not Path(path).is_absolute() for path in self.dependency_files)):
            raise ValueError("explicit sandbox command and exhaustive dependency file inventory required")
        command = tuple(self.command)
        if (not Path(command[0]).is_absolute() or any(arg in {"-c", "-m", "eval", "exec", "--command"}
                or any(token in arg for token in ("exec(", "eval(", "open(", "$(", "`", ";", "\n", "|", "&&")) for arg in command)):
            raise ValueError("formal sandbox rejects inline/dynamic launches and opaque templates")
        executable = Path(command[0]).resolve(strict=True)
        name = executable.name.lower()
        if name in {"env", "xargs", "command"}:
            raise ValueError("formal sandbox requires directly named immutable entrypoint")
        if name.startswith(("python", "pypy")) or name in {"sh", "bash", "dash", "zsh", "node", "ruby", "perl"}:
            arguments = list(command[1:])
            while arguments and arguments[0] in {"-I", "-B", "-u"}:
                arguments.pop(0)
            if not arguments or not Path(arguments[0]).is_absolute() or not Path(arguments[0]).is_file():
                raise ValueError("formal interpreter requires directly named immutable wrapper file")
        dependencies = tuple(str(Path(path).absolute()) for path in self.dependency_files)
        resolved = {str(Path(path).resolve(strict=True)) for path in dependencies}
        if len(dependencies) != len(resolved):
            raise ValueError("sandbox dependency inventory must be unique")
        backend = BackendIdentity.capture(command, self.container_digest, dependencies)
        if not {path for index, path, checksum in backend.files} <= resolved:
            raise ValueError("every executable/wrapper/config input must be explicitly declared")
        for path in dependencies:
            _trusted_dependency_path(Path(path))
            if not Path(path).is_file():
                raise ValueError("sandbox dependencies must be regular files")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "dependency_files", dependencies)


@dataclass(frozen=True)
class IsolationCapability:
    generator_pid: int
    generator_uid: int
    evaluator_uid: int
    private_root: str
    public_root: str
    process_start: str
    _token: object

    @classmethod
    def verify(cls, *, generator_pid, private_root, public_root):
        """Fail-closed Linux distinct-UID capability, checked from live /proc.

        Mount-namespace-only claims are deliberately unsupported until an
        equivalent mount-access attestor is provided by the execution platform.
        """
        proc = Path("/proc") / str(generator_pid)
        status = proc.joinpath("status").read_text()
        ids = next(line.split()[1:] for line in status.splitlines() if line.startswith("Uid:"))
        uid = int(ids[0])
        if uid == os.getuid() or uid == 0 or len(set(ids)) != 1:
            raise ValueError("formal isolation requires verified distinct UID without elevated saved IDs")
        for field in ("CapInh:", "CapPrm:", "CapEff:", "CapAmb:"):
            if any(int(line.split()[1], 16) for line in status.splitlines() if line.startswith(field)):
                raise ValueError("generator capability sets must be empty")
        private, public = Path(private_root).resolve(), Path(public_root).resolve()
        if private == public or private in public.parents or public in private.parents:
            raise ValueError("disjoint generator/evaluator paths required")
        info = private.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("private root must be evaluator-owned and inaccessible to generator UID")
        allowed = {"PATH", "LANG", "LC_ALL", "PYTHONHASHSEED"}
        environment = proc.joinpath("environ").read_bytes().split(b"\0")
        if any(entry.split(b"=", 1)[0].decode() not in allowed for entry in environment if entry):
            raise ValueError("generator environment is not sanitized")
        for fd in proc.joinpath("fd").iterdir():
            target = os.readlink(fd)
            if target != "/dev/null" and not (Path(target).is_absolute() and Path(target).resolve().is_relative_to(public)):
                raise ValueError("generator inherited unapproved file descriptors")
        cwd = proc.joinpath("cwd").resolve()
        if not cwd.is_relative_to(public):
            raise ValueError("generator working directory must be public-only")
        start = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        return cls(generator_pid, uid, os.getuid(), str(private), str(public), start, _CAPABILITY_TOKEN)

    def validate(self):
        if self._token is not _CAPABILITY_TOKEN:
            raise ValueError("verified isolation capability required")
        current = type(self).verify(generator_pid=self.generator_pid, private_root=self.private_root,
                                    public_root=self.public_root)
        if current != self:
            raise ValueError("generator isolation changed or PID was reused")


class TrustedEvaluatorService:
    """External-worker launch interface. Never constructed from an RPC request.

    The deployment provisions the key and signs capabilities; this module only
    verifies them. No key generation or capability-signing endpoint exists.
    """
    def __init__(self, *, key_file, private_root, public_root, nonce_root, sandbox_spec=None,
                 sandbox_command=None, container_digest=None):
        if type(sandbox_spec) is not ExternalSandboxSpec or sandbox_command is not None or container_digest is not None:
            raise ValueError("formal service requires explicit typed external sandbox dependency specification")
        sandbox_command, container_digest = sandbox_spec.command, sandbox_spec.container_digest
        self.sandbox_spec = sandbox_spec
        fd = os.open(key_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid not in {0, os.getuid()} or stat.S_IMODE(info.st_mode) != 0o400:
                raise ValueError("pre-provisioned evaluator/root-owned 0400 trust anchor required")
            self._key = os.read(fd, 4096)
        finally:
            os.close(fd)
        if len(self._key) < 32:
            raise ValueError("trust anchor must contain at least 32 secret bytes")
        self.private_root, self.public_root = Path(private_root).resolve(), Path(public_root).resolve()
        _trusted_path(self.private_root)
        self._private_fd = os.open(self.private_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        private_info = os.fstat(self._private_fd)
        if private_info.st_uid != os.getuid() or stat.S_IMODE(private_info.st_mode) & 0o077:
            raise ValueError("private root must be evaluator-owned and inaccessible to generator")
        self._private_parents = tuple((path, _file_identity(path.stat())) for path in self.private_root.parents)
        self.nonce_root = Path(nonce_root).resolve()
        _trusted_path(self.nonce_root)
        nonce_info = self.nonce_root.stat()
        if nonce_info.st_uid != os.getuid() or stat.S_IMODE(nonce_info.st_mode) != 0o700:
            raise ValueError("pre-provisioned evaluator-owned 0700 nonce directory required")
        self._nonce_fd = os.open(self.nonce_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._nonce_identity = _file_identity(os.fstat(self._nonce_fd))
        self._nonce_parents = tuple((path, _file_identity(path.stat())) for path in self.nonce_root.parents)
        # Seal paths are confined to the launch-configured public root. Signed
        # bytes, not generator filesystem ownership, authorize their contents.
        self._seal_root_fd = os.open(self.public_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._seal_root_identity = _file_identity(os.fstat(self._seal_root_fd))
        self.backend = BackendIdentity.capture(sandbox_command, container_digest, sandbox_spec.dependency_files)
        launch = list(sandbox_command)
        for index, filename, checksum in self.backend.files:
            path = Path(filename)
            _trusted_path(path)
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("formal backend requires regular immutable launch files")
            prefix = launch[index].split("=", 1)[0] + "=" if index and "=" in launch[index] else ""
            launch[index] = prefix + filename
        # Original symlinks are never used for dispatch. Every actual launch
        # file and ancestor is protected against the distinct generator UID.
        self.launch_command = tuple(launch)
        self.service_id = secrets.token_hex(32)
        self.evaluator_pid, self.evaluator_uid = os.getpid(), os.getuid()
        self._private_identity = self._private_stat()

    def _private_stat(self):
        st = os.fstat(self._private_fd)
        return {"dev": st.st_dev, "inode": st.st_ino, "uid": st.st_uid, "mode": stat.S_IMODE(st.st_mode)}

    def _validate_private_root(self):
        info = self.private_root.lstat()
        if (self._private_stat() != self._private_identity
                or _file_identity(info) != _file_identity(os.fstat(self._private_fd))
                or any(_file_identity(path.stat()) != identity for path, identity in self._private_parents)):
            raise ValueError("private evaluator root or trusted ancestor identity changed")
        _trusted_path(self.private_root)

    def binding(self, request, *, generator_pid, nonce, expires):
        """Unsigned measured claims for the independently trusted launcher/signer."""
        self._validate_private_root()
        capability = IsolationCapability.verify(generator_pid=generator_pid, private_root=self.private_root, public_root=self.public_root)
        return {"service_id": self.service_id, "evaluator_pid": self.evaluator_pid, "evaluator_uid": self.evaluator_uid,
            "generator_pid": generator_pid, "generator_uid": capability.generator_uid, "generator_start": capability.process_start,
            "private_root": self._private_stat(), "backend": digest(self.backend.to_dict()),
            "request_hash": digest(request), "authorization": self._claims(request), "nonce": nonce, "expires": expires}

    @staticmethod
    def _claims(request):
        operation, inventory = request.get("operation"), request.get("expected_keys")
        if (operation not in {"hidden", "prediction", "visible"} or not _sha(request.get("fingerprint"))
                or type(inventory) not in (list, tuple) or not inventory
                or any(type(key) is not str or not key for key in inventory) or len(inventory) != len(set(inventory))):
            raise ValueError("formal operation/fingerprint/exact inventory lock required")
        seal_digest = request.get("seal_digest")
        if operation != "visible" and not _sha(seal_digest):
            raise ValueError("formal expected seal-byte digest required")
        return dict(operation=operation, fingerprint=request["fingerprint"], expected_keys=list(inventory), seal_digest=seal_digest)

    def _validate_nonce_directory(self):
        if (self._nonce_identity != _file_identity(os.fstat(self._nonce_fd))
                or self._nonce_identity != _file_identity(self.nonce_root.lstat())
                or any(_file_identity(path.stat()) != identity for path, identity in self._nonce_parents)):
            raise ValueError("nonce directory or trusted parent identity changed")
        _trusted_path(self.nonce_root)

    def seal_snapshot(self, request):
        if _file_identity(self.public_root.lstat()) != self._seal_root_identity:
            raise ValueError("trusted seal root changed")
        relative = Path(request["seal"]).relative_to(self.public_root)
        if not relative.parts or any(part in {".", ".."} for part in relative.parts):
            raise ValueError("seal path escapes trusted root")
        directory = os.dup(self._seal_root_fd)
        try:
            for part in relative.parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ValueError("seal must be a bounded regular file")
                snapshot = bytearray()
                while len(snapshot) <= 4*1024*1024:
                    block = os.read(fd, min(65536, 4*1024*1024+1-len(snapshot)))
                    if not block:
                        break
                    snapshot.extend(block)
            finally:
                os.close(fd)
        finally:
            os.close(directory)
        if len(snapshot) > 4*1024*1024 or hashlib.sha256(snapshot).hexdigest() != request["seal_digest"]:
            raise ValueError("signed seal-byte digest or size mismatch")
        return bytes(snapshot)

    def authorize(self, request, signed):
        body, signature = signed["body"], signed["signature"]
        expected = hmac.new(self._key, canonical_bytes(body), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise ValueError("invalid externally issued formal capability signature")
        if (type(body.get("expires")) not in (int, float) or not math.isfinite(body["expires"])
                or not time.time() < body["expires"] <= time.time()+60 or not _sha(body.get("nonce"))):
            raise ValueError("formal capability expired or invalid nonce/expiry")
        if os.getpid() != self.evaluator_pid or os.getuid() != self.evaluator_uid or self._private_stat() != self._private_identity:
            raise ValueError("trusted evaluator/private-root identity changed")
        self.backend.validate()
        self._validate_private_root()
        self._validate_nonce_directory()
        live = self.binding(request, generator_pid=body["generator_pid"], nonce=body["nonce"], expires=body["expires"])
        if body != live or request.get("backend_identity") != self.backend.to_dict():
            raise ValueError("formal capability is not bound to live process/backend/request identity")
        if not Path(request["manifest"]).resolve().is_relative_to(self.private_root):
            raise ValueError("request escapes verified private root")
        try:
            fd = os.open(body["nonce"], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self._nonce_fd)
        except FileExistsError as error:
            raise ValueError("formal capability replay rejected") from error
        try:
            os.write(fd, b"used")
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(self._nonce_fd)

    def execute(self, request, signed):
        try:
            return _run_request(request, authority=self, signed=signed)
        except Exception:
            # No parser/path/backend exception may disclose private contents.
            pass
        # Raise outside the handler: even __context__ must not retain a JSON
        # parser exception whose .doc can contain the selected file contents.
        raise ValueError("trusted evaluator rejected authorization, seal or execution")


def _rpc(request):
    # The interpreter is pinned to the active environment; source modules are
    # explicitly imported, with no inherited PYTHONPATH, environment or fds.
    source_root = str(Path(__file__).resolve().parents[2])
    script = f"import sys; sys.path.insert(0, {source_root!r}); from pbpf.runner.evaluator import _worker; _worker()"
    with tempfile.TemporaryDirectory(prefix="pbpf-verifier-") as directory:
        result = _bounded_process([sys.executable, "-I", "-c", script], payload=canonical_bytes(request), cwd=directory,
            env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"},
            timeout=request.get("case_seconds", 2.) + request.get("rpc_overhead_seconds", 5.))
    if result.returncode:
        # Never reflect private process stderr, exception text, or test payloads.
        raise ValueError("trusted verifier rejected request (private details withheld)")
    return json.loads(result.stdout)


def _bounded_process(command, *, payload, env, timeout, cwd=None):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        close_fds=True, env=env, cwd=cwd, start_new_session=not _IN_WORKER)
    try:
        stdout, stderr = process.communicate(payload, timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        _kill_process(process)
        process.communicate()
        raise ValueError("bounded verifier unit exceeded deadline") from None
    finally:
        try:
            _kill_process(process)
        except ProcessLookupError:
            pass


def _kill_process(process):
    if not _IN_WORKER:
        os.killpg(process.pid, signal.SIGKILL)
        return
    # Nested sandboxes inherit the worker's containment group so an outer
    # deadline kills all of them. Inner deadlines stop only this process tree.
    def kill_tree(pid):
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            for child in children:
                kill_tree(int(child))
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass
    kill_tree(process.pid)


def _settings(command, container_digest, case_seconds, rpc_overhead_seconds):
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (case_seconds, rpc_overhead_seconds)):
        raise ValueError("positive finite nonboolean case limit and RPC overhead required")
    backend = BackendIdentity.capture(command, container_digest)
    return backend, {"backend_identity": backend.to_dict(), "sandbox_command": list(command) if command else None,
        "case_seconds": case_seconds, "rpc_overhead_seconds": rpc_overhead_seconds}


class VisibleVerifier:
    def __init__(self, manifest, *, manifest_hash, sandbox_command=None, budget_root=None,
                 container_digest="sha256:"+"0"*64, case_seconds=2., rpc_overhead_seconds=5.):
        self.manifest, self.manifest_hash = str(Path(manifest).resolve()), manifest_hash
        self.sandbox_command = sandbox_command
        self.budget_root = str(Path(budget_root or Path(manifest).parent.parent / "verifier-budget").resolve())
        self.backend, self.settings = _settings(sandbox_command, container_digest, case_seconds, rpc_overhead_seconds)

    def execute(self, task, candidate):
        if type(task) is not PublicTask or type(candidate) is not ArmCandidate or task.task_id != candidate.task_id:
            raise TypeError("visible verifier accepts public task and candidate only")
        if not task.visible_tests:
            raise ValueError("nonempty ordered visible test suite required")
        self.backend.validate()
        events = []
        for index, test in enumerate(task.visible_tests):
            result = _rpc(dict(self.settings, operation="visible", manifest=self.manifest, manifest_hash=self.manifest_hash,
                task_id=task.task_id, test_ids=[test.test_id], source=candidate.source,
                budget_root=self.budget_root, work=candidate.version_id, start_suite=index == 0))
            events.extend(result["events"])
        return tuple(VisibleEvent(task.task_id, candidate.version_id,
            work_id([candidate.version_id, row["test_id"], index]), row["test_id"], index,
            row["outcome"], row["feedback"]) for index, row in enumerate(events))


@dataclass(frozen=True, init=False)
class HiddenEvaluator:
    def __init__(self, manifest, *, fingerprint, output_root, mode, manifest_hash,
                 isolation=None, sandbox_command=None, budget_root=None,
                 container_digest="sha256:"+"0"*64, case_seconds=2., rpc_overhead_seconds=5.):
        if mode not in {"smoke", "formal"}:
            raise ValueError("explicit smoke/formal evaluation mode required")
        if mode == "formal":
            raise ValueError("formal execution requires verified external TrustedEvaluatorService, not stdio client")
        backend, settings = _settings(sandbox_command, container_digest, case_seconds, rpc_overhead_seconds)
        for key, value in {"manifest": str(Path(manifest).resolve()), "fingerprint": fingerprint, "output_root": Path(output_root),
            "mode": mode, "manifest_hash": manifest_hash, "isolation": isolation, "sandbox_command": sandbox_command,
            "budget_root": str(Path(budget_root or Path(output_root) / "budget").resolve()),
            "backend": backend, "settings": canonical_bytes(settings)}.items():
            object.__setattr__(self, key, value)

    def evaluate(self, seal, *, code):
        row = _read_seal(seal, "pbpf-final-seal-v1", self.fingerprint)
        _check_code(code, row["candidate_hashes"])
        self.backend.validate()
        base = dict(json.loads(self.settings), **{"operation": "hidden", "manifest": self.manifest, "manifest_hash": self.manifest_hash,
            "seal": str(Path(seal).resolve()), "fingerprint": self.fingerprint, "code": code,
            "output_root": str(self.output_root.resolve()), "confirmatory": self.mode == "formal",
            "budget_root": self.budget_root})
        results, worker_pid = [], None
        for selection in row["selections"]:
            unit = dict(base, work_id=work_id(selection["key"]))
            inventory = _rpc(dict(unit, action="inventory"))
            if "cached_result" in inventory:
                results.append(inventory["cached_result"])
                worker_pid = inventory["worker_pid"]
                continue
            for test_id in inventory["case_ids"]:
                _rpc(dict(unit, action="case", case_id=test_id))
            result = _rpc(dict(unit, action="finalize"))
            results.extend(result["results"])
            worker_pid = result["worker_pid"]
        return {"results": results, "worker_pid": worker_pid, "confirmatory": False, "seal_hash": row["seal_hash"]}


def score_predictions(manifest, seal, *, code, fingerprint, model_lock, manifest_hash, sandbox_command=None,
                      budget_root=None, mode="smoke", isolation=None, container_digest="sha256:"+"0"*64,
                      case_seconds=2., rpc_overhead_seconds=5.):
    if mode not in {"smoke", "formal"}:
        raise ValueError("explicit smoke/formal prediction scoring mode required")
    if mode == "formal":
        raise ValueError("formal scoring requires verified external TrustedEvaluatorService, not stdio client")
    row = _read_seal(seal, "pbpf-prediction-seal-v1", fingerprint)
    if row["model_lock"] != model_lock:
        raise ValueError("prediction model lock mismatch")
    _check_code(code, row["candidate_hashes"])
    backend, settings = _settings(sandbox_command, container_digest, case_seconds, rpc_overhead_seconds)
    base = dict(settings, **{"operation": "prediction", "manifest": str(Path(manifest).resolve()),
        "manifest_hash": manifest_hash, "seal": str(Path(seal).resolve()), "fingerprint": fingerprint,
        "model_lock": model_lock, "code": code,
        "budget_root": str(Path(budget_root or Path(seal).parent / "prediction-budget").resolve()),
        "prediction_root": str(Path(seal).resolve().parent / "prediction-receipts"), "confirmatory": False})
    groups = sorted({work_id([p["key"][0], p["source_hash"], p["key"][4]]) for p in row["predictions"]})
    scores, worker_pid = [], None
    for group in groups:
        backend.validate()
        result = _rpc(dict(base, case_work_id=group))
        scores.extend(result["scores"])
        worker_pid = result["worker_pid"]
    return {"scores": scores, "worker_pid": worker_pid, "prediction_seal_hash": row["seal_hash"], "confirmatory": False}


def _charge(request, work, usage):
    if request.get("budget_root"):
        BudgetLedger(request["budget_root"]).record_attempt(work, usage, accepted=False)


def _metered_execute(request, source, case, work, visibility):
    _charge(request, work + "/case", Usage(**{visibility + "_verifier_cases": 1}))
    start, cpu = time.monotonic(), resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        return _execute(source, case, request.get("sandbox_command"), request["case_seconds"])
    finally:
        end_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        _charge(request, work + "/resources", Usage(wall_seconds=time.monotonic()-start,
            cpu_seconds=max(0., end_cpu.ru_utime + end_cpu.ru_stime - cpu.ru_utime - cpu.ru_stime)))


def _relative_parts(relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise ValueError("private file must remain descriptor-relative")
    return path.parts


def _open_directory_at(root_fd, parts):
    directory = os.dup(root_fd)
    try:
        for part in parts:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return directory
    except BaseException:
        os.close(directory)
        raise


def _snapshot_at(root_fd, relative, *, maximum=64*1024*1024):
    parts = _relative_parts(relative)
    directory = _open_directory_at(root_fd, parts[:-1])
    try:
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("private snapshots require bounded regular files")
            snapshot = bytearray()
            while len(snapshot) <= maximum:
                chunk = os.read(fd, min(65536, maximum+1-len(snapshot)))
                if not chunk:
                    break
                snapshot.extend(chunk)
            if len(snapshot) > maximum:
                raise ValueError("private snapshot exceeds bounded file size")
            return bytes(snapshot)
        finally:
            os.close(fd)
    finally:
        os.close(directory)


def _inventory_at(root_fd, parts=()):
    directory = _open_directory_at(root_fd, parts)
    files = set()
    try:
        def visit(fd, prefix):
            for name in os.listdir(fd):
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    files.add("/".join((*prefix, name)))
                    if len(files) > 10000:
                        raise ValueError("private inventory exceeds bounded file count")
                elif stat.S_ISDIR(info.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        visit(child, (*prefix, name))
                    finally:
                        os.close(child)
                else:
                    raise ValueError("private inventory forbids symlinks and nonregular files")
        visit(directory, ())
        return files
    finally:
        os.close(directory)


def _private_tasks(request, *, authority=None):
    """Hash, validate and parse each bounded private snapshot exactly once."""
    path = Path(request["manifest"])
    if authority is None:
        # Smoke uses the same no-follow snapshot path, without a formal claim.
        root_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        relative, validate = Path(path.name), lambda: None
    else:
        authority._validate_private_root()
        root_fd, relative, validate = authority._private_fd, path.relative_to(authority.private_root), authority._validate_private_root
    total = 0
    def snapshot(name):
        nonlocal total
        validate()
        content = _snapshot_at(root_fd, name)
        total += len(content)
        if total > 256*1024*1024:
            raise ValueError("private publication exceeds bounded snapshot size")
        validate()
        return content
    try:
        manifest_bytes = snapshot(relative)
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        if manifest_hash != request["manifest_hash"]:
            raise ValueError("private manifest checksum mismatch")
        manifest = json.loads(manifest_bytes)
        if manifest["format"] == "pbpf-rpc-evaluator-v1":
            validate()
            return manifest["tasks"]
        if manifest["format"] != "pbpf-evaluator-data-v1":
            raise ValueError("unsupported private manifest")
        parent = relative.parent
        publication = json.loads(snapshot(parent / "publication.json"))
        if (set(publication) != {"format", "experiment_fingerprint", "generator_manifest_hash", "evaluator_manifest_hash"}
                or publication["format"] != "pbpf-data-publication-v1"
                or publication["evaluator_manifest_hash"] != manifest_hash
                or publication["experiment_fingerprint"] != manifest["experiment_fingerprint"]
                or publication["generator_manifest_hash"] != manifest["generator_manifest_hash"]
                or (request.get("fingerprint") and manifest["experiment_fingerprint"] != request["fingerprint"])):
            raise ValueError("private data publication mismatch")
        expected = manifest.get("files")
        if (type(expected) is not dict or "tasks.jsonl" not in expected
                or any(type(name) is not str or not _sha(checksum) for name, checksum in expected.items())):
            raise ValueError("private manifest inventory required")
        validate()
        actual = _inventory_at(root_fd, () if parent == Path(".") else parent.parts)
        if actual != set(expected) | {relative.name, "publication.json"}:
            raise ValueError("private manifest file inventory mismatch")
        snapshots = {}
        for name, checksum in expected.items():
            _relative_parts(name)
            snapshots[name] = snapshot(parent / name)
            if hashlib.sha256(snapshots[name]).hexdigest() != checksum:
                raise ValueError("private file snapshot checksum mismatch")
        rows = [json.loads(line) for line in snapshots["tasks.jsonl"].splitlines() if line.strip()]
        validate()
        return rows
    finally:
        if authority is None:
            os.close(root_fd)


def _execute(source, case, sandbox_command, case_seconds):
    if sandbox_command:
        result = _bounded_process(sandbox_command, payload=canonical_bytes({"source": source, "test": case}),
            env={"PATH": os.defpath, "LANG": "C.UTF-8"}, timeout=case_seconds)
        if result.returncode:
            return None
        outcome = json.loads(result.stdout).get("outcome")
        return outcome if outcome in OUTCOMES else None
    from pbpf.sandbox import LocalPythonSandbox
    test = {"input": "", "output": case.get("expected_output")}
    payload = case.get("payload")
    if isinstance(payload, dict):
        test.update(payload)
    if isinstance(case.get("source"), str) and case["source"].strip():
        source = source + "\n" + case["source"]
    result = LocalPythonSandbox({case["test_id"]: test}, python_executable=sys.executable, timeout_seconds=case_seconds).execute(source, case["test_id"])
    return None if result.infrastructure_failure else result.outcome


def _run_request(request, *, authority=None, signed=None):
    authorized_request = request
    requested_formal = request.get("confirmatory") is True or request.get("mode") == "formal"
    if requested_formal and authority is None:
        raise ValueError("stdio RPC is permanently smoke-only; verified external worker required")
    # Authenticate the entire request before even opening its selected seal.
    if authority is not None:
        authority.authorize(request, signed)
    operation = request["operation"]
    seal = None
    if operation in {"hidden", "prediction"}:
        kind = "pbpf-final-seal-v1" if operation == "hidden" else "pbpf-prediction-seal-v1"
        seal = (_parse_seal(authority.seal_snapshot(request), kind, request["fingerprint"]) if authority is not None
                else _read_seal(request["seal"], kind, request["fingerprint"]))
        if authority is not None and seal["expected_keys"] != request["expected_keys"]:
            raise ValueError("signed seal inventory mismatch")
        _check_code(request["code"], seal["candidate_hashes"])
        if operation == "prediction" and request["model_lock"] != seal["model_lock"]:
            raise ValueError("model lock mismatch")
    request = dict(request, confirmatory=authority is not None)
    if authority is None:
        backend, settings = _settings(request.get("sandbox_command"), request["backend_identity"]["container_digest"],
            request["case_seconds"], request["rpc_overhead_seconds"])
    else:
        authority.backend.validate()
        backend = authority.backend
    if backend.to_dict() != request["backend_identity"]:
        raise ValueError("worker sandbox backend content identity changed")
    if not request.get("budget_root"):
        raise ValueError("trusted verifier requires durable budget ledger")
    if authority is not None:
        # Recheck live evidence immediately before private open without
        # consuming the already-reserved nonce a second time.
        if authority.binding(authorized_request, generator_pid=signed["body"]["generator_pid"],
                nonce=signed["body"]["nonce"], expires=signed["body"]["expires"]) != signed["body"]:
            raise ValueError("formal live authority changed before private open")
        request = dict(request, sandbox_command=authority.launch_command)
    # This is the first private read, after seal/code and live authorization.
    tasks = {task["task_id"]: task for task in _private_tasks(request, authority=authority)}
    sandbox = request.get("sandbox_command")
    if operation == "visible":
        cases = {test["test_id"]: test for test in tasks[request["task_id"]]["tests"] if test["hidden"] is False}
        if len(request["test_ids"]) != 1 or not set(request["test_ids"]) <= set(cases):
            raise ValueError("requested test is not visible")
        events = []
        if request.get("start_suite"):
            _charge(request, "visible/" + request["work"], Usage(visible_verifier_suites=1))
        for test_id in request["test_ids"]:
            outcome = _metered_execute(request, request["source"], cases[test_id],
                "visible/" + request["work"] + "/" + test_id, "visible")
            if outcome is None:
                raise ValueError("visible infrastructure failure")
            events.append({"test_id": test_id, "outcome": outcome, "feedback": "pass" if outcome == "PASS" else outcome.lower()})
        response = {"events": events}
    elif operation == "prediction":
        scores, outcome_cache = [], {}
        predictions = [p for p in seal["predictions"] if work_id([p["key"][0], p["source_hash"], p["key"][4]]) == request["case_work_id"]]
        if not predictions:
            raise ValueError("unknown sealed prediction case unit")
        receipt_path = Path(request["prediction_root"]) / (request["case_work_id"] + ".json")
        identity = _receipt_identity(request, seal, request["case_work_id"])
        def score_unit():
            return _prediction_unit(request, tasks, predictions)
        scores = _cached_unit(receipt_path, identity, score_unit)
        response = {"scores": scores, "prediction_seal_hash": seal["seal_hash"], "confirmatory": request["confirmatory"]}
    elif operation == "hidden":
        root = Path(request["output_root"])
        root.mkdir(parents=True, exist_ok=True)
        selections = [s for s in seal["selections"] if work_id(s["key"]) == request.get("work_id")]
        if len(selections) != 1:
            raise ValueError("exactly one sealed hidden work unit required")
        selection = selections[0]
        key_id = work_id(selection["key"])
        path = root / f"{key_id}.json"
        identity = _receipt_identity(request, seal, key_id)
        if path.exists():
            result = _read_receipt(path, identity)
            response = {"cached_result": result, "results": [result], "confirmatory": request["confirmatory"], "seal_hash": seal["seal_hash"]}
        else:
            cases = [test for test in tasks[selection["key"][2]]["tests"] if test["hidden"] is True]
            if not cases:
                raise ValueError("hidden suite cannot be empty")
            def case_path(case):
                return root / "cases" / (work_id([key_id, case["test_id"]])+".json")
            def case_identity(case):
                return dict(identity, case_id=case["test_id"])
            action = request["action"]
            if action == "inventory":
                response = {"case_ids": [case["test_id"] for case in cases]}
            elif action == "case":
                chosen = [case for case in cases if case["test_id"] == request["case_id"]]
                if len(chosen) != 1:
                    raise ValueError("unknown sealed hidden case")
                case = chosen[0]
                def execute_case():
                    if case is cases[0]:
                        _charge(request, "hidden/" + key_id, Usage(hidden_verifier_suites=1))
                    return {"outcome": _metered_execute(request, request["code"][selection["source_hash"]], case,
                        "hidden/"+key_id+"/"+case["test_id"], "hidden")}
                _cached_unit(case_path(case), case_identity(case), execute_case)
                response = {"accepted_case": case["test_id"]}
            elif action == "finalize":
                outcomes = [_read_receipt(case_path(case), case_identity(case))["outcome"] for case in cases]
                result = dict(key=selection["key"], work_id=key_id, source_hash=selection["source_hash"],
                    outcome=None if None in outcomes else ("PASS" if all(o == "PASS" for o in outcomes) else next(o for o in outcomes if o != "PASS")),
                    infrastructure_failure=None in outcomes, verifier_cases=len(cases), verifier_suites=1)
                result = _cached_unit(path, identity, lambda: result)
                response = {"results": [result], "confirmatory": request["confirmatory"], "seal_hash": seal["seal_hash"]}
            else:
                raise ValueError("unknown hidden unit action")
    else:
        raise ValueError("unknown verifier operation")
    response["worker_pid"] = os.getpid()
    return response


def _receipt_identity(request, seal, key_id):
    return dict(work_id=key_id, seal_hash=seal["seal_hash"], fingerprint=request["fingerprint"],
        manifest_hash=request["manifest_hash"], confirmatory=request["confirmatory"],
        sandbox_identity=digest(request["backend_identity"]), case_seconds=request["case_seconds"],
        rpc_overhead_seconds=request["rpc_overhead_seconds"])


def _read_receipt(path, identity):
    receipt = json.loads(path.read_bytes())
    if receipt["identity"] != identity or receipt["checksum"] != digest(receipt["result"]):
        raise ValueError("immutable evaluator receipt identity/checksum conflict")
    return receipt["result"]


def _cached_unit(path, identity, produce):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return _read_receipt(path, identity)
        result = produce()
        atomic_write(path, canonical_bytes(dict(identity=identity, result=result, checksum=digest(result))), create_once=True)
        return result


def _prediction_unit(request, tasks, predictions):
        scores, outcome_cache = [], {}
        for prediction in predictions:
            task, seed, arm, version, test_id = prediction["key"]
            cases = {test["test_id"]: test for test in tasks[task]["tests"] if test["hidden"] is True}
            case_key = work_id([task, prediction["source_hash"], test_id])
            if case_key not in outcome_cache:
                _charge(request, "future/" + case_key, Usage(hidden_verifier_suites=1))
                outcome_cache[case_key] = _metered_execute(request, request["code"][prediction["source_hash"]],
                    cases[test_id], "future/" + case_key, "hidden")
            outcome = outcome_cache[case_key]
            if outcome is None:
                scores.append({"key": prediction["key"], "infrastructure_failure": True})
                continue
            target = OUTCOMES.index(outcome)
            probabilities = prediction["probabilities"]
            scores.append({"key": prediction["key"], "nll": -math.log(max(probabilities[target], 1e-300)),
                "brier": sum((p - int(i == target))**2 for i, p in enumerate(probabilities)), "infrastructure_failure": False})
        return scores


def _worker():
    global _IN_WORKER
    _IN_WORKER = True
    response = _run_request(json.loads(sys.stdin.buffer.read()))
    sys.stdout.buffer.write(canonical_bytes(response))
