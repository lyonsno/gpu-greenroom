"""Filesystem-backed GPU job queue with flock serialization."""

from __future__ import annotations

from . import dispatch

import ctypes
import fcntl
import hashlib
import json
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from .models import BumpRequest, BumpStatus, ExternalLease, JobRequest, JobState, JobStatus, LeaseStatus


STRUCTURED_COMMAND_CAPABILITY = "structured-command.v1"
DEFAULT_WORKER_CAPABILITIES = frozenset({STRUCTURED_COMMAND_CAPABILITY})
PAUSE_STATE_SCHEMA = "gpu-greenroom.pause-state.v1"
PAUSE_ACK_SCHEMA = "gpu-greenroom.pause-acknowledgement.v1"
TERMINATION_RECEIPT_SCHEMA = "gpu-greenroom.running-job-termination.v1"


class _DarwinProcBsdInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


class PauseStateError(RuntimeError):
    """Pause state cannot be changed under the requested epoch."""

    def __init__(self, message: str, *, expected_epoch: str | None, observed_epoch: str | None):
        super().__init__(message)
        self.expected_epoch = expected_epoch
        self.observed_epoch = observed_epoch


def _safe_substitute(template: str, mapping: dict) -> str:
    """Replace placeholders in one pass so values containing braces stay literal."""

    def replacer(match):
        key = match.group(1)
        if key in mapping:
            return str(mapping[key])
        return match.group(0)

    return re.sub(r"\{(\w+)\}", replacer, template)


def effective_worker_capabilities() -> frozenset[str]:
    configured = os.environ.get("GPU_GREENROOM_WORKER_CAPABILITIES")
    if configured is None:
        return DEFAULT_WORKER_CAPABILITIES | {dispatch.CAPABILITY,'structured-command.v2','checkpoint-continuation.v1'}
    return frozenset(
        capability.strip()
        for capability in configured.split(",")
        if capability.strip()
    )


@lru_cache(maxsize=1)
def _worker_source_identity() -> dict:
    source_root = Path(__file__).resolve().parent.parent
    identity = {
        "source_root": str(source_root),
        "commit": None,
        "git_dirty": None,
    }
    try:
        identity["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        identity["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=source_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return identity


def worker_identity() -> dict:
    return {
        **_worker_source_identity(),
        "pid": os.getpid(),
        "capabilities": sorted(effective_worker_capabilities()),
    }


def required_worker_capabilities(request: JobRequest) -> frozenset[str]:
    required = set(request.required_worker_capabilities)
    if request.command_argv is not None:
        required.add(STRUCTURED_COMMAND_CAPABILITY)
    return frozenset(required)


class GPUQueue:
    """Filesystem-backed job queue.

    Directory layout:
        queue_dir/
            pending/     - submitted jobs awaiting execution
            running/     - at most one job being executed
            done/        - completed jobs
            failed/      - failed jobs
            cancelled/   - cancelled jobs
    """

    VOLATILE_PREFIXES = ("/tmp", "/private/tmp", "/var/tmp")

    def __init__(self, queue_dir: str | Path):
        self.queue_dir = Path(queue_dir)
        self._worker_shutdown_requested = False
        self._owned_child: subprocess.Popen | None = None
        self._owned_process_group: int | None = None
        for sub in ("pending", "running", "done", "failed", "cancelled", "outputs"):
            (self.queue_dir / sub).mkdir(parents=True, exist_ok=True)
        for sub in (
            "leases",
            "leases/receipts",
            "bumps",
            "bumps/receipts",
            "events",
            "control-receipts",
        ):
            (self.queue_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def lock_path(self) -> Path:
        return self.queue_dir / "gpu.lock"

    @property
    def coordination_lock_path(self) -> Path:
        return self.queue_dir / "coordination.lock"

    @property
    def pause_path(self) -> Path:
        return self.queue_dir / "paused"

    @property
    def current_lease_path(self) -> Path:
        return self.queue_dir / "leases" / "current.json"

    @property
    def lease_receipts_dir(self) -> Path:
        return self.queue_dir / "leases" / "receipts"

    @property
    def bumps_dir(self) -> Path:
        return self.queue_dir / "bumps"

    @property
    def bump_receipts_dir(self) -> Path:
        return self.queue_dir / "bumps" / "receipts"

    @property
    def events_dir(self) -> Path:
        return self.queue_dir / "events"

    @property
    def control_receipts_dir(self) -> Path:
        return self.queue_dir / "control-receipts"

    @contextmanager
    def _coordination_lock(self):
        lock_fd = open(self.coordination_lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def _write_text_atomic(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        self._write_text_atomic(path, json.dumps(payload, indent=2))

    def _write_lease_locked(self, lease: ExternalLease) -> None:
        self._write_text_atomic(self.current_lease_path, lease.to_json())

    def _read_lease_locked(self) -> ExternalLease | None:
        if not self.current_lease_path.exists():
            return None
        return ExternalLease.from_json(self.current_lease_path.read_text())

    def _bump_path(self, bump_id: str) -> Path:
        return self.bumps_dir / f"{bump_id}.json"

    def _write_bump_locked(self, bump: BumpRequest) -> None:
        self._write_text_atomic(self._bump_path(bump.bump_id), bump.to_json())

    def _read_bump_locked(self, bump_id: str) -> BumpRequest | None:
        path = self._bump_path(bump_id)
        if not path.exists():
            return None
        return BumpRequest.from_json(path.read_text())

    def _emit_event_locked(self, kind: str, object_id: str, payload: dict) -> None:
        event = {
            "kind": kind,
            "object_id": object_id,
            "emitted_at": time.time(),
            "payload": payload,
        }
        name = f"{time.time_ns()}-{kind}-{object_id}.json"
        self._write_json_atomic(self.events_dir / name, event)

    def _write_receipt_locked(self, directory: Path, name: str, payload: dict) -> None:
        payload = {
            **payload,
            "receipt_written_at": time.time(),
        }
        self._write_json_atomic(directory / name, payload)

    def _pid_alive(self, pid: int | None) -> bool | None:
        if pid is None:
            return None
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except ProcessLookupError:
            return False

    @staticmethod
    def _process_start_identity(pid: int) -> str:
        if sys.platform == "darwin":
            info = _DarwinProcBsdInfo()
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            copied = libproc.proc_pidinfo(
                pid,
                3,  # PROC_PIDTBSDINFO
                0,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if copied != ctypes.sizeof(info):
                return ""
            return (
                f"darwin-proc-bsdinfo-v1:{pid}:"
                f"{info.pbi_start_tvsec}:{info.pbi_start_tvusec}"
            )
        stat_path = Path(f"/proc/{pid}/stat")
        try:
            fields = stat_path.read_text().split()
        except OSError:
            return ""
        if len(fields) < 22:
            return ""
        return f"linux-proc-stat-v1:{pid}:{fields[21]}"

    @staticmethod
    def _process_group_alive(process_group: int | None) -> bool | None:
        if process_group is None:
            return None
        try:
            os.killpg(process_group, 0)
            return True
        except PermissionError:
            return True
        except ProcessLookupError:
            return False

    @classmethod
    def _quiesce_owned_child(
        cls,
        proc: subprocess.Popen | None,
        process_group: int | None = None,
    ) -> bool:
        if process_group is None and proc is not None:
            try:
                process_group = os.getpgid(proc.pid)
            except ProcessLookupError:
                process_group = None
        if process_group is None:
            if proc is not None and proc.poll() is None:
                proc.wait()
            return True
        if cls._process_group_alive(process_group) is False:
            if proc is not None and proc.poll() is None:
                proc.wait()
            return True
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            if proc is not None and proc.poll() is None:
                proc.wait()
            return True

        term_deadline = time.monotonic() + 5
        while time.monotonic() < term_deadline:
            if proc is not None:
                proc.poll()
            if cls._process_group_alive(process_group) is False:
                if proc is not None and proc.poll() is None:
                    proc.wait()
                return True
            time.sleep(0.05)

        try:
            os.killpg(process_group, signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            if proc is not None:
                proc.poll()
            if cls._process_group_alive(process_group):
                return False
            if proc is not None and proc.poll() is None:
                proc.wait()
            return True
        kill_deadline = time.monotonic() + 5
        while time.monotonic() < kill_deadline:
            if proc is not None:
                proc.poll()
            if cls._process_group_alive(process_group) is False:
                if proc is not None and proc.poll() is None:
                    proc.wait()
                return True
            time.sleep(0.05)

        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        return cls._process_group_alive(process_group) is False

    def request_worker_shutdown(self) -> None:
        """Stop admission and ask the currently owned child group to quiesce."""
        self._worker_shutdown_requested = True
        proc = self._owned_child
        if proc is None or proc.poll() is not None:
            return
        try:
            process_group = self._owned_process_group or os.getpgid(proc.pid)
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            return
        # Interrupt proc.wait(); run_one owns the bounded TERM/KILL cleanup and
        # durable failure transition before it releases the execution lock.
        raise KeyboardInterrupt

    @property
    def worker_shutdown_requested(self) -> bool:
        return self._worker_shutdown_requested

    def terminate_running(
        self,
        job_id: str,
        *,
        requested_by: str,
        reason: str,
        signal_number: int = signal.SIGTERM,
    ) -> dict:
        """Pause dispatch and signal one exactly identified running child group."""
        if signal_number not in (signal.SIGTERM, signal.SIGKILL):
            raise ValueError("running job control accepts only SIGTERM or SIGKILL")
        if not requested_by.strip() or not reason.strip():
            raise ValueError("requested_by and reason are required")

        pause_ack = self.pause(
            owner=requested_by,
            contention_class="operator-running-job-termination",
        )
        with self._coordination_lock():
            job_dir = self.queue_dir / "running" / job_id
            status_path = job_dir / "status.json"
            if not status_path.is_file():
                raise RuntimeError(f"running job {job_id} not found")
            state = JobState.from_json(status_path.read_text())
            if state.status != JobStatus.RUNNING:
                raise RuntimeError(f"job {job_id} is not running")
            if not all((
                state.child_pid,
                state.child_process_group,
                state.child_start_identity,
            )):
                raise RuntimeError(
                    f"running job {job_id} has no exact child identity; refusing to signal worker"
                )

            observed_start = self._process_start_identity(state.child_pid)
            if observed_start != state.child_start_identity:
                raise RuntimeError(
                    f"running job {job_id} child identity changed or vanished"
                )
            try:
                observed_group = os.getpgid(state.child_pid)
            except ProcessLookupError as exc:
                raise RuntimeError(
                    f"running job {job_id} child vanished before signal"
                ) from exc
            if observed_group != state.child_process_group:
                raise RuntimeError(
                    f"running job {job_id} process group changed; refusing signal"
                )

            receipt_path = self.control_receipts_dir / (
                f"{time.time_ns()}-{job_id}-signal-{signal_number}.json"
            )
            receipt = {
                "schema": TERMINATION_RECEIPT_SCHEMA,
                "status": "signal_pending",
                "job_id": job_id,
                "requested_by": requested_by,
                "reason": reason,
                "signal": signal_number,
                "child_pid": state.child_pid,
                "child_process_group": state.child_process_group,
                "child_start_identity": state.child_start_identity,
                "effective_route": state.effective_route,
                "pause_epoch": pause_ack.get("epoch"),
                "requested_at": time.time(),
                "receipt_path": str(receipt_path),
            }
            self._write_json_atomic(receipt_path, receipt)
            try:
                os.killpg(state.child_process_group, signal_number)
            except OSError as exc:
                receipt["status"] = "signal_failed"
                receipt["error"] = str(exc)
                self._write_json_atomic(receipt_path, receipt)
                raise
            receipt["status"] = "signal_delivered"
            receipt["delivered_at"] = time.time()
            self._write_json_atomic(receipt_path, receipt)
            return receipt

    def _mark_lease_unknown_locked(self, lease: ExternalLease, reason: str) -> ExternalLease:
        if lease.lifecycle_state == LeaseStatus.OWNERSHIP_UNKNOWN:
            return lease
        lease.lifecycle_state = LeaseStatus.OWNERSHIP_UNKNOWN
        lease.unknown_at = time.time()
        lease.unknown_reason = reason
        self._write_lease_locked(lease)
        self._write_receipt_locked(
            self.lease_receipts_dir,
            f"{lease.lease_id}-ownership-unknown.json",
            {
                "transition": "ownership_unknown",
                "lease_id": lease.lease_id,
                "reason": reason,
                "owner": lease.owner,
                "effective_route": lease.effective_route,
            },
        )
        self._emit_event_locked("lease_ownership_unknown", lease.lease_id, {"reason": reason})
        return lease

    def _refresh_lease_observation_locked(self) -> ExternalLease | None:
        lease = self._read_lease_locked()
        if lease is None:
            return None
        if lease.lifecycle_state in (LeaseStatus.RELEASED, LeaseStatus.OWNERSHIP_UNKNOWN):
            return lease
        if lease.ttl_seconds is not None and time.time() - lease.renewed_at > lease.ttl_seconds:
            return self._mark_lease_unknown_locked(lease, "ttl_expired")
        if lease.lifecycle_state == LeaseStatus.ACTIVE and self._pid_alive(lease.pid) is False:
            return self._mark_lease_unknown_locked(lease, "holder_pid_dead_without_release")
        return lease

    def _external_execution_blocked(self) -> bool:
        with self._coordination_lock():
            return self._external_execution_blocked_locked()

    def _external_execution_blocked_locked(self) -> bool:
        lease = self._refresh_lease_observation_locked()
        return lease is not None and lease.lifecycle_state != LeaseStatus.RELEASED

    def _read_pause_state_locked(self) -> dict | None:
        if not self.pause_path.exists():
            return None
        text = self.pause_path.read_text().strip()
        if not text:
            return {
                "schema": "gpu-greenroom.pause-state.legacy-marker",
                "status": "effective",
                "owner": None,
                "epoch": None,
                "contention_class": None,
                "queue_dir": str(self.queue_dir.resolve()),
                "requested_at": None,
                "effective_at": None,
            }
        payload = json.loads(text)
        if payload.get("schema") != PAUSE_STATE_SCHEMA:
            raise ValueError(f"unsupported pause state schema: {payload.get('schema')!r}")
        return payload

    def pause_state(self) -> dict | None:
        """Return durable effective pause identity, including legacy markers."""
        with self._coordination_lock():
            return self._read_pause_state_locked()

    def pause(
        self,
        *,
        owner: str = "local",
        epoch: str | None = None,
        contention_class: str | None = None,
        requested_at: float | None = None,
    ) -> dict:
        """Linearize queued-to-running pause and return the native acknowledgement."""
        explicit_epoch = epoch is not None
        requested_at = time.time() if requested_at is None else requested_at
        with self._coordination_lock():
            existing = self._read_pause_state_locked()
            if existing is not None:
                observed_epoch = existing.get("epoch")
                if explicit_epoch and observed_epoch != epoch:
                    raise PauseStateError(
                        f"queue is already paused under epoch {observed_epoch!r}",
                        expected_epoch=epoch,
                        observed_epoch=observed_epoch,
                    )
                return {
                    "schema": PAUSE_ACK_SCHEMA,
                    "action": "pause",
                    "owner": existing.get("owner"),
                    "epoch": observed_epoch,
                    "queue_dir": str(self.queue_dir.resolve()),
                    "acknowledged_at": time.time(),
                    "effective_paused": True,
                    "idempotent": True,
                    "pause_state": existing,
                }

            effective_at = time.time()
            state = {
                "schema": PAUSE_STATE_SCHEMA,
                "status": "effective",
                "owner": owner,
                "epoch": epoch or uuid.uuid4().hex,
                "contention_class": contention_class,
                "queue_dir": str(self.queue_dir.resolve()),
                "requested_at": requested_at,
                "effective_at": effective_at,
            }
            self._write_json_atomic(self.pause_path, state)
            return {
                "schema": PAUSE_ACK_SCHEMA,
                "action": "pause",
                "owner": state["owner"],
                "epoch": state["epoch"],
                "queue_dir": state["queue_dir"],
                "acknowledged_at": effective_at,
                "effective_paused": True,
                "idempotent": False,
                "pause_state": state,
            }

    def resume(
        self,
        *,
        owner: str = "local",
        epoch: str | None = None,
    ) -> dict:
        """Resume a paused queue, rejecting a stale explicit pause epoch."""
        with self._coordination_lock():
            existing = self._read_pause_state_locked()
            if existing is None:
                return {
                    "schema": PAUSE_ACK_SCHEMA,
                    "action": "resume",
                    "owner": owner,
                    "epoch": epoch,
                    "queue_dir": str(self.queue_dir.resolve()),
                    "acknowledged_at": time.time(),
                    "effective_paused": False,
                    "idempotent": True,
                    "previous_pause": None,
                }
            observed_epoch = existing.get("epoch")
            if epoch is not None and observed_epoch != epoch:
                raise PauseStateError(
                    f"pause epoch mismatch: requested {epoch!r}, observed {observed_epoch!r}",
                    expected_epoch=epoch,
                    observed_epoch=observed_epoch,
                )
            self.pause_path.unlink()
            return {
                "schema": PAUSE_ACK_SCHEMA,
                "action": "resume",
                "owner": owner,
                "epoch": observed_epoch,
                "queue_dir": str(self.queue_dir.resolve()),
                "acknowledged_at": time.time(),
                "effective_paused": False,
                "idempotent": False,
                "previous_pause": existing,
            }

    def is_paused(self) -> bool:
        return self.pause_path.exists()

    def _try_execution_lock(self):
        lock_fd = open(self.lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_fd
        except BlockingIOError:
            lock_fd.close()
            return None

    def lease_status(self) -> ExternalLease | None:
        """Return the current external lease, refreshing stale observations first."""
        with self._coordination_lock():
            return self._refresh_lease_observation_locked()

    def claim_lease(
        self,
        *,
        owner: str,
        agent_id: str,
        repo_root: str,
        effective_route: str,
        backend: str,
        device: str,
        profile: str,
        supports_checkpoints: bool,
        interruptible: bool,
        lease_id: str | None = None,
        pid: int | None = None,
        process_group: int | None = None,
        ttl_seconds: float = 300.0,
        handoff_bump_id: str | None = None,
        raise_on_blocked: bool = True,
    ) -> ExternalLease | None:
        """Claim a cooperative external GPU lease.

        The claim briefly acquires gpu.lock to prove no Greenroom worker owns
        the execution mutex at the claim boundary. The lease then blocks worker
        dispatch until it is released or deliberately handed off.
        """
        lock_fd = self._try_execution_lock()
        if lock_fd is None:
            if raise_on_blocked:
                raise RuntimeError("gpu.lock is held; cannot claim external lease")
            return None
        try:
            with self._coordination_lock():
                if any((self.queue_dir / "running").iterdir()):
                    if raise_on_blocked:
                        raise RuntimeError(
                            "unresolved running workload blocks external lease claim"
                        )
                    return None
                current = self._refresh_lease_observation_locked()
                replacing_handoff = False
                if current is not None and current.lifecycle_state != LeaseStatus.RELEASED:
                    replacing_handoff = (
                        current.lifecycle_state == LeaseStatus.HANDOFF
                        and handoff_bump_id is not None
                        and current.handoff_bump_id == handoff_bump_id
                    )
                    bump = self._read_bump_locked(handoff_bump_id) if replacing_handoff else None
                    claimant_matches_bump = (
                        bump is not None
                        and bump.status == BumpStatus.GRANTED
                        and owner == bump.requester
                        and agent_id == bump.agent_id
                        and repo_root == bump.repo_root
                        and effective_route == bump.intended_route
                    )
                    if not replacing_handoff or not claimant_matches_bump:
                        if raise_on_blocked:
                            raise RuntimeError("external lease already blocks Greenroom execution")
                        return None

                now = time.time()
                lease = ExternalLease(
                    lease_id=lease_id or uuid.uuid4().hex[:12],
                    owner=owner,
                    agent_id=agent_id,
                    repo_root=repo_root,
                    pid=pid,
                    process_group=process_group,
                    effective_route=effective_route,
                    backend=backend,
                    device=device,
                    profile=profile,
                    supports_checkpoints=supports_checkpoints,
                    interruptible=interruptible,
                    claimed_at=now,
                    renewed_at=now,
                    ttl_seconds=ttl_seconds,
                    handoff_bump_id=handoff_bump_id,
                )
                self._write_lease_locked(lease)
                self._write_receipt_locked(
                    self.lease_receipts_dir,
                    f"{lease.lease_id}-claim.json",
                    {
                        "transition": "claim",
                        "lease_id": lease.lease_id,
                        "owner": owner,
                        "agent_id": agent_id,
                        "repo_root": repo_root,
                        "pid": pid,
                        "process_group": process_group,
                        "effective_route": effective_route,
                        "backend": backend,
                        "device": device,
                        "profile": profile,
                        "supports_checkpoints": supports_checkpoints,
                        "interruptible": interruptible,
                        "handoff_bump_id": handoff_bump_id,
                    },
                )
                if replacing_handoff and current is not None:
                    self._write_receipt_locked(
                        self.lease_receipts_dir,
                        f"{current.lease_id}-handoff-claim.json",
                        {
                            "transition": "handoff_claimed",
                            "lease_id": current.lease_id,
                            "handoff_bump_id": handoff_bump_id,
                            "claimed_by_lease_id": lease.lease_id,
                            "claimed_by_owner": lease.owner,
                        },
                    )
                self._emit_event_locked("lease_claim", lease.lease_id, {"owner": owner, "handoff_bump_id": handoff_bump_id})
                return lease
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def renew_lease(
        self,
        lease_id: str,
        *,
        interruptible: bool | None = None,
        ttl_seconds: float | None = None,
        lifecycle_state: LeaseStatus | str | None = None,
    ) -> ExternalLease:
        """Renew an existing lease without changing its route identity."""
        with self._coordination_lock():
            lease = self._refresh_lease_observation_locked()
            if lease is None or lease.lease_id != lease_id:
                raise KeyError(f"lease {lease_id} not found")
            if lease.lifecycle_state == LeaseStatus.RELEASED:
                raise RuntimeError(f"lease {lease_id} is released")
            lease.renewed_at = time.time()
            if interruptible is not None:
                lease.interruptible = interruptible
            if ttl_seconds is not None:
                lease.ttl_seconds = ttl_seconds
            if lifecycle_state is not None:
                lease.lifecycle_state = LeaseStatus(lifecycle_state)
            self._write_lease_locked(lease)
            self._write_receipt_locked(
                self.lease_receipts_dir,
                f"{lease.lease_id}-renew.json",
                {
                    "transition": "renew",
                    "lease_id": lease.lease_id,
                    "interruptible": lease.interruptible,
                    "ttl_seconds": lease.ttl_seconds,
                    "lifecycle_state": lease.lifecycle_state.value,
                },
            )
            self._emit_event_locked("lease_renew", lease.lease_id, {"lifecycle_state": lease.lifecycle_state.value})
            return lease

    def release_lease(self, lease_id: str, *, released_by: str, reason: str) -> ExternalLease:
        """Release a lease or move it into handoff when a checkpoint grant is waiting."""
        with self._coordination_lock():
            lease = self._refresh_lease_observation_locked()
            if lease is None or lease.lease_id != lease_id:
                raise KeyError(f"lease {lease_id} not found")

            waiting_bump = None
            sticky_handoff_bump = None
            if lease.lifecycle_state == LeaseStatus.HANDOFF and lease.handoff_bump_id:
                bump = self._read_bump_locked(lease.handoff_bump_id)
                if bump is not None and bump.status == BumpStatus.GRANTED:
                    sticky_handoff_bump = bump
            for bump in self.list_bumps_locked():
                if bump.holder_lease_id == lease_id and bump.status == BumpStatus.GRANT_PENDING_CHECKPOINT:
                    waiting_bump = bump
                    break

            lease.released_at = time.time()
            lease.released_by = released_by
            lease.release_reason = reason
            if waiting_bump is not None:
                lease.lifecycle_state = LeaseStatus.HANDOFF
                lease.handoff_bump_id = waiting_bump.bump_id
                waiting_bump.status = BumpStatus.GRANTED
                waiting_bump.updated_at = time.time()
                waiting_bump.granted_at = waiting_bump.granted_at or time.time()
                waiting_bump.quiescence_confirmed = True
                self._write_bump_locked(waiting_bump)
                self._write_receipt_locked(
                    self.bump_receipts_dir,
                    f"{waiting_bump.bump_id}-grant.json",
                    {
                        "transition": "grant",
                        "bump_id": waiting_bump.bump_id,
                        "holder_lease_id": lease_id,
                        "granted_by": waiting_bump.granted_by,
                        "checkpoint": waiting_bump.checkpoint,
                        "quiescence_confirmed": True,
                    },
                )
                self._emit_event_locked("bump_granted", waiting_bump.bump_id, {"holder_lease_id": lease_id})
            elif sticky_handoff_bump is not None:
                lease.lifecycle_state = LeaseStatus.HANDOFF
                lease.handoff_bump_id = sticky_handoff_bump.bump_id
            else:
                lease.lifecycle_state = LeaseStatus.RELEASED
            self._write_lease_locked(lease)
            self._write_receipt_locked(
                self.lease_receipts_dir,
                f"{lease.lease_id}-release.json",
                {
                    "transition": "release",
                    "lease_id": lease.lease_id,
                    "released_by": released_by,
                    "reason": reason,
                    "lifecycle_state": lease.lifecycle_state.value,
                    "handoff_bump_id": lease.handoff_bump_id,
                },
            )
            self._emit_event_locked("lease_release", lease.lease_id, {"lifecycle_state": lease.lifecycle_state.value})
            return lease

    def request_bump(
        self,
        *,
        requester: str,
        agent_id: str,
        repo_root: str,
        intended_route: str,
        workload_class: str,
        memory_pressure: str,
        estimated_occupancy: str,
        full_quiescence_required: bool,
        reason: str,
        callback_address: str,
        bump_id: str | None = None,
    ) -> BumpRequest:
        """Create an idempotent inbound bump request."""
        with self._coordination_lock():
            bump_id = bump_id or uuid.uuid4().hex[:12]
            existing = self._read_bump_locked(bump_id)
            if existing is not None:
                return existing
            bump = BumpRequest(
                bump_id=bump_id,
                requester=requester,
                agent_id=agent_id,
                repo_root=repo_root,
                intended_route=intended_route,
                workload_class=workload_class,
                memory_pressure=memory_pressure,
                estimated_occupancy=estimated_occupancy,
                full_quiescence_required=full_quiescence_required,
                reason=reason,
                callback_address=callback_address,
            )
            self._write_bump_locked(bump)
            self._write_receipt_locked(
                self.bump_receipts_dir,
                f"{bump.bump_id}-request.json",
                {
                    "transition": "request",
                    "bump_id": bump.bump_id,
                    "requester": requester,
                    "agent_id": agent_id,
                    "repo_root": repo_root,
                    "intended_route": intended_route,
                    "workload_class": workload_class,
                    "memory_pressure": memory_pressure,
                    "estimated_occupancy": estimated_occupancy,
                    "estimated_occupancy_authority": bump.estimated_occupancy_authority,
                    "full_quiescence_required": full_quiescence_required,
                    "reason": reason,
                    "callback_address": callback_address,
                },
            )
            self._emit_event_locked("bump_request", bump.bump_id, {"requester": requester})
            return bump

    def list_bumps_locked(self, status: BumpStatus | str | None = None) -> list[BumpRequest]:
        target = BumpStatus(status) if status else None
        bumps = []
        for path in sorted(self.bumps_dir.glob("*.json")):
            bump = BumpRequest.from_json(path.read_text())
            if target is None or bump.status == target:
                bumps.append(bump)
        return bumps

    def list_bumps(self, status: BumpStatus | str | None = None) -> list[BumpRequest]:
        with self._coordination_lock():
            return self.list_bumps_locked(status)

    def get_bump(self, bump_id: str) -> BumpRequest | None:
        with self._coordination_lock():
            return self._read_bump_locked(bump_id)

    def grant_bump(
        self,
        bump_id: str,
        *,
        granted_by: str,
        checkpoint: str,
        quiescence_confirmed: bool,
    ) -> BumpRequest:
        """Grant a bump now or after a named checkpoint.

        A quiesced grant moves the current lease into handoff. A non-quiesced
        grant records a pending checkpoint; release_lease completes the handoff.
        """
        with self._coordination_lock():
            lease = self._refresh_lease_observation_locked()
            bump = self._read_bump_locked(bump_id)
            if bump is None:
                raise KeyError(f"bump {bump_id} not found")
            if bump.status != BumpStatus.PENDING:
                return bump
            if lease is None or lease.lifecycle_state != LeaseStatus.ACTIVE:
                return bump
            bump.holder_lease_id = lease.lease_id
            bump.granted_by = granted_by
            bump.granted_at = time.time()
            bump.updated_at = bump.granted_at
            bump.checkpoint = checkpoint
            bump.quiescence_confirmed = quiescence_confirmed
            if quiescence_confirmed:
                bump.status = BumpStatus.GRANTED
                lease.lifecycle_state = LeaseStatus.HANDOFF
                lease.handoff_bump_id = bump.bump_id
                self._write_lease_locked(lease)
                self._emit_event_locked("bump_granted", bump.bump_id, {"holder_lease_id": lease.lease_id})
            else:
                bump.status = BumpStatus.GRANT_PENDING_CHECKPOINT
                self._emit_event_locked("bump_grant_pending_checkpoint", bump.bump_id, {"checkpoint": checkpoint})
            self._write_bump_locked(bump)
            self._write_receipt_locked(
                self.bump_receipts_dir,
                f"{bump.bump_id}-grant.json",
                {
                    "transition": "grant",
                    "bump_id": bump.bump_id,
                    "holder_lease_id": bump.holder_lease_id,
                    "granted_by": granted_by,
                    "checkpoint": checkpoint,
                    "quiescence_confirmed": quiescence_confirmed,
                    "status": bump.status.value,
                },
            )
            return bump

    def decline_bump(self, bump_id: str, *, declined_by: str, reason: str) -> BumpRequest:
        with self._coordination_lock():
            bump = self._read_bump_locked(bump_id)
            if bump is None:
                raise KeyError(f"bump {bump_id} not found")
            if bump.status in (BumpStatus.GRANTED, BumpStatus.DECLINED, BumpStatus.CLOSED):
                return bump
            bump.status = BumpStatus.DECLINED
            bump.declined_by = declined_by
            bump.declined_at = time.time()
            bump.updated_at = bump.declined_at
            bump.decline_reason = reason
            self._write_bump_locked(bump)
            self._write_receipt_locked(
                self.bump_receipts_dir,
                f"{bump.bump_id}-decline.json",
                {
                    "transition": "decline",
                    "bump_id": bump.bump_id,
                    "declined_by": declined_by,
                    "reason": reason,
                },
            )
            self._emit_event_locked("bump_declined", bump.bump_id, {"declined_by": declined_by})
            return bump

    def wait_for_bump(self, bump_id: str, *, timeout: float | None = None) -> BumpRequest:
        """Wait for a bump to reach a terminal/wake state using filesystem events."""
        deadline = time.time() + timeout if timeout is not None else None

        def current_if_wakeable():
            bump = self.get_bump(bump_id)
            if bump is None:
                raise KeyError(f"bump {bump_id} not found")
            if bump.status in (BumpStatus.GRANTED, BumpStatus.DECLINED, BumpStatus.CLOSED):
                return bump
            return None

        ready = current_if_wakeable()
        if ready is not None:
            return ready

        if hasattr(select, "kqueue"):
            fd = os.open(self.events_dir, os.O_RDONLY)
            kq = select.kqueue()
            try:
                event = select.kevent(
                    fd,
                    filter=select.KQ_FILTER_VNODE,
                    flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE | select.KQ_EV_CLEAR,
                    fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND | select.KQ_NOTE_RENAME,
                )
                kq.control([event], 0, 0)
                ready = current_if_wakeable()
                if ready is not None:
                    return ready
                while True:
                    wait = None if deadline is None else max(0.0, deadline - time.time())
                    if wait == 0.0:
                        raise TimeoutError(f"timed out waiting for bump {bump_id}")
                    kq.control(None, 1, wait)
                    ready = current_if_wakeable()
                    if ready is not None:
                        return ready
            finally:
                kq.close()
                os.close(fd)

        while True:
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(f"timed out waiting for bump {bump_id}")
            time.sleep(0.1)
            ready = current_if_wakeable()
            if ready is not None:
                return ready

    def _is_volatile(self, path: str) -> bool:
        resolved = str(Path(path).resolve())
        return any(resolved == p or resolved.startswith(p + "/")
                    for p in self.VOLATILE_PREFIXES)

    def submit(self, request: JobRequest, *, _staged=False) -> Path:
        """Submit a job. Returns the job directory path.

        If output_dir is empty, auto-assigns a durable path under
        queue_dir/outputs/<job_id>/. If output_dir is under /tmp or
        /private/tmp, records a volatile_output warning.
        """
        dispatch.classify(json.loads(request.to_json()), dispatch.policy(self.queue_dir))
        if type(request.cooperative_checkpoint) is not bool:
            raise ValueError('cooperative_checkpoint must be a boolean')
        if request.cooperative_checkpoint:
            if not request.command_argv or not isinstance(request.agent_id,str) or not request.agent_id.strip():
                raise ValueError('checkpoint continuation requires a structured command and explicit owner')
            if 'checkpoint-continuation.v1' not in request.required_worker_capabilities:
                request.required_worker_capabilities.append('checkpoint-continuation.v1')
            if (request.command_env or {}).get('GPU_GREENROOM_RESUME_CHECKPOINT') and (
                    not request.params.get('continuation_of') or not request.params.get('checkpoint_sha256')):
                raise ValueError('resume checkpoint must be bound to a declared continuation')
        if request.service_class == 'quick' and dispatch.CAPABILITY not in request.required_worker_capabilities:
            request.required_worker_capabilities.append(dispatch.CAPABILITY)
        if (
            request.command_argv is not None
            and STRUCTURED_COMMAND_CAPABILITY
            not in request.required_worker_capabilities
        ):
            request.required_worker_capabilities.append(
                STRUCTURED_COMMAND_CAPABILITY
            )
        if not request.output_dir:
            request.output_dir = str(self.queue_dir / "outputs" / request.job_id)

        job_dir = self.queue_dir / ("continuation-staging" if _staged else "pending") / request.job_id
        job_dir.mkdir(parents=True, exist_ok=not _staged)

        # Write request
        if _staged:
            dispatch.atomic_write(job_dir/'request.json',json.loads(request.to_json()))
        else:
            (job_dir / "request.json").write_text(request.to_json())

        warnings = []
        if self._is_volatile(request.output_dir):
            warnings.append("volatile_output")

        # Write initial status
        state = JobState(
            job_id=request.job_id,
            status=JobStatus.PENDING,
            job_type=request.job_type,
            input_path=request.input_path,
            output_dir=request.output_dir,
            params=request.params,
            submitted_at=request.submitted_at,
            warnings=warnings,
        )
        if _staged:
            dispatch.atomic_write(job_dir/'status.json',json.loads(state.to_json()))
        else:
            (job_dir / "status.json").write_text(state.to_json())

        return job_dir

    def publish_continuation(self, stage):
        stage=Path(stage)
        with self._coordination_lock():
            if stage.parent.resolve()!=(self.queue_dir/'continuation-staging').resolve():
                raise ValueError('continuation staging root differs from this queue')
            if not stage.exists():
                matches=[self.queue_dir/state/stage.name for state in ('pending','running','done','failed','cancelled')
                         if (self.queue_dir/state/stage.name/'checkpoint-handoff.json').is_file()]
                if len(matches)==1:
                    stage=matches[0]
                else:
                    raise ValueError('continuation is neither staged nor uniquely admitted')
            commit=json.loads((stage/'checkpoint-handoff.json').read_text())
            request_bytes=(stage/'request.json').read_bytes()
            request=JobRequest.from_json(request_bytes.decode())
            state=JobState.from_json((stage/'status.json').read_text())
            if (commit.get('schema')!='gpu-greenroom.checkpoint-handoff.v2' or commit.get('status')!='committed'
                    or commit.get('next_job_id')!=stage.name or request.job_id!=stage.name
                    or commit.get('next_request_sha256')!=hashlib.sha256(request_bytes).hexdigest()
                    or request.params.get('continuation_of')!=commit.get('job_id')):
                raise ValueError('continuation commit or request identity is unverified')
            if stage.parent.name!='continuation-staging':
                return stage
            if state.status!=JobStatus.PENDING or state.job_id!=stage.name:
                raise ValueError('prepared continuation state is unverified')
            if commit.get('next_state_sha256')!=hashlib.sha256((stage/'status.json').read_bytes()).hexdigest():
                raise ValueError('prepared continuation state changed after registration')
            if self.get_job(request.job_id) is not None:
                raise ValueError('continuation identity is already admitted')
            destination=stage.rename(self.queue_dir/'pending'/request.job_id)
            for directory in (destination.parent,self.queue_dir/'continuation-staging'):
                fd=os.open(directory,os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            return destination

    def recover_continuations(self):
        directory=self.queue_dir/'continuation-staging'
        recovered=[]
        for stage in directory.iterdir() if directory.is_dir() else ():
            commit=stage/'checkpoint-handoff.json'
            if commit.is_file():
                try:
                    value=json.loads(commit.read_text())
                    if not isinstance(value,dict):
                        raise ValueError('continuation commit must be an object')
                    if value.get('status')=='committed':
                        recovered.append(self.publish_continuation(stage).name)
                except FileNotFoundError as error:
                    with self._coordination_lock():
                        if stage.is_dir():
                            dispatch.atomic_write(stage/'recovery-error.json',
                                {'phase':'continuation-publication','error':str(error),'observed_at':time.time()})
                    continue
                except (ValueError,TypeError,KeyError,OSError) as error:
                    dispatch.atomic_write(stage/'recovery-error.json',
                        {'phase':'continuation-publication','error':str(error),'observed_at':time.time()})
        return recovered

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job. Returns True if cancelled, False if not found/not pending.

        Uses the short transition lock shared with run_one().
        """
        with self._coordination_lock():
            pending_dir = self.queue_dir / "pending" / job_id
            if not pending_dir.exists():
                return False

            state = JobState.from_json((pending_dir / "status.json").read_text())
            state.status = JobStatus.CANCELLED
            state.finished_at = time.time()
            self._write_text_atomic(pending_dir / "status.json", state.to_json())

            dest = self.queue_dir / "cancelled" / job_id
            shutil.move(str(pending_dir), str(dest))
            return True

    def list_jobs(self, status: JobStatus | None = None) -> list[JobState]:
        """List jobs, optionally filtered by status."""
        results = []
        dirs = ["pending", "running", "done", "failed", "cancelled"]
        if status:
            dirs = [status.value]
        for sub in dirs:
            sub_dir = self.queue_dir / sub
            if not sub_dir.exists():
                continue
            for job_dir in sorted(sub_dir.iterdir()):
                status_file = job_dir / "status.json"
                if status_file.exists():
                    results.append(JobState.from_json(status_file.read_text()))
        return results

    def get_job(self, job_id: str) -> JobState | None:
        """Get a specific job's state."""
        for sub in ("pending", "running", "done", "failed", "cancelled"):
            status_file = self.queue_dir / sub / job_id / "status.json"
            if status_file.exists():
                return JobState.from_json(status_file.read_text())
        return None

    def configure_dispatch(self, value, *, owner, reset_fairness=False):
        value=dispatch.validate(value)
        if not isinstance(owner,str) or not owner.strip():
            raise ValueError('dispatch policy owner must be explicit')
        with self._coordination_lock():
            try:
                current=dispatch.policy(self.queue_dir)
            except (ValueError,OSError):
                current={}
            fields={key:item for key,item in current.items() if key not in {'epoch','owner','configured_at'}}
            if value==fields and not reset_fairness:
                return current
            receipt={**value,'epoch':uuid.uuid4().hex,'owner':owner,'configured_at':time.time()}
            dispatch.atomic_write(self.queue_dir/'dispatch-policy.json',receipt)
            if reset_fairness:
                dispatch.atomic_write(self.queue_dir/'dispatch-state.json',
                    {'schema':'gpu-greenroom.dispatch-state.v1','last_class':'normal','reset_by':owner,'at':time.time()})
            self._emit_event_locked('dispatch_policy_changed',receipt['epoch'],receipt)
            return receipt

    def _next_pending(self, config=None) -> Path | None:
        """Get the oldest pending job directory."""
        jobs=dispatch.pending_order(self.queue_dir,config)
        return jobs[0] if jobs else None

    def _write_metadata_sidecar(self, request: JobRequest, state: JobState):
        """Write metadata.json into output_dir for asset browser consumption."""
        out = Path(request.output_dir)
        if not out.is_dir():
            return

        # Derive a human-readable name from input filename if not provided
        name = request.params.get("name", "")
        if not name:
            inp = Path(request.input_path)
            name = inp.stem  # e.g. "dragon" from "dragon.png"

        # Collect output files
        output_files = [
            f.name for f in sorted(out.iterdir())
            if f.is_file() and not f.name.startswith(".")
            and f.name != "metadata.json"
        ]

        metadata = {
            "name": name,
            "job_type": request.job_type,
            "job_id": request.job_id,
            "input_path": request.input_path,
            "input_name": Path(request.input_path).name,
            "params": request.params,
            "output_files": output_files,
            "created_at": state.finished_at,
            "duration_s": round(state.finished_at - state.started_at, 1)
            if state.started_at and state.finished_at else None,
        }

        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def _move_job(self, job_dir: Path, dest_status: str) -> Path:
        """Move a job directory to a new status folder."""
        dest = self.queue_dir / dest_status / job_dir.name
        shutil.move(str(job_dir), str(dest))
        return dest

    @staticmethod
    def _file_artifact(path: Path, *, recorded_path: str | None = None) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"artifact is not a file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": recorded_path if recorded_path is not None else str(path),
            "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size,
        }

    @staticmethod
    def _git_source_snapshot(input_path: Path) -> dict:
        if not input_path.is_file():
            raise FileNotFoundError(f"attested input is not a file: {input_path}")

        def git(*args: str) -> str:
            result = subprocess.run(
                ["git", "-C", str(input_path.parent), *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()

        root = git("rev-parse", "--show-toplevel")
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        return {
            "root": root,
            "commit": commit,
            "clean": not bool(status),
            "status": status.splitlines(),
        }

    @classmethod
    def _terminal_artifact_manifest(cls, output_dir: Path, patterns: list[str]) -> list[dict]:
        manifest = {}
        for pattern in patterns:
            pattern_path = Path(pattern)
            if pattern_path.is_absolute() or ".." in pattern_path.parts:
                raise ValueError(f"artifact manifest pattern must stay under output_dir: {pattern}")
            matches = sorted(output_dir.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"artifact manifest pattern matched no files: {pattern}")
            for path in matches:
                if not path.is_file():
                    raise ValueError(f"artifact manifest matched a non-file: {path}")
                relative = path.relative_to(output_dir).as_posix()
                manifest[relative] = cls._file_artifact(path, recorded_path=relative)
        return [manifest[path] for path in sorted(manifest)]

    def run_one(self, job_types: dict, *, claimant: dict | None = None) -> bool:
        """Pick and run the next pending job under flock.

        job_types: mapping of job_type name -> command template list.
            Template strings may contain {input_path}, {output_dir}, and
            any key from params.

        Returns True if a job was run, False if queue was empty.
        """
        if self._worker_shutdown_requested or self.is_paused():
            return False
        if self._external_execution_blocked():
            return False

        lock_fd = self._try_execution_lock()
        if lock_fd is None:
            return False

        try:
            with self._coordination_lock():
                if self.is_paused():
                    return False
                if self._worker_shutdown_requested:
                    return False
                if self._external_execution_blocked_locked():
                    return False
                if any((self.queue_dir / "running").iterdir()):
                    return False

                effective_claimant = claimant or worker_identity()
                try:
                    dispatch_policy=dispatch.policy(self.queue_dir)
                    if dispatch_policy['mode']=='two-class' and dispatch.CAPABILITY not in effective_claimant.get('capabilities',[]):
                        return False
                    job_dir = self._next_pending(dispatch_policy)
                except (ValueError,KeyError,TypeError,OSError) as error:
                    dispatch.atomic_write(self.queue_dir/'dispatch-error.json',
                        {'schema':'gpu-greenroom.dispatch-error.v1','phase':'dispatch-policy','error':str(error),
                         'observed_at':time.time(),'worker':effective_claimant})
                    return False
                if job_dir is None:
                    return False

                request = JobRequest.from_json((job_dir / "request.json").read_text())
                try:
                    service_class,basis=dispatch.classify(json.loads(request.to_json()),dispatch_policy)
                except ValueError as error:
                    dispatch.atomic_write(self.queue_dir/'dispatch-error.json',
                        {'schema':'gpu-greenroom.dispatch-error.v1','phase':'dispatch-policy','error':str(error),
                         'observed_at':time.time(),'worker':effective_claimant})
                    return False
                effective_claimant = claimant or worker_identity()
                claimant_capabilities = frozenset(
                    effective_claimant.get("capabilities", [])
                )
                if not required_worker_capabilities(request).issubset(
                    claimant_capabilities
                ):
                    return False

                # Pause creation shares this short-lived transition lock, so
                # returning from pause means no later pending-to-running move
                # can belong to the paused epoch.
                job_dir = self._move_job(job_dir, "running")
                state = JobState.from_json((job_dir / "status.json").read_text())
                state.status = JobStatus.RUNNING
                state.started_at = time.time()
                state.pid = os.getpid()
                state.worker_pid = os.getpid()
                state.dispatch={'mode':dispatch_policy['mode'],'service_class':service_class,
                                'policy_epoch':dispatch_policy.get('epoch'),'eligibility_basis':basis}
                if dispatch_policy['mode']=='two-class':
                    dispatch.atomic_write(self.queue_dir/'dispatch-state.json',
                        {'schema':'gpu-greenroom.dispatch-state.v1','last_class':service_class,
                         'job_id':state.job_id,'policy_epoch':dispatch_policy.get('epoch'),'claimed_at':state.started_at})
                self._write_text_atomic(job_dir / "status.json", state.to_json())

            # Structured command jobs carry exact argv and bypass the global
            # job-type registry and all template substitution.
            is_command_job = request.command_argv is not None
            raw_config = job_types.get(request.job_type)
            if not is_command_job and raw_config is None:
                state.status = JobStatus.FAILED
                state.finished_at = time.time()
                state.failure_phase = "dispatch"
                state.error_message = f"Unknown job type: {request.job_type}"
                state.exit_code = -1
                (job_dir / "status.json").write_text(state.to_json())
                self._move_job(job_dir, "failed")
                return True

            if is_command_job:
                if not request.command_argv:
                    state.status = JobStatus.FAILED
                    state.finished_at = time.time()
                    state.failure_phase = "dispatch"
                    state.error_message = "Structured command argv must not be empty"
                    state.exit_code = -1
                    (job_dir / "status.json").write_text(state.to_json())
                    self._move_job(job_dir, "failed")
                    return True
                cmd_template = list(request.command_argv)
                job_cwd = request.command_cwd
                job_env = request.command_env
                job_defaults = {}
                job_timeout = request.command_timeout
            elif isinstance(raw_config, list):
                # Bare list: backwards compat
                cmd_template = raw_config
                job_cwd = None
                job_env = None
                job_defaults = {}
                job_timeout = None
            else:
                # Rich dict config
                cmd_template = raw_config["cmd"]
                job_cwd = raw_config.get("cwd")
                job_env = raw_config.get("env")
                job_defaults = raw_config.get("defaults", {})
                job_timeout = raw_config.get("timeout")  # None = no timeout

            if is_command_job or isinstance(raw_config, list):
                source_attestation_mode = None
                runtime_identity_template = None
                artifact_manifest_patterns = None
            else:
                source_attestation_mode = raw_config.get("source_attestation")
                runtime_identity_template = raw_config.get("runtime_identity_cmd")
                artifact_manifest_patterns = raw_config.get("artifact_manifest")

            if is_command_job:
                cmd = list(cmd_template)
                ignored_params = {}
                substitutions = {}
            else:
                # Per-job overrides: cwd and env from params (removed before template subs)
                override_keys = {"cwd", "env"}
                reserved = {"input_path", "output_dir"}
                if "cwd" in request.params:
                    job_cwd = request.params["cwd"]
                if "env" in request.params and isinstance(request.params.get("env"), dict):
                    job_env = {**(job_env or {}), **request.params["env"]}
                safe_params = {
                    key: value for key, value in request.params.items()
                    if key not in reserved and key not in override_keys
                }
                substitutions = {
                    **job_defaults,
                    **safe_params,
                    "input_path": request.input_path,
                    "output_dir": request.output_dir,
                }

                template_str = " ".join(cmd_template)
                used_keys = {
                    key for key in substitutions
                    if "{" + key + "}" in template_str
                }
                ignored_params = {
                    key: value for key, value in safe_params.items()
                    if key not in used_keys and key not in reserved
                }

                cmd = [_safe_substitute(part, substitutions) for part in cmd_template]
            state.effective_route = shlex.join(cmd)
            (job_dir / "status.json").write_text(state.to_json())

            # Build subprocess environment
            run_env = None
            if job_env:
                run_env = {**os.environ, **job_env}
            if run_env is None:
                run_env=dict(os.environ)
            run_env.pop('GPU_GREENROOM_CONTEXT',None)
            run_env.pop('GPU_GREENROOM_READY_FD',None)
            run_env.pop('GPU_GREENROOM_RESUME_CHECKPOINT',None)
            if request.cooperative_checkpoint and (job_env or {}).get('GPU_GREENROOM_RESUME_CHECKPOINT'):
                if not request.params.get('continuation_of') or not request.params.get('checkpoint_sha256'):
                    state.status=JobStatus.FAILED
                    state.failure_phase='managed_context_preflight'
                    state.error_message='resume checkpoint must be bound to a declared continuation'
                    state.exit_code=-1
                else:
                    run_env['GPU_GREENROOM_RESUME_CHECKPOINT']=job_env['GPU_GREENROOM_RESUME_CHECKPOINT']

            # Evidence-custody preflight: attest the input source and probe the
            # runtime before anything touches output_dir.
            input_artifact = None
            source_attestation = None
            runtime_identity = None
            artifact_manifest = None
            dest_status = "failed"

            if source_attestation_mode not in (None, "git-clean-input"):
                state.status = JobStatus.FAILED
                state.failure_phase = "source_preflight"
                state.error_message = f"Unsupported source attestation mode: {source_attestation_mode}"
                state.exit_code = -1

            if state.status == JobStatus.RUNNING and source_attestation_mode == "git-clean-input":
                try:
                    input_path = Path(request.input_path)
                    input_artifact = self._file_artifact(input_path, recorded_path=request.input_path)
                    snapshot = self._git_source_snapshot(input_path)
                    source_attestation = {
                        "mode": source_attestation_mode,
                        "root": snapshot["root"],
                        "commit": snapshot["commit"],
                        "clean_before": snapshot["clean"],
                        "status_before": snapshot["status"],
                    }
                    if not snapshot["clean"]:
                        raise RuntimeError("attested input source is not clean")
                except Exception as exc:
                    state.status = JobStatus.FAILED
                    state.failure_phase = "source_preflight"
                    state.error_message = str(exc)
                    state.exit_code = -1

            if state.status == JobStatus.RUNNING and runtime_identity_template:
                runtime_cmd = [_safe_substitute(part, substitutions) for part in runtime_identity_template]
                try:
                    probe = subprocess.run(
                        runtime_cmd,
                        capture_output=True,
                        text=True,
                        cwd=job_cwd,
                        env=run_env,
                    )
                    runtime_identity = {
                        "command": runtime_cmd,
                        "exit_code": probe.returncode,
                        "stdout": probe.stdout,
                        "stderr": probe.stderr,
                    }
                    executable = Path(runtime_cmd[0])
                    if not executable.is_file():
                        resolved = shutil.which(runtime_cmd[0])
                        executable = Path(resolved) if resolved else executable
                    if executable.is_file():
                        runtime_identity["executable"] = self._file_artifact(executable)
                    if probe.returncode != 0:
                        raise RuntimeError(f"runtime identity command exited with code {probe.returncode}")
                except Exception as exc:
                    state.status = JobStatus.FAILED
                    state.failure_phase = "runtime_identity"
                    state.error_message = str(exc)
                    state.exit_code = runtime_identity["exit_code"] if runtime_identity else -1

            # Execute
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"
            stdout_path.touch()
            stderr_path.touch()
            proc = None
            shutdown_exception = None
            ownership_unresolved = False
            ready_read=ready_write=None

            if state.status == JobStatus.RUNNING:
                try:
                    os.makedirs(request.output_dir, exist_ok=True)
                    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
                        child_options={}
                        if request.cooperative_checkpoint:
                            ready_read,ready_write=os.pipe()
                            context_path=job_dir/'context.json'
                            dispatch.atomic_write(context_path,{'schema':'gpu-greenroom.job-context.v1',
                                'queue_dir':str(self.queue_dir.resolve()),'job_id':request.job_id,
                                'request_sha256':hashlib.sha256((job_dir/'request.json').read_bytes()).hexdigest(),
                                'ready_inode':os.fstat(ready_read).st_ino,'ready_device':os.fstat(ready_read).st_dev})
                            run_env['GPU_GREENROOM_CONTEXT']=str(context_path.resolve())
                            run_env['GPU_GREENROOM_READY_FD']=str(ready_read)
                            child_options['pass_fds']=(ready_read,)
                        proc = subprocess.Popen(
                            cmd,
                            stdout=out_f,
                            stderr=err_f,
                            cwd=job_cwd,
                            env=run_env,
                            start_new_session=True,
                            **child_options,
                        )
                        if ready_read is not None:
                            os.close(ready_read)
                            ready_read=None
                        self._owned_child = proc
                        self._owned_process_group = os.getpgid(proc.pid)
                        if self._worker_shutdown_requested:
                            raise KeyboardInterrupt
                        state.pid = proc.pid
                        state.child_pid = proc.pid
                        state.child_process_group = self._owned_process_group
                        state.child_start_identity = self._process_start_identity(proc.pid)
                        if not state.child_start_identity:
                            raise RuntimeError("could not record child process start identity")
                        self._write_text_atomic(job_dir / "status.json", state.to_json())
                        if ready_write is not None:
                            try:
                                os.write(ready_write,b'1')
                            except BrokenPipeError:
                                pass
                            os.close(ready_write)
                            ready_write=None
                        state.exit_code = proc.wait(timeout=job_timeout)
                        if self._worker_shutdown_requested:
                            raise KeyboardInterrupt
                    if not self._quiesce_owned_child(proc, state.child_process_group):
                        ownership_unresolved = True
                        state.status = JobStatus.RUNNING
                        state.failure_phase = "completion_quiescence_unresolved"
                        state.error_message = (
                            "Command leader exited but process-group quiescence is unresolved"
                        )
                        state.warnings.append("ownership_unknown:process_group_live")
                        dest_status = None
                    elif state.exit_code == 0:
                        state.status = JobStatus.DONE
                        dest_status = "done"
                    else:
                        state.status = JobStatus.FAILED
                        state.failure_phase = "execution"
                        state.error_message = f"Process exited with code {state.exit_code}"
                        dest_status = "failed"
                except subprocess.TimeoutExpired:
                    if self._quiesce_owned_child(proc, state.child_process_group):
                        state.status = JobStatus.FAILED
                        state.failure_phase = "timeout"
                        state.error_message = f"Job exceeded {job_timeout}s timeout"
                        state.exit_code = -1
                        dest_status = "failed"
                    else:
                        ownership_unresolved = True
                        state.status = JobStatus.RUNNING
                        state.failure_phase = "timeout_quiescence_unresolved"
                        state.error_message = "Timed-out process group could not be quiesced"
                        state.warnings.append("ownership_unknown:process_group_live")
                        dest_status = None
                except KeyboardInterrupt as exc:
                    if self._quiesce_owned_child(proc, state.child_process_group):
                        state.status = JobStatus.FAILED
                        state.failure_phase = "worker_shutdown"
                        state.error_message = "Worker stopped; owned child process group quiesced"
                        state.exit_code = -1
                        dest_status = "failed"
                    else:
                        ownership_unresolved = True
                        state.status = JobStatus.RUNNING
                        state.failure_phase = "worker_shutdown_quiescence_unresolved"
                        state.error_message = (
                            "Worker stopped but process-group quiescence is unresolved"
                        )
                        state.warnings.append("ownership_unknown:process_group_live")
                        dest_status = None
                    shutdown_exception = exc
                except Exception as e:
                    if self._quiesce_owned_child(proc, state.child_process_group):
                        state.status = JobStatus.FAILED
                        state.failure_phase = "launch"
                        state.error_message = str(e)
                        state.exit_code = -1
                        dest_status = "failed"
                    else:
                        ownership_unresolved = True
                        state.status = JobStatus.RUNNING
                        state.failure_phase = "launch_quiescence_unresolved"
                        state.error_message = (
                            f"{e}; process-group quiescence is unresolved"
                        )
                        state.warnings.append("ownership_unknown:process_group_live")
                        dest_status = None

            self._owned_child = None
            self._owned_process_group = None

            if state.status == JobStatus.DONE and source_attestation_mode == "git-clean-input":
                try:
                    snapshot = self._git_source_snapshot(Path(request.input_path))
                    source_attestation.update({
                        "commit_after": snapshot["commit"],
                        "clean_after": snapshot["clean"],
                        "status_after": snapshot["status"],
                    })
                    if (
                        not snapshot["clean"]
                        or snapshot["root"] != source_attestation["root"]
                        or snapshot["commit"] != source_attestation["commit"]
                        or self._file_artifact(
                            Path(request.input_path), recorded_path=request.input_path
                        ) != input_artifact
                    ):
                        raise RuntimeError("attested input source changed during execution")
                except Exception as exc:
                    state.status = JobStatus.FAILED
                    state.failure_phase = "source_postflight"
                    state.error_message = str(exc)
                    state.exit_code = -1
                    dest_status = "failed"

            state.finished_at = None if ownership_unresolved else time.time()

            # Metadata must exist before terminal artifacts are hashed.
            if state.status == JobStatus.DONE:
                try:
                    self._write_metadata_sidecar(request, state)
                except Exception as exc:
                    state.status = JobStatus.FAILED
                    state.failure_phase = "metadata"
                    state.error_message = str(exc)
                    state.exit_code = -1
                    dest_status = "failed"

            if state.status == JobStatus.DONE and artifact_manifest_patterns:
                try:
                    artifact_manifest = self._terminal_artifact_manifest(
                        Path(request.output_dir), artifact_manifest_patterns
                    )
                except Exception as exc:
                    state.status = JobStatus.FAILED
                    state.failure_phase = "artifact_manifest"
                    state.error_message = str(exc)
                    state.exit_code = -1
                    dest_status = "failed"

            if source_attestation is not None and "clean_after" not in source_attestation:
                source_attestation["clean_after"] = None
                source_attestation["status_after"] = None

            self._write_text_atomic(job_dir / "status.json", state.to_json())

            for descriptor in (ready_read,ready_write):
                if descriptor is not None:
                    os.close(descriptor)

            # Write receipt with full route identity
            final_job_dir = (
                job_dir
                if ownership_unresolved
                else self.queue_dir / dest_status / state.job_id
            )
            receipt = {
                "job_id": state.job_id,
                "job_type": state.job_type,
                "dispatch": state.dispatch,
                "status": (
                    "ownership_unknown" if ownership_unresolved else state.status.value
                ),
                "input_path": state.input_path,
                "output_dir": state.output_dir,
                "repo_root": request.repo_root,
                "requested_route": request.route_identity,
                "effective_route": state.effective_route,
                "effective_argv": cmd,
                "effective_cwd": job_cwd,
                "effective_env": job_env,
                "environment_inheritance": "worker-plus-overlay",
                "managed_context": run_env.get('GPU_GREENROOM_CONTEXT'),
                "managed_context_record": str(final_job_dir/'context.json') if request.cooperative_checkpoint else None,
                "reserved_environment_policy": "managed context overwritten; legacy context/ready/resume removed",
                "effective_defaults": job_defaults,
                "effective_timeout": job_timeout,
                "worker_pid": state.worker_pid,
                "child_pid": state.child_pid,
                "child_process_group": state.child_process_group,
                "child_start_identity": state.child_start_identity,
                "ignored_params": ignored_params if ignored_params else None,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
                "exit_code": state.exit_code,
                "failure_phase": state.failure_phase,
                "error_message": state.error_message,
                "warnings": state.warnings if state.warnings else None,
                "request_path": str(final_job_dir / "request.json"),
                "stdout_path": str(final_job_dir / "stdout.log"),
                "stderr_path": str(final_job_dir / "stderr.log"),
                "worker": effective_claimant,
                "input_artifact": input_artifact,
                "source_attestation": source_attestation,
                "runtime_identity": runtime_identity,
                "artifact_manifest": artifact_manifest,
            }
            self._write_json_atomic(job_dir / "receipt.json", receipt)

            if not ownership_unresolved:
                self._move_job(job_dir, dest_status)
            if shutdown_exception is not None:
                raise shutdown_exception
            return True

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def recover_stale(self) -> list[str]:
        """Check for stale running jobs (process no longer alive) and move to failed.

        Acquires flock to prevent race with run_one().
        Returns list of recovered job IDs.
        """
        lock_fd = open(self.lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            recovered = []
            running_dir = self.queue_dir / "running"
            if not running_dir.exists():
                return recovered

            for job_dir in list(running_dir.iterdir()):
                status_file = job_dir / "status.json"
                if not status_file.exists():
                    continue
                try:
                    state = JobState.from_json(status_file.read_text())
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
                exact_identity = all((
                    state.child_pid,
                    state.child_process_group,
                    state.child_start_identity,
                ))
                if not exact_identity:
                    warning = "ownership_unknown:missing_exact_child_identity"
                    if warning not in state.warnings:
                        state.warnings.append(warning)
                        self._write_text_atomic(status_file, state.to_json())
                    continue

                observed_identity = self._process_start_identity(state.child_pid)
                group_alive = self._process_group_alive(state.child_process_group)
                if observed_identity:
                    if observed_identity != state.child_start_identity:
                        warning = "ownership_unknown:child_identity_changed"
                    else:
                        try:
                            observed_group = os.getpgid(state.child_pid)
                        except ProcessLookupError:
                            observed_group = None
                        if observed_group != state.child_process_group:
                            warning = "ownership_unknown:child_process_group_changed"
                        else:
                            warning = (
                                "ownership_unknown:worker_dead_child_live"
                                if self._pid_alive(state.worker_pid) is False
                                else ""
                            )
                    if warning and warning not in state.warnings:
                        state.warnings.append(warning)
                        self._write_text_atomic(status_file, state.to_json())
                    continue
                if group_alive:
                    warning = "ownership_unknown:child_missing_group_live"
                    if warning not in state.warnings:
                        state.warnings.append(warning)
                        self._write_text_atomic(status_file, state.to_json())
                    continue

                state.status = JobStatus.FAILED
                state.finished_at = time.time()
                state.failure_phase = "stale_recovery"
                state.error_message = (
                    f"Exact child {state.child_pid} and process group "
                    f"{state.child_process_group} are no longer alive"
                )
                state.exit_code = -1
                self._write_text_atomic(status_file, state.to_json())

                receipt = {
                    "job_id": state.job_id,
                    "job_type": state.job_type,
                    "status": "failed",
                    "failure_phase": "stale_recovery",
                    "error_message": state.error_message,
                    "child_pid": state.child_pid,
                    "child_process_group": state.child_process_group,
                    "child_start_identity": state.child_start_identity,
                    "started_at": state.started_at,
                    "finished_at": state.finished_at,
                }
                self._write_json_atomic(job_dir / "receipt.json", receipt)

                self._move_job(job_dir, "failed")
                recovered.append(state.job_id)
            return recovered
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
