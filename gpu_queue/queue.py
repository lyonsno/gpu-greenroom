"""Filesystem-backed GPU job queue with flock serialization."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import select
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .models import BumpRequest, BumpStatus, ExternalLease, JobRequest, JobState, JobStatus, LeaseStatus


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
        for sub in ("pending", "running", "done", "failed", "cancelled", "outputs"):
            (self.queue_dir / sub).mkdir(parents=True, exist_ok=True)
        for sub in ("leases", "leases/receipts", "bumps", "bumps/receipts", "events"):
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
            lease = self._refresh_lease_observation_locked()
            return lease is not None and lease.lifecycle_state != LeaseStatus.RELEASED

    def pause(self) -> None:
        """Pause the queue. The worker finishes its current job then waits."""
        self.pause_path.touch()

    def resume(self) -> None:
        """Resume a paused queue."""
        self.pause_path.unlink(missing_ok=True)

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

    def submit(self, request: JobRequest) -> Path:
        """Submit a job. Returns the job directory path.

        If output_dir is empty, auto-assigns a durable path under
        queue_dir/outputs/<job_id>/. If output_dir is under /tmp or
        /private/tmp, records a volatile_output warning.
        """
        if not request.output_dir:
            request.output_dir = str(self.queue_dir / "outputs" / request.job_id)

        job_dir = self.queue_dir / "pending" / request.job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Write request
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
        (job_dir / "status.json").write_text(state.to_json())

        return job_dir

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job. Returns True if cancelled, False if not found/not pending.

        Serializes the pending-state transition without waiting for GPU execution.
        """
        with self._coordination_lock():
            pending_dir = self.queue_dir / "pending" / job_id
            if not pending_dir.exists():
                return False

            state = JobState.from_json((pending_dir / "status.json").read_text())
            state.status = JobStatus.CANCELLED
            state.finished_at = time.time()
            (pending_dir / "status.json").write_text(state.to_json())

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

    def _next_pending(self) -> Path | None:
        """Get the oldest pending job directory."""
        pending = self.queue_dir / "pending"
        jobs = []
        for job_dir in pending.iterdir():
            status_file = job_dir / "status.json"
            if status_file.exists():
                state = JobState.from_json(status_file.read_text())
                jobs.append((state.submitted_at, job_dir))
        if not jobs:
            return None
        jobs.sort(key=lambda x: x[0])
        return jobs[0][1]

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

    def run_one(self, job_types: dict[str, list[str]]) -> bool:
        """Pick and run the next pending job under flock.

        job_types: mapping of job_type name -> command template list.
            Template strings may contain {input_path}, {output_dir}, and
            any key from params.

        Returns True if a job was run, False if queue was empty.
        """
        if self.is_paused():
            return False
        if self._external_execution_blocked():
            return False

        lock_fd = self._try_execution_lock()
        if lock_fd is None:
            return False

        try:
            if self.is_paused():
                return False
            if self._external_execution_blocked():
                return False

            with self._coordination_lock():
                job_dir = self._next_pending()
                if job_dir is None:
                    return False

                request = JobRequest.from_json((job_dir / "request.json").read_text())

                # Claim the pending job before cancellation can observe it.
                job_dir = self._move_job(job_dir, "running")

                state = JobState.from_json((job_dir / "status.json").read_text())
                state.status = JobStatus.RUNNING
                state.started_at = time.time()
                state.pid = os.getpid()
                (job_dir / "status.json").write_text(state.to_json())

            # Resolve job type config — supports bare list or rich dict
            raw_config = job_types.get(request.job_type)
            if raw_config is None:
                state.status = JobStatus.FAILED
                state.finished_at = time.time()
                state.failure_phase = "dispatch"
                state.error_message = f"Unknown job type: {request.job_type}"
                state.exit_code = -1
                (job_dir / "status.json").write_text(state.to_json())
                self._move_job(job_dir, "failed")
                return True

            if isinstance(raw_config, list):
                # Bare list: backwards compat
                cmd_template = raw_config
                job_cwd = None
                job_env = None
                job_defaults = {}
                job_timeout = None
                source_attestation_mode = None
                runtime_identity_template = None
                artifact_manifest_patterns = None
            else:
                # Rich dict config
                cmd_template = raw_config["cmd"]
                job_cwd = raw_config.get("cwd")
                job_env = raw_config.get("env")
                job_defaults = raw_config.get("defaults", {})
                job_timeout = raw_config.get("timeout")  # None = no timeout
                source_attestation_mode = raw_config.get("source_attestation")
                runtime_identity_template = raw_config.get("runtime_identity_cmd")
                artifact_manifest_patterns = raw_config.get("artifact_manifest")

            # Per-job overrides: cwd and env from params (removed before template subs)
            OVERRIDE_KEYS = {"cwd", "env"}
            RESERVED = {"input_path", "output_dir"}
            if "cwd" in request.params:
                job_cwd = request.params["cwd"]
            if "env" in request.params and isinstance(request.params.get("env"), dict):
                job_env = {**(job_env or {}), **request.params["env"]}
            safe_params = {k: v for k, v in request.params.items() if k not in RESERVED and k not in OVERRIDE_KEYS}
            subs = {
                **job_defaults,
                **safe_params,
                "input_path": request.input_path,
                "output_dir": request.output_dir,
            }

            # Detect ignored params: user-supplied keys not consumed by template
            template_str = " ".join(cmd_template)
            used_keys = set()
            for key in subs:
                if "{" + key + "}" in template_str:
                    used_keys.add(key)
            ignored_params = {
                k: v for k, v in safe_params.items()
                if k not in used_keys and k not in RESERVED
            }

            # Safe substitution: replace all placeholders in one pass to prevent
            # chained expansion (e.g. param value "{input_path}" must stay literal)
            import re

            def safe_substitute(template: str, mapping: dict) -> str:
                def replacer(match):
                    key = match.group(1)
                    if key in mapping:
                        return str(mapping[key])
                    return match.group(0)  # leave unrecognized placeholders as-is
                return re.sub(r'\{(\w+)\}', replacer, template)

            cmd = [safe_substitute(part, subs) for part in cmd_template]
            state.effective_route = " ".join(cmd)
            (job_dir / "status.json").write_text(state.to_json())

            # Ensure output directory exists
            run_env = None
            if job_env:
                run_env = {**os.environ, **job_env}

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
                runtime_cmd = [safe_substitute(part, subs) for part in runtime_identity_template]
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

            if state.status == JobStatus.RUNNING:
                os.makedirs(request.output_dir, exist_ok=True)

            # Execute
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"

            try:
                if state.status != JobStatus.RUNNING:
                    raise RuntimeError("preflight failed")
                with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
                    proc = subprocess.run(
                        cmd,
                        stdout=out_f,
                        stderr=err_f,
                        cwd=job_cwd,
                        env=run_env,
                        timeout=job_timeout,
                    )
                state.exit_code = proc.returncode
                if proc.returncode == 0:
                    state.status = JobStatus.DONE
                    dest_status = "done"
                else:
                    state.status = JobStatus.FAILED
                    state.failure_phase = "execution"
                    state.error_message = f"Process exited with code {proc.returncode}"
                    dest_status = "failed"
            except subprocess.TimeoutExpired:
                state.status = JobStatus.FAILED
                state.failure_phase = "timeout"
                state.error_message = f"Job exceeded {job_timeout}s timeout"
                state.exit_code = -1
                dest_status = "failed"
            except Exception as e:
                if state.status != JobStatus.RUNNING:
                    pass
                else:
                    state.status = JobStatus.FAILED
                    state.failure_phase = "launch"
                    state.error_message = str(e)
                    state.exit_code = -1
                    dest_status = "failed"

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

            state.finished_at = time.time()

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

            if state.status == JobStatus.FAILED and dest_status != "failed":
                dest_status = "failed"

            if state.status == JobStatus.RUNNING:
                state.status = JobStatus.FAILED
                state.failure_phase = "launch"
                state.error_message = "job did not reach a terminal state"
                state.exit_code = -1
                dest_status = "failed"
            (job_dir / "status.json").write_text(state.to_json())

            # Write receipt with full route identity
            receipt = {
                "job_id": state.job_id,
                "job_type": state.job_type,
                "status": state.status.value,
                "input_path": state.input_path,
                "output_dir": state.output_dir,
                "effective_route": state.effective_route,
                "effective_cwd": job_cwd,
                "effective_env": job_env,
                "effective_defaults": job_defaults,
                "effective_timeout": job_timeout,
                "ignored_params": ignored_params if ignored_params else None,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
                "exit_code": state.exit_code,
                "failure_phase": state.failure_phase,
                "error_message": state.error_message,
                "warnings": state.warnings if state.warnings else None,
                "input_artifact": input_artifact,
                "source_attestation": source_attestation,
                "runtime_identity": runtime_identity,
                "artifact_manifest": artifact_manifest,
            }
            (job_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))

            self._move_job(job_dir, dest_status)
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
                state = JobState.from_json(status_file.read_text())
                if state.pid is not None:
                    try:
                        os.kill(state.pid, 0)  # check if process is alive
                    except PermissionError:
                        # PID exists but belongs to another user — treat as alive, skip
                        continue
                    except ProcessLookupError:
                        # PID does not exist — stale job
                        state.status = JobStatus.FAILED
                        state.finished_at = time.time()
                        state.failure_phase = "stale_recovery"
                        state.error_message = f"Process {state.pid} no longer alive; recovered by stale detection"
                        state.exit_code = -1
                        (status_file).write_text(state.to_json())

                        receipt = {
                            "job_id": state.job_id,
                            "job_type": state.job_type,
                            "status": "failed",
                            "failure_phase": "stale_recovery",
                            "error_message": state.error_message,
                            "started_at": state.started_at,
                            "finished_at": state.finished_at,
                        }
                        (job_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))

                        self._move_job(job_dir, "failed")
                        recovered.append(state.job_id)
            return recovered
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
