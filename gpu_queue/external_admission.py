"""FIFO admission tickets for work executed by an external page or process.

Workers leave these tickets pending. The requesting client polls queued_claim;
only the head ticket can exchange its queue position for an external lease.
The done receipt concerns admission, never completion of the external workload.
"""

import fcntl
import json
import math
import re
import time

from . import dispatch
from .models import ExternalLease, JobRequest, JobState, JobStatus, LeaseStatus

JOB_TYPE = "external-lease-admission"
# Intentionally absent from every worker's advertised capabilities. This also
# keeps existing capability-aware workers from consuming a client's ticket.
CLIENT_CAPABILITY = "client-claimed-external-lease.v1"


def _identity(lease_id):
    if not isinstance(lease_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", lease_id):
        raise ValueError("queued admission needs a safe explicit lease id")
    return "external-" + lease_id


def _cancellation(queue, lease_id):
    return queue.queue_dir / "external-admission-cancellations" / (lease_id + ".json")


def _assert_admission_identity_locked(queue, lease_id, job_id):
    current = queue._read_lease_locked()
    receipt = queue.lease_receipts_dir / f"{lease_id}-claim.json"
    if receipt.exists() or (current is not None and current.lease_id == lease_id):
        claim = json.loads(receipt.read_text()) if receipt.exists() else {}
        if claim.get("admission_job_id") != job_id:
            raise ValueError("lease ids must be unique; queued admission cannot reuse an earlier claim")


def _finish(queue, request, lease):
    directory = queue.queue_dir / "pending" / request.job_id
    if not directory.exists():
        return
    now = lease.claimed_at
    config = dispatch.policy(queue.queue_dir)
    raw_request, request_digest = dispatch.read_request(directory / "request.json")
    service_class, basis = dispatch.classify(raw_request, config, request_digest=request_digest, queue_dir=queue.queue_dir)
    state = JobState.from_json((directory / "status.json").read_text())
    state.status = JobStatus.DONE
    state.started_at = state.finished_at = now
    state.exit_code = 0
    state.effective_route = lease.effective_route
    state.dispatch = {"mode": config["mode"], "service_class": service_class,
                      "policy_epoch": config.get("epoch"), "eligibility_basis": basis}
    if request.job_id in config.get("job_overrides", {}):
        state.dispatch["operator_override"] = config["job_overrides"][request.job_id]
    dispatch.atomic_write(directory / "status.json", json.loads(state.to_json()))
    dispatch.atomic_write(directory / "receipt.json", {
        "schema": "gpu-greenroom.external-admission.v1", "job_id": request.job_id,
        "status": "done", "scope": "external-lease-admission-only", "inference_completed": False,
        "lease_id": lease.lease_id, "effective_route": lease.effective_route,
        "queue_dir": str(queue.queue_dir), "admitted_at": now,
        "submitted_at": request.submitted_at, "queue_wait_seconds": now - request.submitted_at,
        "holder_pid": lease.pid, "dispatch": state.dispatch,
    })
    if config["mode"] == "two-class":
        dispatch.atomic_write(queue.queue_dir / "dispatch-state.json", {
            "schema": "gpu-greenroom.dispatch-state.v1", "last_class": service_class,
            "job_id": request.job_id, "policy_epoch": config.get("epoch"), "claimed_at": now,
        })
    queue._move_job(directory, "done")


def queued_claim(queue, *, lease_id, **identity):
    """Idempotently enqueue and attempt admission without waiting in the CLI."""
    job_id = _identity(lease_id)
    for field in ("owner", "agent_id", "repo_root", "effective_route", "backend", "device", "profile"):
        if not isinstance(identity.get(field), str) or not identity[field].strip():
            raise ValueError(f"queued admission needs {field}")
    ttl = identity.get("ttl_seconds", 300.0)
    if type(ttl) not in (int, float) or not math.isfinite(ttl) or ttl <= 0:
        raise ValueError("lease TTL must be positive and finite")
    with queue._coordination_lock():
        _assert_admission_identity_locked(queue, lease_id, job_id)
    request = JobRequest(job_type=JOB_TYPE, input_path="", job_id=job_id,
        agent_id=identity["agent_id"], repo_root=identity["repo_root"], route_identity=identity["effective_route"],
        params={"external_lease": {"lease_id": lease_id, **identity}}, service_class="normal",
        required_worker_capabilities=[CLIENT_CAPABILITY])
    directory = queue.submit(request)
    request = JobRequest.from_json((directory / "request.json").read_text())
    lock_fd = queue._try_execution_lock()
    try:
        with queue._coordination_lock():
            state = queue.get_job(job_id)
            try:
                _assert_admission_identity_locked(queue, lease_id, job_id)
            except ValueError as error:
                directory = queue.queue_dir / "pending" / job_id
                if directory.exists():
                    state.status = JobStatus.FAILED
                    state.finished_at = time.time()
                    state.failure_phase = "external_admission_identity"
                    state.error_message = str(error)
                    dispatch.atomic_write(directory / "status.json", json.loads(state.to_json()))
                    dispatch.atomic_write(directory / "receipt.json", {
                        "status": "failed", "scope": "external-lease-admission-only",
                        "failure_phase": state.failure_phase, "error": str(error),
                        "lease_id": lease_id, "job_id": job_id, "inference_completed": False,
                    })
                    queue._move_job(directory, "failed")
                raise
            if _cancellation(queue, lease_id).exists() or state.status == JobStatus.CANCELLED:
                directory = queue.queue_dir / "pending" / job_id
                if directory.exists():
                    state.status = JobStatus.CANCELLED
                    state.finished_at = time.time()
                    dispatch.atomic_write(directory / "status.json", json.loads(state.to_json()))
                    queue._move_job(directory, "cancelled")
                return {"status": "cancelled", "jobId": job_id}
            current = queue._refresh_lease_observation_locked()
            if current and current.lease_id == lease_id:
                if current.lifecycle_state == LeaseStatus.ACTIVE:
                    # Recover a lost response or a crash between lease creation
                    # and the admission receipt. Never start a second lease.
                    _finish(queue, request, current)
                    return {"status": "held", "jobId": job_id, "leaseId": lease_id, "ttlSeconds": current.ttl_seconds}
                return {"status": current.lifecycle_state.value, "jobId": job_id, "leaseId": lease_id}
            if state.status != JobStatus.PENDING:
                return {"status": state.status.value, "jobId": job_id}
            order = dispatch.pending_order(queue.queue_dir)
            ids = [path.name for path in order]
            if job_id not in ids:
                return {"status": "waiting", "jobId": job_id, "position": None, "ahead": [],
                        "queueHeld": True, "reason": "operator queue hold", "queueDir": str(queue.queue_dir)}
            running = sorted(path.name for path in (queue.queue_dir / "running").iterdir())
            blocked = current and current.lifecycle_state != LeaseStatus.RELEASED
            if lock_fd is None or queue.is_paused() or running or blocked or ids[0] != job_id:
                return {"status": "waiting", "jobId": job_id, "position": ids.index(job_id) + 1,
                    "ahead": ids[:ids.index(job_id)], "heldBy": ([current.owner] if blocked else running),
                    "paused": queue.is_paused(), "queueDir": str(queue.queue_dir)}
            now = time.time()
            queue._assert_new_lease_id_locked(lease_id)
            start_identity = queue._capture_lease_start_identity(identity.get("pid"), identity.get("process_group"))
            lease = ExternalLease(lease_id=lease_id, **identity, process_start_identity=start_identity,
                                  claimed_at=now, renewed_at=now)
            queue._write_lease_locked(lease)
            queue._write_receipt_locked(queue.lease_receipts_dir, lease_id + "-claim.json", {
                "transition": "queued_claim", "lease_id": lease_id, "admission_job_id": job_id,
                "process_start_identity": start_identity, **identity,
            })
            queue._emit_event_locked("lease_claim", lease_id, {"owner": lease.owner, "admission_job_id": job_id})
            _finish(queue, request, lease)
            return {"status": "held", "jobId": job_id, "leaseId": lease_id, "ttlSeconds": lease.ttl_seconds}
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()


def withdraw(queue, *, lease_id):
    """Withdraw only before the client starts execution, including a lost grant.

The caller must know that its external workload has not started. Once running,
normal lease release requires the caller's positive settlement evidence.
"""
    job_id = _identity(lease_id)
    with queue._coordination_lock():
        dispatch.atomic_write(_cancellation(queue, lease_id), {
            "lease_id": lease_id, "job_id": job_id, "cancelled_at": time.time(),
            "quiescence_authority": "caller-confirms-external-work-not-started",
        })
        directory = queue.queue_dir / "pending" / job_id
        if directory.exists():
            state = JobState.from_json((directory / "status.json").read_text())
            state.status = JobStatus.CANCELLED
            state.finished_at = time.time()
            dispatch.atomic_write(directory / "status.json", json.loads(state.to_json()))
            queue._move_job(directory, "cancelled")
        lease = queue._refresh_lease_observation_locked()
        held = lease and lease.lease_id == lease_id and lease.lifecycle_state != LeaseStatus.RELEASED
    if held:
        queue.release_lease(lease_id, released_by="admission-client", reason="request withdrawn before external execution")
    return {"status": "cancelled", "jobId": job_id}
