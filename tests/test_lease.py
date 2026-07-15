"""Interactive lease contracts for latency-sensitive GPU work."""

from __future__ import annotations

import fcntl
import json

import pytest

from gpu_queue.models import JobRequest, JobStatus
from gpu_queue.queue import GPUQueue


def _receipt(lease) -> dict:
    return json.loads(lease.receipt_path.read_text())


def test_blocked_lease_stays_requested_without_effective_timestamp(tmp_path):
    from gpu_queue.lease import InteractiveLease

    queue = GPUQueue(tmp_path / "queue")
    blocker = open(queue.lock_path, "w")
    fcntl.flock(blocker, fcntl.LOCK_EX)
    try:
        lease = InteractiveLease(
            queue.queue_dir,
            lease_id="spoke-utterance-1",
            holder="spoke",
            purpose="final-asr",
        )

        lease.request()
        assert lease.acquire(blocking=False) is False

        receipt = _receipt(lease)
        assert receipt["schema"] == "gpu-greenroom.interactive-lease.v1"
        assert receipt["lease_id"] == "spoke-utterance-1"
        assert receipt["holder"] == "spoke"
        assert receipt["purpose"] == "final-asr"
        assert receipt["state"] == "requested"
        assert receipt["requested_at"] is not None
        assert receipt["effective_at"] is None
        assert receipt["released_at"] is None
        assert receipt["last_trustworthy_event"] == "request-persisted"
        assert receipt["current_authority"] == "requires-live-holder-process-and-flock"
        assert receipt["lock_path"] == str(queue.lock_path)
    finally:
        fcntl.flock(blocker, fcntl.LOCK_UN)
        blocker.close()


def test_effective_lease_excludes_jobs_until_release(tmp_path):
    from gpu_queue.lease import InteractiveLease

    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="echo", input_path="/tmp/input.wav")
    queue.submit(request)
    lease = InteractiveLease(
        queue.queue_dir,
        lease_id="spoke-utterance-2",
        holder="spoke",
        purpose="final-asr",
    )

    lease.request()
    assert lease.acquire(blocking=False) is True
    effective = _receipt(lease)
    assert effective["state"] == "effective"
    assert effective["effective_at"] is not None
    assert effective["last_trustworthy_event"] == "gpu-lock-acquired"

    assert queue.run_one({"echo": ["echo", "ok"]}) is False
    assert queue.get_job(request.job_id).status == JobStatus.PENDING

    lease.release()
    released = _receipt(lease)
    assert released["state"] == "released"
    assert released["released_at"] is not None
    assert released["last_trustworthy_event"] == "gpu-lock-released"
    assert queue.run_one({"echo": ["echo", "ok"]}) is True
    assert queue.get_job(request.job_id).status == JobStatus.DONE


def test_receipt_path_is_caller_owned(tmp_path):
    from gpu_queue.lease import InteractiveLease

    queue_dir = tmp_path / "queue"
    receipt_path = tmp_path / "spoke" / "lease-receipt.json"
    lease = InteractiveLease(
        queue_dir,
        lease_id="spoke-utterance-3",
        holder="spoke",
        purpose="final-asr",
        receipt_path=receipt_path,
    )

    lease.request()

    assert lease.receipt_path == receipt_path
    assert json.loads(receipt_path.read_text())["lease_id"] == "spoke-utterance-3"


def test_releasing_requested_lease_never_claims_gpu_lock_release(tmp_path):
    from gpu_queue.lease import InteractiveLease

    queue = GPUQueue(tmp_path / "queue")
    blocker = open(queue.lock_path, "w")
    fcntl.flock(blocker, fcntl.LOCK_EX)
    try:
        lease = InteractiveLease(
            queue.queue_dir,
            lease_id="spoke-utterance-4",
            holder="spoke",
            purpose="final-asr",
        )
        lease.request()
        assert lease.acquire(blocking=False) is False

        lease.release()

        receipt = _receipt(lease)
        assert receipt["state"] == "released-unacquired"
        assert receipt["effective_at"] is None
        assert receipt["last_trustworthy_event"] == "request-cancelled-before-acquisition"
    finally:
        fcntl.flock(blocker, fcntl.LOCK_UN)
        blocker.close()


def test_release_is_idempotent_and_preserves_real_acquisition(tmp_path):
    from gpu_queue.lease import InteractiveLease

    lease = InteractiveLease(
        tmp_path / "queue",
        lease_id="spoke-utterance-5",
        holder="spoke",
        purpose="final-asr",
    )
    lease.request()
    assert lease.acquire(blocking=False) is True

    first = lease.release()
    second = lease.release()

    assert second == first
    assert second["state"] == "released"
    assert second["effective_at"] is not None
    assert second["last_trustworthy_event"] == "gpu-lock-released"


def test_terminal_lease_identity_cannot_reacquire(tmp_path):
    from gpu_queue.lease import InteractiveLease, InteractiveLeaseError

    lease = InteractiveLease(
        tmp_path / "queue",
        lease_id="spoke-utterance-6",
        holder="spoke",
        purpose="final-asr",
    )
    lease.request()
    lease.release()

    with pytest.raises(InteractiveLeaseError, match="terminal"):
        lease.acquire(blocking=False)
