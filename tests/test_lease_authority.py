"""Lease authority regressions, using real host PID/start/group observations."""

import fcntl
import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest

from gpu_queue.models import LeaseStatus
from gpu_queue.queue import GPUQueue
from tests.test_leases_bumps import _bump_kwargs, _lease_kwargs


def expire(queue, lease):
    lease.renewed_at = 0
    queue._write_lease_locked(lease)
    assert queue.lease_status().lifecycle_state == LeaseStatus.OWNERSHIP_UNKNOWN


@pytest.mark.parametrize("operation", ["release", None, "active", "released", "handoff"])
def test_unknown_cannot_be_cleared_by_ordinary_holder_operations(tmp_path, operation):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    expire(queue, lease)
    with pytest.raises(RuntimeError, match="recovery"):
        if operation == "release":
            queue.release_lease(lease.lease_id, released_by="holder", reason="settled")
        else:
            queue.renew_lease(lease.lease_id, lifecycle_state=operation)
    assert queue.lease_status().lifecycle_state == LeaseStatus.OWNERSHIP_UNKNOWN
    assert queue._external_execution_blocked()
    assert not (queue.lease_receipts_dir / f"{lease.lease_id}-release.json").exists()


@pytest.mark.parametrize("state", ["released", "handoff"])
def test_renew_cannot_manufacture_release_or_handoff(tmp_path, state):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    with pytest.raises(ValueError, match="lifecycle"):
        queue.renew_lease(lease.lease_id, lifecycle_state=state)
    assert queue.lease_status().lifecycle_state == LeaseStatus.ACTIVE


@pytest.mark.parametrize("corruption", ["missing", "reused", "wrong-group"])
def test_legacy_or_changed_identity_blocks_release_and_grant(tmp_path, corruption):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    raw = json.loads(lease.to_json())
    if corruption == "missing":
        raw.pop("process_start_identity", None)
    elif corruption == "reused":
        raw["process_start_identity"] = "different-process-at-same-pid"
    else:
        raw["process_group"] += 1
    queue._write_json_atomic(queue.current_lease_path, raw)
    bump = queue.request_bump(**_bump_kwargs())
    with pytest.raises(RuntimeError, match="recovery"):
        queue.release_lease(lease.lease_id, released_by="holder", reason="settled")
    assert queue.grant_bump(bump.bump_id, granted_by="holder", checkpoint="done",
                            quiescence_confirmed=True).status.value == "pending"
    assert queue._external_execution_blocked()


def test_claim_captures_real_start_identity_and_live_holder_can_release(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    assert asdict(lease)["process_start_identity"] == queue._process_start_identity(os.getpid())
    receipt = json.loads((queue.lease_receipts_dir / f"{lease.lease_id}-claim.json").read_text())
    assert receipt["process_start_identity"] == asdict(lease)["process_start_identity"]
    released = queue.release_lease(lease.lease_id, released_by="holder", reason="GPU work settled")
    before = (queue.lease_receipts_dir / f"{lease.lease_id}-release.json").read_bytes()
    assert released.lifecycle_state == LeaseStatus.RELEASED
    assert queue._pid_alive(os.getpid())
    assert queue.release_lease(lease.lease_id, released_by="retry", reason="retry").to_json() == released.to_json()
    assert (queue.lease_receipts_dir / f"{lease.lease_id}-release.json").read_bytes() == before


@pytest.mark.parametrize("overrides", [{"pid": None}, {"pid": -1}, {"process_group": -1}])
def test_new_claim_requires_verifiable_process_identity(tmp_path, overrides):
    queue = GPUQueue(tmp_path / "queue")
    with pytest.raises(ValueError, match="identity"):
        queue.claim_lease(**_lease_kwargs(**overrides))
    assert queue.lease_status() is None


def test_failed_release_receipt_does_not_make_queue_free(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    def fail(*args, **kwargs):
        raise OSError("receipt disk unavailable")
    monkeypatch.setattr(queue, "_write_receipt_locked", fail)
    with pytest.raises(OSError, match="disk"):
        queue.release_lease(lease.lease_id, released_by="holder", reason="settled")
    assert GPUQueue(queue.queue_dir).lease_status().lifecycle_state == LeaseStatus.ACTIVE


def test_recovery_requires_exact_identity_and_dead_process_tree(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                            stdin=subprocess.PIPE, start_new_session=True)
    try:
        lease = queue.claim_lease(**_lease_kwargs(pid=proc.pid, process_group=proc.pid))
        expire(queue, lease)
        identity = json.loads(queue.current_lease_path.read_text()).get("process_start_identity", "pre-fix")
        with pytest.raises(RuntimeError, match="live|quiescen"):
            queue.recover_lease(lease.lease_id, expected_start_identity=identity,
                                recovered_by="operator", reason="tree exited")
        proc.communicate()
        with pytest.raises(RuntimeError, match="identity"):
            queue.recover_lease(lease.lease_id, expected_start_identity="wrong",
                                recovered_by="operator", reason="wrong snapshot")
        fd = queue._try_execution_lock()
        try:
            with pytest.raises(RuntimeError, match="gpu.lock"):
                queue.recover_lease(lease.lease_id, expected_start_identity=identity,
                                    recovered_by="operator", reason="tree exited")
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()
        recovered = queue.recover_lease(lease.lease_id, expected_start_identity=identity,
                                        recovered_by="operator", reason="tree exited")
        assert recovered.lifecycle_state == LeaseStatus.RELEASED
        receipt_path = queue.lease_receipts_dir / f"{lease.lease_id}-recovery.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["transition"] == "recovery"
        assert receipt["process_start_identity"] == identity
        assert receipt["process_group"] == proc.pid
        assert receipt["quiescence_authority"] == "recorded_process_tree_exited"
        assert receipt["previous_state"] == "ownership_unknown"
        before = receipt_path.read_bytes()
        queue.recover_lease(lease.lease_id, expected_start_identity=identity,
                            recovered_by="retry", reason="retry")
        assert receipt_path.read_bytes() == before
        successor = queue.claim_lease(**_lease_kwargs(lease_id="successor"))
        with pytest.raises(KeyError):
            queue.recover_lease(lease.lease_id, expected_start_identity=identity,
                                recovered_by="stale", reason="old lease")
        assert queue.lease_status().lease_id == successor.lease_id
    finally:
        if proc.poll() is None:
            proc.communicate()


def test_unknown_live_descendants_prevent_recovery(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    expire(queue, lease)
    identity = json.loads(queue.current_lease_path.read_text()).get("process_start_identity", "pre-fix")
    monkeypatch.setattr(queue, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(queue, "_process_group_alive", lambda group: True)
    with pytest.raises(RuntimeError, match="live|quiescen"):
        queue.recover_lease(lease.lease_id, expected_start_identity=identity,
                            recovered_by="operator", reason="leader died")
    assert queue._external_execution_blocked()


def test_recovery_preserves_already_granted_handoff(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                            stdin=subprocess.PIPE, start_new_session=True)
    try:
        lease = queue.claim_lease(**_lease_kwargs(pid=proc.pid, process_group=proc.pid))
        bump = queue.request_bump(**_bump_kwargs())
        queue.grant_bump(bump.bump_id, granted_by="holder", checkpoint="done", quiescence_confirmed=True)
        held = queue.lease_status()
        expire(queue, held)
        proc.communicate()
        recovered = queue.recover_lease(lease.lease_id,
            expected_start_identity=asdict(lease)["process_start_identity"],
            recovered_by="operator", reason="holder exited after confirmed handoff")
        assert recovered.lifecycle_state == LeaseStatus.HANDOFF
        assert recovered.handoff_bump_id == bump.bump_id
        assert queue._external_execution_blocked()
    finally:
        if proc.poll() is None:
            proc.communicate()


def test_release_retry_after_receipt_failure_preserves_checkpoint_grant(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    bump = queue.request_bump(**_bump_kwargs())
    queue.grant_bump(bump.bump_id, granted_by="holder", checkpoint="next", quiescence_confirmed=False)
    original = queue._write_receipt_locked
    def fail_release(directory, name, payload):
        if name.endswith("-release.json"):
            raise OSError("release receipt unavailable")
        original(directory, name, payload)
    monkeypatch.setattr(queue, "_write_receipt_locked", fail_release)
    with pytest.raises(OSError):
        queue.release_lease(lease.lease_id, released_by="holder", reason="checkpoint")
    monkeypatch.setattr(queue, "_write_receipt_locked", original)
    assert queue.release_lease(lease.lease_id, released_by="holder", reason="checkpoint").lifecycle_state == LeaseStatus.HANDOFF
    assert queue.lease_status().handoff_bump_id == bump.bump_id


def test_lease_ids_cannot_be_recycled_into_stale_recovery_authority(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs(lease_id="once"))
    queue.release_lease(lease.lease_id, released_by="holder", reason="settled")
    with pytest.raises(ValueError, match="reuse|unique"):
        queue.claim_lease(**_lease_kwargs(lease_id="once"))


def test_expired_queued_withdrawal_does_not_clear_uncertain_ownership(tmp_path):
    from gpu_queue.external_admission import queued_claim, withdraw
    queue = GPUQueue(tmp_path / "queue")
    held = queued_claim(queue, lease_id="page", **_lease_kwargs())
    expire(queue, queue.lease_status())
    with pytest.raises(RuntimeError, match="recovery"):
        withdraw(queue, lease_id=held["leaseId"])
    assert queue._external_execution_blocked()


def test_queued_admission_cannot_recycle_a_direct_lease_after_an_intervening_holder(tmp_path):
    from gpu_queue.external_admission import queued_claim
    queue = GPUQueue(tmp_path / "queue")
    original = queue.claim_lease(**_lease_kwargs(lease_id="once"))
    queue.release_lease("once", released_by="original", reason="settled")
    claim_receipt = (queue.lease_receipts_dir / "once-claim.json").read_bytes()
    queue.claim_lease(**_lease_kwargs(lease_id="intervening"))
    queue.release_lease("intervening", released_by="other", reason="settled")
    with pytest.raises(ValueError, match="unique|reuse"):
        queued_claim(queue, lease_id="once", **_lease_kwargs(owner="replacement"))
    assert (queue.lease_receipts_dir / "once-claim.json").read_bytes() == claim_receipt
    assert queue.lease_status().lease_id == "intervening"
    assert not (queue.queue_dir / "pending/external-once").exists()
    with pytest.raises(KeyError):
        queue.release_lease(original.lease_id, released_by="original", reason="stale retry")


def test_two_pending_checkpoints_keep_first_grant_through_release_retry_and_recovery(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path / "queue")
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                            stdin=subprocess.PIPE, start_new_session=True)
    try:
        lease = queue.claim_lease(**_lease_kwargs(pid=proc.pid, process_group=proc.pid))
        first = queue.request_bump(bump_id="a", **_bump_kwargs())
        second = queue.request_bump(bump_id="b", **_bump_kwargs())
        for bump in (first, second):
            queue.grant_bump(bump.bump_id, granted_by="holder", checkpoint="next", quiescence_confirmed=False)
        write = queue._write_receipt_locked
        def fail_release(directory, name, payload):
            if name.endswith("-release.json"):
                raise OSError("release receipt unavailable")
            write(directory, name, payload)
        monkeypatch.setattr(queue, "_write_receipt_locked", fail_release)
        with pytest.raises(OSError):
            queue.release_lease(lease.lease_id, released_by="holder", reason="checkpoint")
        assert queue.get_bump(first.bump_id).status.value == "granted"
        monkeypatch.setattr(queue, "_write_receipt_locked", write)
        held = queue.release_lease(lease.lease_id, released_by="holder", reason="retry")
        assert held.handoff_bump_id == first.bump_id
        assert queue.get_bump(second.bump_id).status.value == "grant_pending_checkpoint"
        expire(queue, held)
        proc.communicate()
        recovered = queue.recover_lease(lease.lease_id,
            expected_start_identity=asdict(lease)["process_start_identity"],
            recovered_by="operator", reason="recorded tree exited")
        assert recovered.handoff_bump_id == first.bump_id
        assert recovered.lifecycle_state == LeaseStatus.HANDOFF
        assert queue._external_execution_blocked()
    finally:
        if proc.poll() is None:
            proc.communicate()


def test_direct_claim_racing_ticket_publication_fails_ticket_without_overwriting_holder(tmp_path, monkeypatch):
    from gpu_queue.external_admission import queued_claim
    queue = GPUQueue(tmp_path / "queue")
    submit = queue.submit
    def direct_claim_after_submission(request):
        path = submit(request)
        queue.claim_lease(**_lease_kwargs(lease_id="race", owner="direct-holder"))
        queue.release_lease("race", released_by="direct-holder", reason="settled")
        return path
    monkeypatch.setattr(queue, "submit", direct_claim_after_submission)
    with pytest.raises(ValueError, match="unique|reuse"):
        queued_claim(queue, lease_id="race", **_lease_kwargs(owner="queued-holder"))
    assert queue.lease_status().owner == "direct-holder"
    assert queue.get_job("external-race").failure_phase == "external_admission_identity"
    assert not (queue.queue_dir / "pending/external-race").exists()
    claim = json.loads((queue.lease_receipts_dir / "race-claim.json").read_text())
    assert claim["owner"] == "direct-holder"
