"""Cooperative external GPU lease and bump handoff tests."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from gpu_queue.models import BumpStatus, LeaseStatus
from gpu_queue.queue import GPUQueue


REPO_ROOT = Path(__file__).resolve().parent.parent


def _lease_kwargs(**overrides):
    values = {
        "owner": "neural-fire",
        "agent_id": "render-holder",
        "repo_root": "/repo/fire",
        "pid": os.getpid(),
        "process_group": os.getpgrp(),
        "effective_route": "chrome neural-fire --metal",
        "backend": "metal",
        "device": "mps:0",
        "profile": "interactive-render",
        "supports_checkpoints": True,
        "interruptible": False,
        "ttl_seconds": 120,
    }
    values.update(overrides)
    return values


def _bump_kwargs(**overrides):
    values = {
        "requester": "resident-loader",
        "agent_id": "resident-loader",
        "repo_root": "/repo/kaminos",
        "intended_route": "sam31 cold-load --mps",
        "workload_class": "cold-model-load",
        "memory_pressure": "3.32GB",
        "estimated_occupancy": "90s",
        "full_quiescence_required": True,
        "reason": "Need to prepare resident model before capture",
        "callback_address": "file:///tmp/resident-greenroom-callback",
    }
    values.update(overrides)
    return values


def test_claim_lease_records_route_identity_and_receipt(tmp_path):
    queue = GPUQueue(tmp_path / "queue")

    lease = queue.claim_lease(**_lease_kwargs())

    assert lease.lifecycle_state == LeaseStatus.ACTIVE
    assert lease.owner == "neural-fire"
    assert lease.agent_id == "render-holder"
    assert lease.repo_root == "/repo/fire"
    assert lease.pid == os.getpid()
    assert lease.process_group == os.getpgrp()
    assert lease.effective_route == "chrome neural-fire --metal"
    assert lease.backend == "metal"
    assert lease.device == "mps:0"
    assert lease.profile == "interactive-render"
    receipt = json.loads((queue.lease_receipts_dir / f"{lease.lease_id}-claim.json").read_text())
    assert receipt["transition"] == "claim"
    assert receipt["effective_route"] == lease.effective_route


def test_claim_lease_requires_flock_available(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lock_fd = open(queue.lock_path, "w")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)

    try:
        result = queue.claim_lease(**_lease_kwargs(pid=None), raise_on_blocked=False)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    assert result is None
    assert queue.lease_status() is None


def test_renew_updates_timestamps_without_replacing_identity(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs(ttl_seconds=120))
    original_claimed_at = lease.claimed_at
    original_renewed_at = lease.renewed_at
    time.sleep(0.01)

    renewed = queue.renew_lease(lease.lease_id, interruptible=True)

    assert renewed.lease_id == lease.lease_id
    assert renewed.lifecycle_state == LeaseStatus.ACTIVE
    assert renewed.claimed_at == original_claimed_at
    assert renewed.renewed_at > original_renewed_at
    assert renewed.interruptible is True


def test_expired_lease_becomes_ownership_unknown_not_free(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs(ttl_seconds=0.01))
    time.sleep(0.03)

    observed = queue.lease_status()

    assert observed is not None
    assert observed.lease_id == lease.lease_id
    assert observed.lifecycle_state == LeaseStatus.OWNERSHIP_UNKNOWN
    assert observed.unknown_reason == "ttl_expired"
    assert queue.run_one({"echo": ["echo", "nope"]}) is False


def test_dead_pid_becomes_ownership_unknown_but_live_pid_stays_active(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    live = queue.claim_lease(**_lease_kwargs(pid=os.getpid(), ttl_seconds=120))
    assert queue.lease_status().lifecycle_state == LeaseStatus.ACTIVE
    queue.release_lease(live.lease_id, released_by="holder", reason="test reset")

    dead = queue.claim_lease(**_lease_kwargs(pid=99999999, ttl_seconds=120))
    observed = queue.lease_status()

    assert observed.lease_id == dead.lease_id
    assert observed.lifecycle_state == LeaseStatus.OWNERSHIP_UNKNOWN
    assert observed.unknown_reason == "holder_pid_dead_without_release"


def test_release_writes_receipt_and_unblocks_worker(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())

    released = queue.release_lease(lease.lease_id, released_by="holder", reason="checkpoint complete")

    assert released.lifecycle_state == LeaseStatus.RELEASED
    receipt = json.loads((queue.lease_receipts_dir / f"{lease.lease_id}-release.json").read_text())
    assert receipt["transition"] == "release"
    assert receipt["released_by"] == "holder"
    assert receipt["reason"] == "checkpoint complete"


def test_bump_request_records_required_identity_and_is_idempotent(tmp_path):
    queue = GPUQueue(tmp_path / "queue")

    bump = queue.request_bump(bump_id="resident-cold-load", **_bump_kwargs())
    duplicate = queue.request_bump(bump_id="resident-cold-load", **_bump_kwargs(reason="second copy"))

    assert bump.bump_id == "resident-cold-load"
    assert duplicate.bump_id == bump.bump_id
    assert duplicate.reason == bump.reason
    assert duplicate.status == BumpStatus.PENDING
    receipt = json.loads((queue.bump_receipts_dir / "resident-cold-load-request.json").read_text())
    assert receipt["transition"] == "request"
    assert receipt["intended_route"] == bump.intended_route
    assert receipt["estimated_occupancy"] == "90s"
    assert receipt["estimated_occupancy_authority"] == "diagnostic_only"


def test_grant_now_requires_quiescence_and_transitions_lease_to_handoff(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    bump = queue.request_bump(bump_id="resident-cold-load", **_bump_kwargs())

    granted = queue.grant_bump(
        bump.bump_id,
        granted_by="neural-fire",
        checkpoint="capture-42",
        quiescence_confirmed=True,
    )
    observed_lease = queue.lease_status()

    assert granted.status == BumpStatus.GRANTED
    assert granted.granted_by == "neural-fire"
    assert granted.checkpoint == "capture-42"
    assert granted.quiescence_confirmed is True
    assert observed_lease.lease_id == lease.lease_id
    assert observed_lease.lifecycle_state == LeaseStatus.HANDOFF
    assert observed_lease.handoff_bump_id == bump.bump_id
    assert queue.run_one({"echo": ["echo", "must-not-race"]}) is False


def test_grant_after_checkpoint_waits_for_release_before_handoff(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    bump = queue.request_bump(bump_id="resident-cold-load", **_bump_kwargs())

    pending = queue.grant_bump(
        bump.bump_id,
        granted_by="neural-fire",
        checkpoint="next-capture",
        quiescence_confirmed=False,
    )
    assert pending.status == BumpStatus.GRANT_PENDING_CHECKPOINT
    assert queue.lease_status().lifecycle_state == LeaseStatus.ACTIVE

    queue.release_lease(lease.lease_id, released_by="neural-fire", reason="next-capture reached")
    granted = queue.get_bump(bump.bump_id)
    observed_lease = queue.lease_status()

    assert granted.status == BumpStatus.GRANTED
    assert granted.quiescence_confirmed is True
    assert observed_lease.lifecycle_state == LeaseStatus.HANDOFF
    assert observed_lease.handoff_bump_id == bump.bump_id


def test_release_after_grant_now_keeps_handoff_until_requester_claims(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    bump = queue.request_bump(bump_id="resident-cold-load", **_bump_kwargs())
    queue.grant_bump(
        bump.bump_id,
        granted_by="neural-fire",
        checkpoint="capture-42",
        quiescence_confirmed=True,
    )

    released = queue.release_lease(lease.lease_id, released_by="neural-fire", reason="already quiesced")

    assert released.lifecycle_state == LeaseStatus.HANDOFF
    assert released.handoff_bump_id == bump.bump_id
    assert queue.get_bump(bump.bump_id).status == BumpStatus.GRANTED
    assert queue.run_one({"echo": ["echo", "must-not-slip-through"]}) is False


def test_decline_preserves_owner_and_records_reason(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    bump = queue.request_bump(bump_id="resident-cold-load", **_bump_kwargs())

    declined = queue.decline_bump(bump.bump_id, declined_by="neural-fire", reason="training cannot checkpoint")

    assert declined.status == BumpStatus.DECLINED
    assert declined.declined_by == "neural-fire"
    assert declined.decline_reason == "training cannot checkpoint"
    assert queue.lease_status().lease_id == lease.lease_id
    assert queue.lease_status().lifecycle_state == LeaseStatus.ACTIVE


def test_requester_claims_handoff_only_for_matching_granted_bump(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lease = queue.claim_lease(**_lease_kwargs())
    bump = queue.request_bump(bump_id="resident-cold-load", **_bump_kwargs())
    queue.grant_bump(
        bump.bump_id,
        granted_by="neural-fire",
        checkpoint="capture-42",
        quiescence_confirmed=True,
    )

    wrong = queue.claim_lease(**_lease_kwargs(owner="wrong", pid=None), handoff_bump_id="other", raise_on_blocked=False)
    claimed = queue.claim_lease(
        **_lease_kwargs(
            owner="resident-loader",
            agent_id="resident-loader",
            repo_root="/repo/kaminos",
            pid=None,
            effective_route="sam31 cold-load --mps",
            profile="cold-model-load",
            interruptible=True,
        ),
        handoff_bump_id=bump.bump_id,
    )

    assert wrong is None
    assert claimed.owner == "resident-loader"
    assert claimed.handoff_bump_id == bump.bump_id
    assert queue.lease_status().lease_id == claimed.lease_id
    old_receipt = json.loads((queue.lease_receipts_dir / f"{lease.lease_id}-handoff-claim.json").read_text())
    assert old_receipt["transition"] == "handoff_claimed"
    assert old_receipt["claimed_by_lease_id"] == claimed.lease_id


def test_concurrent_grants_allow_only_one_winner(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    queue.claim_lease(**_lease_kwargs())
    first = queue.request_bump(bump_id="first", **_bump_kwargs(reason="first"))
    second = queue.request_bump(bump_id="second", **_bump_kwargs(reason="second"))
    results = []

    def grant(bump_id):
        results.append(queue.grant_bump(bump_id, granted_by="holder", checkpoint=bump_id, quiescence_confirmed=True))

    threads = [threading.Thread(target=grant, args=(first.bump_id,)), threading.Thread(target=grant, args=(second.bump_id,))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    granted = [result for result in results if result.status == BumpStatus.GRANTED]
    still_pending = [queue.get_bump(first.bump_id), queue.get_bump(second.bump_id)]
    assert len(granted) == 1
    assert sum(1 for bump in still_pending if bump.status == BumpStatus.PENDING) == 1


def test_bump_wait_uses_event_file_and_returns_after_grant(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    queue.claim_lease(**_lease_kwargs())
    bump = queue.request_bump(bump_id="resident-cold-load", **_bump_kwargs())

    def grant_later():
        time.sleep(0.1)
        queue.grant_bump(
            bump.bump_id,
            granted_by="neural-fire",
            checkpoint="capture-42",
            quiescence_confirmed=True,
        )

    thread = threading.Thread(target=grant_later)
    thread.start()
    observed = queue.wait_for_bump(bump.bump_id, timeout=2)
    thread.join()

    assert observed.status == BumpStatus.GRANTED


def test_cli_exposes_required_lease_and_bump_commands(tmp_path):
    queue_dir = tmp_path / "cli"
    cmd = [
        sys.executable,
        "-m",
        "gpu_queue.cli",
        "--queue-dir",
        str(queue_dir),
        "lease",
        "claim",
        "--owner",
        "neural-fire",
        "--agent-id",
        "render-holder",
        "--repo-root",
        "/repo/fire",
        "--effective-route",
        "chrome neural-fire --metal",
        "--backend",
        "metal",
        "--device",
        "mps:0",
        "--profile",
        "interactive-render",
        "--supports-checkpoints",
        "--pid",
        str(os.getpid()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr

    bump = subprocess.run(
        [
            sys.executable,
            "-m",
            "gpu_queue.cli",
            "--queue-dir",
            str(queue_dir),
            "bump",
            "request",
            "--bump-id",
            "resident-cold-load",
            "--requester",
            "resident-loader",
            "--agent-id",
            "resident-loader",
            "--repo-root",
            "/repo/kaminos",
            "--intended-route",
            "sam31 cold-load --mps",
            "--workload-class",
            "cold-model-load",
            "--memory-pressure",
            "3.32GB",
            "--estimated-occupancy",
            "90s",
            "--full-quiescence-required",
            "--reason",
            "resident model prep",
            "--callback-address",
            "file:///tmp/resident-loader",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert bump.returncode == 0, bump.stderr
    assert "resident-cold-load" in bump.stdout
