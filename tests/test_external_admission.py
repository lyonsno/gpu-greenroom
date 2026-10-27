"""Browser queue admission through the actual CLI; no GPU work."""
import json
import os
from pathlib import Path
import subprocess
import sys

from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue


def call(root, action="queued-claim", identity="page-one", route="fixture/browser"):
    args = [sys.executable, "-m", "gpu_queue.cli", "--queue-dir", str(root), "lease", action,
            "--lease-id", identity]
    if action == "queued-claim":
        args += ["--owner", "test-page", "--agent-id", "fixture", "--repo-root", str(root.parent),
                 "--pid", str(os.getpid()), "--effective-route", route, "--backend", "webgpu",
                 "--device", "fixture-no-gpu", "--profile", "interactive-generation"]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_admission_is_visible_fifo_and_worker_cannot_consume_ticket(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    batch = JobRequest(job_type="command", input_path="", command_argv=[sys.executable, "-c", "pass"], command_cwd=str(tmp_path))
    queue.submit(batch)
    waiting = call(queue.queue_dir)
    assert waiting["status"] == "waiting"
    assert waiting["position"] == 2
    assert waiting["ahead"] == [batch.job_id]
    ticket = queue.get_job(waiting["jobId"])
    assert ticket.job_type == "external-lease-admission" and ticket.status.value == "pending"
    from gpu_queue.operator_server import queue_snapshot
    row = next(row for row in queue_snapshot(queue)["jobs"] if row["job_id"] == ticket.job_id)
    assert row["agent_id"] == "fixture" and row["requested_route"] == "fixture/browser"
    assert call(queue.queue_dir)["jobId"] == ticket.job_id
    assert len(queue.list_jobs()) == 2
    assert queue.run_one({})
    assert not queue.run_one({}), "only the requesting page can admit its ticket"
    assert not queue.run_one({}, claimant={"capabilities": ["client-claimed-external-lease.v1"]})
    held = call(queue.queue_dir)
    assert held["status"] == "held" and held["leaseId"] == "page-one"
    receipt = json.loads((queue.queue_dir / "done" / ticket.job_id / "receipt.json").read_text())
    assert receipt["scope"] == "external-lease-admission-only"
    assert receipt["inference_completed"] is False
    assert receipt["submitted_at"] == ticket.submitted_at
    later = JobRequest(job_type="command", input_path="", command_argv=[sys.executable, "-c", "pass"], command_cwd=str(tmp_path))
    queue.submit(later)
    assert not queue.run_one({}), "active page lease blocks batch execution"
    assert call(queue.queue_dir)["status"] == "held", "response retry must retain the same lease"
    queue.release_lease("page-one", released_by="fixture", reason="fixture complete")
    assert call(queue.queue_dir)["status"] == "released", "an old request must never requeue itself"
    assert queue.run_one({})


def test_cancel_removes_pending_ticket_and_blocks_late_claim(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    queue.pause(owner="fixture")
    waiting = call(queue.queue_dir)
    assert waiting["status"] == "waiting" and waiting["paused"] is True
    assert call(queue.queue_dir, "withdraw-queued")["status"] == "cancelled"
    assert queue.get_job(waiting["jobId"]).status.value == "cancelled"
    assert call(queue.queue_dir)["status"] == "cancelled"
    assert call(queue.queue_dir, "withdraw-queued", "early-cancel")["status"] == "cancelled"
    assert call(queue.queue_dir, identity="early-cancel")["status"] == "cancelled"


def test_cancel_after_admission_before_execution_releases_reserved_lease(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    assert call(queue.queue_dir)["status"] == "held"
    result = call(queue.queue_dir, "withdraw-queued")
    assert result["status"] == "cancelled"
    assert queue.lease_status().lifecycle_state.value == "released"
    assert call(queue.queue_dir)["status"] == "cancelled"


def test_existing_holder_and_unknown_ownership_block_admission(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    first = call(queue.queue_dir)
    waiting = call(queue.queue_dir, identity="page-two")
    assert waiting["status"] == "waiting" and waiting["heldBy"] == ["test-page"]
    lease = queue.lease_status()
    queue.renew_lease(lease.lease_id, lifecycle_state="ownership_unknown")
    assert call(queue.queue_dir, identity="page-two")["status"] == "waiting"
    assert call(queue.queue_dir)["status"] == "ownership_unknown"
    queue.release_lease(first["leaseId"], released_by="fixture", reason="fixture quiescent")
    assert call(queue.queue_dir, identity="page-two")["status"] == "held"


def test_identity_reuse_does_not_change_route(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    call(queue.queue_dir)
    result = subprocess.run([sys.executable, "-m", "gpu_queue.cli", "--queue-dir", str(queue.queue_dir),
        "lease", "queued-claim", "--lease-id", "page-one", "--owner", "test-page", "--agent-id", "fixture",
        "--repo-root", str(tmp_path), "--pid", str(os.getpid()), "--effective-route", "wrong-route",
        "--backend", "webgpu", "--device", "fixture-no-gpu", "--profile", "interactive-generation"], capture_output=True, text=True)
    assert result.returncode != 0
    assert queue.lease_status().effective_route == "fixture/browser"
