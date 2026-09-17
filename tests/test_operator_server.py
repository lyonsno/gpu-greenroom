import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gpu_queue.models import ExternalLease, JobRequest, JobStatus
from gpu_queue.operator_server import PAGE, make_handler, queue_snapshot
from gpu_queue.queue import GPUQueue
from http.server import ThreadingHTTPServer


def test_operator_snapshot_exposes_pause_and_route(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(
        job_type="command",
        input_path="",
        output_dir=str(tmp_path / "out"),
        repo_root="/work/repo",
        command_argv=["echo", "ok"],
        route_identity="test/route",
    )
    queue.submit(request)
    queue.pause(owner="test")

    snapshot = queue_snapshot(queue)

    assert snapshot["paused"] is True
    assert snapshot["pause_state"]["owner"] == "test"
    assert snapshot["jobs"][0]["requested_route"] == "test/route"
    assert snapshot["jobs"][0]["repo_root"] == "/work/repo"


def test_operator_snapshot_serializes_external_lease(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    lease = ExternalLease(
        owner="chrome",
        agent_id="operator",
        repo_root="/work/repo",
        effective_route="webgpu/fire",
        backend="webgpu",
        device="metal",
        profile="interactive",
        supports_checkpoints=True,
        interruptible=True,
    )
    queue._write_text_atomic(queue.current_lease_path, lease.to_json())

    snapshot = queue_snapshot(queue)

    assert snapshot["lease"]["owner"] == "chrome"
    assert snapshot["lease"]["lifecycle_state"] == "active"
    json.dumps(snapshot)


def test_operator_active_snapshot_does_not_scan_history(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path / "queue")
    historical = queue.queue_dir / "done" / "historical"
    historical.mkdir()
    (historical / "status.json").write_text("not-json")

    snapshot = queue_snapshot(queue, "active")

    assert snapshot["view"] == "active"
    assert snapshot["jobs"] == []


def test_operator_snapshot_preserves_containment_over_stale_declared_status(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="echo", input_path="/tmp/in")
    queue.submit(request)
    pending = queue.queue_dir / "pending" / request.job_id
    cancelled = queue.queue_dir / "cancelled" / request.job_id
    pending.rename(cancelled)

    snapshot = queue_snapshot(queue, "cancelled")

    assert snapshot["jobs"][0]["status"] == "inconsistent"
    assert snapshot["jobs"][0]["declared_status"] == "pending"
    assert snapshot["jobs"][0]["containment_status"] == "cancelled"
    assert snapshot["jobs"][0]["consistent"] is False


def test_operator_history_views_do_not_poll_automatically():
    assert "if(filter==='active')load()" in PAGE


def test_read_only_snapshot_does_not_refresh_lease_authority(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path / "queue")
    queue.coordination_lock_path.write_text("preserve")
    monkeypatch.setattr(queue, "lease_status", lambda: (_ for _ in ()).throw(AssertionError("mutating lease read")))
    snapshot = queue_snapshot(queue, "active", read_only=True)
    assert snapshot["read_only"] is True
    assert snapshot["observed_at"] > 0
    assert queue.coordination_lock_path.read_text() == "preserve"


def test_read_only_api_rejects_mutation(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(queue, "secret", read_only=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with __import__('pytest').raises(HTTPError) as error:
            urlopen(Request(f"http://127.0.0.1:{server.server_address[1]}/api/pause", data=b"{}", headers={"Authorization": "Bearer secret"}))
        assert error.value.code == 403
        assert not queue.is_paused()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_operator_page_renders_both_inconsistent_status_identities():
    assert "j.containment_status" in PAGE
    assert "j.declared_status" in PAGE
    assert "j.consistent&&j.status==='pending'" in PAGE
    assert "j.consistent&&j.status==='running'" in PAGE


def test_operator_api_requires_token_and_cancels_pending(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="echo", input_path="/tmp/in")
    queue.submit(request)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(queue, "secret"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        try:
            urlopen(base + "/api/state")
        except HTTPError as exc:
            assert exc.code == 401
        else:
            raise AssertionError("unauthorized state request succeeded")

        body = json.dumps({"job_id": request.job_id}).encode()
        response = urlopen(Request(
            base + "/api/cancel",
            data=body,
            method="POST",
            headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        ))
        assert json.loads(response.read())["status"] == "cancelled"
        assert queue.get_job(request.job_id).status.value == "cancelled"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_operator_resume_rejects_stale_observed_pause_epoch(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    first = queue.pause(owner="first")
    queue.resume(owner="first", epoch=first["epoch"])
    second = queue.pause(owner="second")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(queue, "secret"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        body = json.dumps({
            "requested_by": "stale-browser",
            "epoch": first["epoch"],
        }).encode()
        try:
            urlopen(Request(
                base + "/api/resume",
                data=body,
                method="POST",
                headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
            ))
        except HTTPError as exc:
            assert exc.code == 409
        else:
            raise AssertionError("stale browser resumed a newer pause epoch")
        assert queue.is_paused()
        assert queue.pause_state()["epoch"] == second["epoch"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
