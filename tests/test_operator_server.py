import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gpu_queue.models import ExternalLease, JobRequest, JobStatus
from gpu_queue.operator_server import make_handler, queue_snapshot
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
    seen = []
    original = queue.list_jobs

    def recording_list(status=None):
        seen.append(status)
        return original(status)

    monkeypatch.setattr(queue, "list_jobs", recording_list)

    snapshot = queue_snapshot(queue, "active")

    assert snapshot["view"] == "active"
    assert seen == [JobStatus.PENDING, JobStatus.RUNNING]


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
