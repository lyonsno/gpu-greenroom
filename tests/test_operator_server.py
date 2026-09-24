import json
import http.client
import re
import socket
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from gpu_queue.models import ExternalLease, JobRequest, JobStatus
from gpu_queue.operator_server import PAGE, make_handler, operator_url, queue_snapshot
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


def test_greenroom_smoke_request_round_trips_operator_response_over_authenticated_api(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(queue, "secret", admission_control=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    request = {
        "schema": "gpu-greenroom.interactive-smoke.v1",
        "id": "9c0a03f6-b2d8-43f4-b7b7-5e848e733661",
        "kind": "interactive-smoke",
        "source": {
            "agent_id": "greenroom-floor-manager",
            "repo_root": str(tmp_path),
        },
        "title": "Inspect the current Greenroom smoke",
        "prompt": "Open the page and report whether the operator controls are legible.",
        "url": "http://127.0.0.1:8766/",
        "availability": "prepared",
        "availability_note": "The monitor is already running.",
    }
    headers = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
    try:
        response = urlopen(Request(base + "/api/smoke-requests", data=json.dumps(request).encode(), headers=headers))
        created = json.load(response)
        assert response.status == 201
        assert created["request"] == request
        assert created["status"] == "operator-needed"

        returned = urlopen(Request(base + "/api/smoke-requests/" + request["id"], headers=headers))
        assert json.load(returned) == created

        answered = urlopen(Request(
            base + "/api/smoke-requests/" + request["id"] + "/response",
            data=json.dumps({"text": "The row hierarchy is clear; the timing column needs a wider viewport."}).encode(),
            headers=headers,
            method="POST",
        ))
        result = json.load(answered)
        assert result["status"] == "responded"
        assert result["response"]["text"] == "The row hierarchy is clear; the timing column needs a wider viewport."
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_read_only_wins_over_admission_control_for_smoke_response_route(tmp_path):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(
        queue_dir, "secret", read_only=True, admission_control=True,
    ))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    body = {
        "schema": "gpu-greenroom.interactive-smoke.v1",
        "id": "9c0a03f6-b2d8-43f4-b7b7-5e848e733661",
        "kind": "interactive-smoke",
        "source": {"agent_id": "example-agent", "repo_root": str(tmp_path)},
        "title": "Inspect Greenroom",
        "prompt": "Report the visible state.",
        "url": "http://127.0.0.1:8766/",
        "availability": "prepared",
        "availability_note": "The monitor is running.",
    }
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(
                base + "/api/smoke-requests",
                data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
            ))
        assert error.value.code == 403
        assert json.loads(error.value.read())["error"] == "read_only_monitor"
        assert not (queue_dir / "smoke-requests").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


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


def test_local_operator_root_bootstraps_current_token_without_weakening_api(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(queue, "process-secret", local_operator=True),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        response = urlopen(base + "/")
        page = response.read().decode()
        assert 'globalThis.__GREENROOM_LOCAL_OPERATOR__=true' in page
        assert 'globalThis.__GREENROOM_BOOTSTRAP_TOKEN__="process-secret"' in page
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert response.headers["X-Frame-Options"] == "DENY"
        with __import__('pytest').raises(HTTPError) as error:
            urlopen(base + "/api/state")
        assert error.value.code == 401
        state = urlopen(Request(
            base + "/api/state",
            headers={"Authorization": "Bearer process-secret"},
        ))
        assert json.load(state)["schema"] == "gpu-greenroom.operator-snapshot.v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_operator_rejects_noncanonical_host_before_disclosing_or_authorizing(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(queue, "process-secret", local_operator=True),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        hostile_root = Request(base + "/", headers={"Host": "attacker.example"})
        with __import__('pytest').raises(HTTPError) as root_error:
            urlopen(hostile_root)
        assert root_error.value.code == 421
        assert b"process-secret" not in root_error.value.read()

        hostile_api = Request(
            base + "/api/state",
            headers={
                "Authorization": "Bearer process-secret",
                "Host": "attacker.example",
            },
        )
        with __import__('pytest').raises(HTTPError) as api_error:
            urlopen(hostile_api)
        assert api_error.value.code == 421
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_operator_rejects_ambiguous_or_non_origin_form_authorities(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(queue, "process-secret", local_operator=True),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    canonical = f"127.0.0.1:{port}"

    def request(method, target, headers, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", port)
        connection.putrequest(method, target, skip_host=True)
        for name, value in headers:
            connection.putheader(name, value)
        if body is not None:
            connection.putheader("Content-Length", str(len(body)))
        connection.endheaders(body)
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    try:
        status, body = request("GET", "/", [("Host", canonical), ("Host", "attacker.example")])
        assert status == 421
        assert b"process-secret" not in body

        status, _ = request(
            "POST",
            "/api/pause",
            [
                ("Host", canonical),
                ("Host", "attacker.example"),
                ("Authorization", "Bearer process-secret"),
                ("Content-Type", "application/json"),
            ],
            b"{}",
        )
        assert status == 421
        assert not queue.is_paused()

        status, body = request("GET", "http://attacker.example/", [("Host", canonical)])
        assert status == 421
        assert b"process-secret" not in body

        status, _ = request(
            "POST",
            "http://attacker.example/api/pause",
            [
                ("Host", canonical),
                ("Authorization", "Bearer process-secret"),
                ("Content-Type", "application/json"),
            ],
            b"{}",
        )
        assert status == 421
        assert not queue.is_paused()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_operator_rejects_raw_network_path_targets_before_dispatch(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(queue, "process-secret", local_operator=True),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    canonical = f"127.0.0.1:{port}"

    def raw_request(method, target, headers=(), body=b""):
        lines = [f"{method} {target} HTTP/1.1", f"Host: {canonical}", "Connection: close"]
        lines.extend(f"{name}: {value}" for name, value in headers)
        if body:
            lines.append(f"Content-Length: {len(body)}")
        request = "\r\n".join(lines).encode() + b"\r\n\r\n" + body
        with socket.create_connection(("127.0.0.1", port)) as connection:
            connection.sendall(request)
            response = b""
            while chunk := connection.recv(4096):
                response += chunk
        return int(response.split(b" ", 2)[1]), response

    try:
        for target in ("//", "//?view=active"):
            status, response = raw_request("GET", target)
            assert status == 421
            assert b"process-secret" not in response

        status, _ = raw_request("GET", "//api/state", [("Authorization", "Bearer process-secret")])
        assert status == 421

        status, _ = raw_request(
            "POST",
            "//api/pause",
            [
                ("Authorization", "Bearer process-secret"),
                ("Content-Type", "application/json"),
            ],
            b"{}",
        )
        assert status == 421
        assert not queue.is_paused()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_default_root_never_discloses_bearer_token(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(queue, "private-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        page = urlopen(f"http://127.0.0.1:{server.server_address[1]}/").read().decode()
        assert "private-token" not in page
        assert "__GREENROOM_LOCAL_OPERATOR__=true" not in page
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_operator_page_reseats_rotated_process_token(tmp_path):
    queue = GPUQueue(tmp_path / "queue")

    def rendered(token):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(queue, token, local_operator=True)
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            return urlopen(f"http://127.0.0.1:{server.server_address[1]}/").read().decode()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    before = rendered("before-restart")
    after = rendered("after-restart")
    assert 'globalThis.__GREENROOM_BOOTSTRAP_TOKEN__="before-restart"' in before
    assert "before-restart" not in after
    assert 'globalThis.__GREENROOM_BOOTSTRAP_TOKEN__="after-restart"' in after
    assert "r.status===401&&globalThis.__GREENROOM_LOCAL_OPERATOR__" in after


def test_local_operator_printed_url_is_stable_and_does_not_disclose_token():
    assert operator_url(8766, "private-token", local_operator=True) == "http://127.0.0.1:8766/"
    assert operator_url(8766, "private-token") == "http://127.0.0.1:8766/#token=private-token"


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
