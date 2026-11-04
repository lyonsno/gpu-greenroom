"""Tests for Metal GPU timeout watchdog auto-pause, stale-recovery WindowServer
correlation + heartbeat, worker env/now-playing attribution, disk headroom guard,
and one-command emergency `gpu-greenroom panic`.
"""

from __future__ import annotations

from collections import namedtuple
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from gpu_queue.models import JobRequest, JobState, JobStatus
from gpu_queue.queue import GPUQueue


def _wait_until(predicate, seconds: float = 8.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("fixture did not reach synchronization point within timeout")


def test_metal_gpu_timeout_triggers_watchdog_auto_pause_and_blocks_next_job(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    next_marker = tmp_path / "next-ran"

    req1 = JobRequest(
        job_type="command",
        input_path="",
        agent_id="fixture-agent",
        route_identity="mlx/switchable-9b",
        command_cwd=str(tmp_path),
        command_argv=[
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.stderr.write('[METAL] Command buffer execution failed: "
                "Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)\\n'); "
                "sys.stderr.flush(); "
                "sys.exit(134)"
            ),
        ],
    )
    req2 = JobRequest(
        job_type="command",
        input_path="",
        agent_id="other-lane",
        route_identity="fixture/next",
        command_cwd=str(tmp_path),
        command_argv=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(next_marker)!r}).touch()",
        ],
    )
    queue.submit(req1)
    queue.submit(req2)

    assert queue.run_one({}) is True

    state1 = queue.get_job(req1.job_id)
    assert state1 is not None
    assert state1.status == JobStatus.FAILED
    assert state1.failure_phase == "metal_gpu_timeout"
    assert "kIOGPUCommandBufferCallbackErrorTimeout" in (state1.error_message or "")
    assert "metal_gpu_timeout" in state1.warnings

    receipt_path = queue.queue_dir / "failed" / req1.job_id / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["failure_phase"] == "metal_gpu_timeout"
    assert receipt["metal_gpu_timeout"] is True
    assert "kIOGPUCommandBufferCallbackErrorTimeout" in receipt["metal_gpu_timeout_signature"]
    assert receipt["watchdog_auto_pause_epoch"]

    assert queue.is_paused() is True
    pause_state = queue.pause_state()
    assert pause_state is not None
    assert pause_state["owner"] == "gpu-greenroom:watchdog"
    assert pause_state["contention_class"] == "metal_gpu_timeout"

    # Subsequent job must NOT be dispatched onto a wedged GPU driver.
    assert queue.run_one({}) is False
    assert not next_marker.exists()
    assert queue.get_job(req2.job_id).status == JobStatus.PENDING


def test_stale_recovery_preserves_heartbeat_and_correlates_windowserver_restart(
    tmp_path, monkeypatch
):
    queue = GPUQueue(tmp_path / "queue")
    queue._heartbeat_interval_seconds = 0.05
    started = tmp_path / "started"
    release = tmp_path / "release"

    ws_identities = {"current": "darwin-proc-bsdinfo-v1:419:1700000000:0"}
    monkeypatch.setattr(queue, "_windowserver_start_identity", lambda: ws_identities["current"])

    req = JobRequest(
        job_type="command",
        input_path="",
        agent_id="fixture-agent",
        route_identity="mlx/svd",
        command_cwd=str(tmp_path),
        command_argv=[
            sys.executable,
            "-c",
            (
                f"import pathlib, time; pathlib.Path({str(started)!r}).touch()\n"
                f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(0.02)\n"
            ),
        ],
    )
    queue.submit(req)

    # Start worker in a thread so it launches the child and writes heartbeat.json
    worker = threading.Thread(target=queue.run_one, args=({},))
    worker.start()
    try:
        _wait_until(started.exists)
        heartbeat_path = queue.queue_dir / "running" / req.job_id / "heartbeat.json"
        _wait_until(heartbeat_path.exists)
        hb_initial = json.loads(heartbeat_path.read_text())
        assert (
            hb_initial["windowserver_start_identity_at_launch"]
            == "darwin-proc-bsdinfo-v1:419:1700000000:0"
        )
        # Wait briefly so at least one periodic heartbeat update lands
        time.sleep(0.12)
        hb_updated = json.loads(heartbeat_path.read_text())
        assert hb_updated["last_observed_alive_at"] >= hb_initial["last_observed_alive_at"]

        # Snapshot the running directory files before the worker closes the job,
        # so we can simulate an abrupt box/worker death while the job was in running/.
        running_job_dir = queue.queue_dir / "running" / req.job_id
        status_snapshot = (running_job_dir / "status.json").read_text()
        heartbeat_snapshot = heartbeat_path.read_text()
        request_snapshot = (running_job_dir / "request.json").read_text()
    finally:
        release.touch()
        worker.join(timeout=5)
        assert not worker.is_alive()

    # Now reconstruct the stale running state after both child and worker died
    # during a WindowServer watchdog reset.
    shutil.rmtree(queue.queue_dir / "done" / req.job_id, ignore_errors=True)
    stale_dir = queue.queue_dir / "running" / req.job_id
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "request.json").write_text(request_snapshot)
    (stale_dir / "heartbeat.json").write_text(heartbeat_snapshot)
    stale_state = JobState.from_json(status_snapshot)
    stale_state.worker_pid = 99999999
    (stale_dir / "status.json").write_text(stale_state.to_json())

    # Simulate WindowServer having been killed and respawned with a new PID/birth time.
    ws_identities["current"] = "darwin-proc-bsdinfo-v1:89100:1700000300:0"

    recovered = queue.recover_stale()
    assert req.job_id in recovered

    recovered_state = queue.get_job(req.job_id)
    assert recovered_state is not None
    assert recovered_state.status == JobStatus.FAILED
    assert recovered_state.failure_phase == "stale_recovery"
    assert recovered_state.last_observed_alive_at is not None
    assert recovered_state.started_at <= recovered_state.last_observed_alive_at <= recovered_state.finished_at
    assert recovered_state.correlated_windowserver_restart is True
    assert "correlated_windowserver_restart" in recovered_state.warnings

    receipt = json.loads((queue.queue_dir / "failed" / req.job_id / "receipt.json").read_text())
    assert receipt["failure_phase"] == "stale_recovery"
    assert receipt["last_observed_alive_at"] == recovered_state.last_observed_alive_at
    assert receipt["recovered_at"] == recovered_state.finished_at
    assert (
        receipt["windowserver_start_identity_at_launch"]
        == "darwin-proc-bsdinfo-v1:419:1700000000:0"
    )
    assert (
        receipt["windowserver_start_identity_at_recovery"]
        == "darwin-proc-bsdinfo-v1:89100:1700000300:0"
    )
    assert receipt["correlated_windowserver_restart"] is True


def test_worker_injects_greenroom_job_id_and_maintains_now_playing(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    observed_path = tmp_path / "observed.json"

    req = JobRequest(
        job_type="command",
        input_path="",
        agent_id="fixture-agent",
        route_identity="mlx/switchable",
        command_cwd=str(tmp_path),
        command_env={"GREENROOM_JOB_ID": "spoofed-id"},
        command_argv=[
            sys.executable,
            "-c",
            (
                "import json, os, pathlib, time\n"
                "qdir = pathlib.Path(os.environ['GPU_GREENROOM_QUEUE_DIR'])\n"
                "now_playing = {}\n"
                "for _ in range(50):\n"
                "    now_playing = json.loads((qdir / 'now-playing.json').read_text())\n"
                "    if now_playing.get('child_pid') is not None:\n"
                "        break\n"
                "    time.sleep(0.01)\n"
                f"pathlib.Path({str(observed_path)!r}).write_text(json.dumps({{\n"
                "    'GREENROOM_JOB_ID': os.environ.get('GREENROOM_JOB_ID'),\n"
                "    'GPU_GREENROOM_JOB_ID': os.environ.get('GPU_GREENROOM_JOB_ID'),\n"
                "    'GPU_GREENROOM_QUEUE_DIR': os.environ.get('GPU_GREENROOM_QUEUE_DIR'),\n"
                "    'now_playing': now_playing,\n"
                "}))\n"
            ),
        ],
    )
    queue.submit(req)
    assert queue.run_one({}) is True

    observed = json.loads(observed_path.read_text())
    assert observed["GREENROOM_JOB_ID"] == req.job_id
    assert observed["GPU_GREENROOM_JOB_ID"] == req.job_id
    assert observed["GPU_GREENROOM_QUEUE_DIR"] == str(queue.queue_dir.resolve())
    assert observed["now_playing"]["active"] is True
    assert observed["now_playing"]["job_id"] == req.job_id
    assert observed["now_playing"]["agent_id"] == "fixture-agent"
    assert observed["now_playing"]["requested_route"] == "mlx/switchable"
    assert observed["now_playing"]["child_pid"] is not None

    after = json.loads((queue.queue_dir / "now-playing.json").read_text())
    assert after["active"] is False
    assert after["last_job_id"] == req.job_id
    assert after["last_status"] == "done"


def test_disk_headroom_circuit_breaker_pauses_before_claiming_job(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path / "queue")
    req = JobRequest(
        job_type="command",
        input_path="",
        agent_id="fixture-disk",
        route_identity="fixture/disk",
        command_cwd=str(tmp_path),
        command_argv=[sys.executable, "-c", "pass"],
    )
    queue.submit(req)

    DiskUsage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: DiskUsage(total=100 * 1024**3, used=99 * 1024**3, free=64 * 1024**2),
    )

    assert queue.run_one({}) is False
    assert queue.get_job(req.job_id).status == JobStatus.PENDING
    assert queue.is_paused() is True
    pause_state = queue.pause_state()
    assert pause_state["owner"] == "gpu-greenroom:watchdog"
    assert pause_state["contention_class"] == "disk_headroom_low"


def test_panic_cli_and_api_pauses_queue_and_sigkills_running_process_group(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    started = tmp_path / "started"
    next_started = tmp_path / "next-started"

    running_req = JobRequest(
        job_type="command",
        input_path="",
        agent_id="runaway-lane",
        route_identity="fixture/runaway",
        command_cwd=str(tmp_path),
        command_argv=[
            sys.executable,
            "-c",
            (
                "import pathlib, signal, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                f"pathlib.Path({str(started)!r}).touch()\n"
                "time.sleep(60)\n"
            ),
        ],
    )
    pending_req = JobRequest(
        job_type="command",
        input_path="",
        agent_id="waiting-lane",
        route_identity="fixture/waiting",
        command_cwd=str(tmp_path),
        command_argv=[
            sys.executable,
            "-c",
            f"import pathlib; pathlib.Path({str(next_started)!r}).touch()",
        ],
    )
    queue.submit(running_req)
    queue.submit(pending_req)

    worker = threading.Thread(target=queue.run_one, args=({},))
    worker.start()
    try:
        _wait_until(
            lambda: started.exists()
            and (queue.get_job(running_req.job_id) is not None)
            and (queue.get_job(running_req.job_id).child_process_group is not None)
        )
        state = queue.get_job(running_req.job_id)
        assert state is not None and state.child_process_group is not None
        pgid = state.child_process_group

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "gpu_queue.cli",
                "--queue-dir",
                str(queue.queue_dir),
                "panic",
                "--owner",
                "operator-emergency",
                "--reason",
                "WindowServer lag emergency stop",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        report = json.loads(proc.stdout)
        assert report["schema"] == "gpu-greenroom.panic-receipt.v1"
        assert report["paused"] is True
        assert report["owner"] == "operator-emergency"
        assert len(report["terminated_jobs"]) == 1
        assert report["terminated_jobs"][0]["job_id"] == running_req.job_id
        assert report["terminated_jobs"][0]["signal"] == int(signal.SIGKILL)
        assert Path(report["receipt_path"]).is_file()

        worker.join(timeout=5)
        assert not worker.is_alive()
        assert queue._process_group_alive(pgid) is False
        assert queue.is_paused() is True
        assert queue.run_one({}) is False
        assert not next_started.exists()
        assert queue.get_job(pending_req.job_id).status == JobStatus.PENDING
    finally:
        if worker.is_alive():
            state = queue.get_job(running_req.job_id)
            if state and state.child_process_group:
                try:
                    os.killpg(state.child_process_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            worker.join(timeout=5)
