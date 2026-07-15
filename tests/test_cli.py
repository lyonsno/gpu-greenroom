"""Tests for the GPU Greenroom CLI."""

import json
import os
import select
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def run_cli(*args, queue_dir=None):
    """Run the CLI as a subprocess and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "gpu_queue.cli"]
    if queue_dir:
        cmd.extend(["--queue-dir", str(queue_dir)])
    cmd.extend(args)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    return result.returncode, result.stdout, result.stderr


@pytest.fixture
def queue_dir(tmp_path):
    return tmp_path / "cli_queue"


class TestCLISubmit:
    def test_submit_prints_job_id(self, queue_dir):
        rc, out, _ = run_cli("submit", "trellis2mlx", "/tmp/test.png", "/tmp/out", queue_dir=queue_dir)
        assert rc == 0
        assert "Submitted job" in out

    def test_submit_creates_files(self, queue_dir):
        run_cli("submit", "trellis2mlx", "/tmp/test.png", "/tmp/out", queue_dir=queue_dir)
        pending = queue_dir / "pending"
        assert pending.exists()
        job_dirs = list(pending.iterdir())
        assert len(job_dirs) == 1
        assert (job_dirs[0] / "request.json").exists()
        assert (job_dirs[0] / "status.json").exists()

    def test_submit_with_params(self, queue_dir):
        run_cli("submit", "trellis2mlx", "/tmp/test.png", "/tmp/out",
                "-p", "seed=123", "resolution=512", queue_dir=queue_dir)
        pending = queue_dir / "pending"
        job_dir = list(pending.iterdir())[0]
        req = json.loads((job_dir / "request.json").read_text())
        assert req["params"]["seed"] == "123"
        assert req["params"]["resolution"] == "512"


class TestCLIList:
    def test_list_empty(self, queue_dir):
        rc, out, _ = run_cli("list", queue_dir=queue_dir)
        assert rc == 0
        assert "No jobs" in out

    def test_list_shows_submitted(self, queue_dir):
        run_cli("submit", "trellis2mlx", "/tmp/test.png", "/tmp/out", queue_dir=queue_dir)
        rc, out, _ = run_cli("list", queue_dir=queue_dir)
        assert rc == 0
        assert "pending" in out
        assert "trellis2mlx" in out


class TestCLIStatus:
    def test_status_nonexistent(self, queue_dir):
        rc, out, _ = run_cli("status", "nonexistent", queue_dir=queue_dir)
        assert rc == 1
        assert "not found" in out


class TestCLICancel:
    def test_cancel_pending(self, queue_dir):
        run_cli("submit", "trellis2mlx", "/tmp/test.png", "/tmp/out", queue_dir=queue_dir)
        job_dir = list((queue_dir / "pending").iterdir())[0]
        job_id = job_dir.name
        rc, out, _ = run_cli("cancel", job_id, queue_dir=queue_dir)
        assert rc == 0
        assert "Cancelled" in out

    def test_cancel_nonexistent(self, queue_dir):
        rc, out, _ = run_cli("cancel", "nonexistent", queue_dir=queue_dir)
        assert rc == 1


class TestCLIPauseResume:
    def test_pause_cli(self, queue_dir):
        rc, out, _ = run_cli("pause", queue_dir=queue_dir)
        assert rc == 0
        assert "paused" in out.lower()
        assert (queue_dir / "paused").exists()

    def test_resume_cli(self, queue_dir):
        run_cli("pause", queue_dir=queue_dir)
        rc, out, _ = run_cli("resume", queue_dir=queue_dir)
        assert rc == 0
        assert "resumed" in out.lower()
        assert not (queue_dir / "paused").exists()

    def test_resume_when_not_paused(self, queue_dir):
        rc, out, _ = run_cli("resume", queue_dir=queue_dir)
        assert rc == 0


class TestCLISubmitDurableOutput:
    def test_submit_without_output_dir(self, queue_dir):
        """Submit without output_dir auto-assigns durable path."""
        rc, out, _ = run_cli("submit", "trellis2mlx", "/tmp/test.png", queue_dir=queue_dir)
        assert rc == 0
        assert "auto-assigned durable" in out
        # Job should exist in pending with an output under queue_dir/outputs/
        pending = queue_dir / "pending"
        job_dir = list(pending.iterdir())[0]
        req = json.loads((job_dir / "request.json").read_text())
        assert str(queue_dir / "outputs") in req["output_dir"]


class TestLoadJobTypes:
    def test_loads_custom_types(self, queue_dir):
        """_load_job_types merges custom config with defaults."""
        from gpu_queue.cli import _load_job_types
        queue_dir.mkdir(parents=True, exist_ok=True)
        config = queue_dir / "job_types.json"
        config.write_text(json.dumps({"custom_type": {"cmd": ["echo", "hi"]}}))
        types = _load_job_types(str(queue_dir))
        assert "custom_type" in types
        assert "trellis2mlx" in types  # default preserved

    def test_hot_reload_picks_up_changes(self, queue_dir):
        """Calling _load_job_types again after file change returns new types."""
        from gpu_queue.cli import _load_job_types
        queue_dir.mkdir(parents=True, exist_ok=True)
        config = queue_dir / "job_types.json"
        config.write_text(json.dumps({"v1": {"cmd": ["echo", "v1"]}}))
        types1 = _load_job_types(str(queue_dir))
        assert "v1" in types1
        assert "v2" not in types1

        config.write_text(json.dumps({"v2": {"cmd": ["echo", "v2"]}}))
        types2 = _load_job_types(str(queue_dir))
        assert "v2" in types2

    def test_malformed_json_falls_back(self, queue_dir):
        """Malformed job_types.json doesn't crash, falls back to defaults."""
        from gpu_queue.cli import _load_job_types
        queue_dir.mkdir(parents=True, exist_ok=True)
        config = queue_dir / "job_types.json"
        config.write_text("{broken json")
        types = _load_job_types(str(queue_dir))
        assert "trellis2mlx" in types  # defaults survived


class TestCLIRecover:
    def test_recover_no_stale(self, queue_dir):
        rc, out, _ = run_cli("recover", queue_dir=queue_dir)
        assert rc == 0
        assert "No stale" in out


def _read_json_line(proc: subprocess.Popen, timeout: float = 3.0) -> dict:
    assert proc.stdout is not None
    readable, _, _ = select.select([proc.stdout], [], [], timeout)
    assert readable, "timed out waiting for lease event"
    line = proc.stdout.readline()
    assert line, f"lease process exited before event: {proc.poll()}"
    return json.loads(line)


def _lease_process(queue_dir: Path, lease_id: str) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gpu_queue.cli",
            "--queue-dir",
            str(queue_dir),
            "lease",
            "acquire",
            "--lease-id",
            lease_id,
            "--holder",
            "spoke",
            "--purpose",
            "final-asr",
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


class TestCLIInteractiveLease:
    def test_poll_interval_must_be_positive(self, queue_dir):
        rc, _, err = run_cli(
            "lease",
            "acquire",
            "--lease-id",
            "bad-poll",
            "--holder",
            "spoke",
            "--purpose",
            "final-asr",
            "--poll-seconds",
            "0",
            queue_dir=queue_dir,
        )

        assert rc == 2
        assert "positive" in err

    def test_preclosed_stdin_never_claims_effective(self, queue_dir):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "gpu_queue.cli",
                "--queue-dir",
                str(queue_dir),
                "lease",
                "acquire",
                "--lease-id",
                "spoke-cli-preclosed",
                "--holder",
                "spoke",
                "--purpose",
                "final-asr",
            ],
            cwd=str(Path(__file__).resolve().parent.parent),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=3,
        )

        assert result.returncode == 0
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert [event["state"] for event in events] == [
            "requested",
            "released-unacquired",
        ]
        assert all(event["effective_at"] is None for event in events)

        receipt_path = (
            queue_dir / "leases" / "spoke-cli-preclosed" / "receipt.json"
        )
        receipt = json.loads(receipt_path.read_text())
        assert receipt["state"] == "released-unacquired"
        assert receipt["last_trustworthy_event"] == (
            "request-cancelled-before-acquisition"
        )

    def test_cancellation_after_lock_claim_prevents_effective_publication(
        self, queue_dir
    ):
        from gpu_queue.cli import _claim_lease_once
        from gpu_queue.lease import InteractiveLease

        lease = InteractiveLease(
            queue_dir,
            lease_id="spoke-cli-cancel-after-claim",
            holder="spoke",
            purpose="final-asr",
        )
        lease.request()
        cancellation_checks = iter((False, True))

        outcome = _claim_lease_once(
            lease,
            cancelled=lambda: next(cancellation_checks),
        )

        assert outcome == "cancelled"
        receipt = json.loads(lease.receipt_path.read_text())
        assert receipt["state"] == "released-unacquired"
        assert receipt["effective_at"] is None
        assert receipt["last_trustworthy_event"] == (
            "gpu-lock-released-before-effective-publication"
        )

    def test_holder_emits_requested_then_effective_and_releases_on_stdin_eof(
        self, queue_dir
    ):
        proc = _lease_process(queue_dir, "spoke-cli-1")
        try:
            requested = _read_json_line(proc)
            effective = _read_json_line(proc)
            assert requested["state"] == "requested"
            assert requested["effective_at"] is None
            assert effective["state"] == "effective"
            assert effective["effective_at"] is not None

            assert proc.stdin is not None
            proc.stdin.close()
            assert proc.wait(timeout=3) == 0

            receipt_path = queue_dir / "leases" / "spoke-cli-1" / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            assert receipt["state"] == "released"
            assert receipt["released_at"] is not None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=3)

    def test_blocked_holder_does_not_emit_effective(self, queue_dir):
        from gpu_queue.queue import GPUQueue
        import fcntl

        queue = GPUQueue(queue_dir)
        blocker = open(queue.lock_path, "w")
        fcntl.flock(blocker, fcntl.LOCK_EX)
        proc = _lease_process(queue_dir, "spoke-cli-blocked")
        try:
            requested = _read_json_line(proc)
            assert requested["state"] == "requested"
            assert proc.stdout is not None
            readable, _, _ = select.select([proc.stdout], [], [], 0.2)
            assert not readable

            receipt_path = queue_dir / "leases" / "spoke-cli-blocked" / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            assert receipt["state"] == "requested"
            assert receipt["effective_at"] is None
        finally:
            proc.kill()
            proc.wait(timeout=3)
            fcntl.flock(blocker, fcntl.LOCK_UN)
            blocker.close()

    def test_sigkill_releases_kernel_lock_without_rewriting_stale_receipt(
        self, queue_dir
    ):
        proc = _lease_process(queue_dir, "spoke-cli-killed")
        _read_json_line(proc)
        effective = _read_json_line(proc)
        assert effective["state"] == "effective"

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=3)

        stale_path = queue_dir / "leases" / "spoke-cli-killed" / "receipt.json"
        stale = json.loads(stale_path.read_text())
        assert stale["state"] == "effective"
        assert stale["current_authority"] == "requires-live-holder-process-and-flock"

        successor = _lease_process(queue_dir, "spoke-cli-successor")
        try:
            assert _read_json_line(successor)["state"] == "requested"
            assert _read_json_line(successor)["state"] == "effective"
            assert successor.stdin is not None
            successor.stdin.close()
            assert successor.wait(timeout=3) == 0
        finally:
            if successor.poll() is None:
                successor.kill()
                successor.wait(timeout=3)
