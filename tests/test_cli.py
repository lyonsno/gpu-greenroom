"""Tests for the GPU Greenroom CLI."""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from gpu_queue.models import JobRequest, JobState, JobStatus
from gpu_queue.queue import GPUQueue


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

    def test_submit_persists_complete_cooperation_contract(self, queue_dir):
        rc, out, err = run_cli(
            "submit", "trellis2mlx", "/tmp/test.png", "/tmp/out",
            "--route-identity", "sjb/grid48-curriculum-r1",
            "--yields-to-waiters",
            "--expected-handoff-seconds", "60",
            "--generation", "3",
            queue_dir=queue_dir,
        )

        assert rc == 0, err
        job_dir = next((queue_dir / "pending").iterdir())
        request = json.loads((job_dir / "request.json").read_text())
        state = json.loads((job_dir / "status.json").read_text())
        assert request["route_identity"] == "sjb/grid48-curriculum-r1"
        assert request["yields_to_waiters"] is True
        assert request["expected_handoff_seconds"] == 60.0
        assert request["generation"] == 3
        assert state["route_identity"] == request["route_identity"]
        assert state["yields_to_waiters"] is True
        assert "Route: sjb/grid48-curriculum-r1" in out
        assert "Cooperation: yields to waiters within <=60s (generation 3)" in out

    @pytest.mark.parametrize(
        "args",
        [
            ("--yields-to-waiters",),
            ("--expected-handoff-seconds", "60"),
            ("--generation", "2"),
            ("--yields-to-waiters", "--route-identity", "sjb/grid48"),
        ],
    )
    def test_submit_rejects_partial_cooperation_contract(self, queue_dir, args):
        rc, _, err = run_cli(
            "submit", "trellis2mlx", "/tmp/test.png", "/tmp/out", *args,
            queue_dir=queue_dir,
        )

        assert rc != 0
        assert "cooperation metadata" in err.lower()
        assert not list((queue_dir / "pending").glob("*")) if (queue_dir / "pending").exists() else True

    def test_submit_notices_current_cooperative_job_and_writes_receipt(self, queue_dir):
        queue = GPUQueue(queue_dir)
        running = JobRequest(
            job_type="command",
            input_path="/tmp/grid48.json",
            route_identity="sjb/grid48-curriculum-r1",
            yields_to_waiters=True,
            expected_handoff_seconds=60.0,
            generation=3,
        )
        pending_dir = queue.submit(running)
        running_dir = queue.queue_dir / "running" / running.job_id
        pending_dir.rename(running_dir)
        state = JobState.from_json((running_dir / "status.json").read_text())
        state.status = JobStatus.RUNNING
        state.started_at = time.time() - 2670.0
        (running_dir / "status.json").write_text(state.to_json())

        rc, out, err = run_cli(
            "submit", "trellis2mlx", "/tmp/waiter.png", "/tmp/waiter-out",
            queue_dir=queue_dir,
        )

        assert rc == 0, err
        assert "Current running job sjb/grid48-curriculum-r1 is yield-aware" in out
        assert "expect the GPU within <=60s" in out
        submitted = [path for path in (queue_dir / "pending").iterdir() if path.name != running.job_id]
        assert len(submitted) == 1
        receipt = json.loads((submitted[0] / "submission_receipt.json").read_text())
        assert receipt["schema"] == "gpu-greenroom.submission-receipt.v1"
        assert receipt["submitted_job"]["yields_to_waiters"] is False
        assert receipt["running_job"]["job_id"] == running.job_id
        assert receipt["running_job"]["generation"] == 3
        assert receipt["notice"] == "current running job is yield-aware; expect the GPU within <=60s"

    def test_submit_warns_when_running_job_cannot_be_verified(self, queue_dir):
        running_dir = queue_dir / "running" / "unreadable-running"
        running_dir.mkdir(parents=True)
        (running_dir / "status.json").write_text("not json")

        rc, out, err = run_cli(
            "submit", "trellis2mlx", "/tmp/waiter.png", "/tmp/waiter-out",
            queue_dir=queue_dir,
        )

        assert rc == 0
        assert "yield-aware" not in out
        assert "could not be verified" in err
        submitted = next((queue_dir / "pending").iterdir())
        receipt = json.loads((submitted / "submission_receipt.json").read_text())
        assert receipt["running_observation"] == "unverified"
        assert "could not read running job" in receipt["running_observation_error"]
        assert receipt["notice"] is None

    def test_submit_warns_instead_of_promising_from_partial_running_metadata(self, queue_dir):
        running_dir = queue_dir / "running" / "partial-cooperation"
        running_dir.mkdir(parents=True)
        state = JobState(
            job_id="partial-cooperation",
            status=JobStatus.RUNNING,
            job_type="command",
            input_path="/tmp/grid48.json",
            output_dir="/tmp/grid48-out",
            route_identity="sjb/grid48-curriculum-r1",
            yields_to_waiters=True,
            expected_handoff_seconds=60,
        )
        (running_dir / "status.json").write_text(state.to_json())

        rc, out, err = run_cli(
            "submit", "trellis2mlx", "/tmp/waiter.png", "/tmp/waiter-out",
            queue_dir=queue_dir,
        )

        assert rc == 0
        assert "yield-aware" not in out
        assert "invalid cooperation metadata" in err
        assert "generation" in err
        submitted = next((queue_dir / "pending").iterdir())
        receipt = json.loads((submitted / "submission_receipt.json").read_text())
        assert receipt["running_observation"] == "observed"
        assert receipt["running_job"]["cooperation_valid"] is False
        assert receipt["notice"] is None

    def test_submit_warns_instead_of_promising_from_stale_running_status(self, queue_dir):
        running_dir = queue_dir / "running" / "stale-running"
        running_dir.mkdir(parents=True)
        state = JobState(
            job_id="stale-running",
            status=JobStatus.DONE,
            job_type="command",
            input_path="/tmp/grid48.json",
            output_dir="/tmp/grid48-out",
            route_identity="sjb/grid48-curriculum-r1",
            yields_to_waiters=True,
            expected_handoff_seconds=60,
            generation=3,
        )
        (running_dir / "status.json").write_text(state.to_json())

        rc, out, err = run_cli(
            "submit", "trellis2mlx", "/tmp/waiter.png", "/tmp/waiter-out",
            queue_dir=queue_dir,
        )

        assert rc == 0
        assert "yield-aware" not in out
        assert "stale status done" in err
        submitted = next((queue_dir / "pending").iterdir())
        receipt = json.loads((submitted / "submission_receipt.json").read_text())
        assert receipt["running_observation"] == "unverified"
        assert receipt["notice"] is None


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

    def test_list_renders_route_yield_promise_and_generation(self, queue_dir):
        queue = GPUQueue(queue_dir)
        request = JobRequest(
            job_type="command",
            input_path="/tmp/grid48.json",
            route_identity="sjb/grid48-curriculum-r1",
            yields_to_waiters=True,
            expected_handoff_seconds=60.0,
            generation=3,
        )
        pending_dir = queue.submit(request)
        running_dir = queue.queue_dir / "running" / request.job_id
        pending_dir.rename(running_dir)
        state = JobState.from_json((running_dir / "status.json").read_text())
        state.status = JobStatus.RUNNING
        state.started_at = time.time() - 2670.0
        (running_dir / "status.json").write_text(state.to_json())

        rc, out, err = run_cli("list", queue_dir=queue_dir)

        assert rc == 0, err
        assert "running" in out
        assert "sjb/grid48-curriculum-r1" in out
        assert "[YIELDS <=60s]" in out
        assert "gen 3" in out

    def test_list_marks_malformed_persisted_cooperation_instead_of_crashing(self, queue_dir):
        queue = GPUQueue(queue_dir)
        request = JobRequest(job_type="command", input_path="/tmp/grid48.json")
        pending_dir = queue.submit(request)
        status_path = pending_dir / "status.json"
        state = json.loads(status_path.read_text())
        state.update({
            "route_identity": "sjb/grid48-curriculum-r1",
            "yields_to_waiters": True,
            "expected_handoff_seconds": "sixty",
            "generation": "third",
        })
        status_path.write_text(json.dumps(state))

        rc, out, err = run_cli("list", queue_dir=queue_dir)

        assert rc == 0, err
        assert "[INVALID COOPERATION METADATA]" in out
        assert "[YIELDS" not in out


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
