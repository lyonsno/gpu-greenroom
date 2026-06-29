"""Tests for the GPU Greenroom CLI."""

import json
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

    def test_list_json_reports_degraded_legacy_rows(self, queue_dir):
        legacy = queue_dir / "failed" / "legacy-provider-route"
        legacy.mkdir(parents=True)
        (legacy / "status.json").write_text(json.dumps({
            "jobId": "legacy-provider-route",
            "jobType": "kaminos.orb-inner-engine.provider-route",
            "status": "failed",
        }))

        rc, out, _ = run_cli("list", "--json", queue_dir=queue_dir)

        assert rc == 0
        payload = json.loads(out)
        assert payload["schema"] == "gpu-greenroom.queue-index.v1"
        assert payload["rows"][0]["job_id"] == "legacy-provider-route"
        assert payload["rows"][0]["status"] == "degraded"


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

    def test_request_checkpoint_pause_cli(self, queue_dir):
        run_cli("submit", "trellis2mlx", "/tmp/test.png", str(queue_dir / "out"), queue_dir=queue_dir)
        job_id = list((queue_dir / "pending").iterdir())[0].name

        rc, out, err = run_cli("request-checkpoint-pause", job_id, queue_dir=queue_dir)

        assert rc == 0, err
        assert "Checkpoint pause requested" in out
        stop_file = queue_dir / "out" / "_control" / "checkpoint-stop"
        request_receipt = queue_dir / "pending" / job_id / "_control" / "checkpoint_pause_request.json"
        assert stop_file.exists()
        assert request_receipt.exists()

        rc, list_out, _ = run_cli("list", "--json", queue_dir=queue_dir)
        payload = json.loads(list_out)
        [row] = payload["rows"]
        assert row["route_job"]["resumability"]["pauseRequested"] is True
        assert row["route_job"]["native"]["checkpoint_pause_request_receipt"] == str(request_receipt)

    def test_request_checkpoint_pause_cli_refuses_non_trellis_job(self, queue_dir):
        run_cli("submit", "echo", "/tmp/test.png", str(queue_dir / "out"), queue_dir=queue_dir)
        job_id = list((queue_dir / "pending").iterdir())[0].name

        rc, out, _ = run_cli("request-checkpoint-pause", job_id, queue_dir=queue_dir)

        assert rc == 1
        assert "does not advertise cooperative checkpoint pause" in out
        assert not (queue_dir / "out" / "_control" / "checkpoint-stop").exists()


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
    def test_default_trellis_job_type_passes_checkpoint_yield_args(self, queue_dir):
        """Native Trellis jobs should expose cooperative checkpoint-yield controls."""
        from gpu_queue.cli import _load_job_types

        types = _load_job_types(str(queue_dir))
        trellis_cmd = types["trellis2mlx"]["cmd"]

        assert "--save-checkpoints" in trellis_cmd
        assert "{checkpoint_dir}" in trellis_cmd
        assert "--checkpoint-stop-file" in trellis_cmd
        assert "{checkpoint_stop_file}" in trellis_cmd

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
