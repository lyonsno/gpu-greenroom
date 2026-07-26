"""One-time queue registration and aggregate execution-start controls."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from gpu_queue.models import JobRequest, JobStatus
from gpu_queue.queue import GPUQueue


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "gpu_queue.cli", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )


def register_queue(registry_path, name, queue_dir):
    return run_cli(
        "--registry",
        registry_path,
        "queues",
        "register",
        name,
        "--queue-dir",
        queue_dir,
        "--contention-class",
        "apple-unified-accelerator",
    )


def test_registry_is_idempotent_and_keeps_native_queue_state_authoritative(tmp_path):
    registry_path = tmp_path / "queues.json"
    queue_dir = tmp_path / "queue-a"

    first = register_queue(registry_path, "science", queue_dir)
    second = register_queue(registry_path, "science", queue_dir)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    stored = json.loads(registry_path.read_text())
    assert stored["schema"] == "gpu-greenroom.queue-registry.v1"
    assert len(stored["queues"]) == 1
    assert set(stored["queues"][0]) == {
        "name", "queue_dir", "contention_class", "adapter"
    }
    queue = GPUQueue(queue_dir)
    queue.submit(JobRequest(job_type="echo", input_path="fixture"))
    result = run_cli("--registry", registry_path, "queues", "status")
    assert result.returncode == 0, result.stderr
    [status] = json.loads(result.stdout)["queues"]
    assert status["name"] == "science"
    assert status["source"] == str(queue_dir.resolve())
    assert status["pending"] == 1
    assert status["running"] == 0
    assert status["paused"] is False


def test_class_pause_only_blocks_start_and_resume_reopens_it(tmp_path):
    registry_path = tmp_path / "queues.json"
    queue_a = GPUQueue(tmp_path / "queue-a")
    queue_b = GPUQueue(tmp_path / "queue-b")
    assert register_queue(registry_path, "science", queue_a.queue_dir).returncode == 0
    assert register_queue(registry_path, "renders", queue_b.queue_dir).returncode == 0

    paused = run_cli(
        "--registry",
        registry_path,
        "queues",
        "pause",
        "--contention-class",
        "apple-unified-accelerator",
    )
    assert paused.returncode == 0, paused.stderr
    pause_receipt = json.loads(paused.stdout)
    assert {row["name"] for row in pause_receipt["queues"]} == {"science", "renders"}
    assert all(row["paused"] for row in pause_receipt["queues"])
    request = JobRequest(job_type="echo", input_path="submitted-during-pause")
    queue_a.submit(request)
    assert queue_a.get_job(request.job_id).status == JobStatus.PENDING
    assert queue_a.run_one({"echo": ["echo", "ok"]}) is False

    resumed = run_cli(
        "--registry",
        registry_path,
        "queues",
        "resume",
        "--contention-class",
        "apple-unified-accelerator",
    )
    assert resumed.returncode == 0, resumed.stderr
    resume_receipt = json.loads(resumed.stdout)
    assert all(not row["paused"] for row in resume_receipt["queues"])
    assert queue_a.run_one({"echo": ["echo", "ok"]}) is True
    assert queue_a.get_job(request.job_id).status == JobStatus.DONE


def test_class_pause_preflights_all_queues_before_mutating_any_marker(tmp_path):
    registry_path = tmp_path / "queues.json"
    queue_a = GPUQueue(tmp_path / "queue-a")
    missing_queue = tmp_path / "missing-queue"
    assert register_queue(registry_path, "science", queue_a.queue_dir).returncode == 0
    assert register_queue(registry_path, "renders", missing_queue).returncode == 0
    shutil.rmtree(missing_queue)

    paused = run_cli(
        "--registry",
        registry_path,
        "queues",
        "pause",
        "--contention-class",
        "apple-unified-accelerator",
    )

    assert paused.returncode != 0
    assert not queue_a.is_paused()


def test_queue_registry_cli_returns_aggregate_json(tmp_path):
    registry_path = tmp_path / "queues.json"
    queue_dir = tmp_path / "queue"
    repo_root = Path(__file__).resolve().parent.parent

    register = register_queue(registry_path, "science", queue_dir)
    assert register.returncode == 0, register.stderr
    registered = json.loads(register.stdout)
    assert registered["name"] == "science"

    status = run_cli("--registry", registry_path, "queues", "status")
    assert status.returncode == 0, status.stderr
    report = json.loads(status.stdout)
    assert report["schema"] == "gpu-greenroom.aggregate-status.v1"
    assert report["registry_path"] == str(registry_path.resolve())
    assert report["queues"][0]["source"] == str(queue_dir.resolve())
