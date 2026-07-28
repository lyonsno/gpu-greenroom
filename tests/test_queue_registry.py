"""One-time queue registration and aggregate execution-start controls."""

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from gpu_queue.cli import cmd_queues_pause
from gpu_queue.control import QueueControlError, QueueRegistry
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
        "--owner",
        "stage-b-test",
        "--epoch",
        "class-pause-1",
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
        "--owner",
        "stage-b-test",
        "--epoch",
        pause_receipt["request"]["epoch"],
    )
    assert resumed.returncode == 0, resumed.stderr
    resume_receipt = json.loads(resumed.stdout)
    assert all(not row["paused"] for row in resume_receipt["queues"])
    assert queue_a.run_one({"echo": ["echo", "ok"]}) is True
    assert queue_a.get_job(request.job_id).status == JobStatus.DONE


def test_pause_drains_current_job_and_resume_consumes_preserved_queue(tmp_path):
    registry = QueueRegistry(tmp_path / "queues.json")
    queue = GPUQueue(tmp_path / "queue")
    registry.register(
        name="science",
        queue_dir=queue.queue_dir,
        contention_class="apple-unified-accelerator",
    )
    current = JobRequest(job_type="slow", input_path="current")
    queued = JobRequest(job_type="echo", input_path="queued-during-pause")
    queue.submit(current)

    worker = threading.Thread(
        target=queue.run_one,
        args=({"slow": ["sleep", "0.2"]},),
    )
    worker.start()
    deadline = time.monotonic() + 2
    while queue.get_job(current.job_id).status != JobStatus.RUNNING:
        assert time.monotonic() < deadline
        time.sleep(0.005)

    pause_receipt = registry.pause(
        "apple-unified-accelerator",
        owner="stage-b-test",
        epoch="drain-current-1",
    )
    queue.submit(queued)
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert queue.get_job(current.job_id).status == JobStatus.DONE
    assert queue.get_job(queued.job_id).status == JobStatus.PENDING
    assert queue.run_one({"echo": ["echo", "ok"]}) is False

    registry.resume(
        "apple-unified-accelerator",
        owner="stage-b-test",
        epoch=pause_receipt["request"]["epoch"],
    )
    assert queue.run_one({"echo": ["echo", "ok"]}) is True
    assert queue.get_job(queued.job_id).status == JobStatus.DONE


def test_pause_cutover_linearizes_against_pending_to_running_transition(
    tmp_path, monkeypatch
):
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="echo", input_path="before-cutover")
    queue.submit(request)
    move_entered = threading.Event()
    release_move = threading.Event()
    pause_returned = threading.Event()
    original_move = queue._move_job

    def block_pending_to_running(job_dir, destination):
        if destination == "running":
            move_entered.set()
            assert release_move.wait(2)
        return original_move(job_dir, destination)

    monkeypatch.setattr(queue, "_move_job", block_pending_to_running)
    worker = threading.Thread(
        target=lambda: queue.run_one({"echo": ["echo", "ok"]}),
        daemon=True,
    )
    worker.start()
    assert move_entered.wait(2)

    pauser = threading.Thread(
        target=lambda: (queue.pause(), pause_returned.set()),
        daemon=True,
    )
    pauser.start()

    assert not pause_returned.wait(0.1), (
        "pause returned while a worker could still cross pending-to-running"
    )
    release_move.set()
    assert pause_returned.wait(2)
    worker.join(2)
    pauser.join(2)
    assert not worker.is_alive()
    assert not pauser.is_alive()
    assert queue.get_job(request.job_id).status == JobStatus.DONE

    later = JobRequest(job_type="echo", input_path="after-cutover")
    queue.submit(later)
    assert queue.run_one({"echo": ["echo", "ok"]}) is False
    assert queue.get_job(later.job_id).status == JobStatus.PENDING


def test_class_pause_records_owner_epoch_and_per_queue_acknowledgements(tmp_path):
    registry = QueueRegistry(tmp_path / "queues.json")
    queue_a = GPUQueue(tmp_path / "queue-a")
    queue_b = GPUQueue(tmp_path / "queue-b")
    registry.register(
        name="science",
        queue_dir=queue_a.queue_dir,
        contention_class="apple-unified-accelerator",
    )
    registry.register(
        name="renders",
        queue_dir=queue_b.queue_dir,
        contention_class="apple-unified-accelerator",
    )

    report = registry.pause(
        "apple-unified-accelerator",
        owner="handy-handy-man",
        epoch="stage-b-epoch-1",
    )

    assert report["request"] == {
        "action": "pause",
        "owner": "handy-handy-man",
        "epoch": "stage-b-epoch-1",
        "contention_class": "apple-unified-accelerator",
        "queues": [
            {
                "name": "renders",
                "queue_dir": str(queue_b.queue_dir.resolve()),
                "adapter": "greenroom-v1",
            },
            {
                "name": "science",
                "queue_dir": str(queue_a.queue_dir.resolve()),
                "adapter": "greenroom-v1",
            },
        ],
    }
    assert report["effective"]["fully_effective"] is True
    acknowledgements = report["effective"]["acknowledgements"]
    assert [row["name"] for row in acknowledgements] == ["renders", "science"]
    assert all(row["action"] == "pause" for row in acknowledgements)
    assert all(row["owner"] == "handy-handy-man" for row in acknowledgements)
    assert all(row["epoch"] == "stage-b-epoch-1" for row in acknowledgements)
    assert all(row["effective_paused"] is True for row in acknowledgements)
    assert all(row["adapter"] == "greenroom-v1" for row in acknowledgements)
    assert all(row["acknowledged_at"] >= report["requested_at"] for row in acknowledgements)
    assert queue_a.pause_state()["epoch"] == "stage-b-epoch-1"
    assert queue_b.pause_state()["owner"] == "handy-handy-man"


def test_resume_rejects_wrong_epoch_and_recovers_after_registry_restart(tmp_path):
    registry_path = tmp_path / "queues.json"
    registry = QueueRegistry(registry_path)
    queue = GPUQueue(tmp_path / "queue")
    registry.register(
        name="science",
        queue_dir=queue.queue_dir,
        contention_class="apple-unified-accelerator",
    )
    registry.pause(
        "apple-unified-accelerator",
        owner="first-controller",
        epoch="durable-pause-7",
    )

    restarted = QueueRegistry(registry_path)
    [status] = restarted.status("apple-unified-accelerator")
    assert status["pause_state"]["owner"] == "first-controller"
    assert status["pause_state"]["epoch"] == "durable-pause-7"

    with pytest.raises(QueueControlError) as raised:
        restarted.resume(
            "apple-unified-accelerator",
            owner="recovery-controller",
            epoch="stale-pause-6",
        )

    assert raised.value.report["status"] == "failed"
    assert raised.value.report["failure_phase"] == "pause-epoch-mismatch"
    assert raised.value.report["effective"]["fully_effective"] is False
    assert queue.is_paused() is True
    assert queue.pause_state()["epoch"] == "durable-pause-7"

    resumed = restarted.resume(
        "apple-unified-accelerator",
        owner="recovery-controller",
        epoch="durable-pause-7",
    )

    assert resumed["status"] == "succeeded"
    assert resumed["request"]["owner"] == "recovery-controller"
    assert resumed["request"]["epoch"] == "durable-pause-7"
    [ack] = resumed["effective"]["acknowledgements"]
    assert ack["action"] == "resume"
    assert ack["previous_pause"]["owner"] == "first-controller"
    assert ack["previous_pause"]["epoch"] == "durable-pause-7"
    assert ack["effective_paused"] is False
    assert queue.is_paused() is False
    assert queue.pause_state() is None


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
        "--owner",
        "stage-b-test",
        "--epoch",
        "preflight-pause-1",
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


def test_partial_marker_failure_writes_durable_per_queue_failure_receipt(
    tmp_path, monkeypatch
):
    registry = QueueRegistry(tmp_path / "queues.json")
    queue_a = GPUQueue(tmp_path / "queue-a")
    queue_b = GPUQueue(tmp_path / "queue-b")
    registry.register(
        name="a-first",
        queue_dir=queue_a.queue_dir,
        contention_class="apple-unified-accelerator",
    )
    registry.register(
        name="b-second",
        queue_dir=queue_b.queue_dir,
        contention_class="apple-unified-accelerator",
    )
    original_pause = GPUQueue.pause

    def fail_second(queue, **kwargs):
        if queue.queue_dir.resolve() == queue_b.queue_dir.resolve():
            raise PermissionError("fixture denies second pause marker")
        return original_pause(queue, **kwargs)

    monkeypatch.setattr(GPUQueue, "pause", fail_second)

    with pytest.raises(Exception) as raised:
        registry.pause(
            "apple-unified-accelerator",
            owner="stage-b-test",
            epoch="partial-pause-1",
        )

    report = raised.value.report
    assert report["status"] == "failed"
    assert report["failure_phase"] == "native-marker-mutation"
    assert report["rollback_attempted"] is False
    assert report["receipt_persisted"] is True
    assert report["request"]["owner"] == "stage-b-test"
    assert report["request"]["epoch"] == "partial-pause-1"
    assert [row["name"] for row in report["request"]["queues"]] == [
        "a-first",
        "b-second",
    ]
    assert report["effective"]["fully_effective"] is False
    assert [
        (row["name"], row["effective_paused"])
        for row in report["effective"]["acknowledgements"]
    ] == [
        ("a-first", True),
        ("b-second", False),
    ]
    assert Path(report["receipt_path"]).exists()
    assert json.loads(Path(report["receipt_path"]).read_text()) == report
    assert [
        (row["name"], row["mutation"], row["observed_paused"])
        for row in report["mutations"]
    ] == [
        ("a-first", "succeeded", True),
        ("b-second", "failed", False),
    ]
    assert queue_a.is_paused() is True
    assert queue_b.is_paused() is False


def test_malformed_pause_state_preserves_partial_failure_receipt(tmp_path):
    registry = QueueRegistry(tmp_path / "queues.json")
    queue_a = GPUQueue(tmp_path / "queue-a")
    queue_b = GPUQueue(tmp_path / "queue-b")
    registry.register(
        name="a-science",
        queue_dir=queue_a.queue_dir,
        contention_class="apple-unified-accelerator",
    )
    registry.register(
        name="b-renders",
        queue_dir=queue_b.queue_dir,
        contention_class="apple-unified-accelerator",
    )
    queue_b.pause_path.write_text('{"schema": "wrong"}')

    with pytest.raises(QueueControlError) as raised:
        registry.pause(
            "apple-unified-accelerator",
            owner="stage-b-test",
            epoch="malformed-state-1",
        )

    report = raised.value.report
    assert report["status"] == "failed"
    assert report["failure_phase"] == "native-marker-mutation"
    assert report["effective"]["fully_effective"] is False
    assert Path(report["receipt_path"]).exists()
    rows = {row["name"]: row for row in report["queues"]}
    assert rows["a-science"]["pause_state"]["epoch"] == "malformed-state-1"
    assert rows["b-renders"]["paused"] is True
    assert rows["b-renders"]["pause_state"] is None
    assert "unsupported pause state schema" in rows["b-renders"]["error"]


def test_second_queue_construction_failure_after_first_mutation_is_receipted(
    tmp_path, monkeypatch
):
    registry = QueueRegistry(tmp_path / "queues.json")
    queue_a = GPUQueue(tmp_path / "queue-a")
    queue_b = GPUQueue(tmp_path / "queue-b")
    registry.register(
        name="a-first",
        queue_dir=queue_a.queue_dir,
        contention_class="apple-unified-accelerator",
    )
    registry.register(
        name="b-second",
        queue_dir=queue_b.queue_dir,
        contention_class="apple-unified-accelerator",
    )
    native_queue = GPUQueue

    def fail_second_construction(queue_dir):
        if Path(queue_dir).resolve() == queue_b.queue_dir.resolve():
            raise PermissionError("fixture denies second queue construction")
        return native_queue(queue_dir)

    monkeypatch.setattr(
        "gpu_queue.control.GPUQueue",
        fail_second_construction,
    )

    with pytest.raises(Exception) as raised:
        registry.pause(
            "apple-unified-accelerator",
            owner="stage-b-test",
            epoch="partial-pause-1",
        )

    report = raised.value.report
    assert report["status"] == "failed"
    assert report["failure_phase"] == "native-marker-mutation"
    assert report["rollback_attempted"] is False
    assert report["receipt_persisted"] is True
    assert Path(report["receipt_path"]).exists()
    assert [
        (row["name"], row["mutation"], row["observed_paused"])
        for row in report["mutations"]
    ] == [
        ("a-first", "succeeded", True),
        ("b-second", "failed", False),
    ]
    assert queue_a.is_paused() is True
    assert queue_b.is_paused() is False


def test_receipt_write_failure_returns_in_memory_mutation_accounting(
    tmp_path, monkeypatch
):
    registry = QueueRegistry(tmp_path / "queues.json")
    queue = GPUQueue(tmp_path / "queue")
    registry.register(
        name="science",
        queue_dir=queue.queue_dir,
        contention_class="apple-unified-accelerator",
    )
    original_write = registry._write_atomic

    def fail_primary_receipt(path, payload):
        if path.parent == registry.receipts_dir:
            raise PermissionError("fixture denies primary receipt directory")
        return original_write(path, payload)

    monkeypatch.setattr(registry, "_write_atomic", fail_primary_receipt)

    with pytest.raises(Exception) as raised:
        registry.pause("apple-unified-accelerator")

    report = raised.value.report
    assert report["status"] == "failed"
    assert report["failure_phase"] == "receipt-write"
    assert report["rollback_attempted"] is False
    assert report["receipt_persisted"] is False
    assert report["failed_receipt_path"].startswith(str(registry.receipts_dir))
    assert report["mutations"][0]["mutation"] == "succeeded"
    assert report["mutations"][0]["observed_paused"] is True
    assert queue.is_paused() is True


def test_queue_control_error_is_structured_stderr_and_nonzero(monkeypatch, capsys):
    report = {
        "schema": "gpu-greenroom.queue-control-receipt.v1",
        "status": "failed",
        "failure_phase": "receipt-write",
        "receipt_persisted": False,
        "failed_receipt_path": "/durable/queue-control-receipts/failed.json",
        "mutations": [{"name": "science", "mutation": "succeeded"}],
    }

    class FailingRegistry:
        def pause(self, _contention_class, **_kwargs):
            raise QueueControlError(report)

    monkeypatch.setattr("gpu_queue.cli.get_registry", lambda _args: FailingRegistry())
    args = type(
        "Args",
        (),
        {
            "contention_class": "apple-unified-accelerator",
            "registry": "/unused/registry.json",
            "owner": "stage-b-test",
            "epoch": "failure-pause-1",
        },
    )()

    with pytest.raises(SystemExit) as raised:
        cmd_queues_pause(args)

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == report
