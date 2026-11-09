"""Source-owned checkpoint declarations require no per-caller switch."""

import json
from pathlib import Path
import sys
import pytest

from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue


def producer(tmp_path):
    root = tmp_path / "producer"
    root.mkdir()
    (root / "generate.py").write_text("print('producer')\n")
    (root / ".gpu-greenroom-producer.json").write_text(json.dumps({
        "schema": "gpu-greenroom.producer-checkpoints.v1",
        "entrypoints": ["generate.py"],
    }))
    return root


def test_ordinary_structured_submission_inherits_source_checkpoint_declaration(tmp_path):
    root = producer(tmp_path)
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="command", input_path="", agent_id="ordinary-caller",
                         repo_root=str(root), command_cwd=str(root),
                         command_argv=[sys.executable, "-u", "generate.py"], route_identity="producer/default")
    job = queue.submit(request)
    saved = json.loads((job / "request.json").read_text())
    assert saved["cooperative_checkpoint"] is True, "ordinary caller lost the producer's checkpoint capability"
    assert "checkpoint-continuation.v1" in saved["required_worker_capabilities"]
    assert saved["params"]["_producer_checkpoint"]["entrypoint"]["path"] == str(root / "generate.py")


def test_other_entrypoints_do_not_inherit_a_neighboring_declaration(tmp_path):
    root = producer(tmp_path)
    (root / "opaque.py").write_text("pass\n")
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="command", input_path="", agent_id="ordinary-caller",
                         repo_root=str(root), command_cwd=str(root),
                         command_argv=[sys.executable, "opaque.py"], route_identity="producer/opaque")
    job = queue.submit(request)
    assert json.loads((job / "request.json").read_text())["cooperative_checkpoint"] is False


def test_ordinary_typed_submission_resolves_the_declared_source_route(tmp_path, capsys, monkeypatch):
    from gpu_queue.cli import main

    root = producer(tmp_path)
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "job_types.json").write_text(json.dumps({"example-producer": {
        "cmd": [sys.executable, "-u", "generate.py", "--output", "{output_dir}/mesh.glb", "--seed", "{seed}"],
        "cwd": str(root), "env": {"PYTHONPATH": "."}, "defaults": {"seed": "17"},
    }}))
    monkeypatch.setattr(sys, "argv", ["gpu-greenroom", "--queue-dir", str(queue), "submit",
                                     "example-producer", "input.png",
                                     "--agent-id", "ordinary-caller"])
    main()
    job = next((queue / "pending").iterdir())
    saved = json.loads((job / "request.json").read_text())
    assert saved["cooperative_checkpoint"] is True, "typed route dropped the source declaration"
    assert saved["repo_root"] == str(root)
    assert saved["command_argv"][-1] == "17"
    assert saved["command_argv"][4] == str(Path(saved["output_dir"]) / "mesh.glb")


def test_changed_declared_entrypoint_fails_before_child_execution(tmp_path):
    root = producer(tmp_path)
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="command", input_path="", agent_id="ordinary-caller",
                         repo_root=str(root), command_cwd=str(root),
                         command_argv=[sys.executable, "generate.py"], route_identity="producer/default")
    queue.submit(request)
    marker = tmp_path / "must-not-execute"
    (root / "generate.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    assert queue.run_one({}) is True
    state = queue.get_job(request.job_id)
    assert state.status.value == "failed"
    assert "changed after submission" in state.error_message
    assert not marker.exists()


def test_automatic_checkpoint_requires_an_owner(tmp_path):
    root = producer(tmp_path)
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="command", input_path="", repo_root=str(root), command_cwd=str(root),
                         command_argv=[sys.executable, "generate.py"], route_identity="producer/default")
    with pytest.raises(ValueError, match="explicit job owner"):
        queue.submit(request)
    assert not list((queue.queue_dir / "pending").iterdir())
