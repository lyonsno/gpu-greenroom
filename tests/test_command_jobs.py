"""Structured arbitrary-command admission and receipt contracts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpu_queue import queue as queue_module
from gpu_queue.cli import _load_job_types, cmd_doctor
from gpu_queue.models import JobRequest, JobStatus
from gpu_queue.queue import GPUQueue


def run_cli(*args, queue_dir, env=None):
    result = subprocess.run(
        [sys.executable, "-m", "gpu_queue.cli", "--queue-dir", str(queue_dir), *args],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


def test_submit_command_manifest_preserves_exact_structured_identity(tmp_path):
    queue_dir = tmp_path / "queue"
    repo_root = tmp_path / "consumer-repo"
    output_dir = tmp_path / "durable-output"
    repo_root.mkdir()
    literal_arg = "literal; shell syntax stays data"
    manifest = {
        "schema": "gpu-greenroom.command.v1",
        "repo_root": str(repo_root),
        "cwd": str(repo_root),
        "env": {"RESULT_NAME": "proof.json"},
        "output_dir": str(output_dir),
        "route_identity": "soviet-judge/grid32-bounded-fixture",
        "argv": [
            sys.executable,
            "-c",
            (
                "import json, os, pathlib, sys; "
                "out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); "
                "(out / os.environ['RESULT_NAME']).write_text("
                "json.dumps({'argv': sys.argv[2:], 'cwd': os.getcwd()}))"
            ),
            str(output_dir),
            literal_arg,
        ],
        "timeout": None,
    }
    manifest_path = tmp_path / "command.json"
    manifest_path.write_text(json.dumps(manifest))

    rc, stdout, stderr = run_cli(
        "submit-command", "--manifest", str(manifest_path), queue_dir=queue_dir
    )

    assert rc == 0, stderr
    response = json.loads(stdout)
    assert response["schema"] == "gpu-greenroom.command-submission.v1"
    assert response["status"] == "pending"
    assert response["effective_queue_dir"] == str(queue_dir.resolve())
    request_path = Path(response["request_path"])
    request = json.loads(request_path.read_text())
    assert request["command_argv"] == manifest["argv"]
    assert request["command_cwd"] == str(repo_root)
    assert request["command_env"] == {"RESULT_NAME": "proof.json"}
    assert request["repo_root"] == str(repo_root)
    assert request["route_identity"] == manifest["route_identity"]
    assert request["command_timeout"] is None
    assert request["required_worker_capabilities"] == ["structured-command.v1"]

    queue = GPUQueue(queue_dir)
    assert queue.run_one(_load_job_types(queue_dir)) is True
    state = queue.get_job(response["job_id"])
    assert state.status == JobStatus.DONE

    proof = json.loads((output_dir / "proof.json").read_text())
    assert proof == {"argv": [literal_arg], "cwd": str(repo_root)}
    receipt = json.loads(
        (queue_dir / "done" / response["job_id"] / "receipt.json").read_text()
    )
    assert receipt["requested_route"] == manifest["route_identity"]
    assert receipt["effective_argv"] == manifest["argv"]
    assert receipt["effective_cwd"] == str(repo_root)
    assert receipt["effective_env"] == {"RESULT_NAME": "proof.json"}
    assert receipt["repo_root"] == str(repo_root)
    assert receipt["effective_timeout"] is None
    assert receipt["stdout_path"].endswith("/stdout.log")
    assert receipt["stderr_path"].endswith("/stderr.log")
    assert receipt["request_path"].endswith("/request.json")
    assert receipt["worker"]["pid"] == os.getpid()
    assert receipt["worker"]["source_root"] == str(
        Path(__file__).resolve().parent.parent
    )
    assert receipt["worker"]["commit"] == subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
    ).strip()
    assert receipt["worker"]["capabilities"] == ["structured-command.v1"]


def test_command_launch_failure_is_durable_before_primary_output(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    repo_root = tmp_path / "consumer-repo"
    repo_root.mkdir()
    request = JobRequest(
        job_type="command",
        input_path="",
        repo_root=str(repo_root),
        command_argv=[str(repo_root / "missing-executable"), "--fixture"],
        command_cwd=str(repo_root),
        command_env={"BACKEND": "mlx"},
        route_identity="soviet-judge/missing-launch-fixture",
    )
    queue.submit(request)

    assert queue.run_one({}) is True

    state = queue.get_job(request.job_id)
    assert state.status == JobStatus.FAILED
    assert state.failure_phase == "launch"
    job_dir = queue.queue_dir / "failed" / request.job_id
    receipt = json.loads((job_dir / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "launch"
    assert receipt["effective_argv"] == request.command_argv
    assert receipt["requested_route"] == request.route_identity
    assert receipt["repo_root"] == str(repo_root)
    assert Path(receipt["request_path"]).exists()
    assert Path(receipt["stdout_path"]).exists()
    assert Path(receipt["stderr_path"]).exists()
    assert not Path(request.output_dir).exists() or not any(Path(request.output_dir).iterdir())


def test_invalid_manifest_writes_submission_failure_report(tmp_path):
    queue_dir = tmp_path / "queue"
    manifest_path = tmp_path / "invalid-command.json"
    manifest_path.write_text(json.dumps({
        "schema": "gpu-greenroom.command.v1",
        "repo_root": str(tmp_path),
        "cwd": str(tmp_path),
        "route_identity": "fixture/missing-argv",
    }))

    rc, stdout, stderr = run_cli(
        "submit-command", "--manifest", str(manifest_path), queue_dir=queue_dir
    )

    assert rc == 2
    assert stdout == ""
    failure = json.loads(stderr)
    assert failure["schema"] == "gpu-greenroom.command-submission-failure.v1"
    assert failure["status"] == "failed"
    assert failure["failure_phase"] == "submission-validation"
    assert failure["manifest_path"] == str(manifest_path.resolve())
    assert failure["effective_queue_dir"] == str(queue_dir.resolve())
    report_path = Path(failure["report_path"])
    assert report_path.exists()
    assert json.loads(report_path.read_text()) == failure
    assert not any((queue_dir / "pending").iterdir())


def test_paused_queue_accepts_command_but_does_not_start_it(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    queue.pause()
    request = JobRequest(
        job_type="command",
        input_path="",
        repo_root=str(repo_root),
        command_argv=[sys.executable, "-c", "print('not yet')"],
        command_cwd=str(repo_root),
        route_identity="fixture/paused-command",
    )

    pending_dir = queue.submit(request)

    assert pending_dir.exists()
    assert queue.get_job(request.job_id).status == JobStatus.PENDING
    assert queue.run_one({}) is False
    assert queue.get_job(request.job_id).status == JobStatus.PENDING


def test_structured_command_composes_pause_recovery_and_capability_claim(
    tmp_path,
):
    queue = GPUQueue(tmp_path / "queue")
    repo_root = tmp_path / "repo"
    output_dir = tmp_path / "output"
    repo_root.mkdir()
    argv = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('composed-ran').write_text('yes')",
    ]
    request = JobRequest(
        job_type="command",
        input_path="",
        output_dir=str(output_dir),
        repo_root=str(repo_root),
        command_argv=argv,
        command_cwd=str(repo_root),
        route_identity="fixture/pause-capability-composition",
        required_worker_capabilities=["structured-command.v1"],
    )
    pending_dir = queue.submit(request)
    before_request = (pending_dir / "request.json").read_bytes()
    before_status = (pending_dir / "status.json").read_bytes()

    pause = queue.pause(
        owner="integration-smoke",
        epoch="integration-pause-1",
        contention_class="apple-unified-accelerator",
    )
    assert pause["effective_paused"] is True
    assert pause["epoch"] == "integration-pause-1"

    restarted = GPUQueue(queue.queue_dir)
    assert restarted.pause_state()["owner"] == "integration-smoke"
    assert restarted.pause_state()["epoch"] == "integration-pause-1"
    incapable_claimant = {
        "pid": 1001,
        "source_root": "/fixture/legacy-worker",
        "commit": "legacy-fixture",
        "git_dirty": False,
        "capabilities": [],
    }
    assert restarted.run_one({}, claimant=incapable_claimant) is False
    assert restarted.get_job(request.job_id).status == JobStatus.PENDING

    resume = restarted.resume(
        owner="integration-smoke",
        epoch="integration-pause-1",
    )
    assert resume["effective_paused"] is False
    assert resume["previous_pause"]["owner"] == "integration-smoke"
    assert resume["previous_pause"]["epoch"] == "integration-pause-1"

    assert restarted.run_one({}, claimant=incapable_claimant) is False
    assert (pending_dir / "request.json").read_bytes() == before_request
    assert (pending_dir / "status.json").read_bytes() == before_status
    assert restarted.get_job(request.job_id).status == JobStatus.PENDING
    assert not (repo_root / "composed-ran").exists()

    capable_claimant = {
        "pid": 2002,
        "source_root": "/fixture/reviewed-worker",
        "commit": "reviewed-fixture",
        "git_dirty": False,
        "capabilities": ["structured-command.v1"],
    }
    assert restarted.run_one({}, claimant=capable_claimant) is True
    assert restarted.get_job(request.job_id).status == JobStatus.DONE
    assert (repo_root / "composed-ran").read_text() == "yes"

    done_dir = restarted.queue_dir / "done" / request.job_id
    assert (done_dir / "request.json").read_bytes() == before_request
    receipt = json.loads((done_dir / "receipt.json").read_text())
    assert receipt["requested_route"] == request.route_identity
    assert receipt["effective_argv"] == argv
    assert receipt["effective_cwd"] == str(repo_root)
    assert receipt["worker"] == capable_claimant


def test_worker_without_required_capability_leaves_fifo_head_pending(
    tmp_path, monkeypatch
):
    queue = GPUQueue(tmp_path / "queue")
    repo_root = tmp_path / "repo"
    output_dir = tmp_path / "command-output"
    repo_root.mkdir()
    command = JobRequest(
        job_type="command",
        input_path="",
        output_dir=str(output_dir),
        repo_root=str(repo_root),
        command_argv=[
            sys.executable,
            "-c",
            "from pathlib import Path; Path('command-ran').write_text('yes')",
        ],
        command_cwd=str(repo_root),
        route_identity="fixture/mixed-version-command",
    )
    command_dir = queue.submit(command)
    command_payload = json.loads((command_dir / "request.json").read_text())
    command_payload["required_worker_capabilities"] = ["structured-command.v1"]
    (command_dir / "request.json").write_text(json.dumps(command_payload, indent=2))

    legacy_output = tmp_path / "legacy-output"
    legacy = JobRequest(
        job_type="write_output",
        input_path="younger",
        output_dir=str(legacy_output),
    )
    queue.submit(legacy)
    before_request = (command_dir / "request.json").read_bytes()
    before_status = (command_dir / "status.json").read_bytes()

    monkeypatch.setenv("GPU_GREENROOM_WORKER_CAPABILITIES", "")
    assert queue.run_one({
        "write_output": ["sh", "-c", "echo legacy > {output_dir}/result.txt"]
    }) is False
    assert (command_dir / "request.json").read_bytes() == before_request
    assert (command_dir / "status.json").read_bytes() == before_status
    assert queue.get_job(command.job_id).status == JobStatus.PENDING
    assert queue.get_job(legacy.job_id).status == JobStatus.PENDING
    assert not (repo_root / "command-ran").exists()
    assert not legacy_output.exists()

    monkeypatch.setenv(
        "GPU_GREENROOM_WORKER_CAPABILITIES", "structured-command.v1"
    )
    assert queue.run_one({}) is True
    assert queue.get_job(command.job_id).status == JobStatus.DONE
    assert (repo_root / "command-ran").read_text() == "yes"

    monkeypatch.setenv("GPU_GREENROOM_WORKER_CAPABILITIES", "")
    assert queue.run_one({
        "write_output": ["sh", "-c", "echo legacy > {output_dir}/result.txt"]
    }) is True
    assert queue.get_job(legacy.job_id).status == JobStatus.DONE
    assert (legacy_output / "result.txt").read_text().strip() == "legacy"


def test_legacy_command_without_capability_field_is_still_gated(
    tmp_path, monkeypatch
):
    queue = GPUQueue(tmp_path / "queue")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    request = JobRequest(
        job_type="command",
        input_path="",
        repo_root=str(repo_root),
        command_argv=[sys.executable, "-c", "print('must not run')"],
        command_cwd=str(repo_root),
        route_identity="fixture/legacy-command-request",
    )
    pending_dir = queue.submit(request)
    request_payload = json.loads((pending_dir / "request.json").read_text())
    request_payload.pop("required_worker_capabilities")
    (pending_dir / "request.json").write_text(json.dumps(request_payload, indent=2))

    monkeypatch.setenv("GPU_GREENROOM_WORKER_CAPABILITIES", "")
    assert queue.run_one({}) is False
    assert queue.get_job(request.job_id).status == JobStatus.PENDING

    monkeypatch.setenv(
        "GPU_GREENROOM_WORKER_CAPABILITIES", "structured-command.v1"
    )
    assert queue.run_one({}) is True
    assert queue.get_job(request.job_id).status == JobStatus.DONE


def test_explicit_claimant_capabilities_govern_admission_and_receipt(
    tmp_path, monkeypatch
):
    queue = GPUQueue(tmp_path / "queue")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    request = JobRequest(
        job_type="command",
        input_path="",
        repo_root=str(repo_root),
        command_argv=[sys.executable, "-c", "print('must not run')"],
        command_cwd=str(repo_root),
        route_identity="fixture/claimant-capability-consistency",
    )
    queue.submit(request)
    claimant = {
        "pid": os.getpid(),
        "source_root": str(repo_root),
        "commit": "fixture",
        "git_dirty": False,
        "capabilities": [],
    }

    monkeypatch.setenv(
        "GPU_GREENROOM_WORKER_CAPABILITIES", "structured-command.v1"
    )
    assert queue.run_one({}, claimant=claimant) is False
    assert queue.get_job(request.job_id).status == JobStatus.PENDING


def test_worker_source_identity_reports_untracked_files(tmp_path, monkeypatch):
    source_root = tmp_path / "worker-source"
    package_dir = source_root / "gpu_queue"
    package_dir.mkdir(parents=True)
    queue_source = package_dir / "queue.py"
    queue_source.write_text("# tracked worker source\n")
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Greenroom Test"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "greenroom-test@example.invalid"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(["git", "add", "gpu_queue/queue.py"], cwd=source_root, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "tracked worker source"],
        cwd=source_root,
        check=True,
    )
    (package_dir / "untracked_helper.py").write_text("# importable local helper\n")

    queue_module._worker_source_identity.cache_clear()
    monkeypatch.setattr(queue_module, "__file__", str(queue_source))
    try:
        identity = queue_module._worker_source_identity()
    finally:
        queue_module._worker_source_identity.cache_clear()

    assert identity["git_dirty"] is True


def test_doctor_reports_cli_queue_and_worker_dispatch(tmp_path):
    queue_dir = tmp_path / "queue"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "gpu-greenroom"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    env = {**os.environ, "PATH": str(bin_dir)}

    rc, stdout, stderr = run_cli(
        "doctor", "--json", queue_dir=queue_dir, env=env
    )

    assert rc == 0, stderr
    report = json.loads(stdout)
    assert report["schema"] == "gpu-greenroom.doctor.v1"
    assert report["healthy"] is True
    assert report["effective_queue_dir"] == str(queue_dir.resolve())
    assert report["checks"]["cli_import"]["ok"] is True
    assert report["checks"]["cli_executable"]["effective"] == str(executable)
    assert report["checks"]["queue_writable"]["ok"] is True
    assert report["checks"]["worker_dispatch_available"]["ok"] is True
    assert report["queue"]["paused"] is False
    assert report["queue"]["pending"] == 0
    assert report["queue"]["running"] == 0


def test_doctor_is_unhealthy_when_installed_cli_route_is_missing(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("gpu_queue.cli.shutil.which", lambda _name: None)
    args = type("Args", (), {"queue_dir": str(tmp_path / "queue")})()

    with pytest.raises(SystemExit) as raised:
        cmd_doctor(args)

    assert raised.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is False
    assert report["effective_queue_dir"] == str((tmp_path / "queue").resolve())
    assert report["cli_executable"] is None
    assert report["checks"]["cli_executable"] == {
        "ok": False,
        "requested": "gpu-greenroom",
        "effective": None,
    }
    assert report["checks"]["worker_dispatch_available"]["ok"] is True
    assert report["checks"]["worker_dispatch_available"]["claim"] == (
        "python-callable-present; no workload dispatched"
    )
