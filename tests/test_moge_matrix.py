"""Tests for the MoGE benchmark matrix Greenroom orchestrator."""

import json
import subprocess
import sys
from pathlib import Path

from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue


REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_dummy_job_types(queue_dir: Path) -> None:
    """Write a fake MoGE benchmark route that can run under GPUQueue."""
    script = (
        "import json, pathlib; "
        "out = pathlib.Path(r'{output_dir}'); "
        "out.mkdir(parents=True, exist_ok=True); "
        "(out / 'benchmark_dummy.json').write_text(json.dumps({"
        "'precision': 'test', "
        "'modelLoadMs': 1, "
        "'firstInferenceMs': 2, "
        "'warmStats': {'median': 3, 'min': 2, 'max': 4, 'mean': 3}"
        "}))"
    )
    config = {
        "moge-bench-mlx": {
            "cmd": [sys.executable, "-c", script],
            "defaults": {"runs": "1"},
            "timeout": 30,
        }
    }
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "job_types.json").write_text(json.dumps(config))


def test_matrix_runs_greenroom_job_in_process_and_collects_result(tmp_path):
    """The matrix harness must not require a separately running worker."""
    queue_dir = tmp_path / "queue"
    output_dir = tmp_path / "matrix"
    image = tmp_path / "input.png"
    image.write_bytes(b"not-a-real-image-for-dummy-route")
    _write_dummy_job_types(queue_dir)

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/moge_matrix.py",
            "--queue-dir", str(queue_dir),
            "--output", str(output_dir),
            "--image", str(image),
            "--runtimes", "mlx",
            "--runs", "1",
            "--timeout", "0.1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((output_dir / "matrix_report.json").read_text())
    assert report["results"]["mlx"]["precision"] == "test"
    assert report["receipts"]["mlx"]["status"] == "done"
    assert report["receipts"]["mlx"]["effective_timeout"] == 30
    assert not list((queue_dir / "pending").iterdir())
    assert list((queue_dir / "done").iterdir())


def test_matrix_provides_repo_local_benchmark_routes_for_fresh_queue(tmp_path):
    """A fresh private queue should not need pre-copied MoGE job types."""
    from benchmarks import moge_matrix

    job_types = moge_matrix.load_matrix_job_types(tmp_path / "fresh-queue", REPO_ROOT)

    assert {"moge-bench-mlx", "moge-bench-pytorch", "moge-bench-webgpu"} <= set(job_types)
    encoded = json.dumps(job_types)
    assert str(REPO_ROOT / "benchmarks" / "bench_mlx.py") in encoded
    assert str(REPO_ROOT / "benchmarks" / "bench_pytorch.py") in encoded
    assert str(REPO_ROOT / "benchmarks" / "moge_webgpu.sh") in encoded
    assert "/Users/" not in encoded.replace(str(REPO_ROOT), "")   # only the resolved repo root, never a hardcoded home path


def test_webgpu_route_and_wrapper_receive_matrix_image(tmp_path):
    """WebGPU must not silently benchmark a different image than the matrix."""
    from benchmarks import moge_matrix

    job_types = moge_matrix.load_matrix_job_types(tmp_path / "fresh-queue", REPO_ROOT)
    cmd = job_types["moge-bench-webgpu"]["cmd"]
    assert "--image" in cmd
    assert "{input_path}" in cmd

    wrapper = (REPO_ROOT / "benchmarks" / "moge_webgpu.sh").read_text()
    assert "--image)" in wrapper
    assert "does not support --image" in wrapper


def test_matrix_exits_nonzero_when_submitted_job_remains_unresolved(tmp_path):
    """A report with a pending/timed-out job must not look like command success."""
    queue_dir = tmp_path / "queue"
    output_dir = tmp_path / "matrix"
    image = tmp_path / "input.png"
    image.write_bytes(b"not-a-real-image-for-unresolved-route")

    queue = GPUQueue(queue_dir)
    blocker = JobRequest(
        job_type="blocker",
        input_path=str(image),
        output_dir=str(tmp_path / "blocker-out"),
    )
    queue.submit(blocker)

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/moge_matrix.py",
            "--queue-dir", str(queue_dir),
            "--output", str(output_dir),
            "--image", str(image),
            "--runtimes", "mlx",
            "--runs", "1",
            "--timeout", "0.1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode != 0
    report = json.loads((output_dir / "matrix_report.json").read_text())
    assert report["success"] is False
    assert report["failures"]["mlx"]["phase"] == "timeout"
    assert report["failures"]["mlx"]["job_id"]
    assert report["failures"]["mlx"]["queue_dir"] == str(queue_dir)


def test_job_types_example_does_not_ship_stale_moge_benchmark_paths():
    """The matrix owns MoGE benchmark routes; examples must not point elsewhere."""
    config = json.loads((REPO_ROOT / "job_types.example.json").read_text())

    assert "moge-bench-mlx" not in config
    assert "moge-bench-pytorch" not in config
    assert "moge-bench-webgpu" not in config
    assert "/Users/" not in json.dumps(config).replace(str(REPO_ROOT), "")


def test_collect_receipt_uses_active_queue_dir(tmp_path, monkeypatch):
    """Receipt fallback must not read DEFAULT_QUEUE_DIR when queue_dir differs."""
    from benchmarks import moge_matrix

    queue_dir = tmp_path / "queue"
    wrong_default = tmp_path / "wrong-default"
    output_dir = tmp_path / "output"
    _write_dummy_job_types(queue_dir)
    monkeypatch.setattr(moge_matrix, "DEFAULT_QUEUE_DIR", str(wrong_default))

    queue = GPUQueue(queue_dir)
    request = JobRequest(
        job_type="moge-bench-mlx",
        input_path=str(tmp_path / "input.png"),
        output_dir=str(output_dir),
        params={"runs": "1"},
    )
    queue.submit(request)
    assert queue.run_one(json.loads((queue_dir / "job_types.json").read_text())) is True

    receipt = moge_matrix.collect_receipt(queue, request.job_id)

    assert receipt is not None
    assert receipt["job_id"] == request.job_id
    assert not wrong_default.exists()


def test_collect_result_stdout_fallback_uses_active_queue_dir(tmp_path, monkeypatch):
    """Stdout JSON fallback must use the active queue directory."""
    from benchmarks import moge_matrix

    queue_dir = tmp_path / "queue"
    wrong_default = tmp_path / "wrong-default"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(moge_matrix, "DEFAULT_QUEUE_DIR", str(wrong_default))

    job_types = {
        "stdout_json": {
            "cmd": [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'precision': 'stdout-only'}))",
            ],
        }
    }
    queue = GPUQueue(queue_dir)
    request = JobRequest(
        job_type="stdout_json",
        input_path=str(tmp_path / "input.png"),
        output_dir=str(output_dir),
    )
    queue.submit(request)
    assert queue.run_one(job_types) is True

    result = moge_matrix.collect_result(queue, request.job_id)

    assert result == {"precision": "stdout-only"}
    assert not wrong_default.exists()
