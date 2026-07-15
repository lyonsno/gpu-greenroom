#!/usr/bin/env python3
"""MoGe benchmark matrix orchestrator.

Submits MLX, PyTorch, and WebGPU benchmark jobs through the GPU Greenroom
queue for contention-free isolated execution. Collects results and produces
a combined comparison table (markdown + JSON).

Usage:
    python benchmarks/moge_matrix.py [--runs 10] [--output /path/to/output]
    python benchmarks/moge_matrix.py --runtimes mlx pytorch  # subset
    python benchmarks/moge_matrix.py --dry-run  # show what would be submitted
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import gpu_queue
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpu_queue.models import JobRequest, JobStatus
from gpu_queue.queue import GPUQueue
from gpu_queue.cli import _load_job_types

DEFAULT_QUEUE_DIR = os.environ.get(
    "GPU_GREENROOM_DIR",
    os.path.expanduser("~/.local/state/gpu-greenroom"),
)

# Benchmark job definitions
BENCHMARK_JOBS = {
    "mlx": {
        "job_type": "moge-bench-mlx",
        "description": "MLX Metal (native compute, fp32)",
    },
    "pytorch": {
        "job_type": "moge-bench-pytorch",
        "description": "PyTorch MPS (fp32)",
    },
    "webgpu": {
        "job_type": "moge-bench-webgpu",
        "description": "WebGPU (browser compute shaders, fp16 weights)",
    },
}


def get_hardware_info() -> dict:
    """Collect hardware info for the benchmark report."""
    info = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }

    # Get Apple Silicon chip info
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["cpu"] = result.stdout.strip()
    except Exception:
        pass

    # Get total memory
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["total_memory_gb"] = int(result.stdout.strip()) / (1024**3)
    except Exception:
        pass

    # Get GPU info (Metal)
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            if displays:
                gpu = displays[0]
                info["gpu"] = gpu.get("sppci_model", "unknown")
                vram = gpu.get("spdisplays_vram_shared", gpu.get("spdisplays_vram", ""))
                if vram:
                    info["vram"] = vram
    except Exception:
        pass

    return info


def get_git_info(repo_path: str) -> dict | None:
    """Get git commit info for a repo."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "log", "-1", "--format=%H %h %s"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(" ", 2)
            return {
                "sha": parts[0],
                "short_sha": parts[1],
                "message": parts[2] if len(parts) > 2 else "",
                "repo": repo_path,
            }
    except Exception:
        pass
    return None


def poll_job(queue: GPUQueue, job_id: str, timeout: float = 600) -> dict | None:
    """Poll for job completion. Returns the status dict or None on timeout."""
    start = time.time()
    last_status = None

    while time.time() - start < timeout:
        state = queue.get_job(job_id)
        if state is None:
            print(f"  WARNING: Job {job_id} not found in queue", file=sys.stderr)
            return None

        if state.status != last_status:
            elapsed = time.time() - start
            print(f"  [{elapsed:.0f}s] {job_id}: {state.status.value}", file=sys.stderr)
            last_status = state.status

        if state.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED):
            return json.loads(state.to_json())

        time.sleep(2)

    print(f"  TIMEOUT: Job {job_id} did not complete in {timeout}s", file=sys.stderr)
    return None


def run_submitted_job_if_next(queue: GPUQueue, job_id: str, job_types: dict) -> None:
    """Run the submitted job in-process when it is next in FIFO.

    This makes single-run benchmark queues self-contained while preserving FIFO:
    if another job is ahead of ours or the lock is held, we leave execution to a
    separately running Greenroom worker instead of jumping the queue.
    """
    pending = sorted(
        queue.list_jobs(JobStatus.PENDING),
        key=lambda state: state.submitted_at,
    )
    if not pending or pending[0].job_id != job_id:
        if pending:
            print(
                f"  Waiting for external worker; next pending job is {pending[0].job_id}",
                file=sys.stderr,
            )
        return

    print(f"  Running {job_id} in-process under Greenroom flock", file=sys.stderr)
    ran = queue.run_one(job_types)
    if not ran:
        print(
            f"  Greenroom worker did not run {job_id}; lock may be held or queue paused",
            file=sys.stderr,
        )


def _job_file(queue: GPUQueue, subdir: str, job_id: str, filename: str) -> Path:
    return Path(queue.queue_dir) / subdir / job_id / filename


def collect_result(queue: GPUQueue, job_id: str) -> dict | None:
    """Read the benchmark JSON from a completed job's output dir."""
    state = queue.get_job(job_id)
    if state is None or state.status != JobStatus.DONE:
        return None

    output_dir = Path(state.output_dir)
    # Look for any benchmark_*.json file
    for p in sorted(output_dir.glob("benchmark_*.json")):
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue

    # Also check stdout for JSON (some jobs write to stdout)
    for sub in ("done", "failed"):
        stdout_path = _job_file(queue, sub, job_id, "stdout.log")
        if stdout_path.exists():
            try:
                text = stdout_path.read_text().strip()
                if text.startswith("{"):
                    return json.loads(text)
            except (json.JSONDecodeError, OSError):
                pass

    return None


def collect_receipt(queue: GPUQueue, job_id: str) -> dict | None:
    """Read the greenroom receipt for a job."""
    for sub in ("done", "failed", "cancelled"):
        receipt_path = _job_file(queue, sub, job_id, "receipt.json")
        if receipt_path.exists():
            try:
                return json.loads(receipt_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
    return None


def format_comparison_table(results: dict[str, dict]) -> str:
    """Format results as a markdown comparison table."""
    lines = []
    lines.append("## MoGe-2 Benchmark Matrix")
    lines.append("")

    # Header
    runtimes = list(results.keys())
    lines.append("| Metric | " + " | ".join(runtimes) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(runtimes)) + "|")

    # Rows
    def get_val(r, key, fmt=".1f", suffix="ms"):
        if r is None:
            return "FAILED"
        val = r.get(key)
        if val is None:
            return "N/A"
        return f"{val:{fmt}}{suffix}"

    metrics = [
        ("Model Load", "modelLoadMs", ".0f", "ms"),
        ("First Inference", "firstInferenceMs", ".0f", "ms"),
        ("Warm Median", lambda r: r.get("warmStats", {}).get("median") if r else None, ".0f", "ms"),
        ("Warm Min", lambda r: r.get("warmStats", {}).get("min") if r else None, ".0f", "ms"),
        ("Warm Max", lambda r: r.get("warmStats", {}).get("max") if r else None, ".0f", "ms"),
        ("Warm Mean", lambda r: r.get("warmStats", {}).get("mean") if r else None, ".0f", "ms"),
        ("Precision", "precision", None, None),
    ]

    for row in metrics:
        label = row[0]
        cells = []
        for rt in runtimes:
            r = results.get(rt)
            if callable(row[1]):
                val = row[1](r)
                if val is None:
                    cells.append("N/A" if r else "FAILED")
                elif row[2] is None:
                    cells.append(str(val))
                else:
                    cells.append(f"{val:{row[2]}}{row[3]}")
            elif row[2] is None:
                cells.append(str(r.get(row[1], "N/A")) if r else "FAILED")
            else:
                cells.append(get_val(r, row[1], row[2], row[3]))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="MoGe benchmark matrix orchestrator",
    )
    parser.add_argument("--runs", type=int, default=10, help="Warm inference runs per runtime")
    parser.add_argument("--output", default=None, help="Output directory for results")
    parser.add_argument("--queue-dir", default=DEFAULT_QUEUE_DIR, help="Greenroom queue directory")
    parser.add_argument("--runtimes", nargs="*", default=None,
                        choices=["mlx", "pytorch", "webgpu"],
                        help="Which runtimes to benchmark (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be submitted")
    parser.add_argument("--image", default=None, help="Override input image path")
    parser.add_argument("--timeout", type=float, default=600, help="Per-job timeout in seconds")
    args = parser.parse_args()

    runtimes = args.runtimes or ["mlx", "pytorch", "webgpu"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.queue_dir) / "benchmark-results" / f"moge-matrix-{timestamp}"

    input_image = args.image or os.path.expanduser(
        "~/dev/moge-webgpu/public/test_fixtures/input.png"
    )

    print(f"MoGe Benchmark Matrix", file=sys.stderr)
    print(f"  Runtimes: {', '.join(runtimes)}", file=sys.stderr)
    print(f"  Runs per runtime: {args.runs}", file=sys.stderr)
    print(f"  Input image: {input_image}", file=sys.stderr)
    print(f"  Output: {output_dir}", file=sys.stderr)
    print(f"  Queue dir: {args.queue_dir}", file=sys.stderr)
    print(f"", file=sys.stderr)

    if args.dry_run:
        print("DRY RUN -- would submit:", file=sys.stderr)
        for rt in runtimes:
            job = BENCHMARK_JOBS[rt]
            print(f"  {job['job_type']}: {job['description']}", file=sys.stderr)
        return

    queue = GPUQueue(args.queue_dir)
    job_types = _load_job_types(args.queue_dir)

    # Collect hardware info and git commits
    hardware = get_hardware_info()
    git_commits = {}
    for name, path in [
        ("gpu-greenroom", str(Path(__file__).resolve().parent.parent)),
        ("moge-mlx", os.path.expanduser("~/dev/moge-mlx")),
        ("moge-webgpu", os.path.expanduser("~/dev/moge-webgpu")),
        ("moge-standalone", os.path.expanduser("~/dev/moge-standalone")),
    ]:
        info = get_git_info(path)
        if info:
            git_commits[name] = info

    # Submit jobs sequentially and wait for each
    results = {}
    receipts = {}
    statuses = {}

    for rt in runtimes:
        job_def = BENCHMARK_JOBS[rt]
        job_type = job_def["job_type"]

        # Build per-job output dir
        job_output = str(output_dir / rt)

        params = {
            "runs": str(args.runs),
        }
        if args.image:
            params["image"] = args.image

        request = JobRequest(
            job_type=job_type,
            input_path=input_image,
            output_dir=job_output,
            params=params,
        )

        print(f"Submitting {rt} ({job_type})...", file=sys.stderr)
        job_dir = queue.submit(request)
        job_id = request.job_id
        print(f"  Job ID: {job_id}", file=sys.stderr)
        print(f"  Output: {job_output}", file=sys.stderr)

        # Execute directly when this benchmark job is next in FIFO. This keeps
        # private benchmark queues self-contained without requiring a background
        # worker, while still refusing to jump ahead of existing queued work.
        run_submitted_job_if_next(queue, job_id, job_types)

        # Poll for completion
        status = poll_job(queue, job_id, timeout=args.timeout)
        statuses[rt] = status

        if status and status.get("status") == "done":
            print(f"  DONE in {status.get('finished_at', 0) - status.get('started_at', 0):.1f}s", file=sys.stderr)
            result = collect_result(queue, job_id)
            if result:
                results[rt] = result
                print(f"  Result collected: {len(result)} keys", file=sys.stderr)
            else:
                print(f"  WARNING: Job completed but no result JSON found", file=sys.stderr)
                results[rt] = None
        else:
            fail_reason = "timeout"
            if status:
                fail_reason = status.get("error_message", status.get("failure_phase", "unknown"))
            print(f"  FAILED: {fail_reason}", file=sys.stderr)
            results[rt] = None

        # Collect receipt
        receipt = collect_receipt(queue, job_id)
        if receipt:
            receipts[rt] = receipt

        print(f"", file=sys.stderr)

    # Build combined report
    report = {
        "timestamp": timestamp,
        "runtimes": runtimes,
        "runs_per_runtime": args.runs,
        "input_image": input_image,
        "hardware": hardware,
        "git_commits": git_commits,
        "results": results,
        "receipts": receipts,
        "statuses": statuses,
    }

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = output_dir / "matrix_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Report saved to {report_path}", file=sys.stderr)

    # Generate comparison table
    table = format_comparison_table(results)
    table_path = output_dir / "comparison.md"
    table_path.write_text(table + "\n")
    print(f"Comparison table saved to {table_path}", file=sys.stderr)

    # Print table to stderr and JSON to stdout
    print(f"\n{table}", file=sys.stderr)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
