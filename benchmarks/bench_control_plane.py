#!/usr/bin/env python3
"""Measure Stage B queue submission and dequeue-check overhead."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue


def _source_identity() -> dict:
    repo_root = Path(__file__).resolve().parent.parent

    def git_output(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "repo_root": str(repo_root),
        "commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def _measure_sample(iterations: int, sample_index: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="gpu-greenroom-bench-") as root:
        root_path = Path(root)

        submit_queue = GPUQueue(root_path / "submit")
        started = time.perf_counter()
        for index in range(iterations):
            submit_queue.submit(
                JobRequest(job_type="echo", input_path=f"fixture-{index}")
            )
        submit_seconds = time.perf_counter() - started

        empty_queue = GPUQueue(root_path / "empty")
        started = time.perf_counter()
        for _ in range(iterations):
            assert empty_queue.run_one({"echo": ["echo", "ok"]}) is False
        empty_dequeue_seconds = time.perf_counter() - started

        paused_queue = GPUQueue(root_path / "paused")
        paused_queue.pause(
            owner="control-plane-benchmark",
            epoch=f"sample-{sample_index}",
        )
        started = time.perf_counter()
        for _ in range(iterations):
            assert paused_queue.run_one({"echo": ["echo", "ok"]}) is False
        paused_dequeue_seconds = time.perf_counter() - started

    return {
        "known_job_submit": submit_seconds,
        "empty_dequeue_check": empty_dequeue_seconds,
        "paused_dequeue_check": paused_dequeue_seconds,
    }


def _distribution(sample_seconds: list[float], iterations: int) -> dict:
    sample_microseconds = [
        seconds * 1_000_000 / iterations for seconds in sample_seconds
    ]
    return {
        "operation_count": iterations * len(sample_seconds),
        "sample_seconds": sample_seconds,
        "sample_microseconds_per_operation": sample_microseconds,
        "microseconds_per_operation": {
            "min": min(sample_microseconds),
            "median": statistics.median(sample_microseconds),
            "max": max(sample_microseconds),
        },
    }


def _measure(iterations: int, samples: int = 1) -> dict:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if samples <= 0:
        raise ValueError("samples must be positive")

    raw_samples = [
        _measure_sample(iterations, sample_index)
        for sample_index in range(samples)
    ]
    measurements = {
        name: _distribution(
            [sample[name] for sample in raw_samples],
            iterations,
        )
        for name in (
            "known_job_submit",
            "empty_dequeue_check",
            "paused_dequeue_check",
        )
    }
    return {
        "schema": "gpu-greenroom.control-plane-benchmark.v1",
        "requested": {
            "iterations_per_sample": iterations,
            "samples": samples,
        },
        "effective": {
            "iterations_per_sample": iterations,
            "samples": samples,
            "operations_per_measurement": iterations * samples,
        },
        "source": _source_identity(),
        **measurements,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    print(json.dumps(_measure(args.iterations, args.samples), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
