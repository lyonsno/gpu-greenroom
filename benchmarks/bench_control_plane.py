#!/usr/bin/env python3
"""Measure known-job submission and paused dispatch-check overhead."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue


def _measure(iterations: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="gpu-greenroom-bench-") as root:
        root_path = Path(root)
        queue = GPUQueue(root_path / "queue")
        started = time.perf_counter()
        for index in range(iterations):
            queue.submit(JobRequest(job_type="echo", input_path=f"fixture-{index}"))
        submit_seconds = time.perf_counter() - started

        queue.pause()
        started = time.perf_counter()
        for _ in range(iterations):
            assert queue.run_one({"echo": ["echo", "ok"]}) is False
        paused_dispatch_seconds = time.perf_counter() - started

    return {
        "schema": "gpu-greenroom.control-plane-benchmark.v1",
        "iterations": iterations,
        "known_job_submit": {
            "seconds": submit_seconds,
            "microseconds_per_operation": submit_seconds * 1_000_000 / iterations,
        },
        "paused_dispatch_check": {
            "seconds": paused_dispatch_seconds,
            "microseconds_per_operation": paused_dispatch_seconds * 1_000_000 / iterations,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    print(json.dumps(_measure(args.iterations), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
