#!/usr/bin/env python3
"""GPU Greenroom CLI — submit, list, status, cancel, run worker."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .models import JobRequest, JobStatus
from .queue import GPUQueue

DEFAULT_QUEUE_DIR = os.environ.get("GPU_GREENROOM_DIR", os.path.expanduser("~/.local/state/gpu-greenroom"))

# Default job type configurations
# Rich config: cmd, cwd, env, defaults
# Bare list also accepted for simple cases
_TRELLIS_ROOT = os.path.expanduser("~/dev/trellis2mlx")
_PERCEPTASIA_ROOT = os.path.expanduser("~/dev/perceptasia")

DEFAULT_JOB_TYPES = {
    "trellis2mlx": {
        "cmd": [
            os.path.join(_TRELLIS_ROOT, ".venv/bin/python"), "-u", "generate.py",
            "--image", "{input_path}",
            "--output", "{output_dir}/seed-{seed}.glb",
            "--seed", "{seed}",
            "--resolution", "{resolution}",
            "--target-faces", "{target_faces}",
            "--texture-size", "{texture_size}",
            "--simplify-first",
            "--save-checkpoints", "{checkpoint_dir}",
            "--checkpoint-stop-file", "{checkpoint_stop_file}",
        ],
        "cwd": _TRELLIS_ROOT,
        "env": {"PYTHONPATH": "."},
        "defaults": {
            "seed": "42",
            "resolution": "512",
            "target_faces": "200000",
            "texture_size": "1024",
        },
    },
    "supermat": {
        "cmd": [
            os.path.join(_PERCEPTASIA_ROOT, ".venv/bin/python"), "-u", "run_supermat.py",
            "--image", "{input_path}",
            "--output-dir", "{output_dir}",
        ],
        "cwd": _PERCEPTASIA_ROOT,
        "env": {"PYTHONPATH": "."},
        "defaults": {},
    },
    "kaminos-moge-webgpu-browser-preview": {
        "cmd": [
            sys.executable, "-m", "gpu_queue.runners.kaminos_browser_preview",
            "--input-path", "{input_path}",
            "--output-dir", "{output_dir}",
            "--result-dir", "{result_dir}",
            "--request-id", "{request_id}",
            "--route-id", "{route_id}",
            "--job-id", "{job_id}",
            "--source-identity-json", "{source_image_identity}",
            "--module-base-url", "{module_base_url}",
            "--kit-version", "{kit_version}",
            "--mode", "{runner_mode}",
            "--kaminos-root", "{kaminos_root}",
            "--kaminos-url", "{kaminos_url}",
            "--chrome-path", "{chrome_path}",
            "--browser-timeout-s", "{browser_timeout_s}",
        ],
        "defaults": {
            "result_dir": "",
            "request_id": "",
            "route_id": "moge.depth-normal.webgpu-local.v0",
            "source_image_identity": "{}",
            "module_base_url": "",
            "kit_version": "unknown",
            "runner_mode": "fixture",
            "kaminos_root": "",
            "kaminos_url": "",
            "chrome_path": "",
            "browser_timeout_s": "120",
        },
    },
}


def get_queue(args) -> GPUQueue:
    return GPUQueue(args.queue_dir)


def cmd_submit(args):
    queue = get_queue(args)
    params = {}
    if args.params:
        for p in args.params:
            k, v = p.split("=", 1)
            params[k] = v
    if args.cwd:
        params["cwd"] = args.cwd

    request = JobRequest(
        job_type=args.job_type,
        input_path=args.input,
        output_dir=args.output_dir,
        params=params,
    )
    job_dir = queue.submit(request)
    print(f"Submitted job {request.job_id}")
    print(f"  Type: {request.job_type}")
    print(f"  Input: {request.input_path}")
    print(f"  Output: {request.output_dir}")
    if not args.output_dir:
        print(f"  (auto-assigned durable output dir)")
    if args.cwd:
        print(f"  Cwd: {args.cwd}")
    print(f"  Dir: {job_dir}")


def cmd_list(args):
    queue = get_queue(args)
    status_filter = JobStatus(args.status) if args.status else None
    if args.json_output:
        print(json.dumps(queue.queue_index_payload(status_filter), indent=2))
        return

    jobs = queue.list_jobs(status_filter)

    if not jobs:
        print("No jobs found.")
        return

    for job in jobs:
        elapsed = ""
        if job.started_at and job.finished_at:
            elapsed = f" ({job.finished_at - job.started_at:.1f}s)"
        elif job.started_at:
            elapsed = f" ({time.time() - job.started_at:.1f}s running)"
        print(f"  {job.job_id}  {job.status.value:10s}  {job.job_type:12s}  {os.path.basename(job.input_path)}{elapsed}")


def cmd_status(args):
    queue = get_queue(args)
    state = queue.get_job(args.job_id)
    if state is None:
        print(f"Job {args.job_id} not found.")
        sys.exit(1)
    print(json.dumps(json.loads(state.to_json()), indent=2))


def cmd_cancel(args):
    queue = get_queue(args)
    if queue.cancel(args.job_id):
        print(f"Cancelled job {args.job_id}")
    else:
        print(f"Could not cancel {args.job_id} (not found or not pending)")
        sys.exit(1)


def _load_job_types(queue_dir):
    """Load job types from defaults + config file. Called per-job so new types are picked up live."""
    job_types = dict(DEFAULT_JOB_TYPES)
    config_path = Path(queue_dir) / "job_types.json"
    if config_path.exists():
        try:
            custom = json.loads(config_path.read_text())
            job_types.update(custom)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not load {config_path}: {e}")
    return job_types


def cmd_worker(args):
    """Run the worker loop — picks and runs jobs sequentially."""
    queue = get_queue(args)

    job_types = _load_job_types(args.queue_dir)

    print(f"GPU Greenroom Worker starting")
    print(f"  Queue dir: {args.queue_dir}")
    print(f"  Job types: {', '.join(job_types.keys())}")
    print(f"  Poll interval: {args.poll}s")

    # Recover stale jobs on startup
    recovered = queue.recover_stale()
    if recovered:
        print(f"  Recovered {len(recovered)} stale job(s): {', '.join(recovered)}")

    was_paused = False
    try:
        while True:
            if queue.is_paused():
                if not was_paused:
                    print("Queue paused. Waiting for resume...")
                    was_paused = True
                time.sleep(args.poll)
                continue
            if was_paused:
                print("Queue resumed.")
                was_paused = False
            job_types = _load_job_types(args.queue_dir)
            ran = queue.run_one(job_types)
            if ran:
                # Check for more immediately
                continue
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nWorker stopped.")


def cmd_pause(args):
    queue = get_queue(args)
    queue.pause()
    print("Queue paused. Worker will finish current job then wait.")


def cmd_resume(args):
    queue = get_queue(args)
    queue.resume()
    print("Queue resumed.")


def cmd_request_checkpoint_pause(args):
    queue = get_queue(args)
    try:
        receipt = queue.request_checkpoint_pause(args.job_id)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)
    if receipt is None:
        print(f"Could not request checkpoint pause for {args.job_id} (not found, not pending/running, or already terminal)")
        sys.exit(1)
    print(f"Checkpoint pause requested for {args.job_id}")
    print(f"  Stop file: {receipt['checkpoint_stop_file']}")
    print(f"  Receipt: {receipt['receipt_path']}")


def cmd_recover(args):
    queue = get_queue(args)
    recovered = queue.recover_stale()
    if recovered:
        print(f"Recovered {len(recovered)} stale job(s):")
        for jid in recovered:
            print(f"  {jid}")
    else:
        print("No stale jobs found.")


def main():
    parser = argparse.ArgumentParser(
        prog="gpu-greenroom",
        description="Filesystem-backed GPU job queue with flock serialization",
    )
    parser.add_argument(
        "--queue-dir", default=DEFAULT_QUEUE_DIR,
        help=f"Queue directory (default: {DEFAULT_QUEUE_DIR})",
    )

    sub = parser.add_subparsers(dest="command")

    # submit
    p_submit = sub.add_parser("submit", help="Submit a job")
    p_submit.add_argument("job_type", help="Job type (e.g. trellis2mlx, supermat)")
    p_submit.add_argument("input", help="Input file path")
    p_submit.add_argument("output_dir", nargs="?", default="", help="Output directory (default: durable path in queue dir)")
    p_submit.add_argument("-p", "--params", nargs="*", help="Key=value params (e.g. seed=42)")
    p_submit.add_argument("--cwd", help="Override working directory (e.g. for branch/worktree)")
    p_submit.set_defaults(func=cmd_submit)

    # list
    p_list = sub.add_parser("list", help="List jobs")
    p_list.add_argument("-s", "--status", choices=["pending", "running", "done", "failed", "checkpoint_paused", "cancelled"])
    p_list.add_argument("--json", action="store_true", dest="json_output", help="Print tolerant queue index JSON")
    p_list.set_defaults(func=cmd_list)

    # status
    p_status = sub.add_parser("status", help="Get job status")
    p_status.add_argument("job_id")
    p_status.set_defaults(func=cmd_status)

    # cancel
    p_cancel = sub.add_parser("cancel", help="Cancel a pending job")
    p_cancel.add_argument("job_id")
    p_cancel.set_defaults(func=cmd_cancel)

    # worker
    p_worker = sub.add_parser("worker", help="Run the worker loop")
    p_worker.add_argument("--poll", type=float, default=2.0, help="Poll interval in seconds")
    p_worker.set_defaults(func=cmd_worker)

    # pause
    p_pause = sub.add_parser("pause", help="Pause the queue (finish current job, then wait)")
    p_pause.set_defaults(func=cmd_pause)

    # resume
    p_resume = sub.add_parser("resume", help="Resume a paused queue")
    p_resume.set_defaults(func=cmd_resume)

    # request-checkpoint-pause
    p_checkpoint_pause = sub.add_parser(
        "request-checkpoint-pause",
        help="Request cooperative stop after the next checkpoint boundary",
    )
    p_checkpoint_pause.add_argument("job_id")
    p_checkpoint_pause.set_defaults(func=cmd_request_checkpoint_pause)

    # recover
    p_recover = sub.add_parser("recover", help="Recover stale running jobs")
    p_recover.set_defaults(func=cmd_recover)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
