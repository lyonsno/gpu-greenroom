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


def _job_summary(job):
    if job is None:
        return None
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "job_type": job.job_type,
        "input_path": job.input_path,
        "input_name": os.path.basename(job.input_path),
        "output_dir": job.output_dir,
        "submitted_at": job.submitted_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "warnings": job.warnings,
    }


def _queue_overview(queue: GPUQueue):
    jobs = queue.list_jobs()
    counts = {status.value: 0 for status in JobStatus}
    for job in jobs:
        counts[job.status.value] += 1

    pending = sorted(
        (job for job in jobs if job.status == JobStatus.PENDING),
        key=lambda job: job.submitted_at,
    )
    running = sorted(
        (job for job in jobs if job.status == JobStatus.RUNNING),
        key=lambda job: job.started_at or job.submitted_at,
    )
    latest_finished = sorted(
        (job for job in jobs if job.status in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)),
        key=lambda job: job.finished_at or job.submitted_at,
        reverse=True,
    )

    return {
        "schema": "gpu-greenroom.queue-status.v0",
        "queue_dir": str(queue.queue_dir),
        "paused": queue.is_paused(),
        "counts": counts,
        "running": [_job_summary(job) for job in running],
        "next_pending": _job_summary(pending[0] if pending else None),
        "latest_finished": _job_summary(latest_finished[0] if latest_finished else None),
    }


def _print_queue_overview_text(overview):
    print(f"Queue: {overview['queue_dir']}")
    print(f"Paused: {'yes' if overview['paused'] else 'no'}")
    print("Counts:")
    for status in ("pending", "running", "done", "failed", "cancelled"):
        print(f"  {status:10s} {overview['counts'][status]}")

    next_pending = overview["next_pending"]
    if next_pending:
        print(f"Next: {next_pending['job_id']} {next_pending['job_type']} {next_pending['input_name']}")
    else:
        print("Next: none")

    running = overview["running"]
    if running:
        print("Running:")
        for job in running:
            elapsed = ""
            if job["started_at"]:
                elapsed = f" ({time.time() - job['started_at']:.1f}s running)"
            print(f"  {job['job_id']} {job['job_type']} {job['input_name']}{elapsed}")


def cmd_status(args):
    queue = get_queue(args)
    if not args.job_id:
        overview = _queue_overview(queue)
        if args.json:
            print(json.dumps(overview, indent=2))
        else:
            _print_queue_overview_text(overview)
        return

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


def _load_job_type_entries(queue_dir):
    """Load job types with source metadata for read-side discovery."""
    config_path = Path(queue_dir) / "job_types.json"
    config_types = {}
    warnings = []
    if config_path.exists():
        try:
            config_types = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"could not load {config_path}: {e}")

    names = sorted(set(DEFAULT_JOB_TYPES) | set(config_types))
    entries = []
    for name in names:
        source = "config" if name in config_types else "default"
        config = config_types.get(name, DEFAULT_JOB_TYPES.get(name))
        if isinstance(config, list):
            cmd = config
            cwd = None
            env = {}
            defaults = {}
            timeout = None
        else:
            cmd = config.get("cmd", [])
            cwd = config.get("cwd")
            env = config.get("env", {}) or {}
            defaults = config.get("defaults", {}) or {}
            timeout = config.get("timeout")
        entries.append({
            "name": name,
            "source": source,
            "cmd_preview": " ".join(str(part) for part in cmd),
            "cwd": cwd,
            "env_keys": sorted(env.keys()),
            "default_keys": sorted(defaults.keys()),
            "timeout": timeout,
        })
    return entries, warnings


def cmd_job_types(args):
    entries, warnings = _load_job_type_entries(args.queue_dir)
    payload = {
        "schema": "gpu-greenroom.job-types.v0",
        "queue_dir": str(Path(args.queue_dir)),
        "config_path": str(Path(args.queue_dir) / "job_types.json"),
        "warnings": warnings,
        "job_types": entries,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("Job types:")
    for entry in entries:
        cwd = entry["cwd"] or "-"
        timeout = "none" if entry["timeout"] is None else str(entry["timeout"])
        print(f"  {entry['name']:16s} {entry['source']:7s} cwd={cwd} timeout={timeout}")
        print(f"    cmd: {entry['cmd_preview']}")
        if entry["default_keys"]:
            print(f"    defaults: {', '.join(entry['default_keys'])}")
        if entry["env_keys"]:
            print(f"    env: {', '.join(entry['env_keys'])}")
    for warning in warnings:
        print(f"Warning: {warning}")


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
    p_list.add_argument("-s", "--status", choices=["pending", "running", "done", "failed", "cancelled"])
    p_list.set_defaults(func=cmd_list)

    # status
    p_status = sub.add_parser("status", help="Get job status")
    p_status.add_argument("job_id", nargs="?")
    p_status.add_argument("--json", action="store_true", help="Print JSON queue overview when no job id is provided")
    p_status.set_defaults(func=cmd_status)

    # job type / route discovery
    p_job_types = sub.add_parser("job-types", aliases=["routes"], help="List configured job types / routes")
    p_job_types.add_argument("--json", action="store_true", help="Print machine-readable route discovery")
    p_job_types.set_defaults(func=cmd_job_types)

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
