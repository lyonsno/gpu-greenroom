#!/usr/bin/env python3
"""GPU Greenroom CLI — submit, list, status, cancel, run worker."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .models import BumpStatus, JobRequest, JobStatus, LeaseStatus
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
        route_identity=args.route_identity,
        yields_to_waiters=args.yields_to_waiters,
        expected_handoff_seconds=args.expected_handoff_seconds,
        generation=args.generation,
    )
    try:
        job_dir = queue.submit(request)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Submitted job {request.job_id}")
    print(f"  Type: {request.job_type}")
    print(f"  Input: {request.input_path}")
    print(f"  Output: {request.output_dir}")
    if not args.output_dir:
        print(f"  (auto-assigned durable output dir)")
    if args.cwd:
        print(f"  Cwd: {args.cwd}")
    if request.route_identity:
        print(f"  Route: {request.route_identity}")
    if request.yields_to_waiters:
        seconds = _format_seconds(request.expected_handoff_seconds)
        print(f"  Cooperation: yields to waiters within <={seconds}s (generation {request.generation})")
    print(f"  Dir: {job_dir}")
    receipt = json.loads((job_dir / "submission_receipt.json").read_text())
    if receipt["running_observation"] == "unverified":
        print(
            "  Warning: current running-job cooperation could not be verified: "
            f"{receipt['running_observation_error']}",
            file=sys.stderr,
        )
    elif receipt["notice"]:
        running = receipt["running_job"]
        route = running["route_identity"] or running["job_id"]
        seconds = _format_seconds(running["expected_handoff_seconds"])
        print(
            f"  Current running job {route} is yield-aware; "
            f"expect the GPU within <={seconds}s"
        )
    elif receipt["running_job"] and receipt["running_job"]["cooperation_error"]:
        print(
            "  Warning: current running job has invalid cooperation metadata: "
            f"{receipt['running_job']['cooperation_error']}",
            file=sys.stderr,
        )


def cmd_list(args):
    queue = get_queue(args)
    status_filter = JobStatus(args.status) if args.status else None
    jobs = queue.list_jobs(status_filter)

    if not jobs:
        print("No jobs found.")
        return

    for job in jobs:
        details = []
        if job.started_at and job.finished_at:
            details.append(f"{job.finished_at - job.started_at:.1f}s")
        elif job.started_at:
            details.append(f"{time.time() - job.started_at:.1f}s running")
        if job.generation is not None:
            details.append(f"gen {job.generation}")
        detail_text = f" ({', '.join(details)})" if details else ""
        route = job.route_identity or job.job_type
        cooperation = ""
        if job.yields_to_waiters and job.cooperation_error() is None:
            cooperation = f" [YIELDS <={_format_seconds(job.expected_handoff_seconds)}s]"
        elif job.cooperation_error():
            cooperation = " [INVALID COOPERATION METADATA]"
        print(
            f"  {job.job_id}  {job.status.value:10s}  {route}"
            f"{cooperation}  {os.path.basename(job.input_path)}{detail_text}"
        )


def _format_seconds(value):
    if value is None:
        return "unknown"
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


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


def cmd_recover(args):
    queue = get_queue(args)
    recovered = queue.recover_stale()
    if recovered:
        print(f"Recovered {len(recovered)} stale job(s):")
        for jid in recovered:
            print(f"  {jid}")
    else:
        print("No stale jobs found.")


def _print_model(model):
    print(json.dumps(json.loads(model.to_json()), indent=2))


def cmd_lease_claim(args):
    queue = get_queue(args)
    lease = queue.claim_lease(
        lease_id=args.lease_id,
        owner=args.owner,
        agent_id=args.agent_id,
        repo_root=args.repo_root,
        pid=args.pid,
        process_group=args.process_group,
        effective_route=args.effective_route,
        backend=args.backend,
        device=args.device,
        profile=args.profile,
        supports_checkpoints=args.supports_checkpoints,
        interruptible=args.interruptible,
        ttl_seconds=args.ttl_seconds,
        handoff_bump_id=args.handoff_bump_id,
        raise_on_blocked=False,
    )
    if lease is None:
        print("Could not claim lease; gpu.lock or existing external lease blocks acquisition.", file=sys.stderr)
        sys.exit(1)
    _print_model(lease)


def cmd_lease_renew(args):
    queue = get_queue(args)
    interruptible = args.interruptible
    if args.not_interruptible:
        interruptible = False
    lease = queue.renew_lease(
        args.lease_id,
        interruptible=interruptible,
        ttl_seconds=args.ttl_seconds,
        lifecycle_state=args.lifecycle_state,
    )
    _print_model(lease)


def cmd_lease_status(args):
    queue = get_queue(args)
    lease = queue.lease_status()
    if lease is None:
        print(json.dumps({"lease": None, "execution_blocked": False}, indent=2))
        return
    _print_model(lease)


def cmd_lease_release(args):
    queue = get_queue(args)
    lease = queue.release_lease(args.lease_id, released_by=args.released_by, reason=args.reason)
    _print_model(lease)


def cmd_bump_request(args):
    queue = get_queue(args)
    bump = queue.request_bump(
        bump_id=args.bump_id,
        requester=args.requester,
        agent_id=args.agent_id,
        repo_root=args.repo_root,
        intended_route=args.intended_route,
        workload_class=args.workload_class,
        memory_pressure=args.memory_pressure,
        estimated_occupancy=args.estimated_occupancy,
        full_quiescence_required=args.full_quiescence_required,
        reason=args.reason,
        callback_address=args.callback_address,
    )
    _print_model(bump)


def cmd_bump_list(args):
    queue = get_queue(args)
    bumps = queue.list_bumps(args.status)
    print(json.dumps([json.loads(bump.to_json()) for bump in bumps], indent=2))


def cmd_bump_grant(args):
    queue = get_queue(args)
    bump = queue.grant_bump(
        args.bump_id,
        granted_by=args.granted_by,
        checkpoint=args.checkpoint,
        quiescence_confirmed=args.quiescence_confirmed,
    )
    _print_model(bump)


def cmd_bump_decline(args):
    queue = get_queue(args)
    bump = queue.decline_bump(args.bump_id, declined_by=args.declined_by, reason=args.reason)
    _print_model(bump)


def cmd_bump_wait(args):
    queue = get_queue(args)
    try:
        bump = queue.wait_for_bump(args.bump_id, timeout=args.timeout)
    except TimeoutError:
        last_bump = queue.get_bump(args.bump_id)
        print(json.dumps({
            "bump_id": args.bump_id,
            "status": "timed_out",
            "failure_phase": "wait",
            "requested_timeout_seconds": args.timeout,
            "effective_queue_dir": str(queue.queue_dir.resolve()),
            "last_trustworthy_bump": (
                json.loads(last_bump.to_json()) if last_bump is not None else None
            ),
        }, indent=2))
        sys.exit(1)
    _print_model(bump)


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
    p_submit.add_argument("--route-identity", help="Stable human/machine route identity for this job")
    p_submit.add_argument("--yields-to-waiters", action="store_true", help="Promise cooperative yield when another job waits")
    p_submit.add_argument("--expected-handoff-seconds", type=float, help="Expected upper bound from waiter arrival to GPU handoff")
    p_submit.add_argument("--generation", type=int, help="Self-resubmitting cooperative lineage generation (starts at 1)")
    p_submit.set_defaults(func=cmd_submit)

    # list
    p_list = sub.add_parser("list", help="List jobs")
    p_list.add_argument("-s", "--status", choices=["pending", "running", "done", "failed", "cancelled"])
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

    # recover
    p_recover = sub.add_parser("recover", help="Recover stale running jobs")
    p_recover.set_defaults(func=cmd_recover)

    # lease
    p_lease = sub.add_parser("lease", help="Manage cooperative external GPU leases")
    lease_sub = p_lease.add_subparsers(dest="lease_command")

    p_lease_claim = lease_sub.add_parser("claim", help="Claim a cooperative external GPU lease")
    p_lease_claim.add_argument("--lease-id")
    p_lease_claim.add_argument("--owner", required=True)
    p_lease_claim.add_argument("--agent-id", required=True)
    p_lease_claim.add_argument("--repo-root", required=True)
    p_lease_claim.add_argument("--pid", type=int)
    p_lease_claim.add_argument("--process-group", type=int)
    p_lease_claim.add_argument("--effective-route", required=True)
    p_lease_claim.add_argument("--backend", required=True)
    p_lease_claim.add_argument("--device", required=True)
    p_lease_claim.add_argument("--profile", required=True)
    p_lease_claim.add_argument("--supports-checkpoints", action="store_true")
    p_lease_claim.add_argument("--interruptible", action="store_true")
    p_lease_claim.add_argument("--ttl-seconds", type=float, default=300.0)
    p_lease_claim.add_argument("--handoff-bump-id")
    p_lease_claim.set_defaults(func=cmd_lease_claim)

    p_lease_renew = lease_sub.add_parser("renew", help="Renew the current external GPU lease")
    p_lease_renew.add_argument("lease_id")
    p_lease_renew.add_argument("--interruptible", action="store_true", default=None)
    p_lease_renew.add_argument("--not-interruptible", action="store_true")
    p_lease_renew.add_argument("--ttl-seconds", type=float)
    p_lease_renew.add_argument("--lifecycle-state", choices=[status.value for status in LeaseStatus])
    p_lease_renew.set_defaults(func=cmd_lease_renew)

    p_lease_status = lease_sub.add_parser("status", help="Show current external GPU lease")
    p_lease_status.set_defaults(func=cmd_lease_status)

    p_lease_release = lease_sub.add_parser("release", help="Release the current external GPU lease")
    p_lease_release.add_argument("lease_id")
    p_lease_release.add_argument("--released-by", required=True)
    p_lease_release.add_argument("--reason", required=True)
    p_lease_release.set_defaults(func=cmd_lease_release)

    # bump
    p_bump = sub.add_parser("bump", help="Manage inbound cooperative GPU bump requests")
    bump_sub = p_bump.add_subparsers(dest="bump_command")

    p_bump_request = bump_sub.add_parser("request", help="Request a cooperative GPU handoff window")
    p_bump_request.add_argument("--bump-id")
    p_bump_request.add_argument("--requester", required=True)
    p_bump_request.add_argument("--agent-id", required=True)
    p_bump_request.add_argument("--repo-root", required=True)
    p_bump_request.add_argument("--intended-route", required=True)
    p_bump_request.add_argument("--workload-class", required=True)
    p_bump_request.add_argument("--memory-pressure", required=True)
    p_bump_request.add_argument("--estimated-occupancy", required=True)
    p_bump_request.add_argument("--full-quiescence-required", action="store_true")
    p_bump_request.add_argument("--reason", required=True)
    p_bump_request.add_argument("--callback-address", required=True)
    p_bump_request.set_defaults(func=cmd_bump_request)

    p_bump_list = bump_sub.add_parser("list", help="List bump requests")
    p_bump_list.add_argument("--status", choices=[status.value for status in BumpStatus])
    p_bump_list.set_defaults(func=cmd_bump_list)

    p_bump_grant = bump_sub.add_parser("grant", help="Grant a bump now or after a checkpoint")
    p_bump_grant.add_argument("bump_id")
    p_bump_grant.add_argument("--granted-by", required=True)
    p_bump_grant.add_argument("--checkpoint", required=True)
    p_bump_grant.add_argument("--quiescence-confirmed", action="store_true")
    p_bump_grant.set_defaults(func=cmd_bump_grant)

    p_bump_decline = bump_sub.add_parser("decline", help="Decline a bump request")
    p_bump_decline.add_argument("bump_id")
    p_bump_decline.add_argument("--declined-by", required=True)
    p_bump_decline.add_argument("--reason", required=True)
    p_bump_decline.set_defaults(func=cmd_bump_decline)

    p_bump_wait = bump_sub.add_parser("wait", help="Wait for a bump grant/decline event")
    p_bump_wait.add_argument("bump_id")
    p_bump_wait.add_argument("--timeout", type=float)
    p_bump_wait.set_defaults(func=cmd_bump_wait)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    if args.command == "lease" and not getattr(args, "lease_command", None):
        p_lease.print_help()
        sys.exit(1)
    if args.command == "bump" and not getattr(args, "bump_command", None):
        p_bump.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
