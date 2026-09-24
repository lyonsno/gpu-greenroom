#!/usr/bin/env python3
"""GPU Greenroom CLI — submit, list, status, cancel, run worker."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import sys
import time
from pathlib import Path

from .control import QueueControlError, QueueRegistry
from .models import BumpStatus, JobRequest, JobStatus, LeaseStatus
from . import gc as gc_mod
from .queue import (
    GPUQueue,
    PauseStateError,
    STRUCTURED_COMMAND_CAPABILITY,
    effective_worker_capabilities,
    worker_identity,
)

DEFAULT_QUEUE_DIR = os.environ.get("GPU_GREENROOM_DIR", os.path.expanduser("~/.local/state/gpu-greenroom"))
DEFAULT_REGISTRY_PATH = os.environ.get(
    "GPU_GREENROOM_REGISTRY",
    os.path.join(DEFAULT_QUEUE_DIR, "queues.json"),
)

# Built-in job types. Real job types live in job_types.json in the queue
# directory and are hot-reloaded every poll; the built-ins exist so a fresh
# install can smoke the queue without any external generator installed.
DEFAULT_JOB_TYPES = {
    "echo": {
        "cmd": ["echo", "greenroom", "{input_path}", "{output_dir}"],
        "defaults": {},
    },
}


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("gpu-greenroom")
    except Exception:
        return "unknown"


def get_queue(args) -> GPUQueue:
    return GPUQueue(args.queue_dir)


def _optional_agent_id(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("agent_id must be a non-empty string when supplied")
    return value


def _agent_id_arg(value):
    try:
        return _optional_agent_id(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


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
        agent_id=_optional_agent_id(args.agent_id),
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
    if request.agent_id:
        print(f"  Agent ID: {request.agent_id}")
    print(f"  Dir: {job_dir}")


def _parse_env_assignments(assignments):
    env = {}
    for assignment in assignments or []:
        if "=" not in assignment:
            raise ValueError(f"environment assignment must be KEY=VALUE: {assignment!r}")
        key, value = assignment.split("=", 1)
        if not key:
            raise ValueError("environment key must not be empty")
        env[key] = value
    return env


def _command_payload(args):
    manifest_path = None
    if args.manifest:
        if args.agent_id is not None:
            raise ValueError("--agent-id cannot be used with --manifest; declare agent_id in the manifest")
        manifest_path = Path(args.manifest).expanduser().resolve()
        payload = json.loads(manifest_path.read_text())
    else:
        argv = list(args.argv or [])
        if argv and argv[0] == "--":
            argv = argv[1:]
        payload = {
            "schema": "gpu-greenroom.command.v1",
            "agent_id": args.agent_id,
            "repo_root": args.repo_root,
            "cwd": args.cwd,
            "env": _parse_env_assignments(args.env),
            "output_dir": args.output_dir,
            "route_identity": args.route_identity,
            "output_class": args.output_class,
            "argv": argv,
            "timeout": args.timeout,
        }

    if not isinstance(payload, dict):
        raise ValueError("command manifest must be a JSON object")
    if payload.get("schema") != "gpu-greenroom.command.v1":
        raise ValueError("command manifest schema must be gpu-greenroom.command.v1")
    agent_id = _optional_agent_id(payload.get("agent_id"))
    argv = payload.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("command manifest argv must be a non-empty list of strings")
    for key in ("repo_root", "cwd", "route_identity"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"command manifest {key} must be a non-empty string")
    repo_root = Path(payload["repo_root"]).expanduser().resolve()
    cwd = Path(payload["cwd"]).expanduser().resolve()
    if not repo_root.is_dir():
        raise ValueError(f"command repo_root is not a directory: {repo_root}")
    if not cwd.is_dir():
        raise ValueError(f"command cwd is not a directory: {cwd}")
    env = payload.get("env") or {}
    if (
        not isinstance(env, dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items())
    ):
        raise ValueError("command manifest env must map strings to strings")
    timeout = payload.get("timeout")
    if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
        raise ValueError("command manifest timeout must be null or a positive number")
    output_dir = payload.get("output_dir") or ""
    if not isinstance(output_dir, str):
        raise ValueError("command manifest output_dir must be a string")
    output_class = payload.get("output_class")
    if output_class is not None and output_class not in gc_mod.CLASSES:
        raise ValueError(f"command manifest output_class must be one of {', '.join(gc_mod.CLASSES)} or absent")
    return {
        "output_class": output_class,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "repo_root": str(repo_root),
        "cwd": str(cwd),
        "env": env,
        "agent_id": agent_id,
        "output_dir": output_dir,
        "route_identity": payload["route_identity"],
        "argv": argv,
        "timeout": timeout,
    }


def _write_submission_failure(queue, args, error):
    failure_dir = queue.queue_dir / "submission-failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    report_path = failure_dir / f"{time.time_ns()}-command-submission.json"
    manifest_path = (
        str(Path(args.manifest).expanduser().resolve())
        if args.manifest else None
    )
    report = {
        "schema": "gpu-greenroom.command-submission-failure.v1",
        "status": "failed",
        "failure_phase": "submission-validation",
        "error_message": str(error),
        "manifest_path": manifest_path,
        "effective_queue_dir": str(queue.queue_dir.resolve()),
        "report_path": str(report_path),
    }
    queue._write_json_atomic(report_path, report)
    return report


def cmd_submit_command(args):
    queue = get_queue(args)
    try:
        payload = _command_payload(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = _write_submission_failure(queue, args, error)
        print(json.dumps(report, indent=2), file=sys.stderr)
        sys.exit(2)

    request = JobRequest(
        job_type="command",
        input_path="",
        output_dir=payload["output_dir"],
        agent_id=payload["agent_id"],
        repo_root=payload["repo_root"],
        command_argv=payload["argv"],
        command_cwd=payload["cwd"],
        command_env=payload["env"],
        route_identity=payload["route_identity"],
        command_timeout=payload["timeout"],
        output_class=payload["output_class"],
        required_worker_capabilities=[STRUCTURED_COMMAND_CAPABILITY],
    )
    job_dir = queue.submit(request)
    response = {
        "schema": "gpu-greenroom.command-submission.v1",
        "job_id": request.job_id,
        "status": "pending",
        "effective_queue_dir": str(queue.queue_dir.resolve()),
        "request_path": str((job_dir / "request.json").resolve()),
        "output_dir": request.output_dir,
        "agent_id": request.agent_id,
        "route_identity": request.route_identity,
        "output_class": request.output_class,
        "required_worker_capabilities": request.required_worker_capabilities,
    }
    print(json.dumps(response, indent=2))


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


def cmd_operator(args):
    from .operator_server import serve

    serve(
        args.queue_dir,
        port=args.port,
        read_only=args.read_only,
        admission_control=args.admission_control,
        identity_label=args.identity_label,
        local_operator=args.local_operator,
    )


def _load_job_types(queue_dir):
    """Load job types from defaults + config file. Called per-job so new types are picked up live."""
    job_types = dict(DEFAULT_JOB_TYPES)
    config_path = Path(queue_dir) / "job_types.json"
    if config_path.exists():
        try:
            custom = json.loads(config_path.read_text())
            job_types.update(custom)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not load job_types.json at {config_path}: {e}", file=sys.stderr)
    return job_types


def cmd_worker(args):
    """Run the worker loop — picks and runs jobs sequentially."""
    queue = get_queue(args)
    claimant = worker_identity()

    job_types = _load_job_types(args.queue_dir)

    print(f"GPU Greenroom Worker starting")
    print(f"  Queue dir: {args.queue_dir}")
    print(f"  Job types: {', '.join(job_types.keys())}")
    print(f"  Capabilities: {', '.join(claimant['capabilities']) or '(none)'}")
    print(f"  Source root: {claimant['source_root']}")
    print(f"  Commit: {claimant['commit'] or '(unavailable)'}")
    print(f"  Git dirty: {claimant['git_dirty']}")
    print(f"  Poll interval: {args.poll}s")

    # Recover stale jobs on startup
    recovered = queue.recover_stale()
    if recovered:
        print(f"  Recovered {len(recovered)} stale job(s): {', '.join(recovered)}")

    def stop_worker(_signum, _frame):
        queue.request_worker_shutdown()

    previous_sigint = signal.signal(signal.SIGINT, stop_worker)
    previous_sigterm = signal.signal(signal.SIGTERM, stop_worker)
    was_paused = False
    try:
        while not queue.worker_shutdown_requested:
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
            ran = queue.run_one(job_types, claimant=claimant)
            if ran:
                # Check for more immediately
                continue
            time.sleep(args.poll)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nWorker stopped.")
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def cmd_pause(args):
    queue = get_queue(args)
    acknowledgement = queue.pause(owner=args.owner, epoch=args.epoch)
    print(json.dumps(acknowledgement, indent=2))


def cmd_resume(args):
    queue = get_queue(args)
    try:
        acknowledgement = queue.resume(owner=args.owner, epoch=args.epoch)
    except PauseStateError as error:
        report = {
            "schema": "gpu-greenroom.pause-control-failure.v1",
            "status": "failed",
            "failure_phase": "pause-epoch-mismatch",
            "action": "resume",
            "owner": args.owner,
            "requested_epoch": error.expected_epoch,
            "observed_epoch": error.observed_epoch,
            "effective_queue_dir": str(queue.queue_dir.resolve()),
            "error_message": str(error),
        }
        print(json.dumps(report, indent=2), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(acknowledgement, indent=2))


def cmd_recover(args):
    queue = get_queue(args)
    recovered = queue.recover_stale()
    if recovered:
        print(f"Recovered {len(recovered)} stale job(s):")
        for jid in recovered:
            print(f"  {jid}")
    else:
        print("No stale jobs found.")


def cmd_doctor(args):
    queue = get_queue(args)
    cli_executable = shutil.which("gpu-greenroom")
    probe_path = queue.queue_dir / f".doctor-write-{os.getpid()}"
    checks = {
        "cli_import": {"ok": callable(main)},
        "cli_executable": {
            "ok": cli_executable is not None,
            "requested": "gpu-greenroom",
            "effective": cli_executable,
        },
        "queue_writable": {"ok": False},
        "worker_dispatch_available": {
            "ok": callable(queue.run_one),
            "claim": "python-callable-present; no workload dispatched",
            "capabilities": sorted(effective_worker_capabilities()),
        },
    }
    try:
        probe_path.write_text("ok")
        checks["queue_writable"]["ok"] = probe_path.read_text() == "ok"
    except OSError as error:
        checks["queue_writable"]["error"] = str(error)
    finally:
        probe_path.unlink(missing_ok=True)

    pending = queue.list_jobs(JobStatus.PENDING)
    running = queue.list_jobs(JobStatus.RUNNING)
    report = {
        "schema": "gpu-greenroom.doctor.v1",
        "healthy": all(check["ok"] for check in checks.values()),
        "effective_queue_dir": str(queue.queue_dir.resolve()),
        "cli_executable": cli_executable,
        "checks": checks,
        "queue": {
            "paused": queue.is_paused(),
            "pending": len(pending),
            "running": len(running),
        },
    }
    print(json.dumps(report, indent=2))
    if not report["healthy"]:
        sys.exit(1)


def get_registry(args):
    return QueueRegistry(args.registry)


def cmd_queues_register(args):
    entry = get_registry(args).register(
        name=args.name,
        queue_dir=args.registered_queue_dir,
        contention_class=args.contention_class,
    )
    print(json.dumps(entry, indent=2))


def cmd_queues_status(args):
    registry = get_registry(args)
    report = {
        "schema": "gpu-greenroom.aggregate-status.v1",
        "registry_path": str(registry.path),
        "queues": registry.status(args.contention_class),
    }
    print(json.dumps(report, indent=2))


def cmd_queues_pause(args):
    try:
        report = get_registry(args).pause(
            args.contention_class,
            owner=args.owner,
            epoch=args.epoch,
        )
    except QueueControlError as error:
        print(json.dumps(error.report, indent=2), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(report, indent=2))


def cmd_queues_resume(args):
    try:
        report = get_registry(args).resume(
            args.contention_class,
            owner=args.owner,
            epoch=args.epoch,
        )
    except QueueControlError as error:
        print(json.dumps(error.report, indent=2), file=sys.stderr)
        sys.exit(2)
    print(json.dumps(report, indent=2))


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


def cmd_gc(args):
    import time as _time
    queue_dir = args.queue_dir
    if args.apply:
        if not args.epoch or not args.owner:
            print("gc --apply requires --epoch <candidates-epoch> and --owner <who>", file=sys.stderr)
            sys.exit(2)
        try:
            summary = gc_mod.apply(queue_dir, epoch=args.epoch, owner=args.owner, now=_time.time())
        except gc_mod.GCRefused as exc:
            print(json.dumps({"status": "refused", "error_message": str(exc)}, indent=2), file=sys.stderr)
            sys.exit(1)
        print(json.dumps(summary, indent=2))
        return
    rows = gc_mod.scan(queue_dir, _load_job_types(queue_dir), now=_time.time(), compute_size=not args.no_size)
    try:
        doc = gc_mod.write_candidates(queue_dir, rows, now=_time.time(), grace_hours=args.grace_hours, authority=args.authority)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    gib = lambda b: (b or 0) / 1073741824
    warnings = []
    if not args.no_size and gib(doc["totals"]["total_bytes"]) > args.warn_over_gib:
        warnings.append(f"outputs/ holds {gib(doc['totals']['total_bytes']):.1f} GiB, over the {args.warn_over_gib:g} GiB threshold")
    try:
        free_gib = shutil.disk_usage(queue_dir).free / 1073741824
        if free_gib < args.free_floor_gib:
            warnings.append(f"free space {free_gib:.1f} GiB is below the {args.free_floor_gib:g} GiB floor")
    except OSError:
        pass
    if args.json:
        print(json.dumps({k: v for k, v in doc.items() if k != "rows"} | {"candidates": [r for r in rows if r["candidate"]], "warnings": warnings}, indent=2))
        return
    totals = doc["totals"]
    print(f"epoch {doc['epoch']}  entries {totals['entry_count']}  total {gib(totals['total_bytes']):.1f} GiB")
    print(f"candidates {totals['candidate_count']} ({gib(totals['candidate_bytes']):.1f} GiB, of which graduated {totals['graduated_count']} / {gib(totals['graduated_bytes']):.1f} GiB)  unclassified {totals['unclassified_count']} ({gib(totals['unclassified_bytes']):.1f} GiB)  pinned {totals['pinned_count']}  active {totals['active_count']}")
    for r in sorted((r for r in rows if r["candidate"]), key=lambda r: -(r["size_bytes"] or 0))[:40]:
        print(f"  {gib(r['size_bytes']):7.2f} GiB  {r['output_class']:12s} {r['age_days']:6.0f} d  {r['owner'] or 'not recorded':24s} {r['name']}")
    for w in warnings:
        print(f"warning: {w}")
    print(f"notices per owner: {Path(queue_dir) / 'gc-notices' / doc['epoch']}")
    print(f"apply after {_time.strftime('%Y-%m-%d %H:%M', _time.localtime(doc['apply_not_before']))} with: gpu-greenroom gc --apply --epoch {doc['epoch']} --owner <who>")


def cmd_retain(args):
    pins = gc_mod.RetentionPins(args.queue_dir)
    if args.list:
        print(json.dumps(pins.entries(), indent=2))
        return
    if not args.name:
        print("retain requires a name (or --list)", file=sys.stderr)
        sys.exit(2)
    if args.unpin:
        print(json.dumps({"name": args.name, "unpinned": pins.unpin(args.name)}))
        return
    if not args.owner or not args.reason:
        print("retain requires --owner and --reason", file=sys.stderr)
        sys.exit(2)
    until = None
    if args.until:
        from datetime import datetime
        try:
            until = datetime.fromisoformat(args.until).timestamp()
        except ValueError:
            print(f"retain --until must be an ISO date or datetime, got {args.until!r}", file=sys.stderr)
            sys.exit(2)
    try:
        entry = pins.pin(args.name, owner=args.owner, reason=args.reason, until=until)
    except (ValueError, gc_mod.PinsUnreadable) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    print(json.dumps({"name": args.name, **entry}, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="gpu-greenroom",
        description="Filesystem-backed GPU job queue with flock serialization",
    )
    parser.add_argument("--version", action="version", version=f"gpu-greenroom {_package_version()}")
    parser.add_argument(
        "--queue-dir", default=DEFAULT_QUEUE_DIR,
        help=f"Queue directory (default: {DEFAULT_QUEUE_DIR})",
    )
    parser.add_argument(
        "--registry", default=DEFAULT_REGISTRY_PATH,
        help=f"Registered queue adapters (default: {DEFAULT_REGISTRY_PATH})",
    )

    sub = parser.add_subparsers(dest="command")

    # submit
    p_submit = sub.add_parser("submit", help="Submit a job")
    p_submit.add_argument("job_type", help="Job type: a key in job_types.json, or the built-in 'echo' smoke type")
    p_submit.add_argument("input", help="Input file path")
    p_submit.add_argument("output_dir", nargs="?", default="", help="Output directory (default: durable path in queue dir)")
    p_submit.add_argument("-p", "--params", nargs="*", help="Key=value params (e.g. seed=42)")
    p_submit.add_argument("--cwd", help="Override working directory (e.g. for branch/worktree)")
    p_submit.add_argument(
        "--agent-id",
        type=_agent_id_arg,
        help="Exact owning agent identity; omitted identity remains not recorded",
    )
    p_submit.set_defaults(func=cmd_submit)

    # submit-command
    p_command = sub.add_parser(
        "submit-command",
        help="Submit an exact structured argv without editing job_types.json",
    )
    p_command.add_argument("--manifest", help="gpu-greenroom.command.v1 JSON manifest")
    p_command.add_argument(
        "--agent-id",
        help="Exact owning agent identity for flag-based submission",
    )
    p_command.add_argument("--repo-root")
    p_command.add_argument("--cwd")
    p_command.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    p_command.add_argument("--output-dir", default="")
    p_command.add_argument("--route-identity")
    p_command.add_argument("--output-class", choices=list(gc_mod.CLASSES), help="Retention class for the output directory (final, witness, intermediate)")
    p_command.add_argument("--timeout", type=float)
    p_command.add_argument("argv", nargs=argparse.REMAINDER)
    p_command.set_defaults(func=cmd_submit_command)

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

    # operator console
    p_operator = sub.add_parser("operator", help="Run the authenticated localhost operator console")
    p_operator.add_argument("--port", type=int, default=8765)
    p_operator.add_argument("--identity-label", default="Agent")
    p_operator.add_argument("--local-operator", action="store_true")
    operator_modes = p_operator.add_mutually_exclusive_group()
    operator_modes.add_argument("--read-only", action="store_true")
    operator_modes.add_argument("--admission-control", action="store_true")
    p_operator.set_defaults(func=cmd_operator)

    # worker
    p_worker = sub.add_parser("worker", help="Run the worker loop")
    p_worker.add_argument("--poll", type=float, default=2.0, help="Poll interval in seconds")
    p_worker.set_defaults(func=cmd_worker)

    # pause
    p_pause = sub.add_parser("pause", help="Pause the queue (finish current job, then wait)")
    p_pause.add_argument("--owner", default="local-cli")
    p_pause.add_argument("--epoch")
    p_pause.set_defaults(func=cmd_pause)

    # resume
    p_resume = sub.add_parser("resume", help="Resume a paused queue")
    p_resume.add_argument("--owner", default="local-cli")
    p_resume.add_argument("--epoch")
    p_resume.set_defaults(func=cmd_resume)

    # recover
    p_recover = sub.add_parser("recover", help="Recover stale running jobs")
    p_recover.set_defaults(func=cmd_recover)

    # doctor
    p_doctor = sub.add_parser("doctor", help="Check CLI, queue, and worker dispatch availability")
    p_doctor.add_argument("--json", action="store_true")
    p_doctor.set_defaults(func=cmd_doctor)

    # registered queue aggregate controls
    p_queues = sub.add_parser("queues", help="Register and control participating queues")
    queues_sub = p_queues.add_subparsers(dest="queues_command")

    p_queues_register = queues_sub.add_parser("register", help="Register a queue adapter")
    p_queues_register.add_argument("name")
    p_queues_register.add_argument("--queue-dir", dest="registered_queue_dir", required=True)
    p_queues_register.add_argument("--contention-class", required=True)
    p_queues_register.set_defaults(func=cmd_queues_register)

    p_queues_status = queues_sub.add_parser("status", help="Aggregate native queue status")
    p_queues_status.add_argument("--contention-class")
    p_queues_status.set_defaults(func=cmd_queues_status)

    p_queues_pause = queues_sub.add_parser(
        "pause", help="Pause queued-to-running transitions for a contention class"
    )
    p_queues_pause.add_argument("--contention-class", required=True)
    p_queues_pause.add_argument("--owner", required=True)
    p_queues_pause.add_argument("--epoch")
    p_queues_pause.set_defaults(func=cmd_queues_pause)

    p_queues_resume = queues_sub.add_parser(
        "resume", help="Resume queued-to-running transitions for a contention class"
    )
    p_queues_resume.add_argument("--contention-class", required=True)
    p_queues_resume.add_argument("--owner", required=True)
    p_queues_resume.add_argument("--epoch", required=True)
    p_queues_resume.set_defaults(func=cmd_queues_resume)

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

    # gc / retain
    p_gc = sub.add_parser("gc", help="Retention: dry-run lists past-TTL outputs into an epoch-bound candidate list; apply deletes exactly that list after the grace window")
    gc_mode = p_gc.add_mutually_exclusive_group(required=True)
    gc_mode.add_argument("--dry-run", action="store_true")
    gc_mode.add_argument("--apply", action="store_true")
    p_gc.add_argument("--epoch", help="Candidate-list epoch to apply")
    p_gc.add_argument("--owner", help="Who is applying (recorded in every receipt)")
    p_gc.add_argument("--grace-hours", type=float, default=72.0)
    p_gc.add_argument("--authority", help="Who approved this collection (recorded in candidates and receipts)")
    p_gc.add_argument("--no-size", action="store_true", help="Skip per-directory size computation")
    p_gc.add_argument("--warn-over-gib", type=float, default=150.0, help="Warn when outputs/ exceeds this size (diagnostic only)")
    p_gc.add_argument("--free-floor-gib", type=float, default=100.0, help="Warn when free space is below this (diagnostic only)")
    p_gc.add_argument("--json", action="store_true")
    p_gc.set_defaults(func=cmd_gc)

    p_retain = sub.add_parser("retain", help="Pin an outputs/ entry so gc never collects it")
    p_retain.add_argument("name", nargs="?")
    p_retain.add_argument("--owner")
    p_retain.add_argument("--reason")
    p_retain.add_argument("--until", help="ISO date/time after which the pin lapses")
    p_retain.add_argument("--list", action="store_true")
    p_retain.add_argument("--unpin", action="store_true")
    p_retain.set_defaults(func=cmd_retain)

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
    if args.command == "queues" and not getattr(args, "queues_command", None):
        p_queues.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
