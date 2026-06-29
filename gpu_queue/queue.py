"""Filesystem-backed GPU job queue with flock serialization."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .models import JobRequest, JobState, JobStatus


class GPUQueue:
    """Filesystem-backed job queue.

    Directory layout:
        queue_dir/
            pending/     - submitted jobs awaiting execution
            running/     - at most one job being executed
            done/        - completed jobs
            failed/      - failed jobs
            cancelled/   - cancelled jobs
    """

    VOLATILE_PREFIXES = ("/tmp", "/private/tmp", "/var/tmp")
    STATUS_DIRS = ("pending", "running", "done", "failed", "checkpoint_paused", "cancelled")
    SCHEDULE_SCHEMA = "gpu-greenroom.schedule.v1"
    INDEX_ROW_SCHEMA = "gpu-greenroom.index-row.v1"
    QUEUE_INDEX_SCHEMA = "gpu-greenroom.queue-index.v1"
    ROUTE_JOB_SCHEMA = "kaminos.route-job.v0"
    TRELLIS_CHECKPOINT_YIELD_SCHEMA = "trellis2mlx.checkpoint_yield.v1"
    TRELLIS_CHECKPOINT_YIELD_EXIT_CODE = 75
    CHECKPOINT_PAUSE_REQUEST_SCHEMA = "gpu-greenroom.checkpoint-pause-request.v1"
    PRIORITY_RANKS = {
        "preview": 0,
        "hero": 1,
        "normal": 2,
        "background": 3,
    }

    def __init__(self, queue_dir: str | Path):
        self.queue_dir = Path(queue_dir)
        for sub in ("pending", "running", "done", "failed", "checkpoint_paused", "cancelled", "outputs"):
            (self.queue_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def lock_path(self) -> Path:
        return self.queue_dir / "gpu.lock"

    @property
    def pause_path(self) -> Path:
        return self.queue_dir / "paused"

    def pause(self) -> None:
        """Pause the queue. The worker finishes its current job then waits."""
        self.pause_path.touch()

    def resume(self) -> None:
        """Resume a paused queue."""
        self.pause_path.unlink(missing_ok=True)

    def is_paused(self) -> bool:
        return self.pause_path.exists()

    def _is_volatile(self, path: str) -> bool:
        resolved = str(Path(path).resolve())
        return any(resolved == p or resolved.startswith(p + "/")
                    for p in self.VOLATILE_PREFIXES)

    def _priority_rank(self, priority_class: str | None) -> int:
        return self.PRIORITY_RANKS.get(priority_class or "normal", self.PRIORITY_RANKS["normal"])

    def _default_schedule(self, request_or_state: JobRequest | JobState) -> dict[str, Any]:
        params = getattr(request_or_state, "params", {}) or {}
        priority_class = str(params.get("priority_class") or params.get("priorityClass") or "normal")
        return {
            "schema": self.SCHEDULE_SCHEMA,
            "priority_class": priority_class,
            "submitted_at": getattr(request_or_state, "submitted_at", 0.0),
        }

    def _job_control_paths(self, request: JobRequest) -> dict[str, str]:
        params = request.params or {}
        checkpoint_dir = (
            params.get("checkpoint_dir")
            or params.get("checkpointDir")
            or str(Path(request.output_dir) / "checkpoints")
        )
        checkpoint_stop_file = (
            params.get("checkpoint_stop_file")
            or params.get("checkpointStopFile")
            or str(Path(request.output_dir) / "_control" / "checkpoint-stop")
        )
        return {
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_stop_file": str(checkpoint_stop_file),
        }

    def _write_schedule(self, job_dir: Path, request: JobRequest) -> dict[str, Any]:
        schedule = self._default_schedule(request)
        (job_dir / "schedule.json").write_text(json.dumps(schedule, indent=2))
        return schedule

    def _read_schedule(self, job_dir: Path, state: JobState | None = None) -> dict[str, Any]:
        schedule_file = job_dir / "schedule.json"
        if schedule_file.exists():
            try:
                schedule = json.loads(schedule_file.read_text())
            except (json.JSONDecodeError, OSError):
                schedule = {}
        else:
            schedule = {}

        fallback = self._default_schedule(state) if state is not None else {
            "schema": self.SCHEDULE_SCHEMA,
            "priority_class": "normal",
            "submitted_at": 0.0,
        }
        return {
            "schema": schedule.get("schema") or self.SCHEDULE_SCHEMA,
            "priority_class": schedule.get("priority_class") or schedule.get("priorityClass") or fallback["priority_class"],
            "submitted_at": schedule.get("submitted_at") or schedule.get("submittedAt") or fallback["submitted_at"],
        }

    def submit(self, request: JobRequest) -> Path:
        """Submit a job. Returns the job directory path.

        If output_dir is empty, auto-assigns a durable path under
        queue_dir/outputs/<job_id>/. If output_dir is under /tmp or
        /private/tmp, records a volatile_output warning.
        """
        if not request.output_dir:
            request.output_dir = str(self.queue_dir / "outputs" / request.job_id)

        job_dir = self.queue_dir / "pending" / request.job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Write request
        (job_dir / "request.json").write_text(request.to_json())
        self._write_schedule(job_dir, request)

        warnings = []
        if self._is_volatile(request.output_dir):
            warnings.append("volatile_output")

        # Write initial status
        state = JobState(
            job_id=request.job_id,
            status=JobStatus.PENDING,
            job_type=request.job_type,
            input_path=request.input_path,
            output_dir=request.output_dir,
            params=request.params,
            submitted_at=request.submitted_at,
            warnings=warnings,
        )
        (job_dir / "status.json").write_text(state.to_json())

        return job_dir

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job. Returns True if cancelled, False if not found/not pending.

        Acquires flock to prevent race with run_one().
        """
        lock_fd = open(self.lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            pending_dir = self.queue_dir / "pending" / job_id
            if not pending_dir.exists():
                return False

            state = JobState.from_json((pending_dir / "status.json").read_text())
            state.status = JobStatus.CANCELLED
            state.finished_at = time.time()
            (pending_dir / "status.json").write_text(state.to_json())

            dest = self.queue_dir / "cancelled" / job_id
            shutil.move(str(pending_dir), str(dest))
            return True
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def list_jobs(self, status: JobStatus | None = None) -> list[JobState]:
        """List jobs, optionally filtered by status."""
        results = []
        dirs = list(self.STATUS_DIRS)
        if status:
            dirs = [status.value]
        for sub in dirs:
            sub_dir = self.queue_dir / sub
            if not sub_dir.exists():
                continue
            for job_dir in sorted(sub_dir.iterdir()):
                status_file = job_dir / "status.json"
                if status_file.exists():
                    try:
                        results.append(JobState.from_json(status_file.read_text()))
                    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                        continue
        return results

    def queue_index_payload(self, status: JobStatus | None = None) -> dict[str, Any]:
        return {
            "schema": self.QUEUE_INDEX_SCHEMA,
            "queue_dir": str(self.queue_dir),
            "rows": self.index_jobs(status),
        }

    def index_jobs(self, status: JobStatus | None = None) -> list[dict[str, Any]]:
        """Return a tolerant JSON index for control-plane consumers.

        Unlike list_jobs(), malformed or legacy status rows are preserved as
        degraded evidence so Kaminos can show that a route exists without
        pretending the old row matches the current receipt model.
        """
        rows: list[dict[str, Any]] = []
        dirs = [status.value] if status else list(self.STATUS_DIRS)
        for sub in dirs:
            sub_dir = self.queue_dir / sub
            if not sub_dir.exists():
                continue
            for job_dir in sorted(sub_dir.iterdir()):
                status_file = job_dir / "status.json"
                if not status_file.exists():
                    continue
                rows.append(self._index_row(job_dir, sub, status_file))
        return rows

    def _index_row(self, job_dir: Path, status_dir: str, status_file: Path) -> dict[str, Any]:
        try:
            raw = json.loads(status_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            return self._degraded_index_row(job_dir, status_dir, {}, exc)

        try:
            state = JobState.from_json(json.dumps(raw))
        except (KeyError, ValueError, TypeError) as exc:
            return self._degraded_index_row(job_dir, status_dir, raw, exc)

        schedule = self._read_schedule(job_dir, state)
        pause_request = self._read_checkpoint_pause_request(job_dir)
        return {
            "schema": self.INDEX_ROW_SCHEMA,
            "job_id": state.job_id,
            "status": state.status.value,
            "status_dir": status_dir,
            "job_type": state.job_type,
            "submitted_at": state.submitted_at,
            "started_at": state.started_at,
            "finished_at": state.finished_at,
            "output_dir": state.output_dir,
            "effective_route": state.effective_route,
            "worker_pid": state.worker_pid,
            "child_pid": state.child_pid,
            "process_group_id": state.process_group_id,
            "schedule": schedule,
            "parse_error": None,
            "checkpoint_pause_request": pause_request,
            "route_job": self._route_job_for_state(state, schedule, job_dir, status_dir, pause_request),
        }

    def _degraded_index_row(
        self,
        job_dir: Path,
        status_dir: str,
        raw: dict[str, Any],
        exc: BaseException,
    ) -> dict[str, Any]:
        job_id = raw.get("job_id") or raw.get("jobId") or job_dir.name
        job_type = raw.get("job_type") or raw.get("jobType")
        schedule = self._read_schedule(job_dir)
        return {
            "schema": self.INDEX_ROW_SCHEMA,
            "job_id": job_id,
            "status": "degraded",
            "status_dir": status_dir,
            "job_type": job_type,
            "submitted_at": raw.get("submitted_at") or raw.get("submittedAt"),
            "started_at": raw.get("started_at") or raw.get("startedAt"),
            "finished_at": raw.get("finished_at") or raw.get("finishedAt"),
            "output_dir": raw.get("output_dir") or raw.get("outputDir"),
            "effective_route": raw.get("effective_route") or raw.get("effectiveRoute"),
            "schedule": schedule,
            "parse_error": str(exc),
            "legacy_status": raw,
            "route_job": {
                "schema": self.ROUTE_JOB_SCHEMA,
                "id": job_id,
                "routeId": job_type or "unknown",
                "executor": {
                    "kind": "native-greenroom",
                    "id": "gpu-greenroom",
                    "nativeQueueDir": str(self.queue_dir),
                },
                "priorityClass": schedule["priority_class"],
                "status": "degraded",
                "inputArtifacts": [],
                "outputPolicy": None,
                "resumability": {"kind": "unknown"},
                "native": {
                    "greenroom_job_id": job_id,
                    "status_dir": status_dir,
                    "job_dir": str(job_dir),
                },
            },
        }

    def _route_job_for_state(
        self,
        state: JobState,
        schedule: dict[str, Any],
        job_dir: Path,
        status_dir: str,
        checkpoint_pause_request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        input_artifacts = []
        if state.input_path:
            input_artifacts.append({"role": "input", "path": state.input_path})
        return {
            "schema": self.ROUTE_JOB_SCHEMA,
            "id": state.job_id,
            "routeId": state.job_type,
            "executor": {
                "kind": "native-greenroom",
                "id": "gpu-greenroom",
                "nativeQueueDir": str(self.queue_dir),
            },
            "priorityClass": schedule["priority_class"],
            "status": state.status.value,
            "inputArtifacts": input_artifacts,
            "outputPolicy": {
                "root": state.output_dir,
                "mode": "caller-owned",
            },
            "resumability": self._resumability_for_state(state, checkpoint_pause_request),
            "native": {
                "greenroom_job_id": state.job_id,
                "status_dir": status_dir,
                "job_dir": str(job_dir),
                "output_dir": state.output_dir,
                **self._native_checkpoint_pause_request_fields(checkpoint_pause_request),
                **self._native_checkpoint_fields(state),
            },
        }

    def _resumability_for_state(
        self,
        state: JobState,
        checkpoint_pause_request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if state.checkpoint_yield:
            receipt = state.checkpoint_yield
            resumability = {
                "kind": "cooperative-checkpoint",
                "state": receipt.get("status"),
                "completedStage": receipt.get("completed_stage"),
                "nextStage": receipt.get("next_stage"),
                "resumeSupported": bool(receipt.get("resume_supported")),
                "checkpointReceipt": receipt.get("receipt_path"),
                "pauseRequested": bool(checkpoint_pause_request),
            }
            if receipt.get("resume_blocker"):
                resumability["resumeBlocker"] = receipt.get("resume_blocker")
            if receipt.get("resume_command_hint"):
                resumability["resumeCommandHint"] = receipt.get("resume_command_hint")
            return resumability
        if checkpoint_pause_request:
            return {
                "kind": "cooperative-checkpoint",
                "state": "pause_requested",
                "pauseRequested": True,
                "resumeSupported": False,
                "checkpointStopFile": checkpoint_pause_request.get("checkpoint_stop_file"),
                "pauseRequestReceipt": checkpoint_pause_request.get("receipt_path"),
            }
        return {"kind": "unknown"}

    def _native_checkpoint_fields(self, state: JobState) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if state.checkpoint_dir:
            fields["checkpoint_dir"] = state.checkpoint_dir
        if state.checkpoint_stop_file:
            fields["checkpoint_stop_file"] = state.checkpoint_stop_file
        if state.checkpoint_yield and state.checkpoint_yield.get("receipt_path"):
            fields["checkpoint_yield_receipt"] = state.checkpoint_yield["receipt_path"]
        return fields

    def _native_checkpoint_pause_request_fields(
        self,
        checkpoint_pause_request: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not checkpoint_pause_request:
            return {}
        fields = {
            "checkpoint_pause_requested": True,
        }
        if checkpoint_pause_request.get("receipt_path"):
            fields["checkpoint_pause_request_receipt"] = checkpoint_pause_request["receipt_path"]
        if checkpoint_pause_request.get("checkpoint_stop_file"):
            fields["checkpoint_stop_file"] = checkpoint_pause_request["checkpoint_stop_file"]
        return fields

    def _read_checkpoint_pause_request(self, job_dir: Path) -> dict[str, Any] | None:
        receipt_path = job_dir / "_control" / "checkpoint_pause_request.json"
        try:
            receipt = json.loads(receipt_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if receipt.get("schema") != self.CHECKPOINT_PAUSE_REQUEST_SCHEMA:
            return None
        if receipt.get("status") != "requested":
            return None
        receipt.setdefault("receipt_path", str(receipt_path))
        return receipt

    def _load_checkpoint_yield_receipt(self, checkpoint_dir: str | None) -> dict[str, Any] | None:
        if not checkpoint_dir:
            return None
        receipt_path = Path(checkpoint_dir) / "_control" / "checkpoint_yield.json"
        try:
            receipt = json.loads(receipt_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if receipt.get("schema") != self.TRELLIS_CHECKPOINT_YIELD_SCHEMA:
            return None
        if receipt.get("status") != "paused_at_checkpoint":
            return None
        if receipt.get("exit_code") != self.TRELLIS_CHECKPOINT_YIELD_EXIT_CODE:
            return None
        if not receipt.get("completed_stage"):
            return None
        receipt.setdefault("receipt_path", str(receipt_path))
        return receipt

    def get_job(self, job_id: str) -> JobState | None:
        """Get a specific job's state."""
        for sub in self.STATUS_DIRS:
            status_file = self.queue_dir / sub / job_id / "status.json"
            if status_file.exists():
                return JobState.from_json(status_file.read_text())
        return None

    def _next_pending(self) -> Path | None:
        """Get the next pending job directory by schedule priority then age."""
        pending = self.queue_dir / "pending"
        jobs = []
        for job_dir in pending.iterdir():
            status_file = job_dir / "status.json"
            if status_file.exists():
                state = JobState.from_json(status_file.read_text())
                schedule = self._read_schedule(job_dir, state)
                jobs.append((
                    self._priority_rank(schedule["priority_class"]),
                    schedule["submitted_at"],
                    state.submitted_at,
                    job_dir,
                ))
        if not jobs:
            return None
        jobs.sort(key=lambda x: (x[0], x[1], x[2], x[3].name))
        return jobs[0][3]

    def request_checkpoint_pause(self, job_id: str) -> dict[str, Any] | None:
        """Request cooperative checkpoint-and-exit for a pending or running job.

        This writes the job's stop file and a request receipt. It does not kill,
        suspend, reorder, or directly modify terminal jobs.
        """
        located = self._find_checkpoint_pause_target(job_id)
        if located is None:
            return None
        job_dir, status_dir = located

        request_file = job_dir / "request.json"
        status_file = job_dir / "status.json"
        state = JobState.from_json(status_file.read_text())
        request = JobRequest.from_json(request_file.read_text()) if request_file.exists() else None

        if not self._cooperative_checkpoint_pause_capable(state.job_type):
            raise ValueError(f"Job type {state.job_type!r} does not advertise cooperative checkpoint pause")

        if request is not None:
            control_paths = self._job_control_paths(request)
        else:
            control_paths = {
                "checkpoint_dir": state.checkpoint_dir,
                "checkpoint_stop_file": state.checkpoint_stop_file,
            }
        checkpoint_stop_file = control_paths.get("checkpoint_stop_file") or state.checkpoint_stop_file
        checkpoint_dir = control_paths.get("checkpoint_dir") or state.checkpoint_dir
        if not checkpoint_stop_file:
            raise ValueError(f"Job {job_id} has no checkpoint stop file")

        stop_path = Path(checkpoint_stop_file)
        stop_path.parent.mkdir(parents=True, exist_ok=True)
        requested_at = time.time()

        receipt_path = job_dir / "_control" / "checkpoint_pause_request.json"
        receipt = {
            "schema": self.CHECKPOINT_PAUSE_REQUEST_SCHEMA,
            "status": "requested",
            "job_id": job_id,
            "job_type": state.job_type,
            "job_status_at_request": state.status.value,
            "requested_at": requested_at,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_stop_file": str(stop_path),
            "receipt_path": str(receipt_path),
            "request_semantics": "cooperative_stop_after_next_checkpoint",
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_receipt = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
        tmp_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(tmp_receipt, receipt_path)

        tmp_stop = stop_path.with_suffix(stop_path.suffix + ".tmp")
        tmp_stop.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(tmp_stop, stop_path)

        if state.status == JobStatus.PENDING:
            state.checkpoint_dir = checkpoint_dir
            state.checkpoint_stop_file = str(stop_path)
            state.checkpoint_pause_request = receipt
            status_file.write_text(state.to_json())
        return receipt

    def _find_checkpoint_pause_target(self, job_id: str) -> tuple[Path, str] | None:
        for status_dir in ("running", "pending"):
            job_dir = self.queue_dir / status_dir / job_id
            if (job_dir / "status.json").exists():
                return job_dir, status_dir
        return None

    def _cooperative_checkpoint_pause_capable(self, job_type: str) -> bool:
        return job_type == "trellis2mlx" or job_type.startswith("trellis2mlx.")

    def _write_metadata_sidecar(self, request: JobRequest, state: JobState):
        """Write metadata.json into output_dir for asset browser consumption."""
        out = Path(request.output_dir)
        if not out.is_dir():
            return

        # Derive a human-readable name from input filename if not provided
        name = request.params.get("name", "")
        if not name:
            inp = Path(request.input_path)
            name = inp.stem  # e.g. "dragon" from "dragon.png"

        # Collect output files
        output_files = [
            f.name for f in sorted(out.iterdir())
            if f.is_file() and not f.name.startswith(".")
            and f.name != "metadata.json"
        ]

        metadata = {
            "name": name,
            "job_type": request.job_type,
            "job_id": request.job_id,
            "input_path": request.input_path,
            "input_name": Path(request.input_path).name,
            "params": request.params,
            "output_files": output_files,
            "created_at": state.finished_at,
            "duration_s": round(state.finished_at - state.started_at, 1)
            if state.started_at and state.finished_at else None,
        }

        (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def _move_job(self, job_dir: Path, dest_status: str) -> Path:
        """Move a job directory to a new status folder."""
        dest = self.queue_dir / dest_status / job_dir.name
        shutil.move(str(job_dir), str(dest))
        return dest

    def run_one(self, job_types: dict[str, list[str]]) -> bool:
        """Pick and run the next pending job under flock.

        job_types: mapping of job_type name -> command template list.
            Template strings may contain {input_path}, {output_dir}, and
            any key from params.

        Returns True if a job was run, False if queue was empty.
        """
        if self.is_paused():
            return False

        lock_fd = open(self.lock_path, "w")
        try:
            # Non-blocking lock attempt
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_fd.close()
            return False

        try:
            if self.is_paused():
                return False

            job_dir = self._next_pending()
            if job_dir is None:
                return False

            request = JobRequest.from_json((job_dir / "request.json").read_text())

            # Move to running
            job_dir = self._move_job(job_dir, "running")

            # Update status
            state = JobState.from_json((job_dir / "status.json").read_text())
            state.status = JobStatus.RUNNING
            state.started_at = time.time()
            state.pid = os.getpid()
            state.worker_pid = os.getpid()
            control_paths = self._job_control_paths(request)
            state.checkpoint_dir = control_paths["checkpoint_dir"]
            state.checkpoint_stop_file = control_paths["checkpoint_stop_file"]

            # Resolve job type config — supports bare list or rich dict
            raw_config = job_types.get(request.job_type)
            if raw_config is None:
                state.status = JobStatus.FAILED
                state.finished_at = time.time()
                state.failure_phase = "dispatch"
                state.error_message = f"Unknown job type: {request.job_type}"
                state.exit_code = -1
                (job_dir / "status.json").write_text(state.to_json())
                self._move_job(job_dir, "failed")
                return True

            if isinstance(raw_config, list):
                # Bare list: backwards compat
                cmd_template = raw_config
                job_cwd = None
                job_env = None
                job_defaults = {}
                job_timeout = None
            else:
                # Rich dict config
                cmd_template = raw_config["cmd"]
                job_cwd = raw_config.get("cwd")
                job_env = raw_config.get("env")
                job_defaults = raw_config.get("defaults", {})
                job_timeout = raw_config.get("timeout")  # None = no timeout

            # Per-job overrides: cwd and env from params (removed before template subs)
            OVERRIDE_KEYS = {"cwd", "env"}
            RESERVED = {"input_path", "output_dir"}
            if "cwd" in request.params:
                job_cwd = request.params["cwd"]
            if "env" in request.params and isinstance(request.params.get("env"), dict):
                job_env = {**(job_env or {}), **request.params["env"]}
            safe_params = {k: v for k, v in request.params.items() if k not in RESERVED and k not in OVERRIDE_KEYS}
            subs = {
                **job_defaults,
                **safe_params,
                "input_path": request.input_path,
                "output_dir": request.output_dir,
                "job_id": request.job_id,
                **control_paths,
            }

            # Detect ignored params: user-supplied keys not consumed by template
            template_str = " ".join(cmd_template)
            used_keys = set()
            for key in subs:
                if "{" + key + "}" in template_str:
                    used_keys.add(key)
            ignored_params = {
                k: v for k, v in safe_params.items()
                if k not in used_keys and k not in RESERVED
            }

            # Safe substitution: replace all placeholders in one pass to prevent
            # chained expansion (e.g. param value "{input_path}" must stay literal)
            import re

            def safe_substitute(template: str, mapping: dict) -> str:
                def replacer(match):
                    key = match.group(1)
                    if key in mapping:
                        return str(mapping[key])
                    return match.group(0)  # leave unrecognized placeholders as-is
                return re.sub(r'\{(\w+)\}', replacer, template)

            cmd = [safe_substitute(part, subs) for part in cmd_template]
            state.effective_route = " ".join(cmd)
            (job_dir / "status.json").write_text(state.to_json())

            # Ensure output directory exists
            os.makedirs(request.output_dir, exist_ok=True)

            # Build subprocess environment
            run_env = None
            if job_env:
                run_env = {**os.environ, **job_env}

            # Execute
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"

            try:
                with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=out_f,
                        stderr=err_f,
                        cwd=job_cwd,
                        env=run_env,
                        start_new_session=True,
                    )
                    state.child_pid = proc.pid
                    try:
                        state.process_group_id = os.getpgid(proc.pid)
                    except OSError:
                        state.process_group_id = proc.pid
                    (job_dir / "status.json").write_text(state.to_json())
                    returncode = proc.wait(timeout=job_timeout)
                state.exit_code = returncode
                if returncode == 0:
                    state.status = JobStatus.DONE
                    dest_status = "done"
                elif returncode == self.TRELLIS_CHECKPOINT_YIELD_EXIT_CODE and (
                    checkpoint_yield := self._load_checkpoint_yield_receipt(state.checkpoint_dir)
                ):
                    state.status = JobStatus.CHECKPOINT_PAUSED
                    state.checkpoint_yield = checkpoint_yield
                    state.failure_phase = None
                    state.error_message = None
                    dest_status = "checkpoint_paused"
                else:
                    state.status = JobStatus.FAILED
                    state.failure_phase = "execution"
                    state.error_message = f"Process exited with code {returncode}"
                    dest_status = "failed"
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
                state.status = JobStatus.FAILED
                state.failure_phase = "timeout"
                state.error_message = f"Job exceeded {job_timeout}s timeout"
                state.exit_code = -1
                dest_status = "failed"
            except Exception as e:
                state.status = JobStatus.FAILED
                state.failure_phase = "launch"
                state.error_message = str(e)
                state.exit_code = -1
                dest_status = "failed"

            state.finished_at = time.time()
            (job_dir / "status.json").write_text(state.to_json())

            # Write receipt with full route identity
            receipt = {
                "job_id": state.job_id,
                "job_type": state.job_type,
                "status": state.status.value,
                "input_path": state.input_path,
                "output_dir": state.output_dir,
                "effective_route": state.effective_route,
                "effective_cwd": job_cwd,
                "effective_env": job_env,
                "effective_defaults": job_defaults,
                "effective_timeout": job_timeout,
                "ignored_params": ignored_params if ignored_params else None,
                "worker_pid": state.worker_pid,
                "child_pid": state.child_pid,
                "process_group_id": state.process_group_id,
                "checkpoint_dir": state.checkpoint_dir,
                "checkpoint_stop_file": state.checkpoint_stop_file,
                "checkpoint_yield": state.checkpoint_yield,
                "checkpoint_pause_request": state.checkpoint_pause_request,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
                "exit_code": state.exit_code,
                "failure_phase": state.failure_phase,
                "error_message": state.error_message,
                "warnings": state.warnings if state.warnings else None,
            }
            (job_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))

            # Write metadata sidecar into output_dir for asset browsers
            if state.status == JobStatus.DONE:
                self._write_metadata_sidecar(request, state)

            self._move_job(job_dir, dest_status)
            return True

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def recover_stale(self) -> list[str]:
        """Check for stale running jobs (process no longer alive) and move to failed.

        Acquires flock to prevent race with run_one().
        Returns list of recovered job IDs.
        """
        lock_fd = open(self.lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            recovered = []
            running_dir = self.queue_dir / "running"
            if not running_dir.exists():
                return recovered

            for job_dir in list(running_dir.iterdir()):
                status_file = job_dir / "status.json"
                if not status_file.exists():
                    continue
                state = JobState.from_json(status_file.read_text())
                if state.pid is not None:
                    try:
                        os.kill(state.pid, 0)  # check if process is alive
                    except PermissionError:
                        # PID exists but belongs to another user — treat as alive, skip
                        continue
                    except ProcessLookupError:
                        # PID does not exist — stale job
                        state.status = JobStatus.FAILED
                        state.finished_at = time.time()
                        state.failure_phase = "stale_recovery"
                        state.error_message = f"Process {state.pid} no longer alive; recovered by stale detection"
                        state.exit_code = -1
                        (status_file).write_text(state.to_json())

                        receipt = {
                            "job_id": state.job_id,
                            "job_type": state.job_type,
                            "status": "failed",
                            "failure_phase": "stale_recovery",
                            "error_message": state.error_message,
                            "started_at": state.started_at,
                            "finished_at": state.finished_at,
                        }
                        (job_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))

                        self._move_job(job_dir, "failed")
                        recovered.append(state.job_id)
            return recovered
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
