"""Filesystem-backed GPU job queue with flock serialization."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

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

    def __init__(self, queue_dir: str | Path):
        self.queue_dir = Path(queue_dir)
        for sub in ("pending", "running", "done", "failed", "cancelled", "outputs"):
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
        dirs = ["pending", "running", "done", "failed", "cancelled"]
        if status:
            dirs = [status.value]
        for sub in dirs:
            sub_dir = self.queue_dir / sub
            if not sub_dir.exists():
                continue
            for job_dir in sorted(sub_dir.iterdir()):
                status_file = job_dir / "status.json"
                if status_file.exists():
                    results.append(JobState.from_json(status_file.read_text()))
        return results

    def get_job(self, job_id: str) -> JobState | None:
        """Get a specific job's state."""
        for sub in ("pending", "running", "done", "failed", "cancelled"):
            status_file = self.queue_dir / sub / job_id / "status.json"
            if status_file.exists():
                return JobState.from_json(status_file.read_text())
        return None

    def _next_pending(self) -> Path | None:
        """Get the oldest pending job directory."""
        pending = self.queue_dir / "pending"
        jobs = []
        for job_dir in pending.iterdir():
            status_file = job_dir / "status.json"
            if status_file.exists():
                state = JobState.from_json(status_file.read_text())
                jobs.append((state.submitted_at, job_dir))
        if not jobs:
            return None
        jobs.sort(key=lambda x: x[0])
        return jobs[0][1]

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
                    proc = subprocess.run(
                        cmd,
                        stdout=out_f,
                        stderr=err_f,
                        cwd=job_cwd,
                        env=run_env,
                        timeout=job_timeout,
                    )
                state.exit_code = proc.returncode
                if proc.returncode == 0:
                    state.status = JobStatus.DONE
                    dest_status = "done"
                else:
                    state.status = JobStatus.FAILED
                    state.failure_phase = "execution"
                    state.error_message = f"Process exited with code {proc.returncode}"
                    dest_status = "failed"
            except subprocess.TimeoutExpired:
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
                "started_at": state.started_at,
                "finished_at": state.finished_at,
                "exit_code": state.exit_code,
                "failure_phase": state.failure_phase,
                "error_message": state.error_message,
                "warnings": state.warnings if state.warnings else None,
            }
            (job_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))

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
